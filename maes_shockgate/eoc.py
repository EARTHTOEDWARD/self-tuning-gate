# MIT License
# Edge-of-chaos control utilities for the Self-Tuning Gate toy experiments.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import numpy as np

Array = np.ndarray


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
@dataclass
class EdgeOfChaosConfig:
    """Tunable parameters for the edge-of-chaos control loop."""

    enabled: bool = False
    period: int = 5
    target_low: float = 0.0
    target_high: float = 0.2
    lambda_ema: float = 0.95
    sigma_ema: float = 0.9
    sigma_a: float = 3.0
    sigma_b: float = 0.5
    apply_to: Tuple[str, ...] = ("u", "input")
    perception: str = "temporal_unsharp"  # gain | temporal_blur | temporal_unsharp | none
    lr_gate: bool = False
    lr_outband_scale: float = 0.3


# ------------------------------------------------------------------------------
# Adaptive perception for streaming features
# ------------------------------------------------------------------------------
class AdaptivePreprocessor1D:
    """
    Per-key temporal preprocessing controlled by scalar sigma>0.
    - gain            : y = tanh(x * (1/sigma))
    - temporal_blur   : y = EMA_sigma(x)  with alpha = sigma / (1 + sigma)
    - temporal_unsharp: y = x + (1/sigma) * (x - EMA_sigma(x))
    """

    def __init__(self, mode: str = "temporal_unsharp"):
        if mode not in ("gain", "temporal_blur", "temporal_unsharp", "none"):
            raise ValueError(f"Unsupported adaptive perception mode '{mode}'")
        self.mode = mode
        self._ema: Dict[str, Array] = {}

    @staticmethod
    def _alpha_from_sigma(sigma: float) -> float:
        return float(sigma / (1.0 + sigma))

    def reset(self):
        self._ema.clear()

    def _ensure_state(self, key: str, x: Array):
        if key not in self._ema:
            self._ema[key] = np.copy(x)

    def __call__(self, key: str, x: Array, sigma: float) -> Array:
        if self.mode == "none":
            return x
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            x = x.reshape(-1)

        if self.mode == "gain":
            g = 1.0 / (1e-6 + sigma)
            return np.tanh(x * g)

        self._ensure_state(key, x)
        alpha = self._alpha_from_sigma(sigma)
        ema = self._ema[key]
        ema = alpha * ema + (1.0 - alpha) * x
        self._ema[key] = ema

        if self.mode == "temporal_blur":
            return ema
        k = 1.0 / (1e-6 + sigma)
        return x + k * (x - ema)


# ------------------------------------------------------------------------------
# Local Lyapunov estimator via power iteration on JVPs
# ------------------------------------------------------------------------------
class LocalLyapunovPower:
    """Tracks the local largest Lyapunov exponent with a one-vector power iteration."""

    def __init__(self, dim: int, ema: float = 0.95, seed: int = 0):
        self.dim = int(dim)
        rng = np.random.RandomState(seed)
        v = rng.randn(dim)
        self.v = v / (np.linalg.norm(v) + 1e-9)
        self.log_ema = 0.0
        self.ema = float(ema)

    def update(self, jvp_fn) -> float:
        w = jvp_fn(self.v)
        gamma = float(np.linalg.norm(w) + 1e-12)
        logg = math.log(gamma)
        self.log_ema = self.ema * self.log_ema + (1.0 - self.ema) * logg
        self.v = w / gamma
        return float(self.log_ema)


# ------------------------------------------------------------------------------
# Sigma controller + optional LR gate
# ------------------------------------------------------------------------------
class SigmaController:
    def __init__(self, a: float, b: float, ema: float):
        self.a = float(a)
        self.b = float(b)
        self.ema = float(ema)
        self.sigma_ema = 1.0

    def step(self, lambda1: float, lo: float, hi: float) -> float:
        if lambda1 < lo:
            s = -(lo - lambda1) / max(1e-6, (hi - lo))
        elif lambda1 > hi:
            s = (lambda1 - hi) / max(1e-6, (hi - lo))
        else:
            s = 0.0
        raw = self.a * s + self.b
        sigma = math.log1p(math.exp(raw)) + 1e-3
        self.sigma_ema = self.ema * self.sigma_ema + (1.0 - self.ema) * sigma
        return float(self.sigma_ema)


class LearningRateGate:
    def __init__(self, outband_scale: float = 0.3):
        self.outband_scale = float(outband_scale)

    def apply(self, base_lr: float, lambda1: float, lo: float, hi: float) -> float:
        out = (lambda1 < lo) or (lambda1 > hi)
        return base_lr * (self.outband_scale if out else 1.0)


# ------------------------------------------------------------------------------
# Reservoir tangent (J @ v) utilities
# ------------------------------------------------------------------------------
def _get_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"Could not find any of {names} on {type(obj).__name__}")


def esn_tangent_jvp_factory(reservoir, x: Array, u: Array):
    """
    Build a JVP closure for the leaky tanh ESN step at (x, u):
        x' = (1-γ) x + γ tanh(W x + U u + b)
        J  = (1-γ) I + γ diag(1 - tanh(a)^2) W
        Jv = (1-γ) v + γ * ((1 - tanh(a)^2) * (W v))
    """

    W = _get_attr(reservoir, "W", "W_res", "W_reservoir")
    Win = _get_attr(reservoir, "Win", "W_in", "U")
    b = _get_attr(reservoir, "bias", "b", "bias_vec", "biases")
    leak = float(_get_attr(reservoir, "leak", "alpha", "leak_rate"))

    a = W.dot(x) + Win.dot(u) + b
    t = np.tanh(a)
    s = (1.0 - t * t)
    one_minus = (1.0 - leak)

    def jvp(v: Array) -> Array:
        return one_minus * v + leak * (s * (W.dot(v)))

    return jvp


# ------------------------------------------------------------------------------
# Edge-of-chaos orchestrator hook
# ------------------------------------------------------------------------------
class EdgeOfChaosHook:
    """
    Adapts selected stream keys, estimates local λ₁, nudges sigma, and optionally gates LR.
    """

    def __init__(self, cfg: EdgeOfChaosConfig, reservoir, dim: Optional[int] = None):
        self.cfg = cfg
        self.reservoir = reservoir
        self.percept = AdaptivePreprocessor1D(cfg.perception)
        self.sigma_ctrl = SigmaController(cfg.sigma_a, cfg.sigma_b, cfg.sigma_ema)
        self.lr_gate = LearningRateGate(cfg.lr_outband_scale) if cfg.lr_gate else None
        if dim is None:
            W = _get_attr(reservoir, "W", "W_res", "W_reservoir")
            dim = int(W.shape[0])
        self.lyap = LocalLyapunovPower(dim=dim, ema=cfg.lambda_ema)
        self.step_idx = 0
        self.lambda1: float = 0.0
        self.sigma: float = 1.0
        self.history: Dict[str, list] = {"lambda1": [], "sigma": []}

    def apply_perception(self, frame: Dict[str, Array]) -> Dict[str, Array]:
        if not self.cfg.enabled:
            return frame
        for k in self.cfg.apply_to:
            if k in frame and frame[k] is not None:
                frame[k] = self.percept(k, np.asarray(frame[k]), self.sigma)
        return frame

    def _make_jvp(self, x: Array, u: Array):
        if hasattr(self.reservoir, "tangent_jvp"):
            def wrap(v):
                return self.reservoir.tangent_jvp(x, u, v)

            return wrap
        if hasattr(self.reservoir, "jvp"):
            return self.reservoir.jvp
        return esn_tangent_jvp_factory(self.reservoir, np.asarray(x), np.asarray(u))

    def update(self, x: Array, u: Array, base_lr: Optional[float] = None) -> Optional[float]:
        if not self.cfg.enabled:
            return None
        self.step_idx += 1
        if (self.step_idx % max(1, self.cfg.period)) != 0:
            return None
        jvp = self._make_jvp(np.asarray(x), np.asarray(u))
        lam = self.lyap.update(jvp)
        self.lambda1 = float(lam)
        self.sigma = self.sigma_ctrl.step(self.lambda1, self.cfg.target_low, self.cfg.target_high)
        self.history["lambda1"].append(self.lambda1)
        self.history["sigma"].append(self.sigma)
        if self.lr_gate is not None and base_lr is not None:
            return self.lr_gate.apply(base_lr, self.lambda1, self.cfg.target_low, self.cfg.target_high)
        return None

    def poised(self) -> bool:
        return self.cfg.target_low <= self.lambda1 <= self.cfg.target_high
