"""Optional plug-in packs for the self-tuning loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Dict, Any
import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _ema_update(val: Optional[float], x: float, a: float = 0.9) -> float:
    return (a * val + (1 - a) * x) if val is not None else x


def _dfa(signal: np.ndarray) -> float:
    """Tiny DFA/Hurst proxy using a handful of box sizes."""
    y = signal - np.mean(signal)
    cum = np.cumsum(y)
    box_sizes = np.array([8, 16, 32, 64, 128])
    box_sizes = box_sizes[box_sizes < len(cum) // 2]
    if len(box_sizes) < 3:
        return float("nan")
    flucts = []
    for n in box_sizes:
        segs = len(cum) // n
        if segs < 2:
            continue
        reshaped = cum[:segs * n].reshape(segs, n)
        t = np.arange(n)
        A = np.vstack([t, np.ones_like(t)]).T
        errs = []
        for row in reshaped:
            a, b = np.linalg.lstsq(A, row, rcond=None)[0]
            trend = a * t + b
            errs.append(np.sqrt(np.mean((row - trend) ** 2)))
        if errs:
            flucts.append(np.mean(errs))
    if len(flucts) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(box_sizes[:len(flucts)] + 1e-9), np.log(np.array(flucts) + 1e-9), 1)
    return float(slope)


@dataclass
class SOCPack:
    """Self-organized criticality thermostat."""

    window: int = 2048
    min_samples: int = 512
    tau_star: Optional[float] = None
    beta_star: Optional[float] = None
    hurst_star: Optional[float] = None
    band_tau: tuple = (1.0, 2.5)
    band_beta: tuple = (0.2, 1.8)
    band_hurst: tuple = (0.4, 0.8)
    k_gamma: float = 0.15

    _residuals: deque = field(default_factory=lambda: deque(maxlen=4096))
    _calibrated: bool = False

    def update(self, residual: float) -> dict:
        """Append new residual magnitude and return SOC diagnostics."""
        self._residuals.append(float(residual))
        if len(self._residuals) < min(self.window, self.min_samples):
            return {
                "gamma_scale": 1.0,
                "loss": 0.0,
                "tau": float("nan"),
                "beta": float("nan"),
                "H": float("nan"),
                "confident": False,
            }

        x = np.asarray(self._residuals, dtype=float)
        x = x[-self.window :]
        tau_hat = self._estimate_avalanche_tau(x)
        beta_hat = self._estimate_psd_beta(x)
        hurst_hat = _dfa(x)

        if not self._calibrated and np.isfinite(tau_hat) and np.isfinite(beta_hat) and np.isfinite(hurst_hat):
            self.tau_star = tau_hat if self.tau_star is None else self.tau_star
            self.beta_star = beta_hat if self.beta_star is None else self.beta_star
            self.hurst_star = hurst_hat if self.hurst_star is None else self.hurst_star
            self._calibrated = True

        dev_tau = self._norm_dev(tau_hat, self.band_tau, self.tau_star)
        dev_beta = self._norm_dev(beta_hat, self.band_beta, self.beta_star)
        dev_hurst = self._norm_dev(hurst_hat, self.band_hurst, self.hurst_star)

        soc_loss = dev_tau ** 2 + dev_beta ** 2 + dev_hurst ** 2
        dev_mean = (dev_tau + dev_beta + dev_hurst) / 3.0
        gamma_scale = _clip(1.0 - self.k_gamma * dev_mean, 0.7, 1.3)

        return {
            "gamma_scale": gamma_scale,
            "loss": float(soc_loss),
            "tau": tau_hat,
            "beta": beta_hat,
            "H": hurst_hat,
            "confident": self._calibrated and np.isfinite(soc_loss),
        }

    def _norm_dev(self, val: float, band: tuple, target: Optional[float]) -> float:
        if not np.isfinite(val) or target is None:
            return 0.0
        lo, hi = band
        rng = (hi - lo) + 1e-6
        return _clip((val - target) / rng, -1.0, 1.0)

    def _estimate_avalanche_tau(self, series: np.ndarray) -> float:
        thr = np.quantile(series, 0.95)
        runs = series > thr
        if not runs.any():
            return float("nan")
        padded = np.concatenate([[0], runs.astype(int), [0]])
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        sizes = ends - starts
        sizes = sizes[sizes > 0]
        if len(sizes) < 5:
            return float("nan")
        sizes.sort()
        tail = sizes[-max(5, len(sizes) // 5) :]
        grid = np.unique(tail)
        ccdf = np.array([np.mean(tail >= g) for g in grid]) + 1e-9
        slope, _ = np.polyfit(np.log(grid + 1e-9), np.log(ccdf), 1)
        return float(1 - slope)

    def _estimate_psd_beta(self, series: np.ndarray) -> float:
        centered = series - np.mean(series)
        f = np.fft.rfft(centered)
        ps = (f * np.conj(f)).real
        freqs = np.fft.rfftfreq(len(centered), d=1.0)
        mask = (freqs > 1 / 256) & (freqs < 0.25) & (ps > 0)
        if mask.sum() < 6:
            return float("nan")
        slope, _ = np.polyfit(np.log(freqs[mask]), np.log(ps[mask]), 1)
        return float(-slope)


@dataclass
class AtlasPack:
    """Maintains regime fingerprints and gently pulls knobs toward stored memories."""

    max_slots: int = 8
    dock_thresh: float = 2.0
    pull_k: float = 0.25
    ttl: int = 10_000
    allow_explore: bool = False

    memories: list[dict] = field(default_factory=list)

    def update(self, lam1: float, beta_psd: float, tau_aval: float, phase: float,
               mode: str, current_knobs: Dict[str, Any]) -> dict:
        """Return knob overrides plus diagnostics."""
        z = self._fingerprint(lam1, beta_psd, tau_aval, phase)
        for mem in self.memories:
            mem["age"] += 1
        self.memories = [m for m in self.memories if m["age"] <= self.ttl]

        best = None
        best_dist = float("inf")
        for mem in self.memories:
            delta = z - mem["mu"]
            dist = float(delta @ mem["cov_inv"] @ delta)
            if dist < best_dist:
                best = mem
                best_dist = dist

        overrides = {}
        docked = False
        mode_allows_action = (mode == "consolidate") or (self.allow_explore and mode == "explore")
        if best is not None and best_dist < self.dock_thresh and mode_allows_action:
            for key in ("leak", "spectral_radius", "attn_temp"):
                if key in best["knobs"] and current_knobs.get(key) is not None:
                    overrides[key] = (
                        (1 - self.pull_k) * current_knobs[key] + self.pull_k * best["knobs"][key]
                    )
            docked = True

        if mode_allows_action:
            self._store_memory(z, current_knobs)

        return {
            "overrides": overrides,
            "docked": docked,
            "best_dist": float(best_dist),
            "slots": len(self.memories),
        }

    def _fingerprint(self, lam1, beta, tau, phase):
        return np.array([
            float(lam1),
            float(beta) if np.isfinite(beta) else 0.0,
            float(tau) if np.isfinite(tau) else 0.0,
            np.sin(float(phase)),
            np.cos(float(phase)),
        ], dtype=float)

    def _store_memory(self, vector: np.ndarray, knobs: Dict[str, Any]):
        cov = np.eye(len(vector)) * 0.5
        cov_inv = np.linalg.inv(cov)
        mem = {
            "mu": vector,
            "cov_inv": cov_inv,
            "age": 0,
            "knobs": {k: knobs.get(k) for k in ("leak", "spectral_radius", "attn_temp")},
        }
        if len(self.memories) < self.max_slots:
            self.memories.append(mem)
        else:
            idx = np.argmax([m["age"] for m in self.memories]) if self.memories else 0
            self.memories[idx] = mem


@dataclass
class AutocatalysisPack:
    """Allows bursts of larger Γ after beneficial corrections, then decays."""

    tau_decay: float = 0.9
    k_sig: float = 2.0
    cap: float = 2.0

    c_t: float = 0.0
    last_mae: Optional[float] = None

    def update(self, mae: float, lambda1: float, last_action: str) -> dict:
        improved = False
        if self.last_mae is not None and last_action == "correct":
            improved = mae < self.last_mae
            if improved:
                delta = _clip((self.last_mae - mae) / (abs(self.last_mae) + 1e-6), 0.0, 1.0)
                lam_bonus = np.exp(-abs(lambda1))
                self.c_t = min(self.cap, self.c_t + 0.5 * delta * lam_bonus)
        self.c_t = self.tau_decay * self.c_t
        self.last_mae = float(mae)

        sig = 1.0 / (1.0 + np.exp(-self.k_sig * self.c_t))
        gamma_scale = _clip(1.0 + 0.3 * (sig - 0.5), 0.8, 1.3)
        return {
            "gamma_scale": float(gamma_scale),
            "c_t": float(self.c_t),
            "improved": bool(improved),
        }


@dataclass
class HazardPack:
    """Hazard-shaped scheduling for shocks using Pareto tail on inter-shock intervals."""

    min_samples: int = 5
    tail_frac: float = 0.3
    kappa: float = 1.0
    threshold: float = 0.3
    gamma_min: float = 0.5
    lambda_band: float = 0.15
    floor: float = 0.0
    deadline_quantile: float = 0.8
    deadline_prob: float = 0.05
    warmup_steps: int = 0
    maxlen: int = 256

    intervals: deque = field(default_factory=lambda: deque(maxlen=256))
    streak: int = 0
    steps: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def update(self, request_shock: bool, gamma: float, lambda1: float) -> dict:
        """Return hazard scale and whether to permit the requested shock."""
        self.steps += 1
        self.streak += 1
        alpha = float("nan")
        tail_len = 0
        hazard_scale = 0.0

        if len(self.intervals) >= self.min_samples:
            tail_len = max(3, int(len(self.intervals) * self.tail_frac))
            tail = np.sort(np.array(self.intervals, dtype=float))[-tail_len:]
            x_min = max(1.0, float(np.min(tail)))
            logs = np.log(tail / x_min + 1e-9)
            denom = np.sum(logs)
            if denom > 0:
                alpha = tail_len / denom
                hazard = alpha / max(x_min, float(self.streak))
                hazard_scale = _clip(self.kappa * hazard, 0.0, 1.0)

        hazard_scale = max(self.floor, hazard_scale)
        deadline = None
        if len(self.intervals) >= self.min_samples:
            deadline = float(np.quantile(self.intervals, self.deadline_quantile))

        permit = True
        executed = False
        warmup = self.steps <= self.warmup_steps
        deadline_trigger = deadline is not None and self.streak >= max(1, int(deadline))
        deadline_override = deadline_trigger and (self.rng.random() < self.deadline_prob)

        if request_shock:
            if warmup:
                permit = True
            else:
                cond_gamma = gamma >= self.gamma_min
                cond_lambda = abs(lambda1) <= self.lambda_band
                cond_hazard = (len(self.intervals) < self.min_samples) or (hazard_scale >= self.threshold)
                permit = cond_gamma and cond_lambda and (cond_hazard or deadline_override)
            if permit:
                self.intervals.append(self.streak or 1)
                self.streak = 0
                executed = True

        return {
            "scale": hazard_scale,
            "permit": permit,
            "alpha": alpha,
            "tail": tail_len,
            "streak": self.streak,
            "executed": executed,
        }
