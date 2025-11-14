"""Lyapunov/TE-aware Gamma autotuning core."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import math
import numpy as np


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class SelfTuningCore:
    """Maintains Γ by watching MAE, λ₁, and TE spikes."""

    W: int = 256
    S: int = 15
    eta: float = 0.05
    eps: float = 1e-4
    k_early: int = 4
    alpha_obj: float = 1.0
    beta_obj: float = 1.0
    mu_smooth: float = 0.05
    w_phase: float = 0.5
    gamma_smooth: float = 0.2
    lam_band: tuple = (0.0, 0.2)
    shock_budget: float = 0.10
    shock_high: float = 0.6
    lr_outband_scale: float = 0.25
    attn_temp_k: float = 0.5
    explore_k: float = 0.2
    leak_k: float = 0.8
    rho_k: float = 0.6
    dropout_k: float = 0.2
    horizon_k: float = 0.4
    lambda_soc: float = 0.0

    _gamma: float = 0.5
    _last_gamma: float = 0.5
    _last_mae: Optional[float] = None
    _te_buf: deque = field(default_factory=lambda: deque(maxlen=512))
    _shock_used: int = 0
    _steps: int = 0
    ctx_cache: dict = field(default_factory=dict)

    def step(self, mae: float, lam1: float, te: Optional[float], gamma_phase: float,
             ctx_key: str, te_mode: str = "explore", valence: Optional[float] = None,
             soc_loss: Optional[float] = None):
        if math.isfinite(te):
            self._te_buf.append(float(te))
        q95 = self._quantile()
        te_spike = math.isfinite(te) and (te > q95)
        trigger = (lam1 > 0.0) or te_spike

        gamma_auto = self._gamma
        if trigger and not self._over_budget():
            slope = 0.0
            if self._last_mae is not None and abs(self._gamma - self._last_gamma) > 1e-6:
                slope = (mae - self._last_mae) / (self._gamma - self._last_gamma + 1e-12)

            def mae_pred(g):
                return float(mae + slope * (g - self._gamma))

            def stability_margin(g, lam):
                return ( _relu(-lam) - _relu(lam)) * g

            def J(g):
                g = _clip01(g)
                te_term = (+1.0 if te_mode == "explore" else -1.0) * (te if math.isfinite(te) else 0.0)
                stab = stability_margin(g, lam1)
                chg = (g - self._gamma) ** 2
                soc_term = 0.0
                if self.lambda_soc > 0.0 and soc_loss is not None and math.isfinite(soc_loss):
                    soc_term = self.lambda_soc * max(0.0, soc_loss) * g
                return (
                    mae_pred(g)
                    + self.alpha_obj * te_term
                    - self.beta_obj * stab
                    + self.mu_smooth * chg
                    + self._budget_penalty()
                    + soc_term
                )

            best_g, bestJ = self._gamma, float("inf")
            for g in np.linspace(0.0, 1.0, self.S):
                Jg = J(g)
                if Jg < bestJ:
                    best_g, bestJ = g, Jg

            gcur = self._gamma
            grad = (J(min(1.0, gcur + 1e-2)) - J(max(0.0, gcur - 1e-2))) / 2e-2
            gamma_auto = _clip01(gcur - self.eta * grad)
            if J(gamma_auto) > bestJ:
                gamma_auto = best_g

        gamma_blend = _clip01(self.w_phase * gamma_phase + (1.0 - self.w_phase) * gamma_auto)
        self._gamma = self.gamma_smooth * self._gamma + (1.0 - self.gamma_smooth) * gamma_blend

        if trigger and self._gamma > self.shock_high:
            self._shock_used += 1
        self._steps += 1
        self.ctx_cache[ctx_key] = self._gamma
        self._last_mae, self._last_gamma = float(mae), float(self._gamma)

        knobs = self._knobs_from_lambda(lam1, scale=float(self._gamma))
        logs = {
            "TE_q95": q95 if math.isfinite(q95) else float("nan"),
            "budget_used": self._shock_used / max(1, self._steps),
            "soc_loss": float(soc_loss) if soc_loss is not None else float("nan"),
        }
        return float(self._gamma), knobs, logs

    # --- helpers ---
    def _quantile(self):
        if not self._te_buf:
            return float("inf")
        arr = np.array(self._te_buf, dtype=float)
        return float(np.quantile(arr, 0.95))

    def _over_budget(self):
        return (self._shock_used / max(1, self._steps)) > self.shock_budget

    def _budget_penalty(self):
        over = (self._shock_used / max(1, self._steps)) - self.shock_budget
        return 10.0 * _relu(over) ** 2

    def _knobs_from_lambda(self, lam1: float, scale: float = 1.0):
        lam_lo, lam_hi = self.lam_band
        if lam1 < lam_lo:
            d = lam1 - lam_lo
        elif lam1 > lam_hi:
            d = lam1 - lam_hi
        else:
            d = 0.0
        d_abs = abs(d)
        return {
            "lr_scale": 1.0 if d == 0.0 else (1.0 - self.lr_outband_scale * min(1.0, d_abs / 0.5)),
            "attn_temp": 1.0 + self.attn_temp_k * d_abs * scale,
            "explore_rate": max(0.0, 1.0 + self.explore_k * (-d)) * scale,
            "leak": None if d == 0.0 else float(np.clip(0.2 + self.leak_k * (-d), 0.05, 0.5)),
            "spectral_radius": None if d == 0.0 else float(np.clip(0.9 * np.exp(-self.rho_k * d), 0.4, 1.0)),
            "dropout": float(np.clip(self.dropout_k * max(0.0, d), 0.0, 0.3)),
            "horizon": None if d == 0.0 else int(max(1, round(10 * np.exp(-self.horizon_k * d))))
        }
