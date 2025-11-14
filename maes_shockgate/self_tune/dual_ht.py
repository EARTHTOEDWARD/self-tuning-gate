"""Dual-subspace tracker + heavy-tail thermostat pack."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional
import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


class TailExponentK:
    """Rolling CCDF tail estimator, returns (k, confidence)."""

    def __init__(self, maxlen: int = 4096, q_tail: float = 0.8, min_tail: int = 50):
        self.buf = deque(maxlen=maxlen)
        self.q_tail = q_tail
        self.min_tail = min_tail

    def update(self, vals):
        v = np.asarray(vals, dtype=float).ravel()
        v = v[np.isfinite(v)]
        for x in v:
            self.buf.append(abs(float(x)))

    def estimate(self):
        if len(self.buf) < self.min_tail:
            return float("nan"), 0.0
        x = np.array(self.buf, dtype=float)
        x = x[x > 0]
        if x.size < self.min_tail:
            return float("nan"), 0.0
        xmin = np.quantile(x, self.q_tail)
        tail = np.sort(x[x >= xmin])
        if tail.size < self.min_tail:
            return float("nan"), 0.0
        ccdf = 1.0 - (np.arange(1, tail.size + 1) / (tail.size + 1))
        mask = ccdf > 0
        lx = np.log(tail[mask])
        lc = np.log(ccdf[mask])
        if lx.size < 10:
            return float("nan"), 0.0
        slope, intercept = np.polyfit(lx, lc, 1)
        pred = slope * lx + intercept
        r2 = 1.0 - np.sum((lc - pred) ** 2) / (np.sum((lc - lc.mean()) ** 2) + 1e-12)
        return float(-slope), float(_clip(r2, 0.0, 1.0))


class OnlinePCA:
    """Simple online covariance tracker to extract top PCs."""

    def __init__(self, dim: int, n: int = 8, decay: float = 0.995):
        self.dim = dim
        self.n = n
        self.decay = decay
        self.mu = np.zeros(dim)
        self.C = np.eye(dim) * 1e-6

    def update(self, x):
        x = np.asarray(x, dtype=float).ravel()
        self.mu = self.decay * self.mu + (1 - self.decay) * x
        xc = x - self.mu
        self.C = self.decay * self.C + (1 - self.decay) * np.outer(xc, xc)

    def pcs(self):
        w, v = np.linalg.eigh(self.C)
        idx = np.argsort(w)[::-1][: self.n]
        return v[:, idx].T  # shape (n, dim)


class DualSubspaceTracker:
    """Maintains cross-covariance between boundary PCs and update proxy, returns param directions."""

    def __init__(self, r: int = 4, maxlen: int = 1024):
        self.r = r
        self.maxlen = maxlen
        self.X = []
        self.G = []

    def update(self, x_pc, g_vec):
        self.X.append(np.asarray(x_pc, dtype=float))
        self.G.append(np.asarray(g_vec, dtype=float))
        if len(self.X) > self.maxlen:
            self.X = self.X[-self.maxlen :]
            self.G = self.G[-self.maxlen :]

    def compute(self):
        if len(self.X) < 64:
            return None
        X = np.stack(self.X, axis=0)
        G = np.stack(self.G, axis=0)
        X = X - X.mean(axis=0, keepdims=True)
        G = G - G.mean(axis=0, keepdims=True)
        C = X.T @ G / max(1, len(X) - 1)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        V_param = Vt[: self.r, :]
        explained = float(np.sum(S[: self.r]) / (np.sum(S) + 1e-12))
        return {"V_param": V_param, "explained": explained}


@dataclass
class DSTHTPack:
    """Dual-subspace tracker + heavy-tail thermostat."""

    n_pc: int = 8
    r: int = 4
    k_band: tuple = (0.8, 1.2)
    k_gain: float = 0.08
    gamma_clip: tuple = (0.8, 1.25)
    q_tail: float = 0.8
    min_tail: int = 50
    explained_floor: float = 0.0

    _pca: Optional[OnlinePCA] = None
    _dst: DualSubspaceTracker = field(default_factory=lambda: DualSubspaceTracker())
    _tail_estimators: list[TailExponentK] = field(default_factory=list)
    last_explained: float = 0.0

    def _ensure(self, dim_x: int, dim_g: int):
        if self._pca is None:
            self._pca = OnlinePCA(dim_x, n=self.n_pc)
        if not self._tail_estimators:
            self._tail_estimators = [
                TailExponentK(q_tail=self.q_tail, min_tail=self.min_tail) for _ in range(self.r)
            ]

    def update(
        self,
        x_t,
        latent_t,
        y_true: float,
        y_hat: float,
        grad_vec=None,
        knobs: Optional[Dict[str, float]] = None,
    ):
        x = np.asarray(x_t, dtype=float).ravel()
        latent = np.asarray(latent_t, dtype=float).ravel()
        error = float(y_true - y_hat)
        g = np.asarray(grad_vec, dtype=float).ravel() if grad_vec is not None else (error * latent)

        self._ensure(x.size, g.size)
        self._pca.update(x)
        pcs = self._pca.pcs()
        x_pc = pcs @ x
        self._dst.update(x_pc, g)

        result = self._dst.compute()
        gamma_scale = 1.0
        k_avg = float("nan")
        k_conf = 0.0
        explained = self.last_explained
        nudges = {}

        if result is not None:
            Vparam = result["V_param"]
            explained = result["explained"]
            self.last_explained = explained
            ks = []
            r2s = []
            for i in range(Vparam.shape[0]):
                proj = float(np.dot(Vparam[i], g))
                self._tail_estimators[i].update([abs(proj)])
                k_i, r2_i = self._tail_estimators[i].estimate()
                ks.append(k_i)
                r2s.append(r2_i)
            good = [k for k, r2 in zip(ks, r2s) if np.isfinite(k) and r2 > 0.3]
            if good and explained >= self.explained_floor:
                k_med = float(np.median(good))
                k_avg = k_med
                k_conf = float(np.mean([r2 for r2 in r2s if r2 > 0.3]))
                k_lo, k_hi = self.k_band
                if k_med < k_lo or k_med > k_hi:
                    dev = (k_med - 0.5 * (k_lo + k_hi)) / (0.5 * (k_hi - k_lo) + 1e-6)
                    gamma_scale = _clip(1.0 - self.k_gain * dev, *self.gamma_clip)
                    if knobs is not None and explained >= self.explained_floor:
                        cool = k_med < k_lo
                        sign = 1.0 if cool else -1.0
                        if knobs.get("spectral_radius") is not None:
                            nudges["spectral_radius"] = float(
                                _clip(knobs["spectral_radius"] * (1.0 - 0.05 * sign), 0.1, 1.2)
                            )
                        if knobs.get("horizon") is not None:
                            nudges["horizon"] = int(max(1, round(knobs["horizon"] * (1.0 - 0.1 * sign))))
                        if knobs.get("dropout") is not None:
                            nudges["dropout"] = float(
                                _clip(knobs["dropout"] + 0.02 * sign, 0.0, 0.5)
                            )
                        if knobs.get("attn_temp") is not None:
                            nudges["attn_temp"] = float(
                                _clip(knobs["attn_temp"] * (1.0 - 0.05 * sign), 0.1, 5.0)
                            )

        return {
            "gamma_scale": gamma_scale,
            "explained": explained,
            "k_avg": k_avg,
            "k_conf": k_conf,
            "nudge": nudges,
        }
