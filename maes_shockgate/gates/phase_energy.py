from dataclasses import dataclass
import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

@dataclass
class PhaseEnergyGate:
    alpha: float = 0.7
    beta: float = 0.2
    seed: int = 0

    def __post_init__(self):
        rng = np.random.RandomState(self.seed + 42)
        self.we = rng.randn(0)
        self.be = 0.0
        self.wp = rng.randn(2) * 0.5
        self.bp = 0.0
        self._Gamma = 0.0

    def step(self, R_t, phi_t):
        R = R_t.ravel()
        if self.we.size == 0:
            rng = np.random.RandomState(self.seed + 99)
            scale = max(1.0, (R.shape[0] ** 0.5))
            self.we = rng.randn(R.shape[0]) / scale
        sinc = np.array([np.sin(phi_t), np.cos(phi_t)], dtype=float)
        e_t = _sigmoid(self.we @ R + self.be)
        p_t = _sigmoid(self.wp @ sinc + self.bp)
        g = self.alpha * e_t + (1 - self.alpha) * p_t
        g = float(np.clip(g, 0.0, 1.0))
        self._Gamma = self.beta * self._Gamma + (1 - self.beta) * g
        return float(self._Gamma)

    def state_dict(self):
        return {"alpha": self.alpha, "beta": self.beta}

    def metrics(self):
        return {"Gamma": float(self._Gamma)}
