from dataclasses import dataclass
import numpy as np

@dataclass
class SRNNContract:
    H: int = 128
    spectral_radius: float = 0.95
    leak: float = 0.1
    input_scale: float = 0.7
    seed: int = 0

    def __post_init__(self):
        rng = np.random.RandomState(self.seed)
        W = rng.randn(self.H, self.H) / (self.H ** 0.5)
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        W = (self.spectral_radius / (s[0] + 1e-12)) * W
        self.W = W
        self.Win = rng.randn(self.H, 0)
        self.h = np.zeros(self.H)
        self._last_preact = np.zeros(self.H)
        self._din = None

    def step(self, x_t):
        x = x_t.astype(float).ravel()
        if self._din is None:
            self._din = x.shape[0]
            rng = np.random.RandomState(self.seed + 3)
            self.Win = self.input_scale * rng.randn(self.H, self._din)
        pre = self.W @ self.h + self.Win @ x
        self._last_preact = pre
        h_new = (1 - self.leak) * self.h + self.leak * np.tanh(pre)
        self.h = h_new
        return self.h.copy()

    def state_dict(self):
        return {"H": self.H, "spectral_radius": self.spectral_radius, "leak": self.leak}

    def metrics(self):
        return {"||h||": float(np.linalg.norm(self.h))}

    def jvp(self, v):
        d = 1.0 - np.tanh(self._last_preact) ** 2
        return (1 - self.leak) * v + self.leak * (d * (self.W @ v))

    def set_spectral_radius(self, target: float):
        target = float(max(0.0, target))
        u, s, vh = np.linalg.svd(self.W, full_matrices=False)
        self.W = (target / (s[0] + 1e-12)) * self.W
        self.spectral_radius = target

    def set_leak(self, leak: float):
        self.leak = float(np.clip(leak, 1e-4, 0.999))
