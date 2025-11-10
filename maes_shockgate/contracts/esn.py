from dataclasses import dataclass
import numpy as np

def _spectral_scale(W, target):
    v = np.random.randn(W.shape[0])
    for _ in range(25):
        v = W @ v
        n = np.linalg.norm(v) + 1e-12
        v = v / n
    lam = np.linalg.norm(W @ v) / (np.linalg.norm(v) + 1e-12)
    if lam > 0:
        W *= (target / lam)
    return W

@dataclass
class ESNContract:
    H: int = 256
    spectral_radius: float = 0.9
    leak: float = 0.2
    input_scale: float = 0.6
    density: float = 0.1
    seed: int = 0

    def __post_init__(self):
        rng = np.random.RandomState(self.seed)
        mask = rng.rand(self.H, self.H) < self.density
        W = rng.randn(self.H, self.H) * mask
        self.W = _spectral_scale(W, self.spectral_radius)
        self.Win = rng.randn(self.H, 0)
        self.b = np.zeros(self.H)
        self.h = np.zeros(self.H)
        self._last_preact = np.zeros(self.H)
        self._din = None

    def step(self, x_t):
        x = x_t.astype(float).ravel()
        if self._din is None:
            self._din = x.shape[0]
            rng = np.random.RandomState(self.seed + 1)
            self.Win = self.input_scale * rng.randn(self.H, self._din)
        pre = self.W @ self.h + self.Win @ x + self.b
        self._last_preact = pre
        h_new = (1 - self.leak) * self.h + self.leak * np.tanh(pre)
        self.h = h_new
        return self.h.copy()

    def state_dict(self):
        return {
            "H": self.H,
            "spectral_radius": self.spectral_radius,
            "leak": self.leak,
            "input_scale": self.input_scale,
            "density": self.density,
        }

    def metrics(self):
        return {"||h||": float(np.linalg.norm(self.h))}

    def jvp(self, v):
        d = 1.0 - np.tanh(self._last_preact) ** 2
        return (1 - self.leak) * v + self.leak * (d * (self.W @ v))

    def set_spectral_radius(self, target: float):
        target = float(max(0.0, target))
        self.W = _spectral_scale(self.W, target)
        self.spectral_radius = target

    def set_leak(self, leak: float):
        self.leak = float(np.clip(leak, 1e-4, 0.999))
