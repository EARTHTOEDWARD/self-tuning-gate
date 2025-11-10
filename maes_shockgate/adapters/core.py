from dataclasses import dataclass
import numpy as np

@dataclass
class NormalizerRMS:
    eps: float = 1e-6
    decay: float = 0.999
    r2: float = 1.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        self.r2 = self.decay * self.r2 + (1 - self.decay) * float(np.mean(x * x))
        scale = (self.r2 + self.eps) ** 0.5
        return x / scale

@dataclass
class Stride:
    k: int = 1
    _i: int = 0

    def allow(self) -> bool:
        self._i += 1
        if self.k <= 1:
            return True
        return (self._i % self.k) == 0

def build_adapters(specs):
    """Return (normalizer, stride) tuple configured from spec strings."""
    norm = None
    stride = Stride(1)
    for s in (specs or []):
        if s.startswith("normalize"):
            norm = NormalizerRMS()
        elif s.startswith("stride:"):
            stride = Stride(int(s.split(":")[1]))
    return norm, stride
