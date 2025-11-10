import numpy as np
try:
    from scipy.signal import hilbert as _hilbert
except Exception:
    _hilbert = None

def analytic_phase(x: np.ndarray) -> float:
    if _hilbert is None:
        if len(x) < 2:
            return 0.0
        dx = x[-1] - x[-2]
        return float(np.arctan2(dx, x[-1]))
    z = _hilbert(x)
    return float(np.angle(z[-1]))
