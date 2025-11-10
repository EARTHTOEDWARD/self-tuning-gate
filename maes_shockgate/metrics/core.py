import numpy as np
from collections import deque

class MetricTracker:
    def __init__(self, te_window=200):
        self._mae_sum = 0.0
        self._mae_n = 0
        self._lyap_sum = 0.0
        self._lyap_n = 0
        self._v = None
        self._te_buf = deque(maxlen=te_window)

    def update_mae(self, y, yhat):
        e = np.abs(np.asarray(y) - np.asarray(yhat))
        mae = float(np.mean(e))
        self._mae_sum += mae
        self._mae_n += 1
        return mae

    def update_lyap1(self, contract):
        H = contract.h.shape[0]
        if self._v is None:
            rng = np.random.RandomState(123)
            self._v = rng.randn(H)
            self._v /= (np.linalg.norm(self._v) + 1e-12)
        w = contract.jvp(self._v)
        ln = np.linalg.norm(w) + 1e-12
        self._v = w / ln
        lam = float(np.log(ln))
        self._lyap_sum += lam
        self._lyap_n += 1
        return lam

    def update_te(self, s_prev, v_prev, v_t, bins=16, eps=1e-3):
        self._te_buf.append((float(s_prev), float(v_prev), float(v_t)))
        if len(self._te_buf) < 32:
            return float("nan")
        arr = np.array(self._te_buf)
        sp, vp, vt = arr[:, 0], arr[:, 1], arr[:, 2]

        def hist3(a, b, c):
            H, _ = np.histogramdd(np.stack([a, b, c], axis=1), bins=bins)
            P = (H + eps) / (np.sum(H) + eps * (bins ** 3))
            return P

        def hist2(a, b):
            H, _ = np.histogramdd(np.stack([a, b], axis=1), bins=bins)
            P = (H + eps) / (np.sum(H) + eps * (bins ** 2))
            return P

        P_vt_vp_sp = hist3(vt, vp, sp)
        P_vt_vp = hist2(vt, vp)[:, :, None]
        P_vp_sp = hist2(vp, sp)[None, :, :]
        P_vp = hist2(vp, vp)[0, :, None]
        ratio = (P_vt_vp_sp / np.maximum(P_vp_sp, eps)) / np.maximum(P_vt_vp / np.maximum(P_vp, eps), eps)
        te = float(np.sum(P_vt_vp_sp * np.log(np.maximum(ratio, eps))))
        return te

    def snapshot(self):
        return {
            "MAE": (self._mae_sum / max(1, self._mae_n)),
            "lambda1": (self._lyap_sum / max(1, self._lyap_n)),
        }
