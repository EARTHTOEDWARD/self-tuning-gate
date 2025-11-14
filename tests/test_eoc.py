import numpy as np

from maes_shockgate.eoc import (
    EdgeOfChaosConfig,
    EdgeOfChaosHook,
    esn_tangent_jvp_factory,
)


class _ToyESN:
    def __init__(self, N=16, M=3, leak=0.2, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(N, N) * 0.2
        self.Win = rng.randn(N, M) * 0.5
        self.bias = rng.randn(N) * 0.01
        self.leak = leak
        self.N = N
        self._state = np.zeros(N)

    def step_state(self, h, u):
        u = np.asarray(u, dtype=float).ravel()
        pre = self.W.dot(h) + self.Win.dot(u) + self.bias
        return (1 - self.leak) * h + self.leak * np.tanh(pre)

    def step(self, u):
        self._state = self.step_state(self._state, u)
        return self._state.copy()


def test_esn_tangent_matches_finite_difference():
    esn = _ToyESN()
    x = np.zeros(esn.N)
    u = np.ones(esn.Win.shape[1]) * 0.1
    v = np.random.RandomState(1).randn(esn.N)
    v /= np.linalg.norm(v) + 1e-9
    jvp = esn_tangent_jvp_factory(esn, x, u)
    w_analytic = jvp(v)
    eps = 1e-6
    x2 = esn.step_state(x + eps * v, u)
    x1 = esn.step_state(x, u)
    w_fd = (x2 - x1) / eps
    rel_err = np.linalg.norm(w_analytic - w_fd) / (np.linalg.norm(w_fd) + 1e-12)
    assert rel_err < 1e-4


def test_edge_of_chaos_hook_updates_lambda_and_sigma():
    esn = _ToyESN()
    cfg = EdgeOfChaosConfig(enabled=True, period=1, perception="gain")
    hook = EdgeOfChaosHook(cfg, esn, dim=esn.N)
    x = np.zeros(esn.N)
    u = np.zeros(esn.Win.shape[1])
    frame = {"u": u.copy()}
    frame = hook.apply_perception(frame)
    esn.step(frame["u"])
    hook.update(x, frame["u"])
    assert np.isfinite(hook.lambda1)
    assert hook.sigma > 0.0
