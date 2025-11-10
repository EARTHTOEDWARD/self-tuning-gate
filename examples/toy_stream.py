"""Minimal example showing manual wiring of the tuner."""

import numpy as np

from maes_shockgate.entrypoints import load
from maes_shockgate.self_tune.core import SelfTuningCore

if __name__ == "__main__":
    contract = load("maes.contracts", "reservoir_esn", H=64)
    gate = load("maes.gates", "phase_energy_blend", alpha=0.6, beta=0.2)
    tuner = SelfTuningCore()
    rng = np.random.RandomState(0)
    gamma = 0.5
    for t in range(10):
        x = rng.randn(3)
        R = contract.step(x)
        phi = 0.1 * t
        gamma_phase = gate.step(R, phi)
        lam = rng.randn() * 0.05
        te = rng.rand() * 0.2
        mae = rng.rand() * 0.1
        gamma, _, _ = tuner.step(mae=mae, lam1=lam, te=te, gamma_phase=gamma_phase,
                                 ctx_key="demo", te_mode="explore", valence=-0.1)
    print("Final Gamma", gamma)
