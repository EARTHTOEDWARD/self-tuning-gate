"""Entry-point friendly wrapper around SelfTuningCore."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from .phase_energy import PhaseEnergyGate
from ..self_tune.core import SelfTuningCore


@dataclass
class AutoTuneGate:
    base: dict | None = None
    tuner: dict | None = None

    def __post_init__(self):
        self._base = PhaseEnergyGate(**(self.base or {}))
        self._core = SelfTuningCore(**(self.tuner or {}))

    def step(self, R_t, phi_t, **signals) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """Return Γ plus optional knob/log dicts if signals provided."""
        gamma_phase = self._base.step(R_t, phi_t)
        if not signals:
            return gamma_phase, {}, {}
        gamma, knobs, logs = self._core.step(
            mae=float(signals.get("mae", 0.0)),
            lam1=float(signals.get("lambda1", 0.0)),
            te=float(signals.get("te", 0.0)),
            gamma_phase=float(gamma_phase),
            ctx_key=signals.get("ctx_key", "default"),
            te_mode=signals.get("te_mode", "explore"),
            valence=float(signals.get("valence", 0.0)) if signals.get("valence") is not None else None,
        )
        return gamma, knobs, logs

    def state_dict(self):
        return {
            "base": self._base.state_dict(),
            "tuner": {
                "W": self._core.W,
                "S": self._core.S,
                "eta": self._core.eta,
                "lam_band": self._core.lam_band,
            },
        }
