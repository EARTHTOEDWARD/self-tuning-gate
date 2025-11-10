"""Entry-point loader with local fallbacks for editable installs."""

import importlib.metadata as md

from .contracts.esn import ESNContract
from .contracts.srnn import SRNNContract
from .gates.phase_energy import PhaseEnergyGate
from .gates.autotune_gate import AutoTuneGate
from .policies.shockgate import ShockGatePolicy

_FALLBACKS = {
    "maes.contracts": {
        "reservoir_esn": ESNContract,
        "reservoir_srnn": SRNNContract,
    },
    "maes.gates": {
        "phase_energy_blend": PhaseEnergyGate,
        "lyap_te_autotune": AutoTuneGate,
    },
    "maes.policies": {
        "shockgate": ShockGatePolicy,
    },
}

def load(group: str, name: str, **kwargs):
    try:
        for ep in md.entry_points(group=group):
            if ep.name == name:
                return ep.load()(**kwargs)
    except Exception:
        pass
    if group in _FALLBACKS and name in _FALLBACKS[group]:
        return _FALLBACKS[group][name](**kwargs)
    raise KeyError(f"{name} not found in entry-point group {group}")
