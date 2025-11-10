import argparse
import yaml
import numpy as np

from .entrypoints import load
from .adapters.core import build_adapters
from .telemetry.writer import TelemetryWriter
from .metrics.core import MetricTracker
from .self_tune.core import SelfTuningCore


def _toy_stream(T=2000, seed=0):
    rng = np.random.RandomState(seed)
    for t in range(T):
        w = 0.02 + 0.005 * np.sin(0.0005 * t)
        phi = w * t
        x = np.array([
            np.sin(phi) + 0.1 * rng.randn(),
            np.cos(phi) + 0.1 * rng.randn(),
            0.5 * np.sin(0.33 * phi) + 0.1 * rng.randn(),
        ])
        y = 0.7 * np.sin(phi + 0.4) + 0.3 * np.sin(0.33 * phi + 0.2)
        yhat = 0.7 * np.sin(phi) + 0.3 * np.sin(0.33 * phi) + 0.1 * rng.randn()
        valence = -(abs(y - yhat))
        yield t, x, phi, float(valence), float(yhat), float(y)

def run_from_config(cfg):
    contract_cfg = cfg.get("contract", {})
    contract_params = {k: v for k, v in contract_cfg.items() if k not in {"type", "adapters"}}
    C = load("maes.contracts", contract_cfg["type"], **contract_params)

    gate_cfg = cfg.get("gate", {})
    gate_params = {k: v for k, v in gate_cfg.items() if k != "type"}
    G = load("maes.gates", gate_cfg["type"], **gate_params)

    policy_cfg = cfg.get("policy", {"type": "shockgate"})
    policy_params = {k: v for k, v in policy_cfg.items() if k != "type"}
    P = load("maes.policies", policy_cfg["type"], **policy_params)

    norm, stride = build_adapters(contract_cfg.get("adapters"))

    tel_cfg = cfg.get("telemetry", {})
    writer = TelemetryWriter(path=tel_cfg.get("path", "logs/run.jsonl"), fmt=tel_cfg.get("fmt", "jsonl"))

    mt = MetricTracker()
    stream = _toy_stream(T=cfg.get("steps", 2000), seed=cfg.get("seed", 0))

    st_cfg = cfg.get("self_tuning", {})
    enable_st = st_cfg.get("enabled", False)
    tuner = SelfTuningCore(**{k: v for k, v in st_cfg.items() if k != "enabled"}) if enable_st else None

    s_prev = 0.0
    v_prev = 0.0
    for t, x_t, phi_t, valence_t, yhat, y in stream:
        if not stride.allow():
            continue
        if norm is not None:
            x_t = norm.apply(x_t)
        R_t = C.step(x_t)

        gate_out = G.step(R_t, phi_t)
        if isinstance(gate_out, (list, tuple)) and len(gate_out) == 3:
            Gamma_phase, _, _ = gate_out
        else:
            Gamma_phase = float(gate_out)

        mae = mt.update_mae(y, yhat)
        lam = mt.update_lyap1(C)
        s_t = float(np.linalg.norm(R_t))
        te = mt.update_te(s_prev, v_prev, valence_t)
        s_prev, v_prev = s_t, valence_t

        phase_bin = int(((np.arctan2(np.sin(phi_t), np.cos(phi_t)) + np.pi) / (2 * np.pi)) * 12)
        mag_bin = int(min(9, max(0, np.linalg.norm(x_t) // 0.5)))
        ctx_key = f"phase{phase_bin}_mag{mag_bin}"
        te_mode = "explore" if valence_t < 0 else "consolidate"

        if enable_st:
            Gamma_final, knobs, st_logs = tuner.step(
                mae=float(mae),
                lam1=float(lam),
                te=float(te) if np.isfinite(te) else float("nan"),
                gamma_phase=float(Gamma_phase),
                ctx_key=ctx_key,
                te_mode=te_mode,
                valence=float(valence_t),
            )
            if knobs.get("leak") is not None and hasattr(C, "set_leak"):
                C.set_leak(knobs["leak"])
            if knobs.get("spectral_radius") is not None and hasattr(C, "set_spectral_radius"):
                C.set_spectral_radius(knobs["spectral_radius"])
        else:
            Gamma_final, knobs, st_logs = float(Gamma_phase), {}, {}

        action_t = P.step(valence_t, Gamma_final)

        row = {
            "t": t,
            "Gamma": float(Gamma_final),
            "Gamma_phase": float(Gamma_phase),
            "phi": float(phi_t),
            "||x||": float(np.linalg.norm(x_t)),
            "||h||": float(np.linalg.norm(R_t)),
            "valence": float(valence_t),
            "action": action_t,
            "MAE": mae,
            "lambda1": lam,
            "TE": te,
            "TE_q95": st_logs.get("TE_q95", np.nan),
            "budget_used": st_logs.get("budget_used", 0.0),
            "lr_scale": knobs.get("lr_scale", 1.0),
            "attn_temp": knobs.get("attn_temp", 1.0),
            "explore_rate": knobs.get("explore_rate", 1.0),
            "leak_cmd": knobs.get("leak", np.nan),
            "rho_cmd": knobs.get("spectral_radius", np.nan),
            "dropout_cmd": knobs.get("dropout", 0.0),
            "horizon_cmd": knobs.get("horizon", np.nan),
        }
        writer.write(row)

    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Self-tuning gate demo runner")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    run_from_config(cfg)

if __name__ == "__main__":
    main()
