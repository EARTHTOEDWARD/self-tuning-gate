import argparse
import yaml
import numpy as np

from .entrypoints import load
from .adapters.core import build_adapters
from .telemetry.writer import TelemetryWriter
from .metrics.core import MetricTracker
from .self_tune.core import SelfTuningCore
from .self_tune.packs import SOCPack, AtlasPack, AutocatalysisPack, HazardPack
from .self_tune.dual_ht import DSTHTPack
from .eoc import EdgeOfChaosConfig, EdgeOfChaosHook


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

    eoc_cfg_raw = cfg.get("eoc", {})
    eoc_hook = None
    if eoc_cfg_raw.get("enabled"):
        eoc_params = {k: v for k, v in eoc_cfg_raw.items()}
        apply_to = eoc_params.get("apply_to")
        if apply_to is not None:
            eoc_params["apply_to"] = tuple(apply_to)
        eoc_config = EdgeOfChaosConfig(**eoc_params)
        eoc_hook = EdgeOfChaosHook(eoc_config, C, dim=getattr(C, "H", None))

    norm, stride = build_adapters(contract_cfg.get("adapters"))

    tel_cfg = cfg.get("telemetry", {})
    writer = TelemetryWriter(path=tel_cfg.get("path", "logs/run.jsonl"), fmt=tel_cfg.get("fmt", "jsonl"))

    mt = MetricTracker()
    stream = _toy_stream(T=cfg.get("steps", 2000), seed=cfg.get("seed", 0))

    st_cfg = cfg.get("self_tuning", {})
    enable_st = st_cfg.get("enabled", False)
    tuner_params = {k: v for k, v in st_cfg.items() if k not in {"enabled", "packs"}}
    tuner = SelfTuningCore(**tuner_params) if enable_st else None

    packs_cfg = st_cfg.get("packs", {})
    soc_cfg = packs_cfg.get("soc", {})
    soc_pack = None
    if soc_cfg.get("enabled"):
        soc_params = {k: v for k, v in soc_cfg.items() if k != "enabled"}
        soc_pack = SOCPack(**soc_params)
    atlas_cfg = packs_cfg.get("atlas", {})
    atlas_pack = None
    if atlas_cfg.get("enabled"):
        atlas_params = {k: v for k, v in atlas_cfg.items() if k != "enabled"}
        atlas_pack = AtlasPack(**atlas_params)
    auto_cfg = packs_cfg.get("autocatalysis", {})
    auto_pack = None
    if auto_cfg.get("enabled"):
        auto_params = {k: v for k, v in auto_cfg.items() if k != "enabled"}
        auto_pack = AutocatalysisPack(**auto_params)
    hazard_cfg = packs_cfg.get("hazard", {})
    hazard_pack = None
    if hazard_cfg.get("enabled"):
        hazard_params = {k: v for k, v in hazard_cfg.items() if k != "enabled"}
        hazard_pack = HazardPack(**hazard_params)
    dst_cfg = packs_cfg.get("dst_htt", {})
    dst_pack = None
    if dst_cfg.get("enabled"):
        dst_params = {k: v for k, v in dst_cfg.items() if k != "enabled"}
        dst_pack = DSTHTPack(**dst_params)

    s_prev = 0.0
    v_prev = 0.0
    last_action = "coast"
    for t, x_t, phi_t, valence_t, yhat, y in stream:
        if not stride.allow():
            continue
        if norm is not None:
            x_t = norm.apply(x_t)
        eoc_state = None
        if eoc_hook is not None:
            frame = {"u": x_t, "input": x_t}
            frame = eoc_hook.apply_perception(frame)
            x_t = frame.get("u", x_t)
            base_state = getattr(C, "h", None)
            if base_state is not None:
                eoc_state = np.copy(base_state)
            elif hasattr(C, "H"):
                eoc_state = np.zeros(int(getattr(C, "H")))
        R_t = C.step(x_t)
        lambda1_eoc = float("nan")
        sigma_eoc = float("nan")
        eoc_poised = False
        if eoc_hook is not None:
            if eoc_state is None:
                eoc_state = np.zeros_like(R_t)
            eoc_hook.update(eoc_state, x_t)
            lambda1_eoc = float(eoc_hook.lambda1)
            sigma_eoc = float(eoc_hook.sigma)
            eoc_poised = eoc_hook.poised()

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

        resid_mag = abs(y - yhat)
        soc_gamma = 1.0
        soc_loss = 0.0
        soc_tau = float("nan")
        soc_beta = float("nan")
        soc_H = float("nan")
        soc_conf = False
        if soc_pack is not None:
            soc_out = soc_pack.update(resid_mag)
            soc_gamma = float(soc_out["gamma_scale"])
            soc_loss = float(soc_out["loss"])
            soc_tau = float(soc_out["tau"])
            soc_beta = float(soc_out["beta"])
            soc_H = float(soc_out["H"])
            soc_conf = bool(soc_out["confident"])

        auto_gamma = 1.0
        auto_c = 0.0
        auto_improved = False
        if auto_pack is not None:
            auto_out = auto_pack.update(mae=float(mae), lambda1=float(lam), last_action=last_action)
            auto_gamma = float(auto_out["gamma_scale"])
            auto_c = float(auto_out["c_t"])
            auto_improved = bool(auto_out["improved"])

        atlas_overrides = {}
        atlas_docked = False
        atlas_dist = float("inf")
        atlas_slots = 0
        # defer atlas update until knobs exist

        if enable_st:
            Gamma_final, knobs, st_logs = tuner.step(
                mae=float(mae),
                lam1=float(lam),
                te=float(te) if np.isfinite(te) else float("nan"),
                gamma_phase=float(Gamma_phase),
                ctx_key=ctx_key,
                te_mode=te_mode,
                valence=float(valence_t),
                soc_loss=soc_loss if soc_pack is not None else None,
            )
        else:
            Gamma_final, knobs, st_logs = float(Gamma_phase), {}, {}

        if atlas_pack is not None:
            atlas_res = atlas_pack.update(
                lam1=float(lam),
                beta_psd=float(soc_beta),
                tau_aval=float(soc_tau),
                phase=float(phi_t),
                mode=te_mode,
                current_knobs={k: knobs.get(k) for k in ("leak", "spectral_radius", "attn_temp")},
            )
            atlas_overrides = atlas_res["overrides"]
            atlas_docked = atlas_res["docked"]
            atlas_dist = atlas_res["best_dist"]
            atlas_slots = atlas_res["slots"]
        for key, value in atlas_overrides.items():
            if value is not None:
                knobs[key] = value

        dst_gamma = 1.0
        dst_explained = float("nan")
        dst_k = float("nan")
        dst_conf = 0.0
        if dst_pack is not None:
            knob_snapshot = {
                "spectral_radius": knobs.get("spectral_radius"),
                "horizon": knobs.get("horizon"),
                "dropout": knobs.get("dropout"),
                "attn_temp": knobs.get("attn_temp"),
            }
            dst_out = dst_pack.update(
                x_t=x_t,
                latent_t=R_t,
                y_true=float(y),
                y_hat=float(yhat),
                grad_vec=None,
                knobs=knob_snapshot,
            )
            dst_gamma = float(dst_out["gamma_scale"])
            dst_explained = float(dst_out["explained"])
            dst_k = float(dst_out["k_avg"])
            dst_conf = float(dst_out["k_conf"])
            for key, value in (dst_out.get("nudge") or {}).items():
                knobs[key] = value

        Gamma_final = float(np.clip(Gamma_final * soc_gamma * auto_gamma * dst_gamma, 0.0, 1.0))
        if enable_st:
            if knobs.get("leak") is not None and hasattr(C, "set_leak"):
                C.set_leak(knobs["leak"])
            if knobs.get("spectral_radius") is not None and hasattr(C, "set_spectral_radius"):
                C.set_spectral_radius(knobs["spectral_radius"])

        action_t = P.step(valence_t, Gamma_final)

        hazard_scale = 0.0
        hazard_permit = True
        hazard_alpha = float("nan")
        hazard_tail = 0
        hazard_streak = hazard_pack.streak if hazard_pack is not None else 0
        if hazard_pack is not None:
            hazard_out = hazard_pack.update(
                request_shock=(action_t == "correct"),
                gamma=float(Gamma_final),
                lambda1=float(lam),
            )
            hazard_scale = float(hazard_out["scale"])
            hazard_permit = bool(hazard_out["permit"])
            hazard_alpha = float(hazard_out["alpha"])
            hazard_tail = int(hazard_out["tail"])
            hazard_streak = int(hazard_out["streak"])
            if action_t == "correct" and not hazard_permit:
                action_t = "coast"

        last_action = action_t

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
            "lambda1_eoc": lambda1_eoc,
            "sigma_eoc": sigma_eoc,
            "poised_eoc": bool(eoc_poised),
            "TE": te,
            "TE_q95": st_logs.get("TE_q95", np.nan),
            "budget_used": st_logs.get("budget_used", 0.0),
            "soc_tau": soc_tau,
            "soc_beta": soc_beta,
            "soc_H": soc_H,
            "soc_confident": soc_conf,
            "soc_loss": soc_loss,
            "soc_gamma_scale": soc_gamma,
            "auto_gamma_scale": auto_gamma,
            "auto_c": auto_c,
            "auto_improved": auto_improved,
            "dst_gamma_scale": dst_gamma,
            "dst_explained": dst_explained,
            "dst_k": dst_k,
            "dst_k_conf": dst_conf,
            "hazard_scale": hazard_scale,
            "hazard_permit": hazard_permit,
            "hazard_alpha": hazard_alpha,
            "hazard_streak": hazard_streak,
            "hazard_tail": hazard_tail,
            "atlas_docked": bool(atlas_docked),
            "atlas_dist": float(atlas_dist),
            "atlas_slots": int(atlas_slots),
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
