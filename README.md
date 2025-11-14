# Self-Tuning Gate

Drop-in, pip-installable scaffold that wraps your MAES/SACP/ABTC modules with a Lyapunov/Transfer-Entropy aware gate (\u03931) plus the classic ShockGate policy. It keeps each subsystem at the edge-of-chaos, enforces a shock budget, and logs criticality metrics for promotion tests.

## Features
- Lightweight ESN/SRNN reservoir "contracts" with Jacobian-vector products for online Lyapunov estimates.
- Phase+energy gate blended with an autotuner that watches local \u03bb\u2081 and TE spikes and keeps \u0393\u2208[0,1] optimal per context.
- SelfTuningCore maps \u03bb-band deviations to live knobs (LR, attention temp, leak, spectral radius, horizon) with safety clamps.
- ShockGate policy converts valence+\u0393 into `correct/consolidate/coast` events so you can gate slow learning.
- Optional SOC thermostat pack that watches avalanche / 1-f / DFA stats and nudges \u0393 plus a \u03bb_SOC\u22c5L_SOC term when it drifts off-band.
- Optional Atlas pack that memoizes regime fingerprints and gently docks leak/\u03c1/attn_temp back to the best match during consolidation.
- Optional Autocatalysis pack that boosts \u0393 briefly after successful corrections, then decays (bursts-then-rest).
- Optional Dual-Subspace + Heavy-Tail pack that aligns shocks with data\u2194update directions and keeps update tails near a target exponent k.
- Optional Hazard pack that fits a Pareto tail to inter-shock times so `correct` only fires when a “due” window arrives.
- JSONL/CSV telemetry with MAE, \u03bb\u2081, TE, knobs, and promotion harness hooks.

## Quickstart
```bash
pip install -e .
self-tuner-run --config configs/self_tuning.self_driving.yaml
```
The toy stream writes `logs/self_tuning_sd.jsonl` that you can feed into the promotion harness.

## Repo layout
- `maes_shockgate/` &mdash; package code (contracts, gates, policies, telemetry, self-tuner).
- `configs/` &mdash; ready-to-run presets (self-driving demo, MAES variant).
- `examples/` &mdash; optional scripts that show how to embed the tuner manually.

## Telemetry schema
Each row contains: `t, y_true, y_hat, MAE, Gamma, Gamma_phase, lambda1, TE, ctrl_update, shockgate_evt, actuator_sat, policy_kind, TE_q95, budget_used, soc_tau, soc_beta, soc_H, soc_confident, soc_loss, soc_gamma_scale, auto_gamma_scale, auto_c, auto_improved, dst_gamma_scale, dst_explained, dst_k, dst_k_conf, hazard_scale, hazard_permit, hazard_alpha, hazard_streak, hazard_tail, atlas_docked, atlas_dist, atlas_slots, lr_scale, attn_temp, explore_rate, leak_cmd, rho_cmd, dropout_cmd, horizon_cmd`.

### SOC pack (optional)
Toggle under `self_tuning.packs.soc` in any config. Supply targets (`tau_star`, `beta_star`, `hurst_star`) or let it auto-calibrate, and set `self_tuning.lambda_soc` to weight the extra loss term.

### Atlas pack (optional)
Enable via `self_tuning.packs.atlas`. It records a small atlas of regime fingerprints (λ₁, PSD β, avalanche τ, phase) and, during consolidation windows, softly blends leak / spectral radius / attention temperature toward the best match.

### Autocatalysis pack (optional)
Toggle `self_tuning.packs.autocatalysis` to allow short “burst” windows: when a `correct` action reduces MAE while λ₁ moves toward 0, Γ is temporarily scaled up (logged as `auto_gamma_scale`) before decaying with `tau_decay`.

### Hazard pack (optional)
Enable via `self_tuning.packs.hazard`. It tracks inter-shock intervals, fits a Pareto tail, and only lets `correct` fire when the scaled hazard rate beats `threshold`, Γ ≥ `gamma_min`, and λ stays within `lambda_band`.

### Dual-Subspace + Heavy-Tail pack (optional)
Enable via `self_tuning.packs.dst_htt`. It keeps a rolling PCA of boundary features, runs the cross-covariance SVD to find the few data↔update directions that matter, estimates the tail exponent `k` along them, scales Γ (`dst_gamma_scale`) to keep `k` inside `k_band`, and nudges knobs (ρ, horizon, dropout, attention temperature) when tails get too heavy or too light.

## Promotion tests (built-in harness compatibility)
1. **MAE reduction** vs fixed-\u0393 baseline (>= configurable %).
2. **TE spikes precede control updates** for >=80% spikes within \u0394t.
3. **Lyapunov guardrail**: \u03bb\u2081 rises only when MAE \u2193 and no actuator saturation.
4. **Shock budget**: moving rate stays within 5-15%.
5. **\u0393 chatter**: variance below a small threshold.

## License
MIT.
