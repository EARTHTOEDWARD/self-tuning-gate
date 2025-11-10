# Self-Tuning Gate

Drop-in, pip-installable scaffold that wraps your MAES/SACP/ABTC modules with a Lyapunov/Transfer-Entropy aware gate (\u03931) plus the classic ShockGate policy. It keeps each subsystem at the edge-of-chaos, enforces a shock budget, and logs criticality metrics for promotion tests.

## Features
- Lightweight ESN/SRNN reservoir "contracts" with Jacobian-vector products for online Lyapunov estimates.
- Phase+energy gate blended with an autotuner that watches local \u03bb\u2081 and TE spikes and keeps \u0393\u2208[0,1] optimal per context.
- SelfTuningCore maps \u03bb-band deviations to live knobs (LR, attention temp, leak, spectral radius, horizon) with safety clamps.
- ShockGate policy converts valence+\u0393 into `correct/consolidate/coast` events so you can gate slow learning.
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
Each row contains: `t, y_true, y_hat, MAE, Gamma, Gamma_phase, lambda1, TE, ctrl_update, shockgate_evt, actuator_sat, policy_kind, TE_q95, budget_used, lr_scale, attn_temp, explore_rate, leak_cmd, rho_cmd, dropout_cmd, horizon_cmd`.

## Promotion tests (built-in harness compatibility)
1. **MAE reduction** vs fixed-\u0393 baseline (>= configurable %).
2. **TE spikes precede control updates** for >=80% spikes within \u0394t.
3. **Lyapunov guardrail**: \u03bb\u2081 rises only when MAE \u2193 and no actuator saturation.
4. **Shock budget**: moving rate stays within 5-15%.
5. **\u0393 chatter**: variance below a small threshold.

## License
MIT.
