# Data dictionary

## Run naming
Each run has a unique name, e.g. `full_lgbm_r400_rep1` (mode `full_lgbm`, offered rate 400 req/s, repetition 1), `A_dec_rep2_r600_rep0` (replica crash with 2 replicas at 600 req/s), `rtt10_open_rep0` (remote PDP with 10 ms round-trip time, open loop). The experiment is given by the directory or table name (`e1_policy`, …, `e8_remote`).

## Raw per-request records (`results/raw_archives/<experiment>.tar.gz`)
Extracting an archive into `results/raw/` creates `results/raw/<experiment>/` with, per run:

**`<run>.client.csv`** — client-side record from the load generator, one row per request.

| column | meaning |
|---|---|
| `seq` | request sequence number within the run |
| `i` | index into the ground-truth table of the replayed trace |
| `code` | HTTP status received by the client (0 = transport error or timeout) |
| `lat_ms` | client-side latency in milliseconds |
| `bytes_in` | response bytes received |
| `ts` | request timestamp (Unix seconds) |
| `err` | load-generator error message, if any |

**`<run>.access.log`** — PoP-side record written by nginx, one line per request, fields separated by `|`:
`msec | trace index | status | request_time (s) | upstream_response_time | t_opa (µs) | t_insp (µs) | decision | escalated (0/1) | replica | bytes_sent`.
`t_opa` is the time spent in the zero-trust decision (decision service to OPA and back), `t_insp` the time spent in threat inspection, and `decision` the outcome (`allow:<reason>`, `deny:<reason>`, `threat`, or `nginx_failopen`).

**`<run>.timeline.csv`** (failure runs only) — per 0.5 s bin: `t` (s since start), `n` requests, `err` errors, `exposed` unauthorised admissions, `p99_ms`, `err_frac`.

**`<run>.ctl<k>.json`** (adaptive runs only) — controller history of replica `k`: list of `[time, in-flight requests, band width w]`.

## Ground truth and traces (`artifacts/`)
**`truth_<workload>.csv`** — one row per request of a workload: `i`, `label` (1 = malicious flow), `attack_cat`, `user`, `role`, `mfa`, `token_valid`, `posture`, `managed`, `geo`, `resource`, `sensitivity`, `ztna_allow` (expected policy decision), `ztna_reason`, `exp_status` (expected HTTP status: 200, 401, 403). Workloads: `unsw_default` (main), `unsw_seed2–4` (context seeds), `unsw_none`/`unsw_none2` (trusted-header identity), `unsw_sz*` (payload sizes), `cic_default`, `nsl_default`, `cicuf_default` (unseen-family protocol).

**`trace_<dataset>.csv`** — replayed flows: `idx`, `label`, `attack_cat`, the dataset's flow features, and light-model scores (`s_*`).

## Summary tables (`results/tables/<experiment>.csv`)
One row per run, recomputed from the raw records by `trident/collect.py`.

| column group | columns |
|---|---|
| configuration | `exp`, `name`, `rate`, `rep`, `mode_name`/`config`, `replicas`, `mode`, `light`, `deep`, `always`, `ztna`, `auth_mode`, `fail_open`, `nginx_failopen`, `target_inflight`, `thr_*`, `opa_rtt_ms`, `dataset`, `bundle` |
| load and latency | `n`, `dur_s`, `rate_offered`, `throughput_nonerr` (req/s), `goodput_mbps`, `err_frac`, `lat_p50`, `lat_p90`, `lat_p95`, `lat_p99`, `lat_mean`, `lat_max` (ms), `slo_attain` (fraction ≤ 25 ms) |
| stage times | `t_opa_ms_p50/p99/mean`, `t_insp_ms_p50/p99/mean`, `req_time_p50/p99/mean` |
| zero-trust | `ztna_decisions`, `ztna_conformance`, `ztna_bypass` (should-deny requests admitted), `ztna_false_deny` |
| detection | `tp`, `fp`, `fn`, `tn`, `recall`, `precision`, `f1`, `fpr`, `esc_rate` (deep-inspection escalation rate) |
| security/availability | `exposure` (unauthorised admissions), `exposure_mal` (malicious flows admitted), `legit`, `legit_served`, `availability_legit`, `false_denials`, `errors_legit`, `status_dist` (JSON) |
| load generator | `vegeta_rate`, `vegeta_p50`, `vegeta_p99`, `vegeta_mean`, `vegeta_bytes_in`, `vegeta_throughput`, `vegeta_duration` |
| service and failures | `svc_metrics` (JSON counters per replica), `events` (JSON list of `[t, event]` for kill/detect/restart), `probe`, `probe_phase`, `scenario` |
| platform fingerprint | `fp_cpu`, `fp_matvec_us`, `fp_pyloop_ms`, `fp_boot` |

`results/<experiment>.jsonl` holds the same rows as written during the campaign. `results/e1_policy.csv` additionally holds the `opa bench` cost per policy (`bench_ns_per_op`).

## Derived tables (`results/tables/` and `results/`)
| file | content |
|---|---|
| `offline_models.csv` | inspector zoo per dataset: threshold, recall, FPR, F1, AUC, single-row cost (Table 6 area, S1 Table A) |
| `operating_points.csv` | recall and FPR at calibrated FPR targets of 5 %, 1 %, 0.1 % |
| `cic_heldout_day.csv` | held-out-day protocol on CIC-IDS2017 |
| `capacity_model.csv` | capacity model: predicted and measured closed-loop capacity |
| `pdp_unloaded.json` | unloaded OPA decision latency for allowed and denied inputs |
| `service_stage_cost_us.json` | single-row inspection cost inside the decision service |
| `novelty_offline.csv`, `novelty_sweep.csv`, `novelty_baselines.csv`, `novelty_cascade_curve_cicuf.csv`, `cicuf_meta.json`, `static_band.json` | unseen-family protocol: novelty detectors, reference-set sweep, baselines at matched FPR, offline cascade curve |
| `e5_recovery.csv`, `e5_load_recovery.csv` | time to detect, time to restore, and losses per failure run |
| `e6_long_p99.csv` | p99 latency with bootstrap 95 % intervals for 60 s runs |
