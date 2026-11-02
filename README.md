# TRIDENT: reproducible joint evaluation of cloud-native SASE security decision pipelines

This repository contains the code and data for the article

> J. Ding, Y. Qin, T. Chen, J. S. Yang, X. Wang. *TRIDENT: A reproducible trace-driven framework for joint security, performance, and reliability evaluation of cloud-native SASE architectures.* (submitted to PLOS ONE)

TRIDENT evaluates the security decision pipeline of a Secure Access Service Edge (SASE) point of presence (PoP) built from production components: an nginx data plane, an Open Policy Agent (OPA) decision point with RS256 token verification, and a machine-learning inspection service. It binds every labelled flow of a public intrusion dataset to a seeded zero-trust access context, replays the requests with an open-loop load generator, and scores every request against ground truth, so that policy conformance, detection, latency, throughput, and failure recovery are measured on the same requests.

The data in this repository are the complete measurement campaign of the article: **559 runs, 3.7 million requests**, with the raw per-request records of every run.

## Repository layout

```
trident/                 framework source (Python): harness, decision service, analysis, experiment drivers, figures
policy/bundle/           Rego zero-trust policy and its data bundle
artifacts/               ground-truth tables (truth_*.csv), replay traces (trace_*.csv), dataset metadata
results/tables/          summary table per experiment (one row per run) and derived tables used in the article
results/*.jsonl          per-run summary rows as written during the campaign, including platform fingerprints
results/raw_archives/    raw per-request records, one archive per experiment (+ campaign logs)
run_all.sh, run_rev*.sh  campaign drivers;  fetch_deps.sh: binaries and public datasets
DATA_DICTIONARY.md       description of every file type and column
```

## Experiments and where they appear in the article

| ID | Experiment | Data | Article |
|----|------------|------|---------|
| E1 | Zero-trust policy decision path | `e1_policy`, `pdp_unloaded.json` | Fig 2, Tables 3–5 |
| E2 | Inspector efficacy–cost frontier, operating points, held-out days | `offline_models.csv`, `operating_points.csv`, `cic_heldout_day.csv` | Fig 3, Tables 6–7, S1 Table A |
| E3 | Pipeline performance, payload, operating point online, remote PDP | `e3_pipeline`, `e3_payload`, `e3_oppoint`, `e8_remote`, `capacity_model.csv` | Figs 4–6, Tables 8–9 |
| E4 | Inspection strategies on UNSW-NB15 (direct comparison, mechanism study, ablation) | `e4_direct`, `e4_cascade`, `e4_ablation` | Fig 7, Table 10, S1 Tables B–C, Figs A–C |
| E5 | Failure injection and recovery | `e5_reliability`, `e5_load`, `e5_recovery.csv`, `e5_load_recovery.csv` | Figs 9–10, Tables 13–14 |
| E6 | Repeatability, context seeds, cross-dataset, long runs | `e6_repro`, `e6_crossdata`, `e6_long`, `e6_long_p99.csv` | S1 Table F, Fig D |
| E7 | Unseen attack families: detector baselines and strategies | `e7_unseen`, `novelty_*.csv` | Tables 11–12, Fig 8, S1 Tables D–E |

`results/raw_archives/_superseded.tar.gz` holds an earlier execution of E7 whose repetitions ran on two different hosts; it is kept for transparency and is not used in the article (see S1 Appendix, Section F).

## Recompute the tables and figures from the raw records

Requires Python 3.12 and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
mkdir -p results/raw
for f in results/raw_archives/*.tar.gz; do tar -xzf "$f" -C results/raw/; done
python3 trident/collect.py      # rebuilds results/tables/<experiment>.csv from the raw records
python3 trident/plots_plos.py   # regenerates the article figures into figures/
```

`collect.py` recomputes every metric from the per-request records and the ground-truth tables in `artifacts/`.

## Re-run the campaign

The reference deployment is a single-vCPU Ubuntu 24.04 virtual machine with nginx 1.24, OPA 1.22 and vegeta 12.12. Absolute throughput depends on the host; every run therefore records a platform fingerprint (`fp_*` columns), and configurations are compared within one fingerprint.

```bash
bash fetch_deps.sh                      # OPA and vegeta into ../bin, public datasets into ../data
python3 trident/prepare_data.py         # inspector models and replay traces
python3 trident/policy_gen.py           # policy bundle
python3 trident/gen_targets.py          # default UNSW-NB15 workload
./run_all.sh                            # E1, E3, E4, E5, E6
python3 trident/offline_extra.py        # operating points; unseen-family protocol
python3 trident/novelty_sweep.py        # novelty reference-set sweep
python3 trident/novelty_baselines.py    # detector baselines (exact/HNSW k-NN, Isolation Forest, LOF)
python3 trident/lof_hnsw.py             # HNSW-based LOF
./run_rev.sh                            # operating point online, higher-load failures, long runs, remote PDP
python3 trident/run_experiments.py e4d  # direct comparison (one session)
python3 trident/run_experiments.py e7c  # unseen families (one session)
python3 trident/collect.py && python3 trident/plots_plos.py
```

Every run is identified by its name; an interrupted campaign resumes without repeating completed runs. Trained models and load-generator target files are not included because they are regenerated deterministically by the scripts above.

## Datasets

The intrusion-detection datasets are publicly available from their original sources and are not redistributed here, apart from the derived replay traces in `artifacts/`:

- UNSW-NB15 — https://research.unsw.edu.au/projects/unsw-nb15-dataset (Moustafa & Slay, MilCIS 2015)
- CIC-IDS2017 — https://www.unb.ca/cic/datasets/ids-2017.html (Sharafaldin et al., ICISSP 2018)
- NSL-KDD — https://www.unb.ca/cic/datasets/nsl.html (Tavallaee et al., CISDA 2009)

## License

Code: MIT License (`LICENSE`). Data in `results/` and `artifacts/`: Creative Commons Attribution 4.0 International (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the article when using the code or data.
