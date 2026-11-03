#!/usr/bin/env python3
"""
Builds clean per-experiment tables (results/tables/<exp>.csv) from
  results/<exp>.jsonl          summary rows written by the runner, and
  results/raw/<exp>/<name>.*   raw per-run records (re-summarised when a row is missing).
Run metadata (config, rate, repetition, mode, ...) is parsed from the run name.
"""
import os, re, json, glob
import numpy as np, pandas as pd
from analyze import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results'); ART = os.path.join(ROOT, 'artifacts')
OUT = os.path.join(RES, 'tables'); os.makedirs(OUT, exist_ok=True)
SLO = 25.0


def truth_for(exp, name):
    if exp == 'e6_crossdata':
        ds = 'cic' if name.startswith('cic') or '_cic' in name else 'nsl'
    return os.path.join(ART, 'truth_unsw_default.csv')


def parse_name(exp, name):
    md = dict(name=name)
    m = re.search(r'_rep(\d+)$', name); md['rep'] = int(m.group(1)) if m else 0
    m = re.search(r'_r(\d+)_rep', name); md['rate'] = int(m.group(1)) if m else (0 if '_closed' in name else None)
    md['loop'] = 'closed' if '_closed' in name else 'open'
    if exp.startswith('e3_pipeline'):
        m = re.match(r'(C\d_[a-z_]+?)_(?:r\d+|closed)_rep', name); md['config'] = m.group(1)
    elif exp == 'e3_payload':
        m = re.match(r'(C\d_\w+?)_(1k|16k|256k|1m)_rep', name); md['config'] = m.group(1); md['sz'] = m.group(2)
    elif exp in ('e4_cascade', 'e4_ablation', 'e6_crossdata', 'e7_unseen', 'e4_direct'):
        m = re.match(r'(?:(cic|nsl)_)?(.+?)_r\d+_rep', name); md['mode_name'] = m.group(2); md['dataset'] = m.group(1) or 'unsw'
    elif exp == 'e1_policy':
        m = re.match(r'(eq|glob|bypass|sw|re)_?(?:pad(\d+))?_?(rs256|none)?_rep', name)
        if m:
            md['style'] = m.group(1); md['pad'] = int(m.group(2) or 0); md['auth_mode'] = m.group(3) or 'none'
    elif exp == 'e8_remote':
        m = re.match(r'rtt(\d+)_(open|closed)_rep', name); md['rtt'] = int(m.group(1)); md['loop'] = m.group(2)
    elif exp == 'e5_load':
        m = re.match(r'A_dec_rep(\d)_r(\d+)_rep', name); md['k'] = int(m.group(1)); md['rate'] = int(m.group(2))
    elif exp == 'e5_reliability':
        md['scenario'] = name.split('_')[0]
        m = re.search(r'_p(\d+)_rep', name); md['probe'] = int(m.group(1)) if m else 0
    return md


def collect(exp):
    rows = {}
    old = os.path.join(RES, f'{exp}.csv')   # summary rows written before the switch to JSONL (E1)
    if exp == 'e1_policy' and os.path.exists(old):
        for _, r in pd.read_csv(old).iterrows():
            rows[r['name']] = {k: r[k] for k in ('name', 'style', 'pad', 'auth_mode', 'bench_ns_per_op', 'rep') if k in r}
            rows[r['name']]['exp'] = exp
    jl = os.path.join(RES, f'{exp}.jsonl')
    if os.path.exists(jl):
        for line in open(jl):
            if line.strip():
                r = json.loads(line); rows[r['name']] = r
    d = os.path.join(RES, 'raw', exp)
    for f in sorted(glob.glob(os.path.join(d, '*.client.csv'))):
        name = os.path.basename(f)[:-len('.client.csv')]
        truth = os.path.join(ART, 'truth_unsw_default.csv')
        if exp == 'e3_payload':
            m = re.search(r'_(1k|16k|256k|1m)_rep', name); truth = os.path.join(ART, f'truth_unsw_sz{m.group(1)}.csv')
        if exp == 'e6_crossdata':
            ds = name.split('_')[0] if name.startswith(('cic_', 'nsl_')) else 'unsw'
            truth = os.path.join(ART, f'truth_{ds}_default.csv')
        if exp == 'e7_unseen':
            truth = os.path.join(ART, 'truth_cicuf_default.csv')
        if exp == 'e6_repro' and name.startswith('seed'):
            truth = os.path.join(ART, f'truth_unsw_{name}.csv')
        try:
            m, _ = summarize(os.path.join(d, name), truth, slo_ms=SLO)   # metrics are always recomputed from raw
        except Exception as e:
            print('resummarise failed', exp, name, e); continue
        r = rows.get(name, dict(exp=exp, name=name))
        r.update({k: v for k, v in m.items() if k != 'status_dist'})
        r['status_dist'] = json.dumps(m['status_dist'])
        rows[name] = r
    out = []
    for name, r in rows.items():
        md = parse_name(exp, name)
        for k, v in md.items():
            r.setdefault(k, v)
        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, f'{exp}.csv'), index=False)
    return df


if __name__ == '__main__':
    for exp in ('e1_policy', 'e3_pipeline', 'e3_payload', 'e3_oppoint', 'e4_cascade', 'e4_ablation', 'e4_direct', 'e5_reliability',
                'e5_load', 'e6_repro', 'e6_crossdata', 'e6_long', 'e7_unseen', 'e8_remote'):
        if os.path.exists(os.path.join(RES, 'raw', exp)) or os.path.exists(os.path.join(RES, f'{exp}.jsonl')):
            df = collect(exp); print(exp, len(df), 'rows')
