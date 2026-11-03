#!/usr/bin/env python3
"""
Metric computation for TRIDENT runs. Joins
  <prefix>.json        vegeta per-request records (client view: code, latency, bytes)
  <prefix>.access.log  nginx per-request records (PoP view: status, stage timings, decision)
  truth_<tag>.csv      ground truth (flow label, expected ZTNA decision, expected status)
and returns a flat metric dictionary plus the joined per-request frame.
"""
import os, json, re
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, 'artifacts')
_i_re = re.compile(r'[?&]i=(\d+)')


def compact(prefix):
    """Reduce vegeta's per-request JSON records to a compact CSV and drop the JSON."""
    c = load_client(prefix)
    c.to_csv(prefix + '.client.csv', index=False, float_format='%.4f')
    os.remove(prefix + '.json')
    return c


def load_client(prefix):
    if os.path.exists(prefix + '.client.csv'):
        return pd.read_csv(prefix + '.client.csv')
    rows = []
    with open(prefix + '.json') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            m = _i_re.search(d['url'])
            rows.append((d['seq'], int(m.group(1)) if m else -1, d['code'], d['latency'] / 1e6, d['bytes_in'], d['timestamp'], d.get('error', '')))
    c = pd.DataFrame(rows, columns=['seq', 'i', 'code', 'lat_ms', 'bytes_in', 'ts', 'err'])
    c['ts'] = pd.to_datetime(c['ts'], format='ISO8601', utc=True).astype('int64') / 1e9
    return c.sort_values('seq').reset_index(drop=True)


def load_pop(prefix):
    rows = []
    with open(prefix + '.access.log') as f:
        for line in f:
            p = line.rstrip('\n').split('|')
            if len(p) < 11: continue
            try:
                rows.append((float(p[0]), int(p[1]) if p[1] not in ('-', '') else -1, int(p[2]), float(p[3]),
                             float(p[5]) / 1000 if p[5] not in ('-', '') else np.nan,
                             float(p[6]) / 1000 if p[6] not in ('-', '') else np.nan,
                             p[7], int(p[8]) if p[8] not in ('-', '') else -1, p[9], int(p[10])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=['msec', 'i', 'status', 'req_time', 't_opa_ms', 't_insp_ms', 'decision', 'esc', 'replica', 'bytes_sent'])


def join(prefix, truth_path):
    c = load_client(prefix); p = load_pop(prefix); t = pd.read_csv(truth_path)
    # when an attack cycles through the trace more than once, align occurrences by order
    c['occ'] = c.groupby('i').cumcount(); p = p.sort_values('msec'); p['occ'] = p.groupby('i').cumcount()
    df = c.merge(p, on=['i', 'occ'], how='left').merge(t, on='i', how='left')
    df['error'] = (df['code'] >= 500) | (df['code'] == 0)
    d = df['decision'].fillna('-')
    df['obs_policy_allow'] = d.str.startswith('allow') | (d == 'threat') | (d == 'nginx_failopen')
    df['obs_reason'] = d.str.replace('deny:', '', regex=False).where(d.str.startswith('deny'), 'allow')
    df['pred_threat'] = (d == 'threat')
    return df


def summarize(prefix, truth_path, slo_ms=None, t_fail=None):
    df = join(prefix, truth_path)
    rep = json.load(open(prefix + '.json'.replace('.json', '.report.json'))) if os.path.exists(prefix + '.report.json') else {}
    n = len(df); ok = ~df['error']
    m = dict(n=n, dur_s=df['ts'].max() - df['ts'].min() + 1e-9)
    m['rate_offered'] = n / m['dur_s']
    m['throughput_nonerr'] = ok.sum() / m['dur_s']
    bs = df.loc[ok, 'bytes_sent'] if 'bytes_sent' in df else df.loc[ok, 'bytes_in']
    m['goodput_mbps'] = float(bs.fillna(0).sum()) * 8 / 1e6 / m['dur_s']   # PoP-side bytes delivered to clients
    m['err_frac'] = float((~ok).mean())
    for q in (50, 90, 95, 99):
        m[f'lat_p{q}'] = float(np.percentile(df['lat_ms'], q))
    m['lat_mean'] = float(df['lat_ms'].mean()); m['lat_max'] = float(df['lat_ms'].max())
    if slo_ms is not None:
        m['slo_attain'] = float((df['lat_ms'] <= slo_ms).mean())
    for col in ('t_opa_ms', 't_insp_ms', 'req_time'):
        v = df[col].dropna()
        if len(v):
            m[col + '_p50'] = float(np.percentile(v, 50)); m[col + '_p99'] = float(np.percentile(v, 99)); m[col + '_mean'] = float(v.mean())
    # ---- zero-trust conformance (on requests that received a policy decision)
    dec = ok & df['decision'].notna() & (df['decision'] != '-')
    exp_allow = df['ztna_allow'] == 1
    conf = (df['obs_policy_allow'] == exp_allow) & (exp_allow | (df['obs_reason'] == df['ztna_reason']))
    m['ztna_decisions'] = int(dec.sum())
    m['ztna_conformance'] = float(conf[dec].mean()) if dec.sum() else np.nan
    m['ztna_bypass'] = int((dec & ~exp_allow & df['obs_policy_allow']).sum())      # should-deny but admitted by policy
    m['ztna_false_deny'] = int((dec & exp_allow & ~df['obs_policy_allow']).sum())
    # ---- threat detection among policy-allowed requests
    ins = dec & exp_allow & df['obs_policy_allow']
    y = df.loc[ins, 'label'] == 1; yh = df.loc[ins, 'pred_threat']
    tp = int((y & yh).sum()); fp = int((~y & yh).sum()); fn = int((y & ~yh).sum()); tn = int((~y & ~yh).sum())
    m.update(tp=tp, fp=fp, fn=fn, tn=tn)
    m['recall'] = tp / max(1, tp + fn); m['precision'] = tp / max(1, tp + fp)
    m['f1'] = 2 * tp / max(1, 2 * tp + fp + fn); m['fpr'] = fp / max(1, fp + tn)
    m['esc_rate'] = float((df.loc[ins, 'esc'] == 1).mean()) if ins.sum() else np.nan
    # ---- coupled security/availability outcomes over the whole run
    admitted = df['code'] == 200
    m['exposure'] = int((admitted & (df['exp_status'] != 200)).sum())            # unauthorised admissions (policy bypass + missed threats)
    m['exposure_mal'] = int((admitted & (df['label'] == 1)).sum())               # malicious flows that reached the origin
    legit = df['exp_status'] == 200
    m['legit'] = int(legit.sum())
    m['legit_served'] = int((legit & admitted).sum())
    m['availability_legit'] = m['legit_served'] / max(1, m['legit'])
    m['false_denials'] = int((legit & ~admitted & ok).sum())
    m['errors_legit'] = int((legit & ~ok).sum())
    m['status_dist'] = df['code'].value_counts().to_dict()
    return m, df


def timeline(df, t0, bin_s=0.5):
    """error / exposure / throughput timeline relative to t0 (seconds)."""
    rel = df['ts'] - t0
    b = (rel // bin_s) * bin_s
    g = df.groupby(b)
    out = pd.DataFrame({'t': g.size().index, 'n': g.size().values,
                        'err': g['error'].sum().values,
                        'exposed': g.apply(lambda x: int(((x['code'] == 200) & (x['exp_status'] != 200)).sum())).values,
                        'p99_ms': g['lat_ms'].quantile(0.99).values})
    out['err_frac'] = out['err'] / out['n']
    return out
