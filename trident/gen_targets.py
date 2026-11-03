#!/usr/bin/env python3
"""
Workload binder: attaches a zero-trust access context (identity token, role, device posture, geo,
target resource) to each labelled flow record of a dataset trace and emits
  targets_<tag>.json : vegeta NDJSON targets (one HTTP request per trace row)
  truth_<tag>.csv    : per-request ground truth (flow label, expected ZTNA decision/reason, expected status)
The access-context generator is seeded, so the workload is bit-for-bit reproducible.
"""
import os, sys, json, time, argparse, random
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'trident'))
from policy_gen import ensure_keys, make_token, Oracle, COUNTRIES_OK, COUNTRIES_BLOCKED, ROLES

ART = os.path.join(ROOT, 'artifacts')


def build(dataset, bundle_dir, tag, seed=1, n=None, auth_mode='rs256', sz='1k', n_users=200, base='http://127.0.0.1:8080',
          p_bad_token=0.03, p_expired=0.03, p_blocked_geo=0.04, n_resources_used=100):
    rng = random.Random(seed)
    data = json.load(open(os.path.join(bundle_dir, 'data.json')))
    oracle = Oracle(data)
    priv, _ = ensure_keys()
    trace = pd.read_csv(os.path.join(ART, f'trace_{dataset}.csv'))
    if n: trace = trace.iloc[:n]
    feats = [c for c in trace.columns if c not in ('idx', 'label', 'attack_cat') and not c.startswith('s_')]
    # ---- user population
    users = []
    now = time.time()
    for u in range(n_users):
        role = rng.choices(ROLES, weights=[0.7, 0.2, 0.1])[0]
        claims = {'sub': f'user{u}', 'role': role, 'dept': rng.choice(['eng', 'sales', 'ops', 'fin']), 'mfa': 1 if rng.random() < 0.75 else 0}
        tok_valid = make_token(priv, claims, valid=True, now=now)
        tok_bad = make_token(priv, claims, valid=False, now=now)
        tok_exp = make_token(priv, claims, valid=True, expired=True, now=now)
        users.append((claims, tok_valid, tok_bad, tok_exp))
    resources = [f'r{i}' for i in range(min(n_resources_used, len(data['resources'])))]
    rows, truth = [], []
    X = trace[feats].values
    for i in range(len(trace)):
        claims, tv, tb, te = users[rng.randrange(n_users)]
        r = rng.random()
        if r < p_bad_token: tok, valid = tb, False
        elif r < p_bad_token + p_expired: tok, valid = te, False
        else: tok, valid = tv, True
        posture = rng.choices(['compliant', 'noncompliant', 'unknown'], weights=[0.8, 0.12, 0.08])[0]
        managed = (posture == 'compliant' and rng.random() < 0.9) or (posture != 'compliant' and rng.random() < 0.3)
        geo = rng.choice(COUNTRIES_BLOCKED) if rng.random() < p_blocked_geo else rng.choice(COUNTRIES_OK)
        res = rng.choice(resources)
        allow, reason = oracle.decide(valid, claims, res, geo, posture, managed)
        label = int(trace.label.iloc[i])
        exp_status = 401 if reason == 'invalid_token' else (403 if not allow else (403 if label else 200))
        flow = ','.join(f'{v:.6g}' for v in X[i])
        hdr = {'X-Flow': [flow], 'X-Trace-Idx': [str(i)], 'X-Resource': [res], 'X-Geo': [geo],
               'X-Device-Posture': [posture], 'X-Device-Managed': ['1' if managed else '0']}
        if auth_mode == 'rs256':
            hdr['Authorization'] = ['Bearer ' + tok]
        else:
            # trusted-header mode: the PEP forwards the claims it decoded without verifying the signature,
            # so forged or expired tokens present their claimed role
            hdr['X-User'] = [claims['sub']]; hdr['X-Role'] = [claims['role']]; hdr['X-Mfa'] = [str(claims['mfa'])]
        rows.append(json.dumps({'method': 'GET', 'url': f'{base}/app/{res}?sz={sz}&i={i}', 'header': hdr}))
        truth.append(dict(i=i, label=label, attack_cat=trace.attack_cat.iloc[i], user=claims['sub'], role=claims['role'],
                          mfa=claims['mfa'], token_valid=int(valid), posture=posture, managed=int(managed), geo=geo,
                          resource=res, sensitivity=data['resources'][res]['sensitivity'], ztna_allow=int(allow),
                          ztna_reason=reason, exp_status=exp_status))
    tpath = os.path.join(ART, f'targets_{tag}.json'); open(tpath, 'w').write('\n'.join(rows) + '\n')
    tr = pd.DataFrame(truth); tr.to_csv(os.path.join(ART, f'truth_{tag}.csv'), index=False)
    print(f'{tag}: {len(rows)} targets; ztna allow {tr.ztna_allow.mean():.3f}; expected status dist {tr.exp_status.value_counts().to_dict()}')
    return tpath


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='unsw')
    ap.add_argument('--bundle', default=os.path.join(ROOT, 'policy', 'bundle'))
    ap.add_argument('--tag', default='unsw_default')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--auth-mode', default='rs256')
    ap.add_argument('--sz', default='1k')
    a = ap.parse_args()
    build(a.dataset, a.bundle, a.tag, a.seed, a.n, a.auth_mode, a.sz)
