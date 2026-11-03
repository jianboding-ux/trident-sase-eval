#!/usr/bin/env python3
"""HNSW-based LOF on the unseen-family protocol: offline metrics appended to novelty_baselines.csv; saved as 'lof_h'."""
import os, sys, time
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_cic, HNSWLOF, SEED, ART, RES
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
d = load_cic(); Xall, yall, lab, dall = d['full']; del d
zoo = joblib.load(os.path.join(ART, 'models_cicuf.joblib')); nov_d = zoo['nov_d']; mu, sd = nov_d.mu, nov_d.sd
tr = np.isin(dall, ['Bot', 'Infiltration', 'WebAttacks']); te = np.isin(dall, ['DDoS', 'PortScan'])
Xf, Xc, yf, yc = train_test_split(Xall[tr], yall[tr], test_size=0.2, random_state=SEED, stratify=lab[tr])
rng = np.random.RandomState(SEED)
idx_te = np.where(te)[0]; ben_te = idx_te[yall[idx_te] == 0]; att_te = idx_te[yall[idx_te] == 1]
samp = np.concatenate([rng.choice(ben_te, 8000, replace=False), rng.choice(att_te[lab[att_te] == 'DDoS'], 3000, replace=False),
                       rng.choice(att_te[lab[att_te] == 'PortScan'], 3000, replace=False)])
Xt, yt, ct = Xall[samp], yall[samp].astype(bool), lab[samp]
ben = Xf[yf == 0]; rng = np.random.RandomState(SEED); perm = rng.permutation(len(ben))
calb = Xc[yc == 0]; calb = calb[rng.choice(len(calb), 5000, replace=False)]
T = lambda X: ((np.sign(X) * np.log1p(np.abs(X)) - mu) / sd).astype(np.float32)
t0 = time.time(); L = HNSWLOF('lof_h', T(ben[perm[:200000]]), mu, sd, 20, 0.99); L.cal = np.sort(L.raw_batch(calb)); print('build s', round(time.time() - t0, 1), flush=True)
s = L.score_batch(Xt); xs = Xt[:300]; t0 = time.perf_counter()
for x in xs: L.score(x)
cost = (time.perf_counter() - t0) / len(xs) * 1e6
r = dict(detector='LOF HNSW, N=2e5', cost_us=cost, auc_all=roc_auc_score(yt, s), auc_ddos=roc_auc_score(yt[(~yt) | (ct == 'DDoS')], s[(~yt) | (ct == 'DDoS')]),
         auc_ps=roc_auc_score(yt[(~yt) | (ct == 'PortScan')], s[(~yt) | (ct == 'PortScan')]))
calsc = L.score_batch(calb)
for a in (0.01, 0.05, 0.10):
    th = np.quantile(s[~yt], 1 - a); v = s > th
    r[f'rec@{a}'] = float(v[yt].mean()); r[f'ddos@{a}'] = float(v[ct == 'DDoS'].mean()); r[f'ps@{a}'] = float(v[ct == 'PortScan'].mean())
    thc = np.quantile(calsc, 1 - a); vc = s > thc; r[f'cal{a}_rec'] = float(vc[yt].mean()); r[f'cal{a}_fpr'] = float(vc[~yt].mean())
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
b = pd.read_csv(os.path.join(RES, 'tables', 'novelty_baselines.csv')); b = b[b.detector != r['detector']]
pd.concat([b, pd.DataFrame([r])]).to_csv(os.path.join(RES, 'tables', 'novelty_baselines.csv'), index=False)
zoo['lof_h'] = L; joblib.dump(zoo, os.path.join(ART, 'models_cicuf.joblib')); print('saved', flush=True)
