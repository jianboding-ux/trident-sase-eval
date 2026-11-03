#!/usr/bin/env python3
"""Second-round baselines for the unseen-family protocol: exact k-NN (N = 1e3, 5e4, 2e5), HNSW (N = 2e5), Isolation Forest,
LOF, and the supervised LightGBM with a retuned threshold; matched-test-FPR and calibrated-target comparison; cost per row.
Adds the HNSW inspector to models_cicuf.joblib and computes the static band width that escalates 30 % of candidates."""
import os, sys, time, json
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_cic, KNNNovelty, HNSWNovelty, SEED, ART, RES
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
d = load_cic(); Xall, yall, lab, dall = d['full']; del d
zoo = joblib.load(os.path.join(ART, 'models_cicuf.joblib')); sup, nov_s, nov_d = zoo['lgbm'], zoo['nov_s'], zoo['nov_d']
tr = np.isin(dall, ['Bot', 'Infiltration', 'WebAttacks']); te = np.isin(dall, ['DDoS', 'PortScan'])
Xf, Xc, yf, yc = train_test_split(Xall[tr], yall[tr], test_size=0.2, random_state=SEED, stratify=lab[tr])
rng = np.random.RandomState(SEED)
idx_te = np.where(te)[0]; ben_te = idx_te[yall[idx_te] == 0]; att_te = idx_te[yall[idx_te] == 1]
samp = np.concatenate([rng.choice(ben_te, 8000, replace=False), rng.choice(att_te[lab[att_te] == 'DDoS'], 3000, replace=False),
                       rng.choice(att_te[lab[att_te] == 'PortScan'], 3000, replace=False)])
Xt, yt, ct = Xall[samp], yall[samp].astype(bool), lab[samp]
ben = Xf[yf == 0]; mu, sd = nov_d.mu, nov_d.sd
rng = np.random.RandomState(SEED); perm = rng.permutation(len(ben))
calb = Xc[yc == 0]; calb = calb[rng.choice(len(calb), 5000, replace=False)]
T = lambda X: ((np.sign(X) * np.log1p(np.abs(X)) - mu) / sd).astype(np.float32)
dets = {'LightGBM (supervised, retuned)': sup, 'k-NN exact, N=1e3': nov_s, 'k-NN exact, N=2e5': nov_d}
m50 = KNNNovelty('nov_m', T(ben[perm[:50000]]), mu, sd, 5, np.zeros(1), 0.99); m50.cal = np.sort(m50.raw_batch(calb)); dets['k-NN exact, N=5e4'] = m50
t0 = time.time(); h = HNSWNovelty('nov_h', T(ben[perm[:200000]]), mu, sd, 5, 0.99); h.cal = np.sort(h.raw_batch(calb)); tb = time.time() - t0
dets['k-NN HNSW, N=2e5'] = h; print('hnsw build+cal s', round(tb, 1), flush=True)
class Wrap:
    def __init__(s, f, cal): s.f, s.cal = f, np.sort(cal)
    def score_batch(s, X): return np.searchsorted(s.cal, s.f(X), side='right') / len(s.cal)
    def score(s, x): return float(s.score_batch(x[None, :])[0])
ifo = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED).fit(T(ben[perm[:200000]]))
dets['Isolation Forest'] = Wrap(lambda X: -ifo.score_samples(T(np.atleast_2d(X))), -ifo.score_samples(T(calb)))
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, algorithm='brute').fit(T(ben[perm[:50000]]))
dets['LOF, N=5e4'] = Wrap(lambda X: -lof.score_samples(T(np.atleast_2d(X))), -lof.score_samples(T(calb)))
print('fitted', flush=True)
rows = []
for name, ins in dets.items():
    s = ins.score_batch(Xt); sc = sup.score_batch(calb) if name.startswith('LightGBM') else None
    calsc = ins.score_batch(calb)
    xs = Xt[:150]; t0 = time.perf_counter()
    for x in xs: ins.score(x)
    cost = (time.perf_counter() - t0) / len(xs) * 1e6
    r = dict(detector=name, cost_us=cost, auc_all=roc_auc_score(yt, s), auc_ddos=roc_auc_score(yt[(~yt) | (ct == 'DDoS')], s[(~yt) | (ct == 'DDoS')]),
             auc_ps=roc_auc_score(yt[(~yt) | (ct == 'PortScan')], s[(~yt) | (ct == 'PortScan')]))
    for a in (0.01, 0.05, 0.10):
        th = np.quantile(s[~yt], 1 - a); v = s > th
        r[f'rec@{a}'] = float(v[yt].mean()); r[f'ddos@{a}'] = float(v[ct == 'DDoS'].mean()); r[f'ps@{a}'] = float(v[ct == 'PortScan'].mean())
        thc = np.quantile(calsc, 1 - a); vc = s > thc
        r[f'cal{a}_rec'] = float(vc[yt].mean()); r[f'cal{a}_fpr'] = float(vc[~yt].mean())
    rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, 'tables', 'novelty_baselines.csv'), index=False)
zoo['nov_h'] = h; joblib.dump(zoo, os.path.join(ART, 'models_cicuf.joblib'))
tr_ = pd.read_csv(os.path.join(ART, 'trace_cicuf.csv')); F = tr_[json.load(open(os.path.join(ART, 'meta_cic.json')))['feats']].values.astype(np.float32)
cand = sup.score_batch(F) < sup.thr; ss = nov_s.score_batch(F[cand])
w30 = float(1 - np.quantile(ss, 0.70)); print('static band w for 30% escalation:', round(w30, 4), 'esc at w=0.3:', round(float((ss >= 0.7).mean()), 3), flush=True)
json.dump({'w30': w30, 'esc_w03': float((ss >= 0.7).mean())}, open(os.path.join(RES, 'tables', 'static_band.json'), 'w'))
print('done', flush=True)
