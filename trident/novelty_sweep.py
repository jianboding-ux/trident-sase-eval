#!/usr/bin/env python3
"""Reference-set size and threshold sweep for the k-NN novelty stage on the CIC-IDS2017 unseen-family protocol;
exports models_cicuf.joblib with the selected configuration."""
import os, sys, time, json
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_cic, Inspector, KNNNovelty, SEED, ART, RES
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
d = load_cic(); Xall, yall, lab, dall = d['full']; del d
zoo = joblib.load(os.path.join(ART, 'models_cicuf.joblib')); sup = zoo['lgbm']
tr = np.isin(dall, ['Bot', 'Infiltration', 'WebAttacks']); te = np.isin(dall, ['DDoS', 'PortScan'])
Xf, Xc, yf, yc = train_test_split(Xall[tr], yall[tr], test_size=0.2, random_state=SEED, stratify=lab[tr])
rng = np.random.RandomState(SEED)
idx_te = np.where(te)[0]; ben_te = idx_te[yall[idx_te] == 0]; att_te = idx_te[yall[idx_te] == 1]
samp = np.concatenate([rng.choice(ben_te, 8000, replace=False), rng.choice(att_te[lab[att_te] == 'DDoS'], 3000, replace=False),
                       rng.choice(att_te[lab[att_te] == 'PortScan'], 3000, replace=False)])
Xt, yt, ct = Xall[samp], yall[samp].astype(bool), lab[samp]
ben = Xf[yf == 0]; T = np.sign(ben) * np.log1p(np.abs(ben)); mu = T.mean(0); sd = T.std(0); sd[sd < 1e-6] = 1.0
rng = np.random.RandomState(SEED); perm = rng.permutation(len(ben))
calb = Xc[yc == 0]; calb = calb[rng.choice(len(calb), 5000, replace=False)]
s_sup = sup.score_batch(Xt); m_sup = s_sup >= sup.thr
rows, ins = [], {}
for n in (1000, 4000, 50000, 200000):
    ref = ben[perm[:min(n, len(ben))]]
    k = KNNNovelty(f'knn{n}', (np.sign(ref) * np.log1p(np.abs(ref)) - mu) / sd, mu, sd, 5, np.zeros(1), 0.99)
    k.cal = np.sort(k.raw_batch(calb)); ins[n] = k
    s = k.score_batch(Xt)
    t0 = time.perf_counter()
    for x in Xt[:100]: k.score(x)
    cost = (time.perf_counter() - t0) / 100 * 1e6
    for thr in (0.99, 0.995, 0.999):
        v = m_sup | (s >= thr)
        rows.append(dict(n_ref=len(ref), thr=thr, cost_us=cost, auc_ddos=roc_auc_score(yt[(~yt) | (ct == 'DDoS')], s[(~yt) | (ct == 'DDoS')]),
                         auc_ps=roc_auc_score(yt[(~yt) | (ct == 'PortScan')], s[(~yt) | (ct == 'PortScan')]),
                         rec_ddos=float(v[ct == 'DDoS'].mean()), rec_ps=float(v[ct == 'PortScan'].mean()), rec_all=float(v[yt].mean()),
                         fpr=float(v[~yt].mean())))
        print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in rows[-1].items()}, flush=True)
    del s
pd.DataFrame(rows).to_csv(os.path.join(RES, 'tables', 'novelty_sweep.csv'), index=False)
sel = {'deep': 200000, 'light': 1000, 'thr': 0.99}   # configuration used in the paper
dp, lt = ins[sel['deep']], ins[sel['light']]; dp.name, lt.name = 'nov_d', 'nov_s'; dp.thr = lt.thr = sel['thr']
s_d = dp.score_batch(Xt); s_s = lt.score_batch(Xt); casc = []
for w in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    th = sel['thr']; lo, hi = th - w * th, th + w * (1 - th); cand = ~m_sup
    esc = cand & (s_s >= lo) & (s_s <= hi); v = m_sup | np.where(esc, s_d >= th, s_s >= th)
    casc.append(dict(w=w, esc=float(esc[cand].mean()), rec_all=float(v[yt].mean()), rec_ddos=float(v[ct == 'DDoS'].mean()), fpr=float(v[~yt].mean())))
    print(casc[-1], flush=True)
pd.DataFrame(casc).to_csv(os.path.join(RES, 'tables', 'novelty_cascade_curve_cicuf.csv'), index=False)
joblib.dump({'lgbm': sup, 'nov_d': dp, 'nov_s': lt}, os.path.join(ART, 'models_cicuf.joblib'))
print('saved', sel, flush=True)
