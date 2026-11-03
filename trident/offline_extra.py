#!/usr/bin/env python3
"""Revision analyses: (A) operating points at calibrated FPR targets 5/1/0.1 %; (B) unseen-family protocols with a
k-NN novelty inspector (CIC-IDS2017 leave-families-out, NSL-KDD); exports the cicuf models and replay trace."""
import os, sys, time, json
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_unsw, load_nsl, load_cic, Inspector, KNNNovelty, SEED, ART, RES, export_trace
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def thr_at(neg, fpr):
    neg = np.sort(neg); return float(min(0.999, max(0.001, neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))])))

def rates(y, yh):
    tp = int((y & yh).sum()); fp = int((~y & yh).sum()); fn = int((y & ~yh).sum()); tn = int((~y & ~yh).sum())
    return dict(recall=tp / max(1, tp + fn), fpr=fp / max(1, fp + tn), precision=tp / max(1, tp + fp), f1=2 * tp / max(1, 2 * tp + fp + fn))

rowsA, rowsB = [], []
cic = None
for name, loader in (('unsw', load_unsw), ('nsl', load_nsl), ('cic', load_cic)):
    d = loader(); zoo = joblib.load(os.path.join(ART, f'models_{name}.joblib'))
    if name == 'cic': cic = d
    Xfit, Xcal, yfit, ycal = train_test_split(d['Xtr'], d['ytr'], test_size=0.2, random_state=SEED, stratify=d['ytr'])
    yte = d['yte'].astype(bool)
    for m in ('lr', 'dt_s', 'lgbm', 'rf', 'mlp'):
        pc = zoo[m].score_batch(Xcal); pt = zoo[m].score_batch(d['Xte'])
        for tgt in (0.05, 0.01, 0.001):
            t = thr_at(pc[ycal == 0], tgt); r = rates(yte, pt >= t)
            rowsA.append(dict(dataset=name, model=m, target_fpr=tgt, thr=t, cal_fpr=float((pc[ycal == 0] >= t).mean()), **r))
        print(name, m, [round(x['recall'], 3) for x in rowsA[-3:]], [round(x['fpr'], 4) for x in rowsA[-3:]], flush=True)
    del Xfit, Xcal
pd.DataFrame(rowsA).to_csv(os.path.join(RES, 'tables', 'operating_points.csv'), index=False)

def novelty_suite(tag, Xtr_fit, ytr_fit, Xtr_cal, ytr_cal, sup, Xte, yte, cte, n_big=50000, n_small=1000, k=5):
    rng = np.random.RandomState(SEED)
    ben = Xtr_fit[ytr_fit == 0]
    T = np.sign(ben) * np.log1p(np.abs(ben)); mu = T.mean(0); sd = T.std(0); sd[sd < 1e-6] = 1.0
    big = ben[rng.choice(len(ben), min(n_big, len(ben)), replace=False)]
    small = ben[rng.choice(len(ben), n_small, replace=False)]
    calb = Xtr_cal[ytr_cal == 0]; calb = calb[rng.choice(len(calb), min(5000, len(calb)), replace=False)]
    out = {}
    for nm, ref in (('nov_d', big), ('nov_s', small)):
        tmp = KNNNovelty(nm, (np.sign(ref) * np.log1p(np.abs(ref)) - mu) / sd, mu, sd, k, np.zeros(1), 0.99)
        tmp.cal = np.sort(tmp.raw_batch(calb)); out[nm] = tmp
    ps = sup.score_batch(Xtr_cal); sup.thr = thr_at(ps[ytr_cal == 0], 0.01)
    s_sup = sup.score_batch(Xte); s_d = out['nov_d'].score_batch(Xte); s_s = out['nov_s'].score_batch(Xte)
    y = yte.astype(bool); res = []
    fams = ['ALL'] + sorted(set(cte[y]))
    for fam in fams:
        sel = (~y) | (cte == fam) if fam != 'ALL' else np.ones(len(y), bool)
        row = dict(protocol=tag, family=fam, n_attack=int((y & sel).sum()))
        for nm, s, t in (('sup', s_sup, sup.thr), ('nov_d', s_d, 0.99), ('nov_s', s_s, 0.99)):
            row[f'auc_{nm}'] = roc_auc_score(y[sel], s[sel]) if len(set(y[sel])) > 1 else np.nan
            row[f'rec_{nm}'] = float((s[sel & y] >= t).mean())
        row['rec_sup_or_d'] = float(((s_sup >= sup.thr) | (s_d >= 0.99))[sel & y].mean())
        row['rec_sup_or_s'] = float(((s_sup >= sup.thr) | (s_s >= 0.99))[sel & y].mean())
        row['fpr_sup'] = float((s_sup[~y] >= sup.thr).mean()); row['fpr_nov_d'] = float((s_d[~y] >= 0.99).mean())
        row['fpr_nov_s'] = float((s_s[~y] >= 0.99).mean())
        row['fpr_sup_or_d'] = float(((s_sup >= sup.thr) | (s_d >= 0.99))[~y].mean())
        row['fpr_sup_or_s'] = float(((s_sup >= sup.thr) | (s_s >= 0.99))[~y].mean())
        res.append(row); print({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items()}, flush=True)
    # offline cascade curve: band on nov_s percentile around 0.99, deep verdict nov_d; sup always on
    for w in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        lo, hi = 0.99 - w * 0.99, 0.99 + w * 0.01
        cand = s_sup < sup.thr
        esc = cand & (s_s >= lo) & (s_s <= hi)
        v = (s_sup >= sup.thr) | np.where(esc, s_d >= 0.99, s_s >= 0.99)
        res.append(dict(protocol=tag, family=f'cascade_w{w}', esc=float(esc[cand].mean()), rec_casc=float(v[y].mean()), fpr_casc=float(v[~y].mean())))
    # single-row costs
    xs = Xte[:200]; cost = {}
    for nm, ins in (('sup', sup), ('nov_d', out['nov_d']), ('nov_s', out['nov_s'])):
        t0 = time.perf_counter()
        for x in xs: ins.score(x)
        cost[nm] = (time.perf_counter() - t0) / len(xs) * 1e6
    print(tag, 'cost_us', cost, flush=True)
    return res, out, cost

# ---- CIC-IDS2017 leave-families-out: train on Bot/Infiltration/WebAttacks days, test on DDoS/PortScan days
Xall, yall, lab, dall = cic['full']; del cic
tr = np.isin(dall, ['Bot', 'Infiltration', 'WebAttacks']); te = np.isin(dall, ['DDoS', 'PortScan'])
Xtr, ytr, labtr = Xall[tr], yall[tr], lab[tr]
Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=labtr)
m = lgb.LGBMClassifier(objective='binary', num_leaves=63, learning_rate=0.1, n_estimators=200, random_state=SEED, num_threads=1, verbose=-1).fit(Xf, yf)
sup = Inspector('lgbm', m.booster_, None, None, 'lgbm'); sup.n_features = Xtr.shape[1]
rng = np.random.RandomState(SEED)
idx_te = np.where(te)[0]
ben_te = idx_te[yall[idx_te] == 0]; att_te = idx_te[yall[idx_te] == 1]
samp = np.concatenate([rng.choice(ben_te, 8000, replace=False), rng.choice(att_te[lab[att_te] == 'DDoS'], 3000, replace=False),
                       rng.choice(att_te[lab[att_te] == 'PortScan'], 3000, replace=False)])
resB, nov, cost = novelty_suite('cic_unseen', Xf, yf, Xc, yc, sup, Xall[samp], yall[samp], lab[samp])
rowsB += resB
zoo_uf = {'lgbm': sup, 'nov_d': nov['nov_d'], 'nov_s': nov['nov_s']}
joblib.dump(zoo_uf, os.path.join(ART, 'models_cicuf.joblib'))
ds_uf = dict(name='cicuf', feats=[f'f{i}' for i in range(Xall.shape[1])], Xte=Xall[te], yte=yall[te], cte=lab[te])
import prepare_data as P
feats = json.load(open(os.path.join(ART, 'meta_cic.json')))['feats']; ds_uf['feats'] = feats
export_trace(ds_uf, {})
json.dump(dict(cost_us=cost, sup_thr=sup.thr), open(os.path.join(RES, 'tables', 'cicuf_meta.json'), 'w'), indent=1)
del Xall
# ---- NSL-KDD (KDDTest+ contains attack types absent from KDDTrain+)
d = load_nsl(); zoo = joblib.load(os.path.join(ART, 'models_nsl.joblib'))
Xf, Xc, yf, yc = train_test_split(d['Xtr'], d['ytr'], test_size=0.2, random_state=SEED, stratify=d['ytr'])
train_types = set(d['ctr']); cte = np.array([c if c in train_types else 'novel' for c in d['cte']])
cte = np.where(d['yte'] == 1, np.where(cte == 'novel', 'novel', 'known'), 'normal')
resN, _, _ = novelty_suite('nsl_test', Xf, yf, Xc, yc, zoo['lgbm'], d['Xte'], d['yte'], cte)
rowsB += resN
pd.DataFrame(rowsB).to_csv(os.path.join(RES, 'tables', 'novelty_offline.csv'), index=False)
print('done', flush=True)
