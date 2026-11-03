#!/usr/bin/env python3
"""
TRIDENT data preparation: loads real NIDS datasets (UNSW-NB15, CIC-IDS2017, NSL-KDD),
cleans/encodes features, trains the inspection-model zoo, measures offline efficacy and
single-row inference cost, and exports replay traces for the SASE harness.

Outputs (under artifacts/):
  models_<ds>.joblib      : dict of Inspector objects (lr, dt_s, dt, rf, lgbm, mlp)
  trace_<ds>.csv          : replay trace (encoded features + label + attack category)
  <results>/offline_models.csv : offline efficacy + cost per (dataset, model)
"""
import os, sys, time, json, glob, argparse
import numpy as np, pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
ART = os.path.join(ROOT, 'artifacts')
RES = os.path.join(ROOT, 'results')
os.makedirs(ART, exist_ok=True); os.makedirs(RES, exist_ok=True)
SEED = 42


class Inspector:
    """Single-row scorer with embedded preprocessing. score(x) -> P(malicious)."""
    def __init__(self, name, model, mean=None, std=None, kind='tree'):
        self.name, self.model, self.kind = name, model, kind
        self.mean, self.std = mean, std
        self.n_features = None
        self.thr = 0.5          # operating threshold (calibrated to a target false-positive rate)

    def _prep(self, x):
        if self.mean is not None:
            return (x - self.mean) / self.std
        return x

    def score(self, x):
        """x: 1-D numpy float array (encoded features)."""
        X = self._prep(x).reshape(1, -1)
        if self.kind == 'lgbm':
            return float(self.model.predict(X, num_threads=1)[0])
        return float(self.model.predict_proba(X)[0, 1])

    def score_batch(self, X):
        X = self._prep(X)
        if self.kind == 'lgbm':
            return self.model.predict(X, num_threads=1)
        return self.model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- datasets
def load_unsw():
    # The 175,341-record partition is the training set and the 82,332-record partition the
    # test set, following the original description of the UNSW-NB15 split.
    a = pd.read_csv(os.path.join(DATA, 'unsw/UNSW_NB15_testing-set.csv'))
    b = pd.read_csv(os.path.join(DATA, 'unsw/UNSW_NB15_training-set.csv'))
    tr, te = (a, b) if len(a) > len(b) else (b, a)
    cats = ['proto', 'service', 'state']
    feats = [c for c in tr.columns if c not in ('id', 'attack_cat', 'label')]
    maps = {}
    for c in cats:
        vals = sorted(tr[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    def enc(df):
        X = df[feats].copy()
        for c in cats:
            X[c] = df[c].astype(str).map(maps[c]).fillna(-1).astype(int)
        X = X.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X.values.astype(np.float32), df['label'].values.astype(int), df['attack_cat'].astype(str).values
    Xtr, ytr, ctr = enc(tr); Xte, yte, cte = enc(te)
    return dict(name='unsw', feats=feats, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, cte=cte, ctr=ctr)


def _clean_cic_label(s):
    s = str(s)
    if s == 'BENIGN': return 'BENIGN'
    if s.startswith('Web Attack'):
        if 'Brute' in s: return 'WebAttack-BruteForce'
        if 'XSS' in s: return 'WebAttack-XSS'
        return 'WebAttack-SQLi'
    return s


def load_cic(train_frac=0.7, max_train=300000, max_benign_per_file=100000):
    files = {
        'DDoS': 'cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'PortScan': 'cicids2017/ML_Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Bot': 'cicids2017/ML_Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Infiltration': 'cicids2017/ML_Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'WebAttacks': 'cicids2017/ML_Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    }
    rng = np.random.RandomState(SEED)
    Xs, labs, days = [], [], []
    feats = None
    for day, f in files.items():
        hdr = pd.read_csv(os.path.join(DATA, f), nrows=0, encoding='latin1')
        cols = [c.strip() for c in hdr.columns]
        drop = {'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Protocol', 'Timestamp'}
        keep = [c for c in cols if c not in drop]
        fcols = [c for c in keep if c != 'Label']
        if feats is None: feats = fcols
        assert fcols == feats, day
        usecols = [hdr.columns[i] for i, c in enumerate(cols) if c in keep]
        df = pd.read_csv(os.path.join(DATA, f), encoding='latin1', usecols=usecols, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        lab = df['Label'].map(_clean_cic_label).values
        X = df[feats].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        del df
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dup = X.duplicated(keep='first').values & pd.Series(lab).duplicated(keep='first').values
        X = X.values; keep_mask = ~dup
        X, lab = X[keep_mask], lab[keep_mask]
        ben = np.where(lab == 'BENIGN')[0]
        if len(ben) > max_benign_per_file:
            drop_idx = rng.choice(ben, len(ben) - max_benign_per_file, replace=False)
            m = np.ones(len(lab), bool); m[drop_idx] = False
            X, lab = X[m], lab[m]
        Xs.append(X); labs.append(lab); days.append(np.array([day] * len(lab)))
        print(f'[cic] {day}: {len(lab)} rows after dedup/cap, {int((lab != "BENIGN").sum())} attacks', flush=True)
    Xall = np.concatenate(Xs); lab = np.concatenate(labs); dall = np.concatenate(days)
    del Xs
    y = (lab != 'BENIGN').astype(int)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=1 - train_frac, random_state=SEED, stratify=lab)
    if len(idx_tr) > max_train:
        idx_tr = rng.choice(idx_tr, max_train, replace=False)
    d = dict(name='cic', feats=feats, Xtr=Xall[idx_tr], ytr=y[idx_tr], Xte=Xall[idx_te], yte=y[idx_te],
             cte=lab[idx_te], ctr=lab[idx_tr], day_tr=dall[idx_tr], day_te=dall[idx_te])
    d['full'] = (Xall, y, lab, dall)
    return d


def load_nsl():
    cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
            'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
            'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
            'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']
    tr = pd.read_csv(os.path.join(DATA, 'nslkdd/KDDTrain+.txt'), names=cols)
    te = pd.read_csv(os.path.join(DATA, 'nslkdd/KDDTest+.txt'), names=cols)
    cats = ['protocol_type', 'service', 'flag']
    feats = [c for c in cols if c not in ('label', 'difficulty')]
    maps = {c: {v: i for i, v in enumerate(sorted(tr[c].astype(str).unique()))} for c in cats}
    def enc(df):
        X = df[feats].copy()
        for c in cats:
            X[c] = df[c].astype(str).map(maps[c]).fillna(-1).astype(int)
        return X.astype(float).values.astype(np.float32), (df['label'] != 'normal').astype(int).values, df['label'].astype(str).values
    Xtr, ytr, ctr = enc(tr); Xte, yte, cte = enc(te)
    return dict(name='nsl', feats=feats, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, cte=cte, ctr=ctr)


# ----------------------------------------------------------------------------- models
def train_zoo(ds, which=None, target_fpr=0.05):
    Xfit, Xcal, yfit, ycal = train_test_split(ds['Xtr'], ds['ytr'], test_size=0.2, random_state=SEED, stratify=ds['ytr'])
    Xtr, ytr = Xfit, yfit
    sc = StandardScaler().fit(Xtr)
    mean, std = sc.mean_.astype(np.float32), np.where(sc.scale_ < 1e-9, 1.0, sc.scale_).astype(np.float32)
    specs = {
        'lr':   lambda: Inspector('lr', LogisticRegression(max_iter=2000, C=1.0), mean, std, 'linear'),
        'dt_s': lambda: Inspector('dt_s', DecisionTreeClassifier(max_depth=4, random_state=SEED), None, None, 'tree'),
        'dt':   lambda: Inspector('dt', DecisionTreeClassifier(max_depth=12, random_state=SEED), None, None, 'tree'),
        'rf':   lambda: Inspector('rf', RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEED), None, None, 'tree'),
        'lgbm': lambda: Inspector('lgbm', None, None, None, 'lgbm'),
        'mlp':  lambda: Inspector('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, random_state=SEED), mean, std, 'mlp'),
    }
    zoo, rows = {}, []
    for name, mk in specs.items():
        if which and name not in which: continue
        ins = mk()
        t0 = time.time()
        if name == 'lgbm':
            params = dict(objective='binary', num_leaves=63, learning_rate=0.1, n_estimators=200,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                          num_threads=1, verbose=-1)
            m = lgb.LGBMClassifier(**params).fit(Xtr, ytr)
            ins.model = m.booster_
        else:
            ins.model.fit(ins._prep(Xtr), ytr)
        ttrain = time.time() - t0
        ins.n_features = Xtr.shape[1]
        # operating point: smallest threshold whose calibration-split FPR <= target_fpr
        pc = ins.score_batch(Xcal); neg = np.sort(pc[ycal == 0])
        ins.thr = float(min(0.999, max(0.001, neg[min(len(neg) - 1, int(np.ceil((1 - target_fpr) * len(neg))))])))
        zoo[name] = ins
        print(f'[{ds["name"]}] trained {name} in {ttrain:.1f}s', flush=True)
        rows.append((name, ttrain))
    return zoo, dict(rows)


def evaluate(ds, zoo, ttrain):
    Xte, yte = ds['Xte'], ds['yte']
    rows = []
    rng = np.random.RandomState(0)
    probe = rng.choice(len(Xte), 300, replace=False)
    for name, ins in zoo.items():
        t0 = time.time(); p = ins.score_batch(Xte); tb = (time.time() - t0) / len(Xte) * 1e6
        yhat05 = (p >= 0.5).astype(int)
        yhat = (p >= ins.thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
        # single-row latency (what the service pays per request)
        lat = []
        for i in probe:
            x = Xte[i]
            t0 = time.perf_counter(); ins.score(x); lat.append((time.perf_counter() - t0) * 1e6)
        lat = np.array(lat)
        rows.append(dict(dataset=ds['name'], model=name, thr=ins.thr, acc=accuracy_score(yte, yhat),
                         precision=precision_score(yte, yhat, zero_division=0), recall=recall_score(yte, yhat),
                         f1=f1_score(yte, yhat), auc=roc_auc_score(yte, p), fpr=fp / max(1, fp + tn),
                         acc05=accuracy_score(yte, yhat05), recall05=recall_score(yte, yhat05), f1_05=f1_score(yte, yhat05),
                         fpr05=float(((yhat05 == 1) & (yte == 0)).sum() / max(1, (yte == 0).sum())),
                         train_s=ttrain[name], batch_us=tb, single_us_med=np.median(lat), single_us_p95=np.percentile(lat, 95)))
        print(rows[-1], flush=True)
    return rows


def export_trace(ds, zoo, n=24000, mal_frac=0.3):
    """Stratified replay trace drawn from the test partition with a fixed malicious fraction."""
    rng = np.random.RandomState(SEED)
    Xte, yte, cte = ds['Xte'], ds['yte'], ds['cte']
    n_mal = int(n * mal_frac); n_ben = n - n_mal
    ben = np.where(yte == 0)[0]; mal = np.where(yte == 1)[0]
    sel_b = rng.choice(ben, min(n_ben, len(ben)), replace=False)
    # proportional stratification over attack categories (min 1 of each if present)
    cats = pd.Series(cte[mal]).value_counts()
    sel_m = []
    for c, cnt in cats.items():
        k = max(1, int(round(n_mal * cnt / len(mal))))
        pool = mal[cte[mal] == c]
        sel_m.extend(rng.choice(pool, min(k, len(pool)), replace=False))
    sel = np.concatenate([sel_b, np.array(sel_m)])
    rng.shuffle(sel)
    df = pd.DataFrame(Xte[sel], columns=ds['feats'])
    df.insert(0, 'attack_cat', cte[sel]); df.insert(0, 'label', yte[sel]); df.insert(0, 'idx', np.arange(len(sel)))
    # stage scores from the zoo for offline cascade design
    for name in ('dt_s', 'lr'):
        if name in zoo:
            df[f's_{name}'] = zoo[name].score_batch(Xte[sel])
    path = os.path.join(ART, f'trace_{ds["name"]}.csv')
    df.to_csv(path, index=False, float_format='%.6g')
    print(f'trace {ds["name"]}: {len(df)} rows, {df.label.mean():.3f} malicious -> {path}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='unsw,cic,nsl')
    args = ap.parse_args()
    all_rows = []
    loaders = {'unsw': load_unsw, 'cic': load_cic, 'nsl': load_nsl}
    for name in args.datasets.split(','):
        ds = loaders[name]()
        print(f'{name}: train {ds["Xtr"].shape} ({ds["ytr"].mean():.3f} mal), test {ds["Xte"].shape} ({ds["yte"].mean():.3f} mal)', flush=True)
        zoo, ttrain = train_zoo(ds)
        rows = evaluate(ds, zoo, ttrain)
        all_rows.extend(rows)
        joblib.dump(zoo, os.path.join(ART, f'models_{name}.joblib'))
        export_trace(ds, zoo)
        meta = dict(name=name, feats=ds['feats'], n_train=int(len(ds['ytr'])), n_test=int(len(ds['yte'])),
                    mal_train=float(ds['ytr'].mean()), mal_test=float(ds['yte'].mean()),
                    test_cats=pd.Series(ds['cte']).value_counts().to_dict())
        json.dump(meta, open(os.path.join(ART, f'meta_{name}.json'), 'w'), indent=1, default=str)
        if name == 'cic':
            # held-out-day robustness: train on all other days, test on the held-out day (lgbm + rf)
            Xall, yall, call, dall = ds['full']
            hrows = []
            for day in sorted(set(dall)):
                tr = dall != day; te = dall == day
                rng = np.random.RandomState(SEED)
                tri = np.where(tr)[0]
                if len(tri) > 300000: tri = rng.choice(tri, 300000, replace=False)
                m = lgb.LGBMClassifier(objective='binary', num_leaves=63, learning_rate=0.1, n_estimators=200,
                                       random_state=SEED, num_threads=1, verbose=-1).fit(Xall[tri], yall[tri])
                p = m.predict_proba(Xall[te])[:, 1]; yh = (p >= 0.5).astype(int); yt = yall[te]
                tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
                hrows.append(dict(heldout_day=day, n=int(te.sum()), n_mal=int(yt.sum()),
                                  recall=tp / max(1, tp + fn), fpr=fp / max(1, fp + tn),
                                  precision=tp / max(1, tp + fp), f1=f1_score(yt, yh, zero_division=0)))
                print(hrows[-1], flush=True)
            pd.DataFrame(hrows).to_csv(os.path.join(RES, 'cic_heldout_day.csv'), index=False)
    pd.DataFrame(all_rows).to_csv(os.path.join(RES, 'offline_models.csv'), index=False)
    print('done')


if __name__ == '__main__':
    main()


class KNNNovelty:
    """Nearest-neighbour novelty inspector. score(x) = percentile of the mean distance to the k nearest
    benign reference flows (signed-log, standardised features) within benign calibration distances.
    Cost grows linearly with the reference-set size."""
    def __init__(self, name, R, mu, sd, k, cal_sorted, thr):
        self.name, self.kind, self.k = name, 'knn', k
        self.R = np.ascontiguousarray(R, dtype=np.float32); self.Rn = (self.R ** 2).sum(1)
        self.mu, self.sd = mu.astype(np.float32), sd.astype(np.float32)
        self.cal = np.sort(cal_sorted).astype(np.float32); self.thr = thr; self.n_features = len(mu)

    def _t(self, X):
        return ((np.sign(X) * np.log1p(np.abs(X)) - self.mu) / self.sd).astype(np.float32)

    def raw_batch(self, X, chunk=256):
        Z = self._t(np.atleast_2d(X)); out = np.empty(len(Z), np.float32)
        for i in range(0, len(Z), chunk):
            z = Z[i:i + chunk]
            d2 = self.Rn[None, :] - 2.0 * (z @ self.R.T) + (z ** 2).sum(1)[:, None]
            np.maximum(d2, 0, out=d2)
            part = np.partition(d2, self.k, axis=1)[:, :self.k]
            out[i:i + chunk] = np.sqrt(part).mean(1)
        return out

    def score_batch(self, X):
        return np.searchsorted(self.cal, self.raw_batch(X), side='right') / len(self.cal)

    def score(self, x):
        z = self._t(x)
        d2 = self.Rn - 2.0 * (self.R @ z) + float(z @ z)
        np.maximum(d2, 0, out=d2)
        s = float(np.sqrt(np.partition(d2, self.k)[:self.k]).mean())
        return float(np.searchsorted(self.cal, s, side='right')) / len(self.cal)


class HNSWNovelty(KNNNovelty):
    """k-NN novelty inspector on a FAISS HNSW graph (approximate search); same transform and calibration as KNNNovelty."""
    def __init__(self, name, Rz, mu, sd, k, thr, M=32, ef_construction=80, ef_search=64):
        import faiss
        faiss.omp_set_num_threads(1)
        self.name, self.kind, self.k, self.thr, self.ef_search = name, 'hnsw', k, thr, ef_search
        self.mu, self.sd = mu.astype(np.float32), sd.astype(np.float32); self.n_features = len(mu)
        idx = faiss.IndexHNSWFlat(Rz.shape[1], M); idx.hnsw.efConstruction = ef_construction
        idx.add(np.ascontiguousarray(Rz, dtype=np.float32)); idx.hnsw.efSearch = ef_search
        self._idx = idx; self.blob = faiss.serialize_index(idx); self.cal = np.zeros(1, np.float32)

    def __getstate__(self):
        d = dict(self.__dict__); d.pop('_idx', None); return d

    def __setstate__(self, d):
        import faiss
        faiss.omp_set_num_threads(1)
        self.__dict__.update(d); self._idx = faiss.deserialize_index(self.blob); self._idx.hnsw.efSearch = self.ef_search

    def raw_batch(self, X, chunk=4096):
        Z = self._t(np.atleast_2d(X)); out = np.empty(len(Z), np.float32)
        for i in range(0, len(Z), chunk):
            D, _ = self._idx.search(Z[i:i + chunk], self.k); out[i:i + chunk] = np.sqrt(np.maximum(D, 0)).mean(1)
        return out

    def score(self, x):
        D, _ = self._idx.search(self._t(x)[None, :], self.k)
        return float(np.searchsorted(self.cal, float(np.sqrt(np.maximum(D, 0)).mean()), side='right')) / len(self.cal)


class HNSWLOF(HNSWNovelty):
    """Local outlier factor on an HNSW graph: reference k-distances and local reachability densities are precomputed;
    a query needs one approximate k-NN search. score = percentile of LOF among benign calibration flows."""
    def __init__(self, name, Rz, mu, sd, k, thr, **kw):
        super().__init__(name, Rz, mu, sd, k, thr, **kw)
        self.kind = 'hnsw_lof'
        D, I = self._idx.search(np.ascontiguousarray(Rz, dtype=np.float32), k + 1)
        D = np.sqrt(np.maximum(D[:, 1:], 0)); I = I[:, 1:]
        self.kdist = D[:, -1].astype(np.float32)
        reach = np.maximum(D, self.kdist[I]); self.lrd = (1.0 / np.maximum(reach.mean(1), 1e-9)).astype(np.float32)

    def _lof(self, D, I):
        D = np.sqrt(np.maximum(D, 0)); reach = np.maximum(D, self.kdist[I])
        lrd_q = 1.0 / np.maximum(reach.mean(1), 1e-9)
        return self.lrd[I].mean(1) / lrd_q

    def raw_batch(self, X, chunk=4096):
        Z = self._t(np.atleast_2d(X)); out = np.empty(len(Z), np.float32)
        for i in range(0, len(Z), chunk):
            D, I = self._idx.search(Z[i:i + chunk], self.k); out[i:i + chunk] = self._lof(D, I)
        return out

    def score(self, x):
        D, I = self._idx.search(self._t(x)[None, :], self.k)
        return float(np.searchsorted(self.cal, float(self._lof(D, I)[0]), side='right')) / len(self.cal)
