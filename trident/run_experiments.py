#!/usr/bin/env python3
"""
Experiment driver. Each experiment is a function that brings up stack configurations, drives load,
and appends one row per run to results/<exp>.csv (raw per-run records are kept in results/raw/<exp>/).

usage: run_experiments.py <exp> [--reps N] [--quick]
"""
import os, sys, time, json, argparse, itertools, subprocess, random
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'trident'))
from harness import Stack, attack, RUN, BIN
from analyze import summarize, timeline
from policy_gen import write_bundle
from gen_targets import build

ART = os.path.join(ROOT, 'artifacts'); RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)
SLO_MS = 25.0


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def append_row(exp, row):
    path = os.path.join(RES, f'{exp}.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}, default=str) + '\n')


def raw_prefix(exp, name):
    d = os.path.join(RES, 'raw', exp); os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def ensure_targets(tag, dataset='unsw', bundle=None, seed=1, auth_mode='rs256', sz='1k', n=None):
    bundle = bundle or os.path.join(ROOT, 'policy', 'bundle')
    tpath = os.path.join(ART, f'targets_{tag}.json')
    if not os.path.exists(tpath):
        build(dataset, bundle, tag, seed=seed, n=n, auth_mode=auth_mode, sz=sz)
    return tpath, os.path.join(ART, f'truth_{tag}.csv')


_DONE = {}


def fingerprint():
    """platform fingerprint recorded with every run: CPU model, a memory-bound matvec and a Python loop."""
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?')
    a = np.random.RandomState(0).rand(50000, 78).astype(np.float32); x = a[0]
    t0 = time.perf_counter()
    for _ in range(20):
        a @ x
    mv = (time.perf_counter() - t0) / 20 * 1e6
    t0 = time.perf_counter(); acc = 0
    for i in range(300000):
        acc += i
    return dict(fp_cpu=cpu, fp_matvec_us=mv, fp_pyloop_ms=(time.perf_counter() - t0) * 1e3, fp_boot=open('/proc/sys/kernel/random/boot_id').read().strip())


def already_done(exp, name):
    """a run is complete when its compact client record and PoP log exist."""
    d = os.path.join(RES, 'raw', exp)
    return os.path.exists(os.path.join(d, name + '.client.csv')) and os.path.exists(os.path.join(d, name + '.access.log'))


def run_one(exp, name, stack_kwargs, targets, truth, rate, duration, extra=None, slo=SLO_MS, max_workers=None,
            chaos=None, probe=None, keep_stack=None):
    """chaos: list of (t_offset_s, component) kills; probe: supervisor probe interval (s)."""
    if already_done(exp, name):
        log(f'{exp}/{name}: skip (done)')
        return None
    s = keep_stack or Stack(**stack_kwargs)
    if keep_stack is None:
        s.up()
    try:
        if probe is not None:
            # probe phase drawn per run (seeded by the run name) so that detection delay is uniform in [0, probe)
            phase = random.Random(name).uniform(0, probe)
            s.start_supervisor(probe_interval=probe, phase=phase)
            extra = dict(extra or {}, probe_phase=phase)
        prefix = raw_prefix(exp, name)
        fp = fingerprint()
        if chaos:
            # run the attack in a subprocess-like thread so kills can be timed against t0
            import threading
            res = {}
            def go():
                res['rep'] = attack(targets, rate, duration, prefix, max_workers=max_workers)
            th = threading.Thread(target=go); th.start()
            t0 = time.time()
            for off, comp in chaos:
                time.sleep(max(0, t0 + off - time.time()))
                s.kill(comp)
            th.join()
            rep = res['rep']
        else:
            rep = attack(targets, rate, duration, prefix, max_workers=max_workers)
        if probe is not None:
            s.stop_supervisor()
        m, df = summarize(prefix, truth, slo_ms=slo)
        row = dict(exp=exp, name=name, rate=rate, duration=duration, **{k: v for k, v in stack_kwargs.items()})
        row.update({k: v for k, v in m.items() if k != 'status_dist'})
        row['status_dist'] = json.dumps(m['status_dist'])
        row['vegeta_rate'] = rep['rate']; row['vegeta_p99'] = rep['latencies']['99th'] / 1e6; row['vegeta_p50'] = rep['latencies']['50th'] / 1e6
        row['vegeta_mean'] = rep['latencies']['mean'] / 1e6; row['vegeta_bytes_in'] = rep['bytes_in']['total']
        row['vegeta_throughput'] = rep['throughput']; row['vegeta_duration'] = rep['duration'] / 1e9
        met = s.metrics(); row['svc_metrics'] = json.dumps({k: {kk: vv for kk, vv in (v or {}).items() if kk != 'ctl_hist'} for k, v in met.items()})
        if extra: row.update(extra)
        row.update(fp)
        if chaos:
            row['events'] = json.dumps([(round(t - rep['t_start'], 3), e) for t, e in s.events])
            tl = timeline(df, rep['t_start'], 0.5); tl.to_csv(prefix + '.timeline.csv', index=False)
            # controller history for adaptive runs
        for k, v in met.items():
            if v and v.get('ctl_hist'):
                json.dump(v['ctl_hist'], open(prefix + f'.ctl{k}.json', 'w'))
        append_row(exp, row)
        log(f'{exp}/{name}: rate={rate} p50={row["lat_p50"]:.2f} p99={row["lat_p99"]:.2f} err={row["err_frac"]:.3f} '
            f'thr={row["throughput_nonerr"]:.0f} conf={row["ztna_conformance"]:.4f} rec={row["recall"]:.3f} fpr={row["fpr"]:.3f} '
            f'esc={row["esc_rate"]:.2f} expo={row["exposure"]}')
        return row
    finally:
        if keep_stack is None:
            s.down()


# ============================================================================ E1: policy decision path
def e1_policy(reps=2, quick=False):
    exp = 'e1_policy'
    rate, dur = 200, 12
    pads = [0, 100, 1000, 10000] if not quick else [0, 1000]
    for rep in range(reps):
        for style in ('eq', 'glob'):
            for pad in pads:
                if style == 'glob' and pad == 0: continue
                bdir = os.path.join(ROOT, 'policy', f'bundle_{style}_{pad}')
                write_bundle(bdir, K=100, pad=pad, pad_style=style)
                tag = 'unsw_default'
                targets, truth = ensure_targets(tag)
                # OPA micro-benchmark (pure evaluation cost, no HTTP)
                inp = {"auth_mode": "rs256", "token": json.loads(open(targets).readline())['header']['Authorization'][0][7:],
                       "now": int(time.time()), "resource": "r1", "path": "/app/r1", "geo": "US",
                       "device": {"posture": "compliant", "managed": True}}
                json.dump(inp, open(os.path.join(RUN, 'bench_in.json'), 'w'))
                b = subprocess.run([os.path.join(BIN, 'opa'), 'bench', '-d', bdir, '-i', os.path.join(RUN, 'bench_in.json'),
                                    'data.ztna.decision', '-f', 'json', '--count', '1'], capture_output=True, text=True)
                try:
                    bj = json.loads(b.stdout); ns = bj['T'] / bj['N']
                except Exception:
                    ns = np.nan
                run_one(exp, f'{style}_pad{pad}_rs256_rep{rep}', dict(bundle=bdir, mode='none', replicas=1, auth_mode='rs256'),
                        targets, truth, rate, dur, extra=dict(style=style, pad=pad, rep=rep, bench_ns_per_op=ns))
        # JWT ablation: trusted-header identity (no signature verification)
        bdir = os.path.join(ROOT, 'policy', 'bundle_eq_0'); write_bundle(bdir, K=100)
        targets, truth = ensure_targets('unsw_none', auth_mode='none')
        run_one(exp, f'eq_pad0_none_rep{rep}', dict(bundle=bdir, mode='none', replicas=1, auth_mode='none'),
                targets, truth, rate, dur, extra=dict(style='eq', pad=0, rep=rep, bench_ns_per_op=np.nan))
        # policy bypass baseline (no PDP call at all)
        run_one(exp, f'bypass_rep{rep}', dict(bundle=bdir, mode='none', replicas=1, ztna=0),
                targets, truth, rate, dur, extra=dict(style='bypass', pad=0, rep=rep, bench_ns_per_op=np.nan))


def e1b_policy(reps=2, quick=False):
    """non-indexable rule styles (startswith / regex) and the corrected trusted-header identity mode."""
    exp = 'e1_policy'
    rate, dur = 200, 12
    targets, truth = ensure_targets('unsw_default')
    for rep in range(reps):
        for style in ('sw', 're'):
            for pad in ([100, 1000, 10000] if not quick else [1000]):
                bdir = os.path.join(ROOT, 'policy', f'bundle_{style}_{pad}')
                write_bundle(bdir, K=100, pad=pad, pad_style=style)
                inp = {"auth_mode": "rs256", "token": json.loads(open(targets).readline())['header']['Authorization'][0][7:],
                       "now": int(time.time()), "resource": "r1", "path": "/app/r1", "geo": "US",
                       "device": {"posture": "compliant", "managed": True}}
                json.dump(inp, open(os.path.join(RUN, 'bench_in.json'), 'w'))
                b = subprocess.run([os.path.join(BIN, 'opa'), 'bench', '-d', bdir, '-i', os.path.join(RUN, 'bench_in.json'),
                                    'data.ztna.decision', '-f', 'json', '--count', '1'], capture_output=True, text=True)
                try:
                    bj = json.loads(b.stdout); ns = bj['T'] / bj['N']
                except Exception:
                    ns = np.nan
                run_one(exp, f'{style}_pad{pad}_rs256_rep{rep}', dict(bundle=bdir, mode='none', replicas=1, auth_mode='rs256'),
                        targets, truth, rate, dur, extra=dict(style=style, pad=pad, rep=rep, bench_ns_per_op=ns))
        bdir = os.path.join(ROOT, 'policy', 'bundle_eq_0'); write_bundle(bdir, K=100)
        t2, tr2 = ensure_targets('unsw_none2', auth_mode='none')
        run_one(exp, f'eq_pad0_none2_rep{rep}', dict(bundle=bdir, mode='none', replicas=1, auth_mode='none'),
                t2, tr2, rate, dur, extra=dict(style='eq', pad=0, rep=rep, bench_ns_per_op=np.nan, auth_mode='none'))


# ============================================================================ E3: pipeline performance
CONFIGS = {
    'C0_passthrough': dict(ztna=0, mode='none'),
    'C1_ztna': dict(ztna=1, mode='none'),
    'C2_ztna_light': dict(ztna=1, mode='light', light='lr'),
    'C3_ztna_lgbm': dict(ztna=1, mode='deep', deep='lgbm'),
    'C4_ztna_deep': dict(ztna=1, mode='deep', deep='rf'),
}


def e3_pipeline(reps=3, quick=False):
    exp = 'e3_pipeline'
    targets, truth = ensure_targets('unsw_default')
    rates = [50, 100, 200, 400, 800, 1200] if not quick else [100, 400]
    dur = 15
    for rep in range(reps):
        for cname, ckw in CONFIGS.items():
            s = Stack(replicas=1, **ckw); s.up()
            try:
                for rate in rates:
                    run_one(exp, f'{cname}_r{rate}_rep{rep}', dict(replicas=1, **ckw), targets, truth, rate, dur,
                            extra=dict(config=cname, rep=rep, loop='open'), keep_stack=s)
                # closed-loop capacity probe
                run_one(exp, f'{cname}_closed_rep{rep}', dict(replicas=1, **ckw), targets, truth, 0, 10,
                        extra=dict(config=cname, rep=rep, loop='closed'), max_workers=64, keep_stack=s)
            finally:
                s.down()


def e3_payload(reps=2, quick=False):
    """data-plane goodput vs payload size (closed loop)."""
    exp = 'e3_payload'
    for rep in range(reps):
        for cname in ('C0_passthrough', 'C4_ztna_deep', 'C3_ztna_lgbm'):
            ckw = CONFIGS[cname]
            s = Stack(replicas=1, **ckw); s.up()
            try:
                for sz in (['1k', '16k', '256k', '1m'] if not quick else ['1k', '1m']):
                    targets, truth = ensure_targets(f'unsw_sz{sz}', sz=sz, n=6000)
                    run_one(exp, f'{cname}_{sz}_rep{rep}', dict(replicas=1, **ckw), targets, truth, 0, 10,
                            extra=dict(config=cname, rep=rep, sz=sz), max_workers=32, keep_stack=s)
            finally:
                s.down()


# ============================================================================ E4: adaptive cascade
MODES = {
    'none': dict(mode='none'),
    'light': dict(mode='light'),
    'deep': dict(mode='deep'),
    'sample50': dict(mode='sample:0.5'),
    'cascade50': dict(mode='cascade:0.5'),
    'adaptive': dict(mode='adaptive', target_inflight=4),
}


def e4_cascade(reps=3, quick=False, dataset='unsw', modes=None, rates=None, exp='e4_cascade', light='lr', deep='rf'):
    targets, truth = ensure_targets(f'{dataset}_default', dataset=dataset)
    rates = rates or ([100, 150, 200, 300, 400, 600] if not quick else [150, 400])
    dur = 15
    modes = modes or MODES
    for rep in range(reps):
        for mname, mkw in modes.items():
            kw = dict(replicas=1, ztna=1, dataset=dataset, light=light, deep=deep, **mkw)
            s = Stack(**kw); s.up()
            try:
                for rate in rates:
                    rname = (f'{dataset}_' if dataset != 'unsw' else '') + f'{mname}_r{rate}_rep{rep}'
                    run_one(exp, rname, kw, targets, truth, rate, dur,
                            extra=dict(mode_name=mname, rep=rep, dataset=dataset), keep_stack=s)
            finally:
                s.down()


def e4_ablation(reps=2, quick=False):
    """controller target, static band width, light-model choice."""
    modes = {}
    for tgt in (2, 8, 16):
        modes[f'adaptive_t{tgt}'] = dict(mode='adaptive', target_inflight=tgt)
    for w in (0.25, 0.75):
        modes[f'cascade{int(w*100)}'] = dict(mode=f'cascade:{w}')
    modes['sample25'] = dict(mode='sample:0.25')
    e4_cascade(reps=reps, quick=quick, modes=modes, rates=[150, 300, 600], exp='e4_ablation')
    e4_cascade(reps=reps, quick=quick, modes={'adaptive_dts': dict(mode='adaptive', target_inflight=4),
                                              'cascade50_dts': dict(mode='cascade:0.5')},
               rates=[150, 300, 600], exp='e4_ablation', light='dt_s')


# ============================================================================ E5: reliability
def e5_reliability(reps=3, quick=False):
    exp = 'e5_reliability'
    targets, truth = ensure_targets('unsw_default')
    rate, dur, t_kill = 150, 40, 10
    probes = [1, 2, 5] if not quick else [1, 5]
    for rep in range(reps):
        # A: decision-replica crash, fail-closed data plane, 1 vs 2 replicas
        for replicas in (1, 2):
            for probe in probes:
                kw = dict(replicas=replicas, ztna=1, mode='deep', deep='lgbm')
                run_one(exp, f'A_dec_rep{replicas}_p{probe}_rep{rep}', kw, targets, truth, rate, dur,
                        extra=dict(scenario='A_replica_crash', probe=probe, rep=rep), chaos=[(t_kill, 'dec1')], probe=probe)
        # B: PDP (OPA) crash, fail-open vs fail-closed policy path
        for fo in (0, 1):
            for probe in probes:
                kw = dict(replicas=1, ztna=1, mode='deep', deep='lgbm', fail_open=fo)
                run_one(exp, f'B_opa_fo{fo}_p{probe}_rep{rep}', kw, targets, truth, rate, dur,
                        extra=dict(scenario='B_pdp_crash', probe=probe, rep=rep), chaos=[(t_kill, 'opa')], probe=probe)
        # C: data-plane fail-open (bypass authz on 5xx) with replica crash
        kw = dict(replicas=1, ztna=1, mode='deep', deep='lgbm', nginx_failopen=True)
        run_one(exp, f'C_nginx_failopen_p2_rep{rep}', kw, targets, truth, rate, dur,
                extra=dict(scenario='C_dataplane_failopen', probe=2, rep=rep), chaos=[(t_kill, 'dec1')], probe=2)
        # baseline without failure
        kw = dict(replicas=1, ztna=1, mode='deep', deep='lgbm')
        run_one(exp, f'nofail_rep{rep}', kw, targets, truth, rate, dur, extra=dict(scenario='none', probe=0, rep=rep))


# ============================================================================ E6: robustness / reproducibility
def e6_robustness(reps=5, quick=False):
    exp = 'e6_repro'
    targets, truth = ensure_targets('unsw_default')
    kw = dict(replicas=1, ztna=1, mode='deep', deep='lgbm')
    s = Stack(**kw); s.up()
    try:
        for rep in range(reps):
            run_one(exp, f'ref_rep{rep}', kw, targets, truth, 200, 15, extra=dict(kind='repeat', rep=rep), keep_stack=s)
        for seed in (2, 3, 4):
            t2, tr2 = ensure_targets(f'unsw_seed{seed}', seed=seed)
            run_one(exp, f'seed{seed}', kw, t2, tr2, 200, 15, extra=dict(kind='seed', rep=seed), keep_stack=s)
    finally:
        s.down()
    # cross-dataset: cascade comparison on CIC-IDS2017 and NSL-KDD traces
    for ds in ('cic', 'nsl'):
        e4_cascade(reps=2 if not quick else 1, quick=quick, dataset=ds, rates=[150, 400],
                   modes={k: MODES[k] for k in ('light', 'deep', 'cascade50', 'adaptive')}, exp='e6_crossdata')


# ============================================================================ revision experiments
def e4c_direct(reps=3, quick=False):
    """full LightGBM inspection on every flow vs. the cascade modes, same harness and loads (UNSW-NB15)."""
    e4_cascade(reps=reps, modes={'full_lgbm': dict(mode='deep')}, rates=[100, 150, 200, 300, 400, 600, 800], exp='e4_cascade', deep='lgbm')
    e4_cascade(reps=reps, modes={'adaptive': MODES['adaptive'], 'light': MODES['light']}, rates=[800], exp='e4_cascade')


E7_MODES = {
    'sup': dict(mode='none'),                      # supervised LightGBM only
    'deep': dict(mode='deep'),                     # supervised + 200k-reference k-NN novelty on every admitted flow
    'sample30': dict(mode='sample:0.3'),           # novelty on a random 30 %
    'static30': dict(mode='cascade:0.3'),          # novelty on the 30 % most novel flows (light k-NN ranking)
    'adaptive': dict(mode='adaptive', target_inflight=4),
}


def e7_unseen(reps=2, quick=False):
    """unique-capability deep stage: unseen attack families (CIC-IDS2017 leave-families-out trace)."""
    exp = 'e7_unseen'
    targets, truth = ensure_targets('cicuf_default', dataset='cicuf')
    for rep in range(reps):
        for mname, mkw in E7_MODES.items():
            kw = dict(replicas=1, ztna=1, dataset='cicuf', light='nov_s', deep='nov_d', always='lgbm', thr_light=1.0, thr_deep=0.99, **mkw)
            s = Stack(**kw); s.up(timeout=120)
            try:
                for rate in [100, 200, 400, 600]:
                    run_one(exp, f'{mname}_r{rate}_rep{rep}', kw, targets, truth, rate, 15, extra=dict(mode_name=mname, rep=rep), keep_stack=s)
            finally:
                s.down()


def e3c_oppoint(reps=3, quick=False):
    """ZTNA + LightGBM at calibrated FPR targets 5 %, 1 %, 0.1 % (UNSW-NB15, 200 req/s)."""
    exp = 'e3_oppoint'
    op = pd.read_csv(os.path.join(RES, 'tables', 'operating_points.csv'))
    op = op[(op.dataset == 'unsw') & (op.model == 'lgbm')].set_index('target_fpr')['thr']
    targets, truth = ensure_targets('unsw_default')
    for tgt in (0.05, 0.01, 0.001):
        kw = dict(replicas=1, ztna=1, mode='deep', deep='lgbm', thr_deep=float(op[tgt]))
        s = Stack(**kw); s.up()
        try:
            for rep in range(reps):
                run_one(exp, f'fpr{tgt}_rep{rep}', kw, targets, truth, 200, 15, extra=dict(target_fpr=tgt, rep=rep), keep_stack=s)
        finally:
            s.down()


def e5b_load(reps=3, quick=False):
    """replica crash at higher load: k=2 at 300/600 req/s and k=1 at 600 req/s (probe 2 s)."""
    exp = 'e5_load'
    targets, truth = ensure_targets('unsw_default')
    for rep in range(reps):
        for k, rate in ((2, 300), (2, 600), (1, 600)):
            kw = dict(replicas=k, ztna=1, mode='deep', deep='lgbm')
            run_one(exp, f'A_dec_rep{k}_r{rate}_rep{rep}', kw, targets, truth, rate, 40,
                    extra=dict(scenario='A_replica_crash', probe=2, rep=rep), chaos=[(10, 'dec1')], probe=2)


def e6b_long(reps=3, quick=False):
    exp = 'e6_long'
    targets, truth = ensure_targets('unsw_default')
    kw = dict(replicas=1, ztna=1, mode='deep', deep='lgbm')
    s = Stack(**kw); s.up()
    try:
        for rep in range(reps):
            run_one(exp, f'long_rep{rep}', kw, targets, truth, 200, 60, extra=dict(rep=rep), keep_stack=s)
    finally:
        s.down()


def e8_remote(reps=2, quick=False):
    """remote PDP placement: emulated round-trip time between decision service and PDP."""
    exp = 'e8_remote'
    targets, truth = ensure_targets('unsw_default')
    for rtt in (0, 2, 5, 10, 20):
        kw = dict(replicas=1, ztna=1, mode='deep', deep='lgbm', opa_rtt_ms=float(rtt))
        s = Stack(**kw); s.up()
        try:
            for rep in range(reps):
                run_one(exp, f'rtt{rtt}_open_rep{rep}', kw, targets, truth, 200, 15, extra=dict(rtt=rtt, rep=rep, loop='open'), keep_stack=s)
                run_one(exp, f'rtt{rtt}_closed_rep{rep}', kw, targets, truth, 0, 10, extra=dict(rtt=rtt, rep=rep, loop='closed'),
                        max_workers=64, keep_stack=s)
        finally:
            s.down()


def e4d_direct(reps=2, quick=False):
    """direct comparison within one session: full LightGBM vs. the forest-based modes (UNSW-NB15)."""
    exp = 'e4_direct'
    targets, truth = ensure_targets('unsw_default')
    modes = {'full_lgbm': dict(mode='deep', deep='lgbm'), 'adaptive': dict(mode='adaptive', target_inflight=4, deep='rf'),
             'deep_rf': dict(mode='deep', deep='rf'), 'light': dict(mode='light', deep='rf')}
    for rep in range(reps):
        for mname, mkw in modes.items():
            kw = dict(replicas=1, ztna=1, light='lr', **mkw)
            s = Stack(**kw); s.up()
            try:
                for rate in [200, 400, 600, 800]:
                    run_one(exp, f'{mname}_r{rate}_rep{rep}', kw, targets, truth, rate, 15, extra=dict(mode_name=mname, rep=rep), keep_stack=s)
            finally:
                s.down()


E7C_MODES = {
    'sup': dict(mode='none', deep='nov_d'),
    'knn_full': dict(mode='deep', deep='nov_d'),
    'hnsw_full': dict(mode='deep', deep='nov_h'),
    'lof_full': dict(mode='deep', deep='lof_h'),
    'adaptive': dict(mode='adaptive', target_inflight=4, deep='nov_d'),
    'random30': dict(mode='sample:0.3', deep='nov_d'),
    'static_e30': dict(mode='cascade:0.043', deep='nov_d'),
    'random60': dict(mode='sample:0.6', deep='nov_d'),
    'static30': dict(mode='cascade:0.3', deep='nov_d'),
    'adaptive_lof': dict(mode='adaptive', target_inflight=4, deep='lof_h'),
}


def e7c_unseen(reps=2, quick=False):
    """unseen families, second round: efficient and density-based baselines and budget-matched pairs, one session."""
    exp = 'e7_unseen'
    targets, truth = ensure_targets('cicuf_default', dataset='cicuf')
    for rep in range(reps):
        for mname, mkw in E7C_MODES.items():
            kw = dict(replicas=1, ztna=1, dataset='cicuf', light='nov_s', always='lgbm', thr_light=1.0, thr_deep=0.99, **mkw)
            s = Stack(**kw); s.up(timeout=180)
            try:
                for rate in [200, 400, 600]:
                    run_one(exp, f'{mname}_r{rate}_rep{rep}', kw, targets, truth, rate, 15, extra=dict(mode_name=mname, rep=rep), keep_stack=s)
            finally:
                s.down()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exp'); ap.add_argument('--reps', type=int, default=None); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    fn = {'e1': e1_policy, 'e1b': e1b_policy, 'e3': e3_pipeline, 'e3p': e3_payload, 'e4': e4_cascade, 'e4a': e4_ablation,
          'e5': e5_reliability, 'e6': e6_robustness, 'e4c': e4c_direct, 'e7': e7_unseen, 'e3c': e3c_oppoint,
          'e5b': e5b_load, 'e6b': e6b_long, 'e8': e8_remote, 'e4d': e4d_direct, 'e7c': e7c_unseen}[a.exp]
    kw = dict(quick=a.quick)
    if a.reps is not None: kw['reps'] = a.reps
    fn(**kw)
    log('finished ' + a.exp)
