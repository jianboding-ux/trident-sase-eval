#!/usr/bin/env python3
"""Generates the paper figures from results/tables/*.csv and raw per-run records."""
import os, json, glob, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, 'results', 'tables'); RAW = os.path.join(ROOT, 'results', 'raw')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3, 'lines.markersize': 4})
CFG_LABEL = {'C0_passthrough': 'C0 passthrough', 'C1_ztna': 'C1 ZTNA', 'C2_ztna_light': 'C2 ZTNA+light (LR)',
             'C3_ztna_lgbm': 'C3 ZTNA+LightGBM', 'C4_ztna_deep': 'C4 ZTNA+deep (RF)'}
MODE_LABEL = {'none': 'none', 'light': 'light (LR)', 'deep': 'deep (RF)', 'sample50': 'sample 50%', 'cascade50': 'static cascade w=0.5',
              'adaptive': 'adaptive cascade'}
MODE_STYLE = {'none': ('0.5', ':'), 'light': ('tab:green', '--'), 'deep': ('tab:red', '-'), 'sample50': ('tab:purple', '-.'),
              'cascade50': ('tab:orange', '--'), 'adaptive': ('tab:blue', '-')}


def savefig(name):
    from PIL import Image
    fig = plt.gcf(); axes = []
    for a in fig.axes:
        if not any(abs(a.get_position().x0 - b.get_position().x0) < 1e-3 and abs(a.get_position().y0 - b.get_position().y0) < 1e-3 for b in axes):
            axes.append(a)
    plt.tight_layout()
    if len(axes) > 1:
        for k, a in enumerate(axes):
            a.text(-0.16, 1.10, 'ABCDEFGH'[k], transform=a.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')
        plt.tight_layout()
    plt.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight', pad_inches=0.04)
    png = os.path.join(FIG, name + '.png'); plt.savefig(png, dpi=600, bbox_inches='tight', pad_inches=0.04); plt.close()
    im = Image.open(png).convert('RGB'); im.save(os.path.join(FIG, name + '.tif'), compression='tiff_lzw', dpi=(600, 600)); os.remove(png)
    print('wrote', name, im.size)


def shared_legend(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    if ax.get_legend(): ax.get_legend().remove()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False)


def load(exp):
    p = os.path.join(T, f'{exp}.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


# --------------------------------------------------------------------------- E1
def fig_policy():
    d = load('e1_policy')
    if d is None: return
    d = d[d['style'].notna()]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.75))
    styles = [('eq', 'equality (indexed)'), ('glob', 'glob (indexed)'), ('sw', 'prefix (startswith)'), ('re', 'regex')]
    for st, lab in styles:
        s = d[(d['style'] == st) & (d['auth_mode'] == 'rs256')]
        if not len(s): continue
        g = s.groupby('pad')
        x = g['pad'].first().values; x = np.where(x == 0, 1, x)
        ax[0].plot(x, g['t_opa_ms_p50'].mean().values, marker='o', label=lab)
        ax[1].plot(x, g['t_opa_ms_p99'].mean().values, marker='o', label=lab)
        b = g['bench_ns_per_op'].mean().values / 1000
        ax[2].plot(x, b, marker='s', label=lab)
    for a, t in zip(ax, ['decision path $t_{opa}$, p50 (ms)', 'decision path $t_{opa}$, p99 (ms)', 'OPA evaluation (opa bench), µs/op']):
        a.set_xscale('log'); a.set_xlabel('padding rules $P$ (1 = none)'); a.set_title(t, fontsize=8)
    ax[0].set_yscale('log'); ax[1].set_yscale('log'); ax[2].set_yscale('log')
    ax[0].legend(loc='upper left')
    savefig('fig_policy')
    # identity mode bars
    rows = []
    for st, am, lab in [('bypass', None, 'PDP bypassed'), ('eq', 'none', 'trusted header'), ('eq', 'rs256', 'RS256 verified')]:
        s = d[(d['style'] == st) & (d['pad'] == 0)]
        if am: s = s[s['auth_mode'] == am]
        if st == 'eq' and am == 'none':
            s2 = s[s['name'].str.contains('none2')]
            s = s2 if len(s2) else s
        if len(s): rows.append((lab, s['lat_p50'].mean(), s['lat_p99'].mean(), s['ztna_conformance'].mean(), s['ztna_bypass'].mean()))
    pd.DataFrame(rows, columns=['mode', 'p50', 'p99', 'conformance', 'bypass']).to_csv(os.path.join(T, 'e1_identity_modes.csv'), index=False)


# --------------------------------------------------------------------------- E2
def fig_frontier():
    d = pd.read_csv(os.path.join(ROOT, 'results', 'offline_models.csv'))
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.75))
    style = {'lr': ('o', 'tab:blue', 'LR'), 'dt_s': ('v', 'tab:orange', 'DT (depth 4)'), 'dt': ('^', 'tab:green', 'DT (depth 12)'),
             'rf': ('s', 'tab:red', 'RF'), 'lgbm': ('D', 'tab:purple', 'LightGBM'), 'mlp': ('P', 'tab:brown', 'MLP')}
    for a, ds, title in zip(ax, ['unsw', 'cic', 'nsl'], ['UNSW-NB15', 'CIC-IDS2017', 'NSL-KDD']):
        s_ = d[d.dataset == ds]
        for _, r in s_.iterrows():
            m, c, lab = style[r['model']]
            a.scatter(r['single_us_med'], r['f1'], marker=m, color=c, s=36, label=lab, zorder=3)
        a.set_xscale('log'); a.set_xlabel('inference cost (µs, median)'); a.set_title(title, fontsize=9)
        a.set_xlim(15, 15000)
    ax[0].set_ylabel('$F_1$ at calibrated threshold')
    shared_legend(fig, ax[0], 6)
    savefig('fig_frontier')


# --------------------------------------------------------------------------- E3
def fig_pipeline():
    d = load('e3_pipeline')
    if d is None: return
    o = d[d.loop == 'open']
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.85))
    for cfg, lab in CFG_LABEL.items():
        s = o[o.config == cfg].groupby('rate')
        if not len(s): continue
        x = s['rate'].first().values
        ax[0].plot(x, s['lat_p50'].mean().values, marker='o', label=lab)
        ax[1].plot(x, s['lat_p99'].mean().values, marker='o', label=lab)
        ax[2].plot(x, s['throughput_nonerr'].mean().values, marker='o', label=lab)
    ax[2].plot([50, 1200], [50, 1200], 'k:', lw=0.8, label='offered')
    for a in ax: a.set_xscale('log'); a.set_xlabel('offered rate (req/s)')
    ax[0].set_yscale('log'); ax[1].set_yscale('log'); ax[2].set_yscale('log')
    ax[0].set_title('client latency p50 (ms)', fontsize=8); ax[1].set_title('client latency p99 (ms)', fontsize=8)
    ax[2].set_title('non-error throughput (req/s)', fontsize=8); shared_legend(fig, ax[2], 3)
    savefig('fig_pipeline')
    # decomposition at 200 req/s and payload goodput
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.85))
    cfgs = list(CFG_LABEL.keys()); opa = []; insp = []; rest = []; tot = []
    for cfg in cfgs:
        s_ = o[(o.config == cfg) & (o.rate == 100)]
        to, ti, p50 = s_['t_opa_ms_p50'].mean(), s_['t_insp_ms_p50'].mean(), s_['lat_p50'].mean()
        to = to if to > 0.05 else 0.0; ti = ti if ti > 0.05 else 0.0
        opa.append(to); insp.append(ti); rest.append(max(0, p50 - to - ti)); tot.append(p50)
    x = np.arange(len(cfgs))
    ax[0].bar(x, rest, label='data plane + client', color='0.7')
    ax[0].bar(x, opa, bottom=rest, label='ZTNA decision ($t_{opa}$)', color='tab:blue')
    ax[0].bar(x, insp, bottom=np.array(rest) + np.array(opa), label='inspection ($t_{insp}$)', color='tab:red')
    for xi, t in zip(x, tot): ax[0].text(xi, t + 0.15, f'{t:.2f}', ha='center', va='bottom', fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['C0', 'C1', 'C2', 'C3', 'C4']); ax[0].set_ylabel('median latency at 100 req/s (ms)')
    ax[0].set_ylim(0, max(tot) * 1.18); ax[0].legend(loc='upper left')
    p = load('e3_payload')
    if p is not None:
        order = ['1k', '16k', '256k', '1m']
        for cfg in ['C0_passthrough', 'C3_ztna_lgbm', 'C4_ztna_deep']:
            s = p[p.config == cfg].groupby('sz')
            if not len(s): continue
            y = [s['goodput_mbps'].mean().get(z, np.nan) for z in order]
            ax[1].plot(range(4), y, marker='o', label=CFG_LABEL[cfg])
        ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['1 KB', '16 KB', '256 KB', '1 MB']); ax[1].set_yscale('log')
        ax[1].set_ylabel('closed-loop goodput (Mbit/s)'); ax[1].set_xlabel('response payload'); ax[1].legend(loc='lower right')
    savefig('fig_decomp_goodput')


# --------------------------------------------------------------------------- E4
def fig_cascade():
    d = load('e4_cascade')
    if d is None: return
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.75))
    for mode, lab in MODE_LABEL.items():
        s = d[d.mode_name == mode].groupby('rate')
        if not len(s): continue
        c, ls = MODE_STYLE[mode]; x = s['rate'].first().values
        ax[0].plot(x, s['lat_p99'].mean().values, marker='o', color=c, ls=ls, label=lab)
        ax[1].plot(x, s['recall'].mean().values, marker='o', color=c, ls=ls, label=lab)
        ax[2].plot(x, s['esc_rate'].mean().values, marker='o', color=c, ls=ls, label=lab)
        ax[3].plot(x, s['slo_attain'].mean().values, marker='o', color=c, ls=ls, label=lab)
    ax[0].set_yscale('log'); ax[0].set_title('client p99 latency (ms)', fontsize=8); ax[0].axhline(25, color='k', lw=0.6, ls=':')
    ax[1].set_title('recall (policy-allowed flows)', fontsize=8); ax[2].set_title('escalation rate $e$', fontsize=8)
    ax[3].set_title('SLO attainment (25 ms)', fontsize=8)
    for a in ax: a.set_xlabel('offered rate (req/s)')
    ax[1].legend(loc='lower left', fontsize=8)
    savefig('fig_cascade')
    # controller trace
    files = sorted(glob.glob(os.path.join(RAW, 'e4_cascade', 'adaptive_r400_rep0.ctl*.json')))
    if files:
        h = np.array(json.load(open(files[0])))
        if len(h):
            fig, ax1 = plt.subplots(figsize=(3.7, 2.6))
            t = h[:, 0] - h[0, 0]
            ax1.plot(t, h[:, 2], color='tab:blue', label='band width $w$'); ax1.set_ylabel('band width $w$'); ax1.set_xlabel('time (s)')
            ax2 = ax1.twinx(); ax2.plot(t, h[:, 1], color='tab:red', alpha=0.6, lw=0.8, label='in-flight $q$'); ax2.set_ylabel('in-flight requests $q$')
            ax2.grid(False)
            h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax1.legend(h1 + h2, l1 + l2, loc='upper right')
            savefig('fig_controller')
    # frontier recall vs p99 at selected rates
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.85))
    for a, rate in zip(ax, [300, 400]):
        for mode, lab in MODE_LABEL.items():
            s = d[(d.mode_name == mode) & (d.rate == rate)]
            if not len(s): continue
            c, _ = MODE_STYLE[mode]
            a.scatter(s['lat_p99'], s['recall'], color=c, label=lab)
        a.set_xscale('log'); a.set_xlabel(f'client p99 latency at {rate} req/s (ms)'); a.set_ylabel('recall')
        a.axvline(25, color='k', lw=0.6, ls=':')
    ax[0].legend(fontsize=8, loc='lower left')
    savefig('fig_cascade_frontier')


# --------------------------------------------------------------------------- E5
def fig_reliability():
    d = load('e5_reliability')
    if d is None: return
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.85))
    def tl(name):
        p = os.path.join(RAW, 'e5_reliability', name + '.timeline.csv')
        return pd.read_csv(p) if os.path.exists(p) else None
    for name, lab, c in [('A_dec_rep1_p2_rep0', 'replica crash, $k$=1', 'tab:red'), ('A_dec_rep2_p2_rep0', 'replica crash, $k$=2', 'tab:blue')]:
        t = tl(name)
        if t is not None: ax[0].plot(t['t'], t['err_frac'], label=lab, color=c)
    ax[0].axvline(10, color='k', ls=':', lw=0.8); ax[0].set_title('error fraction (0.5 s bins), $\\pi$=2 s', fontsize=8); ax[0].set_xlabel('time (s)'); ax[0].legend()
    for name, lab, c in [('B_opa_fo0_p2_rep0', 'fail-closed: errors', 'tab:red'), ('B_opa_fo1_p2_rep0', 'fail-open: exposed', 'tab:orange')]:
        t = tl(name)
        if t is None: continue
        if 'fo0' in name: ax[1].plot(t['t'], t['err'], label=lab, color=c)
        else: ax[1].plot(t['t'], t['exposed'], label=lab, color=c)
    ax[1].axvline(10, color='k', ls=':', lw=0.8); ax[1].set_title('PDP crash, requests per 0.5 s, $\\pi$=2 s', fontsize=8); ax[1].set_xlabel('time (s)'); ax[1].legend()
    # exposure-availability scatter
    for sc, lab, mk in [('A_dec_rep1', 'replica crash $k$=1', 'o'), ('A_dec_rep2', 'replica crash $k$=2', 's'), ('B_opa_fo0', 'PDP crash fail-closed', '^'),
                        ('B_opa_fo1', 'PDP crash fail-open', 'v'), ('C_nginx', 'data-plane fail-open', 'D')]:
        s = d[d['name'].str.startswith(sc)]
        if len(s): ax[2].scatter(s['errors_legit'], s['exposure'], label=lab, marker=mk)
    ax[2].set_xlabel('legitimate requests errored'); ax[2].set_ylabel('unauthorised admissions'); ax[2].set_yscale('symlog')
    ax[2].legend(fontsize=8); ax[2].set_title('exposure vs availability loss', fontsize=8)
    savefig('fig_reliability')
    # MTTR vs probe interval
    rows = []
    for _, r in d.iterrows():
        if not isinstance(r.get('events'), str): continue
        ev = json.loads(r['events'])
        kill = [t for t, e in ev if e.startswith('kill')]; det = [t for t, e in ev if e.startswith('detect')]; rst = [t for t, e in ev if e.startswith('restart')]
        t = tl(r['name'])
        if not kill or t is None: continue
        after = t[t['t'] >= kill[0]]
        ok = after[(after['err'] == 0) & (after['n'] > 0)]
        # time to restore: first bin after the kill from which no further errors occur
        restore = np.nan
        errs = after[after['err'] > 0]['t']
        if len(errs): restore = errs.max() + 0.5 - kill[0]
        elif len(after): restore = 0.0
        rows.append(dict(name=r['name'], scenario=r.get('scenario'), probe=r.get('probe'), ttd=(det[0] - kill[0]) if det else np.nan,
                         ttr=(rst[0] - kill[0]) if rst else np.nan, restore=restore, err_legit=r['errors_legit'], exposure=r['exposure'],
                         avail=r['availability_legit']))
    rec = pd.DataFrame(rows); rec.to_csv(os.path.join(T, 'e5_recovery.csv'), index=False)
    if len(rec):
        fig, ax = plt.subplots(figsize=(3.7, 2.6))
        for sc, lab in [('A_dec_rep1', 'replica crash $k$=1'), ('B_opa_fo0', 'PDP crash fail-closed'), ('B_opa_fo1', 'PDP crash fail-open')]:
            s = rec[rec['name'].str.startswith(sc)].groupby('probe')['restore'].agg(['mean', 'std'])
            if len(s): ax.errorbar(s.index, s['mean'], yerr=s['std'].fillna(0), marker='o', capsize=2, label=lab)
        ax.set_xlabel('liveness probe interval $\\pi$ (s)'); ax.set_ylabel('time to restore (s)'); ax.legend(fontsize=8)
        savefig('fig_mttr')


# --------------------------------------------------------------------------- E6
def fig_robust():
    r = load('e6_repro'); c = load('e6_crossdata')
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.85))
    if r is not None:
        rep = r[r['name'].str.startswith('ref')]; seed = r[r['name'].str.startswith('seed')]
        for i, (col, lab) in enumerate([('lat_p50', 'p50 (ms)'), ('lat_p99', 'p99 (ms)'), ('f1', '$F_1$'), ('ztna_conformance', 'conformance')]):
            v = rep[col].values; vs = seed[col].values
            ax[0].errorbar(i, v.mean() / v.mean(), yerr=v.std() / v.mean(), fmt='o', color='tab:blue', capsize=3, label='5 repetitions' if i == 0 else None)
            ax[0].errorbar(i + 0.25, vs.mean() / v.mean(), yerr=vs.std() / v.mean(), fmt='s', color='tab:orange', capsize=3, label='3 context seeds' if i == 0 else None)
        ax[0].set_xticks(np.arange(4) + 0.12); ax[0].set_xticklabels(['p50', 'p99', '$F_1$', 'conf.']); ax[0].set_ylabel('relative to repetition mean'); ax[0].legend(fontsize=8)
        ax[0].set_title('repeatability (ZTNA+LightGBM, 200 req/s)', fontsize=8)
    if c is not None:
        for a, ds, title in zip(ax[1:], ['cic', 'nsl'], ['CIC-IDS2017 trace', 'NSL-KDD trace']):
            s = c[c.dataset == ds]
            modes = ['light', 'deep', 'cascade50', 'adaptive']; x = np.arange(len(modes))
            for j, rate in enumerate([150, 400]):
                g = s[s.rate == rate].groupby('mode_name')
                rec_ = [g['recall'].mean().get(m, np.nan) for m in modes]; p99 = [g['lat_p99'].mean().get(m, np.nan) for m in modes]
                a.bar(x + (j - 0.5) * 0.35, rec_, width=0.35, label=f'recall @ {rate} req/s', alpha=0.85)
                for xi, pv in zip(x + (j - 0.5) * 0.35, p99):
                    if not np.isnan(pv): a.text(xi, 0.02, f'{pv:.0f}', ha='center', va='bottom', fontsize=8, rotation=90, color='white')
            a.set_xticks(x); a.set_xticklabels(['light', 'deep', 'casc.0.5', 'adaptive'], fontsize=8); a.set_ylim(0, 1.05); a.set_title(title, fontsize=8)
            a.legend(fontsize=8, loc='lower right')
        ax[1].set_ylabel('recall (p99 ms printed in bars)')
    savefig('fig_robust')


def _modes_panel(d, modes, fname, titles=('recall', 'client latency p50 (ms)', 'error fraction'), cols=('recall', 'lat_p50', 'err_frac')):
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.85))
    for m, lab, c, ls in modes:
        s = d[d.mode_name == m].groupby('rate')
        if not len(s): continue
        x = s['rate'].first().values
        for a, col in zip(ax, cols):
            a.plot(x, s[col].mean().values, marker='o', color=c, ls=ls, label=lab)
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=8); a.set_xlabel('offered rate (req/s)')
    ax[1].set_yscale('log'); shared_legend(fig, ax[0], 4)
    savefig(fname)


def fig_unseen():
    d = load('e7_unseen')
    if d is None: return
    _modes_panel(d, [('sup', 'supervised only', '0.5', ':'), ('knn_full', 'exact k-NN, every flow', 'tab:red', '-'),
                     ('hnsw_full', 'HNSW k-NN, every flow', 'tab:green', '-'), ('lof_full', 'HNSW LOF, every flow', 'tab:olive', '-'),
                     ('static_e30', 'exact k-NN, band e=0.30', 'tab:orange', '--'), ('random30', 'exact k-NN, random 30%', 'tab:purple', '-.'),
                     ('adaptive', 'exact k-NN, adaptive', 'tab:blue', '-')], 'fig_unseen')


def fig_direct():
    d = load('e4_direct')
    if d is None: return
    _modes_panel(d, [('full_lgbm', 'full LightGBM', 'tab:green', '-'), ('adaptive', 'adaptive cascade (LR→RF)', 'tab:blue', '-'),
                     ('deep_rf', 'full RF', 'tab:red', '--'), ('light', 'light (LR)', '0.5', ':')], 'fig_direct',
                 titles=('recall', 'client latency p99 (ms)', 'error fraction'), cols=('recall', 'lat_p99', 'err_frac'))


def fig_relload():
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for name, lab, c in [('A_dec_rep2_r300_rep0', '$k$=2, 300 req/s', 'tab:green'), ('A_dec_rep1_r600_rep0', '$k$=1, 600 req/s', 'tab:red'),
                         ('A_dec_rep2_r600_rep0', '$k$=2, 600 req/s', 'tab:blue')]:
        p = os.path.join(RAW, 'e5_load', name + '.timeline.csv')
        if os.path.exists(p):
            t = pd.read_csv(p); ax.plot(t['t'], t['err_frac'], label=lab, color=c)
    ax.axvline(10, color='k', ls=':', lw=0.8); ax.set_xlabel('time (s)'); ax.set_ylabel('error fraction'); ax.legend(fontsize=8)
    savefig('fig_relload')


def fig_remote():
    d = load('e8_remote')
    if d is None: return
    o = d[d.loop == 'open'].groupby('rtt'); c = d[d.loop == 'closed'].groupby('rtt')
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.75))
    x = o['rtt'].first().values
    ax[0].plot(x, o['lat_p50'].mean().values, marker='o', label='client p50'); ax[0].plot(x, o['lat_p99'].mean().values, marker='s', label='client p99')
    ax[0].plot(x, x + o['lat_p50'].mean().values[0], 'k:', lw=0.8, label='p50 at 0 ms + RTT')
    ax[0].set_xlabel('emulated PDP round-trip time (ms)'); ax[0].set_ylabel('latency at 200 req/s (ms)'); ax[0].legend(fontsize=8)
    xc = c['rtt'].first().values
    ax[1].plot(xc, c['throughput_nonerr'].mean().values, marker='o', color='tab:green')
    ax[1].set_ylim(0, 1.3 * c['throughput_nonerr'].mean().max()); ax[1].set_xlabel('emulated PDP round-trip time (ms)')
    ax[1].set_ylabel('closed-loop throughput (req/s)')
    savefig('fig_remote')


if __name__ == '__main__':
    fig_policy(); fig_frontier(); fig_pipeline(); fig_cascade(); fig_reliability(); fig_robust()
    fig_unseen(); fig_direct(); fig_relload(); fig_remote()
