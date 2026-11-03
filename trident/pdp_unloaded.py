#!/usr/bin/env python3
"""Unloaded (sequential, single client) per-decision latency of the OPA server for allowed and denied inputs."""
import os, sys, time, json, subprocess, urllib.request, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_gen import ensure_keys, make_token, write_bundle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BIN = os.path.join(os.path.dirname(ROOT), 'bin')
priv, _ = ensure_keys()
tok = make_token(priv, {'sub': 'u1', 'role': 'employee', 'dept': 'eng', 'mfa': 1})
base = {"auth_mode": "rs256", "token": tok, "now": int(time.time()), "resource": "r1", "path": "/app/r1",
        "device": {"posture": "compliant", "managed": True}}
inputs = {'allowed': dict(base, geo='US'), 'denied': dict(base, geo='XA')}

def post(inp, timeout=30):
    req = urllib.request.Request('http://127.0.0.1:8181/v1/data/ztna/decision', data=json.dumps({'input': inp}).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())['result']
    return (time.perf_counter() - t0) * 1000, out

rows = []
for style, pad in [('eq', 10000), ('sw', 10000), ('re', 100), ('re', 1000), ('re', 10000)]:
    bdir = os.path.join(ROOT, 'policy', f'bundle_{style}_{pad}')
    if not os.path.exists(os.path.join(bdir, 'ztna.rego')): write_bundle(bdir, K=100, pad=pad, pad_style=style)
    p = subprocess.Popen([os.path.join(BIN, 'opa'), 'run', '--server', '--addr', '127.0.0.1:8181', '--log-level', 'error', bdir],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen('http://127.0.0.1:8181/health', timeout=0.5); break
        except Exception: time.sleep(0.2)
    for kind, inp in inputs.items():
        lat = []; t_start = time.time(); dec = None
        for i in range(25):
            ms, dec = post(inp); lat.append(ms)
            if time.time() - t_start > 60: break
        rows.append(dict(style=style, pad=pad, kind=kind, n=len(lat), first_ms=lat[0], median_ms=statistics.median(lat[1:]),
                         max_ms=max(lat[1:]), decision=dec['reason']))
        print(rows[-1], flush=True)
    p.kill(); p.wait(); time.sleep(0.5)
json.dump(rows, open(os.path.join(ROOT, 'results', 'tables', 'pdp_unloaded.json'), 'w'), indent=1)
