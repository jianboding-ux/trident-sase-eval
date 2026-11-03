#!/usr/bin/env python3
"""
TRIDENT decision service (policy-plane microservice of the reference SASE PoP).

For each proxied request (nginx auth_request subrequest) it performs
  1. Zero-trust access decision   : OPA (Rego) evaluation of identity/device/geo/resource context
  2. Threat inspection            : ML inspector on the flow feature vector carried by the request
and returns 200 (allow) / 401 (identity) / 403 (policy or threat) / 503 (dependency failure, fail-closed).

Inspection modes
  none      : no inspection
  light     : light model only (e.g. dt_s)
  deep      : deep model on every request (e.g. rf)
  sample:p  : deep model on a random fraction p of requests, light model otherwise
  cascade:w : static two-stage cascade; light score s in band [0.5-w*0.5, 0.5+w*0.5] escalates to deep
  adaptive  : cascade whose band width w is controlled from the in-flight request count (congestion)

Per-request timing is exported in response headers (X-T-Opa, X-T-Insp microseconds) and logged by nginx.
"""
import os, sys, time, json, random, asyncio, argparse, signal
import numpy as np
import joblib
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'trident'))
from prepare_data import Inspector  # noqa: F401 (needed to unpickle)


class Controller:
    """Congestion-driven band controller for the adaptive cascade.
    Additive-increase / multiplicative-decrease on band width w in [0,1]."""
    def __init__(self, target=4, w0=1.0, inc=0.02, dec=0.85, period=0.05):
        self.target, self.w, self.inc, self.dec, self.period = target, w0, inc, dec, period
        self.last = time.monotonic()
        self.hist = []  # (t, inflight, w)

    def update(self, inflight):
        now = time.monotonic()
        if now - self.last < self.period:
            return self.w
        self.last = now
        if inflight > self.target:
            self.w = max(0.0, self.w * self.dec)
        else:
            self.w = min(1.0, self.w + self.inc)
        self.hist.append((now, inflight, self.w))
        return self.w


class Service:
    def __init__(self, a):
        self.a = a
        zoo = joblib.load(os.path.join(ROOT, 'artifacts', f'models_{a.dataset}.joblib'))
        self.light = zoo[a.light]; self.deep = zoo[a.deep]
        self.always = zoo[a.always] if a.always else None     # always-on supervised inspector (optional)
        if a.thr_light is not None: self.light.thr = a.thr_light
        if a.thr_deep is not None: self.deep.thr = a.thr_deep
        if self.always is not None and a.thr_always is not None: self.always.thr = a.thr_always
        self.n_feat = self.light.n_features
        mode = a.mode
        self.mode, self.param = (mode.split(':') + [None])[:2]
        self.param = float(self.param) if self.param is not None else None
        self.ctl = Controller(target=a.target_inflight)
        self.inflight = 0
        self.rng = random.Random(a.seed)
        self.counters = dict(req=0, allow=0, deny_policy=0, deny_token=0, threat=0, opa_err=0, esc=0)
        self.sess = None
        self.opa_url = f'{a.opa_url}/v1/data/ztna/decision'
        self.now = int(time.time())

    async def start(self):
        self.sess = ClientSession(connector=TCPConnector(limit=0, keepalive_timeout=60),
                                  timeout=ClientTimeout(total=self.a.opa_timeout))

    # ----------------------------------------------------------------- ZTNA
    async def opa_eval(self, inp):
        if self.a.opa_rtt_ms > 0:          # emulated network round trip to a remote PDP
            await asyncio.sleep(self.a.opa_rtt_ms / 1000.0)
        try:
            async with self.sess.post(self.opa_url, json={'input': inp}) as r:
                if r.status != 200:
                    return None
                d = await r.json()
                return d.get('result')
        except Exception:
            return None

    # ----------------------------------------------------------------- inspection
    def inspect(self, x):
        """returns (malicious: bool, escalated: int, score)"""
        m = self.mode
        if self.always is not None:
            s0 = self.always.score(x)
            if s0 >= self.always.thr:
                return True, 0, s0
        if m == 'none':
            return False, 0, 0.0
        td, tl = self.deep.thr, self.light.thr
        if m == 'deep':
            s = self.deep.score(x); return s >= td, 1, s
        s1 = self.light.score(x)
        if m == 'light':
            return s1 >= tl, 0, s1
        if m == 'sample':
            if self.rng.random() < self.param:
                s = self.deep.score(x); return s >= td, 1, s
            return s1 >= tl, 0, s1
        w = self.param if m == 'cascade' else self.ctl.update(self.inflight)
        lo, hi = tl - w * tl, tl + w * (1.0 - tl)
        if lo <= s1 <= hi:
            s = self.deep.score(x); return s >= td, 1, s
        return s1 >= tl, 0, s1

    # ----------------------------------------------------------------- handler
    async def authz(self, req):
        self.inflight += 1
        t0 = time.perf_counter()
        try:
            h = req.headers
            self.counters['req'] += 1
            hdr = {'X-Replica': str(self.a.port)}
            # ---- zero-trust access decision
            res = h.get('X-Resource', '')
            inp = {'auth_mode': self.a.auth_mode, 'now': int(time.time()),
                   'resource': res, 'path': h.get('X-Original-URI', ''), 'geo': h.get('X-Geo', ''),
                   'device': {'posture': h.get('X-Device-Posture', 'unknown'), 'managed': h.get('X-Device-Managed') == '1'}}
            if self.a.auth_mode == 'rs256':
                auth = h.get('Authorization', '')
                inp['token'] = auth[7:] if auth.startswith('Bearer ') else ''
            else:
                inp['claims'] = {'sub': h.get('X-User', ''), 'role': h.get('X-Role', ''), 'mfa': int(h.get('X-Mfa', '0'))}
            if self.a.ztna:
                dec = await self.opa_eval(inp)
            else:
                dec = {'allow': True, 'reason': 'bypass'}
            t1 = time.perf_counter()
            hdr['X-T-Opa'] = str(int((t1 - t0) * 1e6))
            if dec is None:
                self.counters['opa_err'] += 1
                if self.a.fail_open:
                    dec = {'allow': True, 'reason': 'fail_open'}
                else:
                    hdr['X-Decision'] = 'opa_unavailable'
                    return web.Response(status=503, headers=hdr)
            if not dec['allow']:
                hdr['X-Decision'] = 'deny:' + dec['reason']
                if dec['reason'] == 'invalid_token':
                    self.counters['deny_token'] += 1
                    return web.Response(status=401, headers=hdr)
                self.counters['deny_policy'] += 1
                return web.Response(status=403, headers=hdr)
            # ---- threat inspection
            t2 = time.perf_counter()
            flow = h.get('X-Flow')
            esc, score = 0, 0.0
            if flow and (self.mode != 'none' or self.always is not None):
                x = np.fromstring(flow, sep=',', dtype=np.float32)
                if x.shape[0] == self.n_feat:
                    mal, esc, score = self.inspect(x)
                    self.counters['esc'] += esc
                    if mal:
                        self.counters['threat'] += 1
                        hdr['X-Decision'] = 'threat'
                        hdr['X-Esc'] = str(esc); hdr['X-Score'] = f'{score:.3f}'
                        hdr['X-T-Insp'] = str(int((time.perf_counter() - t2) * 1e6))
                        return web.Response(status=403, headers=hdr)
            hdr['X-T-Insp'] = str(int((time.perf_counter() - t2) * 1e6))
            hdr['X-Esc'] = str(esc); hdr['X-Score'] = f'{score:.3f}'
            hdr['X-Decision'] = 'allow:' + dec['reason']
            self.counters['allow'] += 1
            return web.Response(status=200, headers=hdr)
        finally:
            self.inflight -= 1

    async def health(self, req):
        return web.json_response({'ok': True, 'port': self.a.port, 'inflight': self.inflight})

    async def metrics(self, req):
        d = dict(self.counters); d['w'] = self.ctl.w; d['inflight'] = self.inflight
        d['ctl_hist'] = self.ctl.hist[-2000:]
        return web.json_response(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9001)
    ap.add_argument('--dataset', default='unsw')
    ap.add_argument('--mode', default='deep')
    ap.add_argument('--light', default='lr')
    ap.add_argument('--deep', default='rf')
    ap.add_argument('--ztna', type=int, default=1)
    ap.add_argument('--auth-mode', default='rs256')
    ap.add_argument('--fail-open', type=int, default=0)
    ap.add_argument('--opa-url', default='http://127.0.0.1:8181')
    ap.add_argument('--opa-timeout', type=float, default=2.0)
    ap.add_argument('--target-inflight', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--always', default=None)
    ap.add_argument('--thr-light', type=float, default=None)
    ap.add_argument('--thr-deep', type=float, default=None)
    ap.add_argument('--thr-always', type=float, default=None)
    ap.add_argument('--opa-rtt-ms', type=float, default=0.0)
    a = ap.parse_args()
    svc = Service(a)
    app = web.Application()
    app.add_routes([web.get('/authz', svc.authz), web.get('/health', svc.health), web.get('/metrics', svc.metrics)])

    async def on_start(app):
        await svc.start()

    async def on_cleanup(app):
        if svc.sess:
            await svc.sess.close()
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host='127.0.0.1', port=a.port, access_log=None, print=None)


if __name__ == '__main__':
    main()
