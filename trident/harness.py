#!/usr/bin/env python3
"""
TRIDENT harness: brings up the reference SASE PoP (nginx data plane, OPA policy decision point,
decision-service replicas, origin), drives open-loop load with vegeta, optionally injects
component failures with an orchestrator-like liveness supervisor, and collects
client-side (vegeta) and PoP-side (nginx access log) per-request records.
"""
import os, sys, time, json, signal, subprocess, threading, shutil, random
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(os.path.dirname(ROOT), 'bin')
RUN = os.path.join(ROOT, 'run')
PAY = os.path.join(RUN, 'payload')
PY = sys.executable
OPA_PORT, POP_PORT, ORIGIN_PORT = 8181, 8080, 9100

NGINX_CONF = '''
worker_processes 1;
daemon off;
pid {run}/nginx.pid;
error_log {run}/nginx_error.log warn;
events {{ worker_connections 16384; }}
http {{
  access_log off;
  sendfile on;
  tcp_nopush on;
  log_format trident '$msec|$http_x_trace_idx|$status|$request_time|$upstream_response_time|$t_opa|$t_insp|$decision|$esc|$replica|$bytes_sent';
  upstream decision {{
{servers}
    keepalive 128;
  }}
  upstream origin {{ server 127.0.0.1:{origin_port}; keepalive 128; }}
  server {{
    listen 127.0.0.1:{origin_port} backlog=4096;
    root {payload};
    location /app/ {{ try_files /$arg_sz.bin =404; }}
  }}
  server {{
    listen 127.0.0.1:{pop_port} backlog=8192;
    access_log {run}/access.log trident buffer=256k flush=500ms;
    location = /_authz {{
      internal;
      proxy_pass http://decision/authz;
      proxy_pass_request_body off;
      proxy_set_header Content-Length "";
      proxy_set_header X-Original-URI $request_uri;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_connect_timeout 500ms;
      proxy_read_timeout 15s;
      proxy_next_upstream error timeout http_502 http_503;
      proxy_next_upstream_tries 3;
    }}
    location /app/ {{
      auth_request /_authz;
      auth_request_set $t_opa $upstream_http_x_t_opa;
      auth_request_set $t_insp $upstream_http_x_t_insp;
      auth_request_set $decision $upstream_http_x_decision;
      auth_request_set $esc $upstream_http_x_esc;
      auth_request_set $replica $upstream_http_x_replica;
      proxy_pass http://origin;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      {failopen}
    }}
    location @failopen {{
      set $decision "nginx_failopen";
      proxy_pass http://origin;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
    }}
    location = /health {{ return 200 'ok'; }}
  }}
}}
'''


def http_ok(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def http_json(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


class Stack:
    def __init__(self, dataset='unsw', bundle='policy/bundle', replicas=1, mode='deep', light='lr', deep='rf',
                 ztna=1, auth_mode='rs256', fail_open=0, nginx_failopen=False, target_inflight=4, origin_replicas=1,
                 always=None, thr_light=None, thr_deep=None, thr_always=None, opa_rtt_ms=0.0):
        self.cfg = dict(dataset=dataset, bundle=bundle, replicas=replicas, mode=mode, light=light, deep=deep, ztna=ztna,
                        auth_mode=auth_mode, fail_open=fail_open, nginx_failopen=nginx_failopen, target_inflight=target_inflight,
                        always=always, thr_light=thr_light, thr_deep=thr_deep, thr_always=thr_always, opa_rtt_ms=opa_rtt_ms)
        self.bundle = bundle if os.path.isabs(bundle) else os.path.join(ROOT, bundle)
        self.procs = {}
        self.events = []  # (t, event)
        self.supervisor = None
        self.stop_sup = threading.Event()

    # ------------------------------------------------------------- processes
    def _spawn(self, name, cmd, logname=None):
        log = open(os.path.join(RUN, f'{logname or name}.log'), 'ab')
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        self.procs[name] = p
        return p

    def start_opa(self):
        self._spawn('opa', [os.path.join(BIN, 'opa'), 'run', '--server', '--addr', f'127.0.0.1:{OPA_PORT}',
                            '--log-level', 'error', self.bundle])

    def start_replica(self, k):
        c = self.cfg
        port = 9000 + k
        self._spawn(f'dec{k}', [PY, os.path.join(ROOT, 'trident', 'decision_service.py'), '--port', str(port),
                                '--dataset', c['dataset'], '--mode', c['mode'], '--light', c['light'], '--deep', c['deep'],
                                '--ztna', str(c['ztna']), '--auth-mode', c['auth_mode'], '--fail-open', str(c['fail_open']),
                                '--target-inflight', str(c['target_inflight']), '--seed', str(k),
                                '--opa-rtt-ms', str(c['opa_rtt_ms'])]
                    + (['--always', c['always']] if c['always'] else [])
                    + sum([[f'--{a.replace("_", "-")}', str(c[a])] for a in ('thr_light', 'thr_deep', 'thr_always') if c[a] is not None], []))

    def write_payloads(self):
        os.makedirs(PAY, exist_ok=True)
        rng = random.Random(0)
        for name, size in (('1k', 1024), ('16k', 16 * 1024), ('256k', 256 * 1024), ('1m', 1024 * 1024), ('4m', 4 * 1024 * 1024)):
            p = os.path.join(PAY, f'{name}.bin')
            if not os.path.exists(p) or os.path.getsize(p) != size:
                open(p, 'wb').write(bytes(rng.getrandbits(8) for _ in range(min(size, 4096))) * (size // min(size, 4096)))

    def start_nginx(self):
        self.write_payloads()
        servers = '\n'.join(f'    server 127.0.0.1:{9000 + k + 1} max_fails=1 fail_timeout=1s;' for k in range(self.cfg['replicas']))
        failopen = 'error_page 500 502 503 504 = @failopen;' if self.cfg['nginx_failopen'] else ''
        conf = NGINX_CONF.format(run=RUN, servers=servers, origin_port=ORIGIN_PORT, pop_port=POP_PORT, payload=PAY, failopen=failopen)
        cpath = os.path.join(RUN, 'nginx.conf'); open(cpath, 'w').write(conf)
        self._spawn('nginx', ['nginx', '-c', cpath, '-p', RUN])

    def up(self, timeout=60):
        os.makedirs(RUN, exist_ok=True)
        import socket
        for port in [OPA_PORT, POP_PORT, ORIGIN_PORT] + [9000 + k for k in range(1, self.cfg['replicas'] + 1)]:
            sk = socket.socket(); sk.settimeout(0.3)
            try:
                sk.connect(('127.0.0.1', port)); sk.close()
                raise RuntimeError(f'port {port} is already in use: a stale stack is running')
            except (ConnectionRefusedError, socket.timeout, OSError):
                sk.close()
        self.start_opa()
        for k in range(1, self.cfg['replicas'] + 1):
            self.start_replica(k)
        self.start_nginx()
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok = http_ok(f'http://127.0.0.1:{OPA_PORT}/health') and http_ok(f'http://127.0.0.1:{POP_PORT}/health')
            ok = ok and all(http_ok(f'http://127.0.0.1:{9000 + k}/health') for k in range(1, self.cfg['replicas'] + 1))
            if ok:
                time.sleep(0.5)
                return True
            time.sleep(0.5)
        raise RuntimeError('stack failed to start: ' + str({k: p.poll() for k, p in self.procs.items()}))

    def down(self):
        self.stop_supervisor()
        for name, p in list(self.procs.items()):
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for name, p in list(self.procs.items()):
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        self.procs.clear()
        subprocess.run(['pkill', '-f', 'decision_service.py'], capture_output=True)
        subprocess.run(['pkill', '-f', 'opa run --server'], capture_output=True)
        time.sleep(0.5)

    # ------------------------------------------------------------- failure injection
    def kill(self, name):
        p = self.procs.get(name)
        if p is None: return
        self.events.append((time.time(), f'kill:{name}'))
        p.kill()
        p.wait()

    def restart(self, name):
        if name == 'opa':
            self.start_opa()
        elif name.startswith('dec'):
            self.start_replica(int(name[3:]))
        self.events.append((time.time(), f'restart:{name}'))

    def start_supervisor(self, probe_interval=1.0, probe_timeout=0.5, failure_threshold=1, phase=0.0, grace=5.0):
        """Emulates an orchestrator liveness probe: polls /health of OPA and replicas every
        `probe_interval` seconds (first probe after `phase` seconds, so that the probe phase is not
        aligned with the failure); after `failure_threshold` consecutive failures a component is
        restarted and left alone for a start-up grace period of `grace` seconds."""
        self.stop_sup.clear()
        fails, grace_until = {}, {}

        def loop():
            self.stop_sup.wait(phase)
            while not self.stop_sup.is_set():
                comps = {'opa': f'http://127.0.0.1:{OPA_PORT}/health'}
                for k in range(1, self.cfg['replicas'] + 1):
                    comps[f'dec{k}'] = f'http://127.0.0.1:{9000 + k}/health'
                for name, url in comps.items():
                    if time.time() < grace_until.get(name, 0):
                        continue
                    if http_ok(url, probe_timeout):
                        fails[name] = 0
                    else:
                        fails[name] = fails.get(name, 0) + 1
                        if fails[name] >= failure_threshold:
                            self.events.append((time.time(), f'detect:{name}'))
                            self.restart(name)
                            grace_until[name] = time.time() + grace
                            fails[name] = 0
                self.stop_sup.wait(probe_interval)
        self.supervisor = threading.Thread(target=loop, daemon=True)
        self.supervisor.start()

    def stop_supervisor(self):
        self.stop_sup.set()
        if self.supervisor:
            self.supervisor.join(timeout=3)
            self.supervisor = None

    # ------------------------------------------------------------- metrics
    def metrics(self):
        return {k: http_json(f'http://127.0.0.1:{9000 + k}/metrics') for k in range(1, self.cfg['replicas'] + 1)}


# ----------------------------------------------------------------- load generation
def truncate_access_log():
    p = os.path.join(RUN, 'access.log')
    open(p, 'w').close()
    # nginx keeps its fd; truncation is enough since it appends


def attack(targets, rate, duration, out_prefix, max_workers=None, timeout='15s', keepalive=True, workers=10):
    """Runs vegeta (open loop at `rate` req/s; rate=0 -> closed loop with `max_workers`). Returns report dict."""
    truncate_access_log()
    bin_out = out_prefix + '.bin'
    cmd = [os.path.join(BIN, 'vegeta'), 'attack', '-format=json', f'-targets={targets}', f'-rate={rate}/1s',
           f'-duration={duration}s', f'-timeout={timeout}', f'-workers={workers}', f'-output={bin_out}', '-max-body=0']
    if max_workers: cmd.append(f'-max-workers={max_workers}')
    if not keepalive: cmd.append('-keepalive=false')
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT)
    t1 = time.time()
    time.sleep(1.2)  # let nginx flush its buffered access log
    rep = subprocess.run([os.path.join(BIN, 'vegeta'), 'report', '-type=json', bin_out], capture_output=True, text=True, check=True)
    report = json.loads(rep.stdout)
    report['t_start'], report['t_end'] = t0, t1
    with open(out_prefix + '.json', 'w') as f:
        subprocess.run([os.path.join(BIN, 'vegeta'), 'encode', '--to=json', bin_out], stdout=f, check=True)
    shutil.copy(os.path.join(RUN, 'access.log'), out_prefix + '.access.log')
    os.remove(bin_out)
    from analyze import compact
    compact(out_prefix)
    return report


if __name__ == '__main__':
    # smoke test
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='deep'); ap.add_argument('--rate', type=int, default=50)
    ap.add_argument('--duration', type=int, default=5); ap.add_argument('--replicas', type=int, default=1)
    ap.add_argument('--targets', default=os.path.join(ROOT, 'artifacts', 'targets_unsw_default.json'))
    a = ap.parse_args()
    s = Stack(mode=a.mode, replicas=a.replicas)
    s.up()
    try:
        rep = attack(a.targets, a.rate, a.duration, os.path.join(RUN, 'smoke'))
        print(json.dumps({k: rep[k] for k in ('requests', 'rate', 'throughput', 'success', 'status_codes')}, indent=1))
        print('latencies(ms):', {k: v / 1e6 for k, v in rep['latencies'].items()})
        print(s.metrics())
    finally:
        s.down()
