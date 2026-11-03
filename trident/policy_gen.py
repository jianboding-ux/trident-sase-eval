#!/usr/bin/env python3
"""
ZTNA policy generation for TRIDENT.

Produces an OPA bundle directory with
  ztna.rego   : the zero-trust access policy (identity, device posture, geo, resource sensitivity)
  data.json   : resource catalogue (K resources), blocked countries, signing key
and a Python reference oracle implementing the same specification (used for conformance checks).

Policy scaling knobs:
  --resources K        number of resources in the catalogue (data-driven lookup, O(1))
  --pad-rules N        number of additional per-resource rule bodies appended to the policy
  --pad-style STYLE    'eq' (input.path == "...", indexable) or 'glob' (glob.match, non-indexable)
  --jwt MODE           'rs256' (verify signature + exp in Rego), 'none' (trusted identity headers)
"""
import os, json, argparse, random, time
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import jwt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, 'artifacts')

ROLES = ['employee', 'contractor', 'admin']
SENS = ['low', 'medium', 'high']
COUNTRIES_OK = ['US', 'DE', 'GB', 'FR', 'JP', 'SG', 'AU', 'CA', 'NL', 'IE']
COUNTRIES_BLOCKED = ['XA', 'XB', 'XC']


def ensure_keys():
    os.makedirs(ART, exist_ok=True)
    priv_path = os.path.join(ART, 'jwt_private.pem'); pub_path = os.path.join(ART, 'jwt_public.pem')
    if not os.path.exists(priv_path):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        open(priv_path, 'wb').write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                      serialization.NoEncryption()))
        open(pub_path, 'wb').write(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                                serialization.PublicFormat.SubjectPublicKeyInfo))
    return open(priv_path, 'rb').read(), open(pub_path, 'rb').read()


def make_catalogue(K, seed=7):
    rng = random.Random(seed)
    res = {}
    for i in range(K):
        s = rng.choices(SENS, weights=[0.5, 0.35, 0.15])[0]
        roles = ['admin']
        if rng.random() < 0.9: roles.append('employee')
        if s != 'high' and rng.random() < 0.5: roles.append('contractor')
        res[f'r{i}'] = {'sensitivity': s, 'roles': sorted(roles)}
    return res


REGO_HEAD = '''package ztna

import rego.v1

default allow := false
default reason := "no_rule"

# ---------------------------------------------------------------- identity
claims := c if {
\tinput.auth_mode == "rs256"
\t[valid, _, payload] := io.jwt.decode_verify(input.token, {"cert": data.jwt_public_pem, "alg": "RS256"})
\tvalid
\tpayload.exp > input.now
\tc := payload
}

claims := c if {
\tinput.auth_mode == "none"
\tc := input.claims
}

token_ok if { claims }

# ---------------------------------------------------------------- context
resource := data.resources[input.resource]

geo_ok if { not input.geo in data.blocked_countries }

role_ok if { claims.role in resource.roles }

posture_ok if {
\tresource.sensitivity == "low"
}

posture_ok if {
\tresource.sensitivity == "medium"
\tinput.device.posture != "noncompliant"
}

posture_ok if {
\tresource.sensitivity == "high"
\tinput.device.posture == "compliant"
\tinput.device.managed == true
\tclaims.mfa == 1
}

contractor_ok if { claims.role != "contractor" }

contractor_ok if {
\tclaims.role == "contractor"
\tinput.device.posture == "compliant"
}

# ---------------------------------------------------------------- decision
allow if {
\ttoken_ok
\tresource
\tgeo_ok
\trole_ok
\tposture_ok
\tcontractor_ok
}

reason := "invalid_token" if { not token_ok }
reason := "unknown_resource" if { token_ok; not resource }
reason := "geo_blocked" if { token_ok; resource; not geo_ok }
reason := "role" if { token_ok; resource; geo_ok; not role_ok }
reason := "posture" if { token_ok; resource; geo_ok; role_ok; not posture_ok }
reason := "contractor_device" if { token_ok; resource; geo_ok; role_ok; posture_ok; not contractor_ok }
reason := "allow" if { allow }

decision := {"allow": allow, "reason": reason}
'''


def pad_rules(n, style):
    """Extra per-application rule bodies. 'eq' uses equality on input.path (rule-indexable);
    'glob' uses glob.match (not indexable: every body is evaluated)."""
    out = []
    for i in range(n):
        if style == 'eq':
            out.append(f'allow if {{\n\tinput.path == "/legacy/app{i}/"\n\ttoken_ok\n\tclaims.role == "admin"\n}}\n')
        elif style == 'glob':
            out.append(f'allow if {{\n\tglob.match("/legacy/app{i}/*", ["/"], input.path)\n\ttoken_ok\n\tclaims.role == "admin"\n}}\n')
        elif style == 'sw':
            out.append(f'allow if {{\n\tstartswith(input.path, "/legacy/app{i}/")\n\ttoken_ok\n\tclaims.role == "admin"\n}}\n')
        else:
            out.append(f'allow if {{\n\tregex.match("^/legacy/app{i}/.*$", input.path)\n\ttoken_ok\n\tclaims.role == "admin"\n}}\n')
    return '\n'.join(out)


def write_bundle(outdir, K=100, pad=0, pad_style='eq'):
    priv, pub = ensure_keys()
    os.makedirs(outdir, exist_ok=True)
    rego = REGO_HEAD + '\n# ---------------------------------------------------------------- padding rules\n' + pad_rules(pad, pad_style)
    open(os.path.join(outdir, 'ztna.rego'), 'w').write(rego)
    data = {'resources': make_catalogue(K), 'blocked_countries': COUNTRIES_BLOCKED, 'jwt_public_pem': pub.decode()}
    json.dump(data, open(os.path.join(outdir, 'data.json'), 'w'))
    return data


class Oracle:
    """Reference implementation of the ZTNA specification (ground truth for conformance)."""
    def __init__(self, data):
        self.res = data['resources']; self.blocked = set(data['blocked_countries'])

    def decide(self, claims_valid, claims, resource, geo, posture, managed):
        if not claims_valid: return False, 'invalid_token'
        r = self.res.get(resource)
        if r is None: return False, 'unknown_resource'
        if geo in self.blocked: return False, 'geo_blocked'
        if claims['role'] not in r['roles']: return False, 'role'
        s = r['sensitivity']
        if s == 'medium' and posture == 'noncompliant': return False, 'posture'
        if s == 'high' and not (posture == 'compliant' and managed and claims['mfa'] == 1): return False, 'posture'
        if claims['role'] == 'contractor' and posture != 'compliant': return False, 'contractor_device'
        return True, 'allow'


def make_token(priv, claims, valid=True, expired=False, now=None):
    now = now or time.time()
    c = dict(claims); c['iat'] = int(now); c['exp'] = int(now - 3600) if expired else int(now + 86400 * 30)
    if valid:
        return jwt.encode(c, priv, algorithm='RS256')
    # bad signature: sign with a throw-away key
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return jwt.encode(c, k, algorithm='RS256')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'policy', 'bundle'))
    ap.add_argument('--resources', type=int, default=100)
    ap.add_argument('--pad-rules', type=int, default=0)
    ap.add_argument('--pad-style', default='eq')
    a = ap.parse_args()
    write_bundle(a.out, a.resources, a.pad_rules, a.pad_style)
    print('bundle written to', a.out)
