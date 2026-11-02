package ztna

import rego.v1

default allow := false
default reason := "no_rule"

# ---------------------------------------------------------------- identity
claims := c if {
	input.auth_mode == "rs256"
	[valid, _, payload] := io.jwt.decode_verify(input.token, {"cert": data.jwt_public_pem, "alg": "RS256"})
	valid
	payload.exp > input.now
	c := payload
}

claims := c if {
	input.auth_mode == "none"
	c := input.claims
}

token_ok if { claims }

# ---------------------------------------------------------------- context
resource := data.resources[input.resource]

geo_ok if { not input.geo in data.blocked_countries }

role_ok if { claims.role in resource.roles }

posture_ok if {
	resource.sensitivity == "low"
}

posture_ok if {
	resource.sensitivity == "medium"
	input.device.posture != "noncompliant"
}

posture_ok if {
	resource.sensitivity == "high"
	input.device.posture == "compliant"
	input.device.managed == true
	claims.mfa == 1
}

contractor_ok if { claims.role != "contractor" }

contractor_ok if {
	claims.role == "contractor"
	input.device.posture == "compliant"
}

# ---------------------------------------------------------------- decision
allow if {
	token_ok
	resource
	geo_ok
	role_ok
	posture_ok
	contractor_ok
}

reason := "invalid_token" if { not token_ok }
reason := "unknown_resource" if { token_ok; not resource }
reason := "geo_blocked" if { token_ok; resource; not geo_ok }
reason := "role" if { token_ok; resource; geo_ok; not role_ok }
reason := "posture" if { token_ok; resource; geo_ok; role_ok; not posture_ok }
reason := "contractor_device" if { token_ok; resource; geo_ok; role_ok; posture_ok; not contractor_ok }
reason := "allow" if { allow }

decision := {"allow": allow, "reason": reason}

# ---------------------------------------------------------------- padding rules
