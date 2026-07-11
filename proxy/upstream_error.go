package proxy

import (
	"errors"
	"strings"
)

// improperlyFormedMarker is the substring AWS returns in the 400 response body
// when it rejects the request payload itself ("Improperly formed request.").
// Matched case-insensitively so a minor upstream wording/casing change does not
// silently bypass the permanent-error short-circuit. Ported from kiro-tutu.
const improperlyFormedMarker = "improperly formed request"

// isImproperlyFormedRejection reports whether an upstream error body marks the
// request as improperly formed. Both the endpoint short-circuit (kiro.go) and
// handleAccountFailure key off this single predicate so they can never disagree.
func isImproperlyFormedRejection(errBody string) bool {
	return strings.Contains(strings.ToLower(errBody), improperlyFormedMarker)
}

// upstreamPermanentErrorCode marks a request that the upstream rejects for a
// reason inherent to the request itself (e.g. "Improperly formed request").
// Retrying it — across endpoints or accounts — cannot succeed and only
// amplifies the damage (wasted upstream hits, scattered cache affinity,
// unwarranted account health penalties), so it must short-circuit both layers.
const upstreamPermanentErrorCode = "upstream_permanent_rejection"

// upstreamPermanentError signals an upstream rejection that is inherent to the
// request and therefore not retryable. Both the endpoint loop (kiro.go) and the
// account loop short-circuit on it: no other endpoint/account is tried, account
// health is not penalised (handleAccountFailure treats it as neutral).
type upstreamPermanentError struct {
	status  int
	code    string
	message string
}

func newUpstreamPermanentError(status int, message string) error {
	return &upstreamPermanentError{
		status:  status,
		code:    upstreamPermanentErrorCode,
		message: message,
	}
}

func (e *upstreamPermanentError) Error() string {
	return e.message
}

func asUpstreamPermanentError(err error) (*upstreamPermanentError, bool) {
	var target *upstreamPermanentError
	if errors.As(err, &target) {
		return target, true
	}
	return nil, false
}

func isUpstreamPermanentError(err error) bool {
	_, ok := asUpstreamPermanentError(err)
	return ok
}
