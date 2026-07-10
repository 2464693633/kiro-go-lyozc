package proxy

// Production hardening middleware — minimal, zero-dependency subset of the
// kiro-tutu production chain. Wraps the existing *Handler without touching its
// large ServeHTTP, preserving all current routing/behaviour.
//
// From outermost to innermost:
//
//	recover -> request-id -> security headers -> body-cap -> handler
//
// Deliberately NOT included (they need deps / infrastructure absent from this
// repo): Prometheus /metrics, rate limiting, SQLite/JSONL audit, cluster drain.
// Adding any of those means pulling modernc.org/sqlite, redis, etc. — out of
// scope for the zero-dep P0 port.
import (
	"context"
	"io"
	"kiro-go/logger"
	"net/http"
	"runtime/debug"
	"strings"

	"github.com/google/uuid"
)

// maxInboundRequestBytes caps every inbound request body so a hostile/oversized
// upload cannot exhaust memory via io.ReadAll on the API endpoints. 64 MiB is
// well above any real Anthropic/OpenAI request (including base64 images).
const maxInboundRequestBytes = 64 << 20

type ctxKeyRequestID struct{}

func requestIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxKeyRequestID{}).(string)
	return v
}

// WrapHardening wraps inner with the zero-dep hardening chain.
func WrapHardening(inner http.Handler) http.Handler {
	h := inner
	h = bodyCapMiddleware(h)
	h = securityHeadersMiddleware(h)
	h = requestIDMiddleware(h)
	h = recoverMiddleware(h)
	return h
}

// recoverMiddleware turns an unexpected panic in the handler chain into a clean
// 500 instead of crashing the connection (which leaves an SSE client hanging
// forever). http.ErrAbortHandler is re-panicked to preserve net/http's
// intentional abort semantics.
func recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			rec := recover()
			if rec == nil {
				return
			}
			if rec == http.ErrAbortHandler {
				panic(rec)
			}
			logger.Errorf("[Recover] panic on %s %s (req=%s): %v\n%s",
				r.Method, r.URL.Path, requestIDFromContext(r.Context()), rec, debug.Stack())
			func() {
				defer func() { _ = recover() }() // guard against write-after-headers panics
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"internal server error"}}`)
			}()
		}()
		next.ServeHTTP(w, r)
	})
}

// requestIDMiddleware propagates an inbound X-Request-Id or mints a fresh UUID,
// echoes it back on the response and carries it through context for log lines.
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rid := strings.TrimSpace(r.Header.Get("X-Request-Id"))
		if rid == "" {
			rid = uuid.NewString()
		}
		w.Header().Set("x-request-id", rid)
		ctx := context.WithValue(r.Context(), ctxKeyRequestID{}, rid)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// securityHeadersMiddleware sets baseline browser-security headers (admin panel
// clickjacking / MIME-sniffing protection).
func securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Referrer-Policy", "no-referrer")
		next.ServeHTTP(w, r)
	})
}

// bodyCapMiddleware bounds the request body. When the limit is exceeded the
// subsequent io.ReadAll inside a handler returns an error, which each handler
// already maps to a 400 "Failed to read request body" — the goal (memory bound)
// is met without changing per-endpoint error handling.
func bodyCapMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, maxInboundRequestBytes)
		}
		next.ServeHTTP(w, r)
	})
}
