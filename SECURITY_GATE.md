# Brahman — Production Security Gate

> Last audit: 2026-04-28  
> Status: **PASS** (all P0 + P1 controls implemented)

---

## P0 Controls (Must Pass)

| # | Control | Status | Evidence |
|---|---------|--------|----------|
| 1 | App API endpoints require auth | ✅ PASS | `app/api/verify.py`, `app/api/solnet.py` — `Depends(require_api_key)` |
| 2 | Centralized auth module | ✅ PASS | `app/security/auth.py` — shared by all endpoints |
| 3 | Sovereign Node endpoints require auth | ✅ PASS | `kernel/sovereign_node.py` — `Depends(get_api_key)` |
| 4 | Non-default API key enforcement | ✅ PASS | Startup aborts in production if key is missing or matches known defaults |
| 5 | Model artifact integrity (fail-closed) | ✅ PASS | `run.py` — mandatory `BRAHMAN_MODEL_SHA256` in production; abort + delete on mismatch |
| 6 | Deployment entrypoint alignment | ✅ PASS | `railway.toml` → `Dockerfile` → single canonical startup path |
| 7 | On-chain authorization guards | ✅ PASS | `lib.rs` — `AUTHORIZED_DEPLOYER`, registered node whitelist, admin-only finalize |

## P1 Controls (Enterprise Baseline)

| # | Control | Status | Evidence |
|---|---------|--------|----------|
| 1 | HTTP security headers | ✅ PASS | `SecurityHeadersMiddleware` in both `app/main.py` and `sovereign_node.py` |
| 2 | HSTS (production only) | ✅ PASS | Enabled when `BRAHMAN_ENV=production` |
| 3 | X-Content-Type-Options | ✅ PASS | `nosniff` |
| 4 | X-Frame-Options | ✅ PASS | `DENY` |
| 5 | Content-Security-Policy | ✅ PASS | `default-src 'self'; frame-ancestors 'none'` |
| 6 | Referrer-Policy | ✅ PASS | `strict-origin-when-cross-origin` |
| 7 | CORS restricted | ✅ PASS | Env-configurable origins, no wildcard+credentials |
| 8 | WebSocket auth (no query-param) | ✅ PASS | First-message auth protocol |
| 9 | Path traversal guard | ✅ PASS | Basename-only + `.json` whitelist + `resolve().is_relative_to()` |
| 10 | HTTP rate limiting | ✅ PASS | Per-IP sliding window (10 req/sec) on `/verify`, `/verify/kp`, `/cartridge/load` |
| 11 | WebSocket rate limiting | ✅ PASS | Per-IP sliding window in WS handler |
| 12 | Request body size limit | ✅ PASS | `RequestSizeLimitMiddleware` — 1MB max |
| 13 | Container non-root | ✅ PASS | `Dockerfile` — `USER brahman` with dedicated group |
| 14 | Multi-stage build | ✅ PASS | `build-essential` not in runtime image |
| 15 | TrustedHostMiddleware | ✅ PASS | Configurable via `BRAHMAN_TRUSTED_HOSTS` |
| 16 | Healthcheck alignment | ✅ PASS | `Dockerfile` CMD and HEALTHCHECK align with `railway.toml` (`run.py` on 8080) |

## P2 Controls (Operations)

| # | Control | Status | Evidence |
|---|---------|--------|----------|
| 1 | Startup config validation | ✅ PASS | Both services validate env vars and abort on misconfiguration |
| 2 | Security logging | ✅ PASS | Auth failures, rate limits, WS connects logged with client IP |
| 3 | Gate documentation | ✅ PASS | This document |

## Accepted Risks

| Risk | Severity | Rationale | Review Date |
|------|----------|-----------|-------------|
| `RUSTSEC-2024-0344` curve25519-dalek 3.2.1 | HIGH | Transitive via `anchor-lang` — no direct upgrade path. Formally ignored in `.cargo/audit.toml`. Compensating control: on-chain authorization guards prevent exploitation. | 2026-07-28 |
| JS dev-dependency vulns (serialize-javascript, etc.) | HIGH/MODERATE | Reduced from 11 to 5 via `yarn` resolutions. All remaining are strictly in the test/dev toolchain (`devDependencies`) and never shipped in the production Docker runtime. | 2026-07-28 |
| Python dependency audit incomplete | LOW | `pip-audit` not run in CI yet. Manual review shows no known CVEs in runtime deps. | 2026-06-28 |

## Required Production Environment Variables

```bash
# MANDATORY
export BRAHMAN_API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export BRAHMAN_ENV="production"

# MANDATORY for model bootstrap (run.py)
export BRAHMAN_MODEL_SHA256="<sha256-of-model-file>"

# RECOMMENDED
export BRAHMAN_CORS_ORIGINS="https://yourdomain.com"
export BRAHMAN_TRUSTED_HOSTS="yourdomain.com,api.yourdomain.com"
export BRAHMAN_NODE_ID="sovereign-prod-01"
```

## Re-audit Cadence

- **Quarterly**: Full security gate re-audit
- **On dependency update**: `cargo audit` + `yarn audit` + `pip-audit`
- **On architecture change**: Full threat model review
