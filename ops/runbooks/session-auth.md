# Runbook: sessions, MFA, and auth rate limits

## Symptoms

- Members cannot sign in; API returns 429.
- Administrator cannot enter workspace without MFA enrolment.
- Unexpected mass sign-out after password change or refresh-token reuse.

## Clear durable auth rate limits (Postgres)

From `dataforge.backend`:

```bash
uv run python scripts/clear_auth_rate_limits.py
```

SlowAPI in-memory counters reset on process restart. With `REDIS_URL`, flush the SlowAPI keys in Redis or restart replicas after intentional abuse testing.

## MFA

- **Optional** for non-admin roles (member can enable under Security settings).
- **Required for `admin`**: password sign-in returns `mfaEnrollmentRequired` until TOTP is confirmed via `POST /api/v1/auth/mfa`.
- Demo sign-in (`DEMO_MODE`) bypasses enrolment so local demos keep working; production must keep `DEMO_MODE=false`.

Wrong MFA codes burn the challenge; the member signs in again.

## Session revoke

- Password change bumps `session_version` and revokes refresh tokens.
- Reusing a rotated refresh cookie revokes the whole token family.
- Manual: sign the member out from admin tools or revoke refresh rows for that user.

## Related endpoints

| Endpoint | Notes |
|---|---|
| `POST /auth/login` | Durable + SlowAPI limits |
| `POST /auth/mfa` | Completes MFA or admin enrolment |
| `POST /auth/refresh` | Cookie-scoped; durable limit |
| `POST /auth/logout` | Clears cookie and revokes family |
