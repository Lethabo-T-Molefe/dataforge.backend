# Load smoke (k6)

## Prerequisites

Install [k6](https://k6.io/docs/get-started/installation/).

## Run against staging or local API

```bash
k6 run \
  -e BASE_URL=http://127.0.0.1:8000 \
  -e EMAIL=thandi@dataforge.local \
  -e PASSWORD='DataForge!2026' \
  ops/load/k6-smoke.js
```

Use a non-admin demo user so MFA enrolment does not block the script. For production-like staging, create a dedicated load-test account with a verified email.

## Last run summary

Paste the k6 summary here after a staging run:

```
_pending — date, p95, failure rate, environment URL_
```
