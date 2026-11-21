# EC2 deployment

The API runs on one EC2 instance (af-south-1) without Docker: Python 3.12 in a uv virtualenv, the
`dataforge` systemd service (migrations run on start) and Caddy in front for HTTPS at
`https://<public-ip-with-dashes>.sslip.io`.

## Continuous deployment

Every push to `main` runs the CI checks in `.github/workflows/ci.yml`. When they pass, the `Deploy to EC2`
job signs in to AWS with GitHub OIDC as `dataforge-github-deploy`, a role that can only send Systems Manager
commands to this instance, and runs `/opt/dataforge/deploy.sh <commit>` there. The script checks out that
commit, syncs dependencies, restarts the service and waits for `/api/v1/health`. If the new version is not
healthy it rolls the code back to the previous commit and the job fails. Database migrations are not rolled back.

## Files

- `setup.sh`: one-time host setup on a fresh Ubuntu instance (run as root).
- `deploy.sh`: deploys one commit; installed on the host as `/opt/dataforge/deploy.sh`.

## AWS pieces

- IAM OIDC provider `token.actions.githubusercontent.com`.
- Role `dataforge-github-deploy`: trusted only for `repo:Lethabo-T-Molefe/dataforge.backend:ref:refs/heads/main`;
  allowed `ssm:SendCommand` on this instance with `AWS-RunShellScript`, and to read command results.
- Instance profile `dataforge-ec2-ssm` (AmazonSSMManagedInstanceCore) on the instance, so the SSM agent can
  receive commands. No inbound port is needed for deploys.

## On the host

- Settings: `/opt/dataforge/app/.env` (never committed).
- Logs: `journalctl -u dataforge -f`.
- Manual deploy of the latest main: `sudo /opt/dataforge/update.sh`.
