#!/bin/bash
# One-time setup of the DataForge API on a fresh Ubuntu EC2 instance (no Docker): Python via uv, a systemd
# service and Caddy for HTTPS. Run as root. Then put the settings in /opt/dataforge/app/.env (see .env.example)
# and run: systemctl start dataforge
set -eux
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y git curl caddy

# App user, code, Python environment
id dataforge >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/dataforge --shell /usr/sbin/nologin dataforge
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
[ -d /opt/dataforge/app/.git ] || git clone --depth 1 https://github.com/Lethabo-T-Molefe/dataforge.backend /opt/dataforge/app
mkdir -p /opt/dataforge/var/landing
chown -R dataforge:dataforge /opt/dataforge
cd /opt/dataforge/app
sudo -u dataforge env UV_PYTHON_INSTALL_DIR=/opt/dataforge/python /usr/local/bin/uv sync --frozen --no-dev

# HTTPS hostname from the public IP (free sslip.io DNS + Let's Encrypt via Caddy)
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
HOST="${IP//./-}.sslip.io"
echo "$HOST" > /opt/dataforge/hostname
printf '%s {\n\treverse_proxy 127.0.0.1:8000\n}\n' "$HOST" > /etc/caddy/Caddyfile
systemctl enable caddy
systemctl restart caddy

cat > /etc/systemd/system/dataforge.service <<EOF
[Unit]
Description=DataForge API
After=network-online.target
Wants=network-online.target
ConditionPathExists=/opt/dataforge/app/.env

[Service]
User=dataforge
WorkingDirectory=/opt/dataforge/app
Environment=ENVIRONMENT=production
Environment=STORAGE_DIR=/opt/dataforge/var/landing
Environment=PUBLIC_API_URL=https://$HOST
Environment=PATH=/opt/dataforge/app/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStartPre=/opt/dataforge/app/.venv/bin/dataforge migrate
ExecStart=/opt/dataforge/app/.venv/bin/uvicorn dataforge_backend.main:app --host 127.0.0.1 --port 8000 --proxy-headers
Restart=always
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

cp /opt/dataforge/app/ops/ec2/deploy.sh /opt/dataforge/deploy.sh
chmod 755 /opt/dataforge/deploy.sh

cat > /opt/dataforge/update.sh <<'EOF'
#!/bin/bash
set -e
cd /opt/dataforge/app
sudo -u dataforge git pull --ff-only
sudo -u dataforge env UV_PYTHON_INSTALL_DIR=/opt/dataforge/python /usr/local/bin/uv sync --frozen --no-dev
sudo systemctl restart dataforge
EOF
chmod +x /opt/dataforge/update.sh

systemctl daemon-reload
systemctl enable dataforge
echo "NATIVE SETUP DONE -> https://$HOST"
