#!/usr/bin/env bash
# One-time provisioning for a fresh Ubuntu 22.04/24.04 Oracle Cloud "Always
# Free" instance. Run AFTER copying the project onto the server (see
# deploy/README.md for the exact command) and BEFORE starting the service.
#
# Usage, on the server:
#   sudo bash deploy/setup.sh
set -euo pipefail

PROJECT_ROOT="/opt/trading_bot"
PROJECT_DIR="$PROJECT_ROOT/trading_bot"
SERVICE_USER="trading-bot"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Expected the project at $PROJECT_DIR but it's not there."
    echo "Copy it there first (see deploy/README.md), then re-run this script."
    exit 1
fi

echo "== Installing system packages =="
apt-get update
apt-get install -y python3 python3-venv python3-pip build-essential

echo "== Creating a dedicated, unprivileged service user =="
# Runs as its own account, not root and not your login user -- this process
# holds live brokerage/market-data credentials, so it gets its own reduced
# blast radius if anything on this box is ever compromised.
id -u "$SERVICE_USER" &>/dev/null || useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"

echo "== Setting ownership =="
chown -R "$SERVICE_USER":"$SERVICE_USER" "$PROJECT_ROOT"

echo "== Creating venv and installing requirements (as $SERVICE_USER) =="
sudo -u "$SERVICE_USER" python3 -m venv "$PROJECT_DIR/venv"
sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/pip" install --upgrade pip --quiet
sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "== Creating log directory =="
mkdir -p /var/log/trading-bot
chown "$SERVICE_USER":"$SERVICE_USER" /var/log/trading-bot

echo "== Installing systemd units (main service + market-hours timers) =="
cp "$PROJECT_DIR/deploy/trading-bot.service" /etc/systemd/system/trading-bot.service
cp "$PROJECT_DIR/deploy/trading-bot-start.timer" /etc/systemd/system/trading-bot-start.timer
cp "$PROJECT_DIR/deploy/trading-bot-start.service" /etc/systemd/system/trading-bot-start.service
cp "$PROJECT_DIR/deploy/trading-bot-stop.timer" /etc/systemd/system/trading-bot-stop.timer
cp "$PROJECT_DIR/deploy/trading-bot-stop.service" /etc/systemd/system/trading-bot-stop.service
systemctl daemon-reload

cat <<'EOF'

== Setup done. Before starting anything: ==

1. Create the .env file ON THIS SERVER (never copy a filled-in .env over an
   untrusted channel -- type the values directly here):
     sudo -u trading-bot nano /opt/trading_bot/trading_bot/.env
   Same variables as your local .env: ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD,
   ROBINHOOD_MFA_SECRET, ROBINHOOD_ACCOUNT_NUMBER, ALPACA_API_KEY_ID,
   ALPACA_API_SECRET_KEY. Then lock it down:
     sudo chmod 600 /opt/trading_bot/trading_bot/.env

2. Confirm config/settings.yaml has options.enabled: true, and
   options.live_trading_enabled: false -- stay dry-run and watch the logs
   for a while before ever flipping that to true on a server you're not
   physically in front of.

3. Enable the MARKET-HOURS TIMERS (not the main service directly -- the
   timers start/stop it automatically at 9:30/16:00 America/New_York,
   weekdays, so nothing needs to be open on your end at all):
     sudo systemctl enable --now trading-bot-start.timer trading-bot-stop.timer

4. Optional: to test right now instead of waiting for the next scheduled
   window, start the bot service directly (the timers won't stop this
   early one until the next scheduled 16:00, so stop it yourself when done
   testing):
     sudo systemctl start trading-bot
     sudo systemctl stop trading-bot

5. Check the timers are actually scheduled:
     systemctl list-timers trading-bot-*

6. Watch it (whenever it's running):
     sudo journalctl -u trading-bot -f
   or:
     tail -f /var/log/trading-bot/run_live.log
EOF
