# Deploying to an always-on server (e.g. Oracle Cloud Free Tier)

Prerequisites you handle yourself (Claude can't do these — they need your
own account/payment verification): a running Ubuntu 22.04/24.04 instance,
its public IP, and the SSH private key Oracle generated when you created
it.

## 1. Connect

```bash
chmod 600 /path/to/your-oracle-key.pem   # SSH refuses a key with loose permissions
ssh -i /path/to/your-oracle-key.pem ubuntu@<your-instance-ip>
```

(Default login user is `ubuntu` on Oracle's Ubuntu images; `opc` if you
picked Oracle Linux instead.)

## 2. Copy the project onto the server

From your local machine (a new terminal, not the SSH session above) —
**this copies your local `.env` too if you have one; check it doesn't
before running this**, since credentials shouldn't cross a network in a
plain rsync unless you're already relying on SSH's own encryption (which
you are here, but it's still worth being deliberate rather than copying it
by accident):

```bash
rsync -avz --exclude 'venv' --exclude '.env' --exclude '__pycache__' \
    -e "ssh -i /path/to/your-oracle-key.pem" \
    "C:/Users/cheng/Downloads/trading_bot/trading_bot/" \
    ubuntu@<your-instance-ip>:/tmp/trading_bot_upload/
```

Then, back in the SSH session:

```bash
sudo mkdir -p /opt/trading_bot
sudo mv /tmp/trading_bot_upload /opt/trading_bot/trading_bot
```

## 3. Run setup

```bash
cd /opt/trading_bot/trading_bot
sudo bash deploy/setup.sh
```

This installs Python/build tools, creates a dedicated unprivileged
`trading-bot` system user (so the process holding your live credentials
isn't running as root or your own login), builds the venv, and installs
the systemd service — but does NOT start it. It prints the exact next
steps (creating `.env` on the server, confirming `settings.yaml`, starting
the service) when it finishes.

## 4. Create `.env` on the server

Type it directly on the server — don't transfer a filled-in `.env` over
any channel, even an encrypted one, as a matter of habit:

```bash
sudo -u trading-bot nano /opt/trading_bot/trading_bot/.env
sudo chmod 600 /opt/trading_bot/trading_bot/.env
```

Same variables as local: `ROBINHOOD_USERNAME`, `ROBINHOOD_PASSWORD`,
`ROBINHOOD_MFA_SECRET`, `ROBINHOOD_ACCOUNT_NUMBER`, `ALPACA_API_KEY_ID`,
`ALPACA_API_SECRET_KEY`.

## 5. Enable the market-hours schedule

Rather than running 24/7, `trading-bot.service` is started and stopped
automatically by two `systemd` timers -- 9:30am and 4:00pm America/New_York,
weekdays, DST handled automatically since that's a real timezone, not a
fixed UTC offset. Nothing needs to be open on your end for this to work --
not Claude, not your phone, nothing:

```bash
sudo systemctl enable --now trading-bot-start.timer trading-bot-stop.timer
systemctl list-timers trading-bot-*        # confirms both are scheduled
sudo journalctl -u trading-bot -f          # watch it live once it's running
```

`enable` on the *timers* is what survives a server reboot — the main
`trading-bot.service` itself is deliberately not `enable`d directly, since
we don't want it starting immediately on every boot regardless of time of
day.

**To test right now** instead of waiting for the next scheduled window:
`sudo systemctl start trading-bot` (and `stop` it yourself when done — the
stop timer won't touch an out-of-schedule run you started manually until
its own next scheduled 16:00).

**One known gap, inherited from `orchestration/market_hours.py`**: no
holiday calendar. On a market holiday that falls on a weekday, the start
timer still fires -- harmless (it'll just poll a market that isn't moving
and do nothing), but not truly holiday-aware. Building a holiday calendar
is a small, separate task if it ever actually matters to you.

## Staying safe while this runs unattended

- `config/settings.yaml`: leave `options.live_trading_enabled: false` for a
  good while after first deploying. Dry-run trades still get logged
  (`orchestration/trade_log.py`) and show up in the dashboard — watch that
  before ever flipping it to true on a machine you're not in front of.
- Updating the code later: re-run the `rsync` step from #2 (still
  excluding `venv`/`.env`), then `sudo systemctl restart trading-bot`.
- Checking on it without SSH-ing in every time: run
  `dashboard/live_account_dashboard.py` from your own machine (or also
  deploy it here on its own service) — it just needs Robinhood
  credentials, not to be colocated with the trading loop.
