#!/bin/bash
# Stvelli Street — Server Deployment Script
# Run as root on Ubuntu/Debian:  bash deploy.sh
set -e

REPO="https://github.com/mdavisyoung-ctrl/stvelli-street.git"
BRANCH="claude/stock-arbitrage-scanner-rzPxb"
INSTALL_DIR="/opt/stvelli-street"
SERVICE="stvelli-street"
PYTHON="python3"

echo "=== Stvelli Street Deployment ==="

# ── System deps ──
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git python3 python3-pip python3-venv curl

# ── Clone or update repo ──
echo "[2/6] Cloning repository..."
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Repo exists — pulling latest..."
    git -C "$INSTALL_DIR" fetch origin
    git -C "$INSTALL_DIR" checkout "$BRANCH"
    git -C "$INSTALL_DIR" pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO" "$INSTALL_DIR"
fi

# ── Python venv + deps ──
echo "[3/6] Setting up Python environment..."
$PYTHON -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q
# CPU-only torch for FinBERT
"$INSTALL_DIR/venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu -q

# ── .env file ──
echo "[4/6] Writing .env..."
cat > "$INSTALL_DIR/.env" <<EOF
EODHD_API_KEY=69af52dd739130.88973111
TELEGRAM_TOKEN=${TELEGRAM_TOKEN:-}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-}
EOF
echo "  .env written. Edit $INSTALL_DIR/.env to add your Telegram credentials."

# ── Pre-download FinBERT model so first scan is instant ──
echo "[5/6] Pre-loading FinBERT model (downloads ~430MB once)..."
cd "$INSTALL_DIR"
"$INSTALL_DIR/venv/bin/python" -c "
from transformers import pipeline
pipe = pipeline('text-classification', model='ProsusAI/finbert', device=-1, top_k=None)
print('  FinBERT ready.')
" || echo "  FinBERT preload failed — will download on first scan."

# ── Systemd service ──
echo "[6/6] Installing systemd service..."
cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=Stvelli Street Arbitrage Scanner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/scan_once.py --loop
Restart=always
RestartSec=30
StandardOutput=append:$INSTALL_DIR/scanner.log
StandardError=append:$INSTALL_DIR/scanner.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Scanner status:  systemctl status $SERVICE"
echo "Live logs:       journalctl -u $SERVICE -f"
echo "Stop scanner:    systemctl stop $SERVICE"
echo "Restart:         systemctl restart $SERVICE"
echo ""
echo "Next step — add Telegram so your phone gets trade alerts:"
echo "  1. Message @BotFather on Telegram → /newbot → copy token"
echo "  2. Message @userinfobot on Telegram → copy your chat_id"
echo "  3. Edit $INSTALL_DIR/.env and fill in:"
echo "       TELEGRAM_TOKEN=your_token"
echo "       TELEGRAM_CHAT_ID=your_chat_id"
echo "  4. systemctl restart $SERVICE"
