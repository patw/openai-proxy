#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  LLM Proxy - Ubuntu Install"
echo "============================================"
echo ""

# ---- System packages (available via apt) ----
APT_PACKAGES=(
    python3-flask
    python3-httpx
    python3-dotenv
    python3-matplotlib
)

echo "📦 Installing system packages via apt..."
echo "   Packages: ${APT_PACKAGES[*]}"
echo ""

sudo apt update
sudo apt install -y "${APT_PACKAGES[@]}"

# ---- Pip packages (not in apt) ----
echo ""
echo "📦 Installing pip packages..."
echo "   (moofile — embedded BSON document store)"
echo ""

pip install moofile --break-system-packages

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env:"
echo "       cp .env.example .env"
echo ""
echo "  2. (Optional) Edit .env to change the port or bind address."
echo ""
echo "  3. Run the proxy + web UI:"
echo "       ./start.sh"
echo ""
echo "  4. Open http://localhost:8086/ and add your first model!"
echo ""
