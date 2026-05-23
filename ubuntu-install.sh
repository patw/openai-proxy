#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  OpenAI Compatible Proxy - Ubuntu Install"
echo "============================================"
echo ""

REQUIRED_PACKAGES=(
    python3-fastapi
    python3-uvicorn
    python3-httpx
    python3-dotenv
)

echo "📦 Installing system packages via apt..."
echo "   Packages: ${REQUIRED_PACKAGES[*]}"
echo ""

sudo apt update
sudo apt install -y "${REQUIRED_PACKAGES[@]}"

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and configure it:"
echo "       cp .env.example .env"
echo "       nano .env"
echo ""
echo "  2. Run the proxy:"
echo "       ./start.sh"
echo ""
