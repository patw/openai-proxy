#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for .env
if [ ! -f .env ]; then
    echo "❌ .env file not found. Copy .env.example to .env and fill in your values."
    exit 1
fi

# Load .env
set -a
source .env
set +a

# Ensure data directory exists
mkdir -p data

# Run the Flask app
echo "🚀 Starting LLM Proxy (Web UI + Proxy)..."
python3 app.py
