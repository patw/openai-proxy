#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for .env
if [ ! -f .env ]; then
    echo "❌ .env file not found. Copy .env.example to .env and fill in your values."
    exit 1
fi

# Activate virtual environment if present
if [ -f venv/bin/activate ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

echo "🚀 Starting OpenAI Compatible Proxy..."
python3 main.py
