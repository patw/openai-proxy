# 🔄 OpenAI Compatible Proxy

A lightweight, low-dependency proxy that presents an OpenAI-compatible API while forwarding all requests to any real backend — forcing your chosen model on every request.

> **Use case:** You have a client hardcoded to call `gpt-4` but you actually want it hitting `accounts/fireworks/models/deepseek-v4-pro` (or any other model/backend). This proxy sits in the middle and rewrites the model name transparently. Supports streaming (SSE) responses too.

---

## ✨ Features

- **OpenAI-compatible API** — Drop-in replacement for any client expecting `/v1/chat/completions`
- **Model forcing** — Every incoming model name gets rewritten to your configured model
- **Streaming support** — SSE streaming works out of the box (no timeouts on long LLM requests)
- **Smart path stitching** — Handles nested base paths (e.g., `/inference/v1`) without double-prefixing
- **Deployed via apt** — Every dependency is available as a native Ubuntu package, no pip needed

---

## 🚀 Quick Start

### Option A: Ubuntu native (no pip)

```bash
# Install dependencies via apt
./ubuntu-install.sh

# Copy and edit your config
cp .env.example .env
nano .env

# Run
./start.sh
```

### Option B: pip / uv

```bash
pip install -r requirements.txt
# or: uv pip install -r requirements.txt

cp .env.example .env
nano .env

./start.sh
```

---

## ⚙️ Configuration

All config lives in `.env`:

| Variable | Description | Example |
|---|---|---|
| `REAL_BASE_URL` | The actual backend to forward to | `https://api.fireworks.ai/inference/v1` |
| `REAL_API_KEY` | Your real API key for the backend | `sk-...` |
| `REAL_MODEL_NAME` | The model name to force | `accounts/fireworks/models/deepseek-v4-pro` |
| `PROXY_PORT` | Port the proxy listens on | `8086` |

---

## 📡 Usage

Once running, point any OpenAI-compatible client at your proxy:

```bash
# Example with curl
curl http://localhost:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key-works" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The `model` field will be silently rewritten to whatever you set in `REAL_MODEL_NAME`.

---

## 🧱 Dependency Mapping

| pip package | Ubuntu apt package |
|---|---|
| `fastapi` | `python3-fastapi` |
| `uvicorn` | `python3-uvicorn` |
| `httpx` | `python3-httpx` |
| `python-dotenv` | `python3-dotenv` |

---

## 📄 License

MIT — do whatever you want with it.
