# 🔄 OpenAI Compatible Proxy

A lightweight, low-dependency proxy that presents an OpenAI-compatible API while forwarding all requests to any real backend — forcing your chosen model on every request.

> **Use case:** You have a client hardcoded to call `gpt-4` but you actually want it hitting `accounts/fireworks/models/deepseek-v4-pro` (or any other model/backend). This proxy sits in the middle and rewrites the model name transparently. Supports streaming (SSE) responses too.

---

## ✨ Features

- **OpenAI-compatible API** — Drop-in replacement for any client expecting `/v1/chat/completions`
- **Model forcing** — Every incoming model name gets rewritten to your configured model
- **Streaming support** — SSE streaming works out of the box (no timeouts on long LLM requests)
- **Smart path stitching** — Handles nested base paths (e.g., `/inference/v1`) without double-prefixing
- **Bind to any interface** — Bind to a specific LAN IP to expose the proxy to your local network while keeping it off the WAN side (great for running on a router!)
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
| `BIND_HOST` | IP address the server binds to (set to LAN IP to avoid WAN exposure) | `192.168.1.1` |

### 🔒 Binding to LAN only

By default `BIND_HOST=0.0.0.0` listens on **all** network interfaces — which includes the WAN side if you're on a router. To keep the proxy accessible from your LAN but hidden from the internet, set `BIND_HOST` to your router's LAN IP:

```ini
# Only listen on the LAN interface — WAN stays blind
BIND_HOST=192.168.1.1
```

Now devices on your local network can hit `http://192.168.1.1:8086` but the outside world cannot.

---

## 📡 Usage

Once running, point any OpenAI-compatible client at your proxy:

```bash
# From the same machine (if BIND_HOST is 0.0.0.0 or 127.0.0.1)
curl http://localhost:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key-works" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# From another device on the LAN (if BIND_HOST is your LAN IP)
curl http://192.168.1.1:8086/v1/chat/completions \
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

## 🛡️ Security Note

- **Never commit `.env` to git.** It contains your real API key. The proxy itself doesn't validate incoming auth — it replaces whatever the client sends with your real key. Keep this thing on localhost or behind a firewall/VPN.
- **Use `BIND_HOST` to lock down exposure.** If running on a router or multi-interface machine, set `BIND_HOST` to your LAN IP so the proxy isn't reachable from the WAN side. The `.gitignore` already covers your live `.env`.

---

## 📄 License

MIT — do whatever you want with it.
