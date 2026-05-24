# 🔄 OpenAI Compatible Proxy

A lightweight, low-dependency proxy that presents an OpenAI-compatible API while forwarding all requests to any real backend — forcing your chosen model on every request.

> **Use case:** You have a client hardcoded to call `gpt-4` but you actually want it hitting `accounts/fireworks/models/deepseek-v4-pro` (or any other model/backend). This proxy sits in the middle and rewrites the model name transparently. Supports streaming (SSE) responses too.

---

## ✨ Features

- **OpenAI-compatible API** — Drop-in replacement for any client expecting `/v1/chat/completions`
- **Model forcing** — Every incoming model name gets rewritten to your configured model
- **Fast & Smart model routing** — Request `model="fast"` for your cheap/fast model, `model="smart"` for your capable/slow model (optional — falls back to fast if not configured)
- **Automatic fallback on failure** — If one backend times out, returns a server error (5xx), or rate-limits you (429), the proxy automatically retries on the other model. Works bidirectionally: fast↔smart.
- **Streaming support** — SSE streaming works out of the box (no timeouts on long LLM requests)
- **Configurable timeout** — Per-request timeout (default 120s) so stalled backends trigger fallback instead of hanging forever
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

### Fast model (default)

| Variable | Description | Example |
|---|---|---|
| `FAST_BASE_URL` | The actual backend to forward to | `https://api.fireworks.ai/inference/v1` |
| `FAST_API_KEY` | Your real API key for the backend | `sk-...` |
| `FAST_MODEL_NAME` | The model name to force | `accounts/fireworks/models/deepseek-v4-pro` |

> **Backward compatibility:** If you have an existing `.env` using the old `REAL_*` names (`REAL_BASE_URL`, `REAL_API_KEY`, `REAL_MODEL_NAME`), those still work as fallbacks.

### Smart model (optional)

| Variable | Description | Example |
|---|---|---|
| `SMART_BASE_URL` | Backend for "smart" requests (falls back to `FAST_BASE_URL`) | `https://api.openai.com/v1` |
| `SMART_API_KEY` | API key for the smart backend (falls back to `FAST_API_KEY`) | `sk-...` |
| `SMART_MODEL_NAME` | Model to use for "smart" requests (falls back to `FAST_MODEL_NAME`) | `gpt-4o` |

If none of the `SMART_*` variables are set, requesting `model="smart"` will behave identically to `model="fast"` — and **fallback will be disabled** since there's nowhere else to fall back to.

### Server settings

| Variable | Description | Example |
|---|---|---|
| `PROXY_PORT` | Port the proxy listens on | `8086` |
| `BIND_HOST` | IP address the server binds to (set to LAN IP to avoid WAN exposure) | `192.168.1.1` |
| `PROXY_TIMEOUT` | Seconds before a backend request times out and triggers fallback (0 = no timeout) | `120` |

### 🔄 Fallback behavior

When both fast and smart models are configured (i.e., they point to different backends or different model names), the proxy will **automatically retry on the other model** if the primary one fails due to:

- **Connection errors** — can't reach the backend at all
- **Timeouts** — backend doesn't respond within `PROXY_TIMEOUT` seconds
- **Server errors** — backend returns HTTP 5xx
- **Rate limiting** — backend returns HTTP 429

The client sees only the successful response from whichever backend worked. If both fail, the proxy returns the primary backend's error (504 for timeouts, or the actual error status otherwise).

> **Note:** For streaming responses that have already started sending data to the client, mid-stream failures cannot be recovered — the fallback only applies to errors that occur *before* the first byte is sent.

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
# Use the fast model explicitly
curl http://localhost:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key-works" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "Tell me a joke!"}]
  }'

# Use the smart model for heavier tasks
curl http://localhost:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key-works" \
  -d '{
    "model": "smart",
    "messages": [{"role": "user", "content": "Write a detailed analysis of..."}]
  }'

# Any unrecognized model name goes to the fast backend
curl http://localhost:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key-works" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The `model` field will be silently rewritten to your configured `FAST_MODEL_NAME` or `SMART_MODEL_NAME`. If the primary backend fails, the proxy transparently retries on the other one.

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
