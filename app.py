"""
LLM Proxy + Web UI — Flask application.

Serves:
  /                     Model dashboard
  /models/new           Add a model
  /models/<name>/edit   Edit a model
  /models/<name>/delete Delete a model
  /settings             Global settings (electricity, wattage)
  /reports              Usage charts and tables
  /v1/chat/completions  OpenAI-compatible proxy endpoint
  /v1/models            List available models (OpenAI-compatible)
"""

import os
import json
import hmac
from datetime import date, timedelta

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Load environment variables from .env before any config is read
load_dotenv()

from storage import get_settings, save_settings
from models_config import (
    list_models, get_model, save_model, delete_model,
    validate_model_form, model_tags_summary,
)
from proxy import handle_proxy_request
from reporting import (
    get_daily_usage, get_per_model_summary,
    daily_cost_chart, monthly_cost_chart, model_breakdown_chart,
    token_volume_chart,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB request body limit

_DEFAULT_SECRET = "llm-proxy-dev-key-change-me"
_PLACEHOLDER_SECRET = "change-me-to-something-random"
app.secret_key = os.getenv("FLASK_SECRET_KEY", _DEFAULT_SECRET)

if not app.secret_key or app.secret_key in (_DEFAULT_SECRET, _PLACEHOLDER_SECRET):
    raise RuntimeError(
        "FLASK_SECRET_KEY is not set in .env or is still the default placeholder. "
        "Please set it to a random value before starting the server."
    )

PROXY_PORT = int(os.getenv("PROXY_PORT", 8086))
# Default to loopback so a fresh install isn't exposed to the network. Set
# BIND_HOST to your LAN IP (or 0.0.0.0) in .env to reach it from other devices.
BIND_HOST = os.getenv("BIND_HOST", "127.0.0.1")

# Optional shared-secret gate for the proxy API. When PROXY_API_KEY is unset
# (the default) the /v1/* endpoints are open — fine for localhost-only use.
# Set it in .env to require `Authorization: Bearer <key>` on every proxy call,
# which is what you want once BIND_HOST is exposed to your LAN.
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "").strip()

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


# ---------------------------------------------------------------------------
# Proxy API authentication
# ---------------------------------------------------------------------------

@app.before_request
def _require_proxy_key():
    """Enforce the shared key on /v1/* requests when one is configured.

    No key configured → open (localhost default). The web UI and /health are
    never gated here.
    """
    if not request.path.startswith("/v1/"):
        return
    if not PROXY_API_KEY:
        return
    expected = f"Bearer {PROXY_API_KEY}"
    provided = request.headers.get("Authorization", "")
    if not hmac.compare_digest(provided, expected):
        return jsonify({"error": "Unauthorized"}), 401


# ---------------------------------------------------------------------------
# Template filters
# ---------------------------------------------------------------------------

@app.template_filter("format_number")
def format_number(n):
    """Format large integers with commas."""
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


# ---------------------------------------------------------------------------
# Web UI routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    models = list_models()
    tags = model_tags_summary()
    return render_template("index.html", models=models, tags=tags,
                           host=BIND_HOST, port=PROXY_PORT)


@app.route("/health")
def health():
    """Basic health check with tag/model status."""
    models = list_models(enabled_only=True)
    tags = model_tags_summary()
    return jsonify({
        "status": "ok",
        "models": {
            "total": len(models),
            "enabled": len(models),
            "fast": tags.get("fast"),
            "smart": tags.get("smart"),
            "local": tags.get("local"),
        },
        "api": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
        }
    })


# ---- Model CRUD ----


def _form_to_model_preview(form):
    """Build a model-shaped dict from form data so the UI can repopulate."""
    tag_value = form.get("tag", "").strip()
    return {
        "name": form.get("name", "").strip(),
        "display_name": form.get("display_name", "").strip(),
        "provider": form.get("provider", "").strip(),
        "type": form.get("type", "remote"),
        "tags": [tag_value] if tag_value in ("fast", "smart", "local") else [],
        "base_url": form.get("base_url", "").strip(),
        "api_key": form.get("api_key", ""),
        "api_key_header": form.get("api_key_header", "").strip(),
        "api_model_name": form.get("api_model_name", "").strip(),
        "input_price_per_million": form.get("input_price_per_million", "0") or "0",
        "output_price_per_million": form.get("output_price_per_million", "0") or "0",
        "cached_price_per_million": form.get("cached_price_per_million", "0") or "0",
        "enabled": form.get("enabled") == "1",
    }


@app.route("/models/new", methods=["GET", "POST"])
def model_new():
    if request.method == "POST":
        form = request.form
        errors = validate_model_form(form, editing=False)
        if errors:
            return render_template("model_form.html", editing=False,
                                   model=_form_to_model_preview(form), errors=errors)

        tag_value = form.get("tag", "").strip()
        tags = [tag_value] if tag_value in ("fast", "smart", "local") else []

        data = {
            "name": form["name"].strip(),
            "display_name": form.get("display_name", "").strip(),
            "provider": form.get("provider", "").strip(),
            "type": form.get("type", "remote"),
            "tags": tags,
            "base_url": form["base_url"].strip(),
            "api_key": form.get("api_key", "").strip(),
            "api_key_header": form.get("api_key_header", "").strip(),
            "api_model_name": form["api_model_name"].strip(),
            "input_price_per_million": float(form.get("input_price_per_million", 0) or 0),
            "output_price_per_million": float(form.get("output_price_per_million", 0) or 0),
            "cached_price_per_million": float(form.get("cached_price_per_million", 0) or 0),
            "enabled": form.get("enabled") == "1",
        }
        save_model(data)
        flash(f"Model '{data['name']}' added.", "success")
        return redirect(url_for("index"))

    return render_template("model_form.html", editing=False, model=None, errors=[])


@app.route("/models/<name>/clone", methods=["GET"])
def model_clone(name):
    """Pre-fill the 'add model' form with all fields from an existing model."""
    source = get_model(name)
    if not source:
        flash(f"Model '{name}' not found.", "error")
        return redirect(url_for("index"))

    form_data = {
        "name": "",
        "display_name": source.get("display_name", ""),
        "provider": source.get("provider", ""),
        "type": source.get("type", "remote"),
        "tags": source.get("tags", []),
        "base_url": source.get("base_url", ""),
        "api_key": source.get("api_key", ""),
        "api_key_header": source.get("api_key_header", ""),
        "api_model_name": source.get("api_model_name", ""),
        "input_price_per_million": source.get("input_price_per_million", 0),
        "output_price_per_million": source.get("output_price_per_million", 0),
        "cached_price_per_million": source.get("cached_price_per_million", 0),
        "enabled": source.get("enabled", True),
    }

    return render_template("model_form.html", editing=False,
                           model=form_data, errors=[])


@app.route("/models/<name>/edit", methods=["GET", "POST"])
def model_edit(name):
    model = get_model(name)
    if not model:
        flash(f"Model '{name}' not found.", "error")
        return redirect(url_for("index"))

    if request.method == "POST":
        form = request.form
        errors = validate_model_form(form, editing=True)
        if errors:
            # Merge submitted values into the model so the form repopulates
            merged = dict(model)
            merged.update(_form_to_model_preview(form))
            merged["name"] = name  # keep read-only
            return render_template("model_form.html", editing=True,
                                   model=merged, errors=errors)

        tag_value = form.get("tag", "").strip()
        tags = [tag_value] if tag_value in ("fast", "smart", "local") else []

        data = {
            "name": name,  # name is read-only on edit
            "display_name": form.get("display_name", "").strip(),
            "provider": form.get("provider", "").strip(),
            "type": form.get("type", "remote"),
            "tags": tags,
            "base_url": form["base_url"].strip(),
            "api_key": form.get("api_key", "").strip(),
            "api_key_header": form.get("api_key_header", "").strip(),
            "api_model_name": form["api_model_name"].strip(),
            "input_price_per_million": float(form.get("input_price_per_million", 0) or 0),
            "output_price_per_million": float(form.get("output_price_per_million", 0) or 0),
            "cached_price_per_million": float(form.get("cached_price_per_million", 0) or 0),
            "enabled": form.get("enabled") == "1",
        }
        save_model(data)
        flash(f"Model '{name}' updated.", "success")
        return redirect(url_for("index"))

    return render_template("model_form.html", editing=True, model=model,
                           errors=[])


@app.route("/models/<name>/delete", methods=["POST"])
def model_delete(name):
    deleted = delete_model(name)
    if deleted:
        flash(f"Model '{name}' deleted.", "success")
    else:
        flash(f"Model '{name}' not found.", "error")
    return redirect(url_for("index"))


# ---- Settings ----

@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    saved = False
    if request.method == "POST":
        try:
            updates = {
                "electricity_cost_per_kwh": float(request.form.get("electricity_cost_per_kwh", 0.12)),
                "local_model_max_wattage": int(request.form.get("local_model_max_wattage", 300)),
                "proxy_timeout_seconds": int(request.form.get("proxy_timeout_seconds", 120)),
            }
        except (ValueError, TypeError):
            flash("Invalid settings value.", "error")
            return redirect(url_for("settings_page"))

        save_settings(updates)
        saved = True

    return render_template("settings.html", settings=get_settings(), saved=saved)


# ---- Reports ----

@app.route("/reports")
def reports():
    days = 30

    daily = get_daily_usage(days)
    per_model = get_per_model_summary(days)

    # Totals
    today_str = date.today().isoformat()
    today_cost = sum(d["cost"] for d in daily if d["date"] == today_str)

    week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
    week_cost = sum(d["cost"] for d in daily if d["date"] >= week_start)

    month_start = date.today().replace(day=1).isoformat()
    month_cost = sum(d["cost"] for d in daily if d["date"] >= month_start)

    # Charts
    daily_chart = daily_cost_chart(days)
    monthly_chart = monthly_cost_chart()
    model_chart = model_breakdown_chart(days)
    token_chart = token_volume_chart(days)

    return render_template(
        "reports.html",
        days=days,
        daily=daily,
        per_model=per_model,
        today_cost=today_cost,
        week_cost=week_cost,
        month_cost=month_cost,
        daily_chart=daily_chart,
        monthly_chart=monthly_chart,
        model_chart=model_chart,
        token_chart=token_chart,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible API routes
# ---------------------------------------------------------------------------

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    body = request.get_data()
    headers = dict(request.headers)
    return handle_proxy_request("POST", "/v1/chat/completions", headers, body)


@app.route("/v1/models", methods=["GET"])
def list_models_api():
    """OpenAI-compatible model list, including tag aliases."""
    models = list_models(enabled_only=True)
    tags = model_tags_summary()
    seen = set()
    data = []
    for m in models:
        name = m["name"]
        seen.add(name)
        data.append({
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": m.get("provider", "unknown"),
        })

    # Add tag aliases so clients can request "fast", "smart", or "local"
    for tag in ("fast", "smart", "local"):
        model_name = tags.get(tag)
        if model_name and tag not in seen:
            data.append({
                "id": tag,
                "object": "model",
                "created": 0,
                "owned_by": "proxy",
            })
            seen.add(tag)

    return jsonify({"object": "list", "data": data})


# ---- Catch-all for other /v1/ paths (forward to fast model if possible) ----

@app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_catchall(path):
    """
    Catch-all for any other /v1/... paths.
    Forwards to the fast-tagged model (best-effort passthrough).
    """
    from models_config import get_model_by_tag

    model = get_model_by_tag("fast")
    if not model:
        return jsonify({"error": "No fast model configured for passthrough."}), 502

    body = request.get_data()
    headers = dict(request.headers)
    return handle_proxy_request(request.method, f"/v1/{path}", headers, body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import waitress
    if BIND_HOST not in _LOOPBACK_HOSTS and not PROXY_API_KEY:
        print(f"⚠️  WARNING: bound to non-loopback {BIND_HOST} with no PROXY_API_KEY set — "
              "the proxy API is reachable on your network with no authentication.")
        print("   Set PROXY_API_KEY in .env to require a shared key.")
    print(f"Starting LLM Proxy on {BIND_HOST}:{PROXY_PORT}")
    print(f"Web UI:  http://{BIND_HOST}:{PROXY_PORT}/")
    print(f"Proxy:   http://{BIND_HOST}:{PROXY_PORT}/v1/chat/completions")
    waitress.serve(app, host=BIND_HOST, port=PROXY_PORT, threads=8)
