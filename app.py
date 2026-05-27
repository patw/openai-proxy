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
from datetime import date, timedelta

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "llm-proxy-dev-key-change-me")
PROXY_PORT = int(os.getenv("PROXY_PORT", 8086))
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")


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


# ---- Model CRUD ----

@app.route("/models/new", methods=["GET", "POST"])
def model_new():
    if request.method == "POST":
        form = request.form
        errors = validate_model_form(form, editing=False)
        if errors:
            return render_template("model_form.html", editing=False,
                                   form=form, errors=errors)

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
            "api_model_name": form["api_model_name"].strip(),
            "input_price_per_million": float(form.get("input_price_per_million", 0) or 0),
            "output_price_per_million": float(form.get("output_price_per_million", 0) or 0),
            "cached_price_per_million": float(form.get("cached_price_per_million", 0) or 0),
            "enabled": form.get("enabled") == "1",
        }
        save_model(data)
        flash(f"Model '{data['name']}' added.", "success")
        return redirect(url_for("index"))

    return render_template("model_form.html", editing=False, form={}, errors=[])


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
            return render_template("model_form.html", editing=True,
                                   model=model, form=form, errors=errors)

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
                           form={}, errors=[])


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
        updates = {
            "electricity_cost_per_kwh": float(request.form.get("electricity_cost_per_kwh", 0.12)),
            "local_model_max_wattage": int(request.form.get("local_model_max_wattage", 300)),
        }
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
    """OpenAI-compatible model list."""
    models = list_models(enabled_only=True)
    data = []
    for m in models:
        data.append({
            "id": m["name"],
            "object": "model",
            "created": 0,
            "owned_by": m.get("provider", "unknown"),
        })
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
    print(f"Starting LLM Proxy on {BIND_HOST}:{PROXY_PORT}")
    print(f"Web UI:  http://{BIND_HOST}:{PROXY_PORT}/")
    print(f"Proxy:   http://{BIND_HOST}:{PROXY_PORT}/v1/chat/completions")
    app.run(host=BIND_HOST, port=PROXY_PORT, debug=False, threaded=True)
