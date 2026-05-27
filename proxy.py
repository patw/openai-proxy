"""
Proxy forwarding logic — sends requests to the real backend, captures usage,
and implements automatic fallback between fast ↔ smart models.
"""

import json
import time
import httpx
from urllib.parse import urlparse
from flask import Response, stream_with_context, jsonify

from usage_tracker import extract_usage_from_json, extract_usage_from_sse_line, record_usage
from models_config import get_model, get_model_by_tag
from storage import get_settings

TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


# ---------------------------------------------------------------------------
# Path resolution (ported from original)
# ---------------------------------------------------------------------------

def _resolve_path(incoming_path: str, base_url: str) -> str:
    """Stitch the incoming path onto the base URL, avoiding duplicate segments."""
    incoming = [s for s in incoming_path.split("/") if s]
    parsed = urlparse(base_url)
    base_segments = [s for s in parsed.path.split("/") if s]

    if incoming and base_segments and incoming[0] == base_segments[-1]:
        incoming = incoming[1:]

    remainder = "/".join(incoming)
    base = base_url.rstrip("/")
    return f"{base}/{remainder}" if remainder else base


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------

def resolve_model(requested_model: str | None) -> dict | None:
    """
    Resolve the requested model name to a full config dict.
    Returns None if the model is unknown.
    """
    if not requested_model:
        return None

    if requested_model in ("fast", "smart", "local"):
        return get_model_by_tag(requested_model)

    return get_model(requested_model)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def calculate_cost(model: dict, usage: dict, duration_seconds: float, settings: dict) -> float:
    """
    Calculate the cost of a request.

    For remote models:  token-based pricing.
    For local models:   wattage × duration × electricity cost.
    """
    if model.get("type") == "local":
        wattage = settings.get("local_model_max_wattage", 300)
        price_kwh = settings.get("electricity_cost_per_kwh", 0.12)
        hours = duration_seconds / 3600.0
        kwh = (wattage / 1000.0) * hours
        return kwh * price_kwh

    # Remote model — token pricing
    it = usage.get("input_tokens", 0)
    ot = usage.get("output_tokens", 0)
    ct = usage.get("cached_tokens", 0)

    iprice = model.get("input_price_per_million", 0)
    oprice = model.get("output_price_per_million", 0)
    cprice = model.get("cached_price_per_million", 0)

    cost = (it / 1_000_000) * iprice
    cost += (ot / 1_000_000) * oprice
    cost += (ct / 1_000_000) * cprice
    return cost


# ---------------------------------------------------------------------------
# Forwarding — non-streaming
# ---------------------------------------------------------------------------

def _forward_non_streaming(model: dict, method: str, path: str,
                           headers: dict, body: bytes):
    """
    Forward a non-streaming request to the backend.

    Returns (response, usage_dict_or_None).
    """
    url = _resolve_path(path, model["base_url"])

    # Rewrite the model name in the body using string replacement to
    # avoid altering JSON whitespace, key ordering, or encoding.
    new_body = body
    requested_model = "?"
    if method == "POST" and body:
        try:
            data = json.loads(body)
            requested_model = data.get("model", "?")
            # String-replace the model value in the original body so we
            # don't disturb any other formatting the backend might expect.
            old_name = json.dumps(requested_model)  # quoted JSON string
            new_name = json.dumps(model["api_model_name"])
            new_body = body.replace(
                b'"model":' + old_name.encode("utf-8"),
                b'"model":' + new_name.encode("utf-8"),
                1,
            )
            # Fallback: if string replace didn't work (unusual spacing),
            # fall back to parse-rewrite-serialize.
            if new_body == body:
                data["model"] = model["api_model_name"]
                new_body = json.dumps(data).encode("utf-8")
        except Exception:
            pass

    # Build headers — keep everything the client sent except hop-by-hop
    # headers and anything we explicitly override below.
    fwd_headers = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding", "connection",
                  "authorization"):
            continue
        fwd_headers[k] = v

    # Ensure Content-Type is set for POST requests
    if method == "POST" and "content-type" not in {k.lower() for k in fwd_headers}:
        fwd_headers["Content-Type"] = "application/json"

    fwd_headers["host"] = urlparse(model["base_url"]).netloc
    if model.get("api_key"):
        fwd_headers["authorization"] = f"Bearer {model['api_key']}"

    print(f"[{model['name']}] {method} {url}  (requested: '{requested_model}' "
          f"→ using: '{model['api_model_name']}')")

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.request(method, url, headers=fwd_headers, content=new_body)

    # Log response for non-2xx so we can diagnose backend errors
    if resp.status_code >= 400:
        try:
            body_preview = resp.text[:500]
            print(f"[{model['name']}] ← {resp.status_code}: {body_preview}")
        except Exception:
            print(f"[{model['name']}] ← {resp.status_code} (body unreadable)")

    # Try to extract usage from response body
    usage = None
    if resp.status_code < 400:
        try:
            resp_data = resp.json()
            usage = extract_usage_from_json(resp_data)
        except Exception:
            pass

    return resp, usage


# ---------------------------------------------------------------------------
# Forwarding — streaming (SSE)
# ---------------------------------------------------------------------------

def _forward_streaming(model: dict, method: str, path: str,
                       headers: dict, body: bytes):
    """
    Forward a streaming request, yielding raw bytes while capturing
    usage data from the final SSE chunk.

    Returns (generator_or_Response, usage_container_list).
    Caller should use stream_with_context().
    """
    url = _resolve_path(path, model["base_url"])

    new_body = body
    requested_model = "?"
    if method == "POST" and body:
        try:
            data = json.loads(body)
            requested_model = data.get("model", "?")
            old_name = json.dumps(requested_model)
            new_name = json.dumps(model["api_model_name"])
            new_body = body.replace(
                b'"model":' + old_name.encode("utf-8"),
                b'"model":' + new_name.encode("utf-8"),
                1,
            )
            if new_body == body:
                data["model"] = model["api_model_name"]
                new_body = json.dumps(data).encode("utf-8")
        except Exception:
            pass

    fwd_headers = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding", "connection",
                  "authorization"):
            continue
        fwd_headers[k] = v
    if method == "POST" and "content-type" not in {k.lower() for k in fwd_headers}:
        fwd_headers["Content-Type"] = "application/json"
    fwd_headers["host"] = urlparse(model["base_url"]).netloc
    if model.get("api_key"):
        fwd_headers["authorization"] = f"Bearer {model['api_key']}"

    print(f"[{model['name']}] {method} {url} (stream)  (requested: '{requested_model}' "
          f"→ using: '{model['api_model_name']}')")

    usage_container = [None]  # mutable container for the generator to write into

    def generate():
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream(method, url, headers=fwd_headers, content=new_body) as resp:
                current_data = ""
                for line in resp.iter_lines():
                    # Yield the line back to the client
                    yield (line + "\n").encode("utf-8")

                    # Try to parse SSE data lines for usage
                    if line.startswith("data: "):
                        sse_data = line[6:]  # strip "data: "
                        if sse_data != "[DONE]":
                            current_data = sse_data

                # After the stream ends, extract usage from the last data line
                if current_data:
                    usage = extract_usage_from_sse_line(current_data)
                    if usage:
                        usage_container[0] = usage

    return generate, usage_container


# ---------------------------------------------------------------------------
# Top-level proxy handler
# ---------------------------------------------------------------------------

def handle_proxy_request(method: str, path: str, headers: dict, body: bytes):
    """
    Full proxy request handler.

    1. Resolve the requested model to a backend config.
    2. Forward the request (with fallback for fast↔smart).
    3. Capture and record usage.
    4. Return a Flask Response.
    """
    # ---- Resolve model ----
    try:
        req_data = json.loads(body)
        requested_model = req_data.get("model")
    except Exception:
        requested_model = None

    model = resolve_model(requested_model)
    if model is None:
        return jsonify({"error": f"Unknown model: '{requested_model}'"}), 400

    if not model.get("enabled", True):
        return jsonify({"error": f"Model '{requested_model}' is disabled."}), 400

    is_streaming = False
    try:
        is_streaming = json.loads(body).get("stream", False)
    except Exception:
        pass

    settings = get_settings()

    # ---- Determine fallback ----
    primary = model
    fallback = None
    primary_label = None
    fallback_label = None

    tags = primary.get("tags", [])
    if "fast" in tags:
        primary_label = "fast"
        fallback = get_model_by_tag("smart")
        fallback_label = "smart"
    elif "smart" in tags:
        primary_label = "smart"
        fallback = get_model_by_tag("fast")
        fallback_label = "fast"

    # Don't fallback to the same model
    if fallback and fallback["name"] == primary["name"]:
        fallback = None

    # ---- Forward to primary ----
    resp, usage, duration, error = _attempt_forward(
        primary, method, path, headers, body, is_streaming, settings
    )

    # ---- Fallback on failure ----
    if error and fallback:
        print(f"[fallback] Primary '{primary['name']}' failed ({error}), "
              f"trying '{fallback['name']}'...")
        resp, usage, duration, error2 = _attempt_forward(
            fallback, method, path, headers, body, is_streaming, settings
        )
        if error2:
            # Both failed — return the primary error
            return _build_error_response(error, resp)
        # Fallback succeeded — record usage under the fallback model
        record_usage_for_model(fallback, usage, duration, settings)
        return resp

    if error:
        return _build_error_response(error, resp)

    # Primary succeeded
    record_usage_for_model(primary, usage, duration, settings)
    return resp


def _attempt_forward(model: dict, method: str, path: str, headers: dict,
                     body: bytes, is_streaming: bool, settings: dict):
    """
    Try to forward a request to *model*.

    Returns (response, usage_dict, duration_seconds, error_str).
    Exactly one of (response, usage) or error will be meaningful.
    """
    t0 = time.time()

    try:
        if is_streaming:
            gen, container = _forward_streaming(model, method, path, headers, body)
            duration = time.time() - t0
            resp = Response(
                stream_with_context(gen()),
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            return resp, container[0], duration, None
        else:
            hx_resp, usage = _forward_non_streaming(model, method, path, headers, body)
            duration = time.time() - t0

            status = hx_resp.status_code
            # Convert httpx.Response → Flask Response
            resp_headers = dict(hx_resp.headers)
            resp_headers.pop("transfer-encoding", None)
            resp_headers.pop("content-encoding", None)
            flask_resp = Response(
                hx_resp.content,
                status=status,
                headers=resp_headers,
            )

            if _should_fallback(status):
                return None, usage, duration, f"HTTP {status}"
            return flask_resp, usage, duration, None

    except httpx.ConnectError as e:
        return None, None, time.time() - t0, f"Connection error: {e}"
    except httpx.ReadTimeout as e:
        return None, None, time.time() - t0, f"Timeout: {e}"
    except httpx.RemoteProtocolError as e:
        return None, None, time.time() - t0, f"Protocol error: {e}"
    except Exception as e:
        return None, None, time.time() - t0, f"Unexpected error: {e}"


def _should_fallback(status_code: int) -> bool:
    return status_code >= 500 or status_code == 429


def _build_error_response(error: str, resp):
    """Build a Flask error response when both primary and fallback fail."""
    if resp is not None:
        return Response(
            resp.content,
            status=resp.status_code,
            headers=dict(resp.headers),
        )
    return jsonify({"error": error}), 502


def record_usage_for_model(model: dict, usage: dict | None,
                           duration_seconds: float, settings: dict):
    """Persist usage data, if available."""
    if usage is None:
        return

    cost = calculate_cost(model, usage, duration_seconds, settings)
    record_usage(
        model_name=model["name"],
        model_display=model.get("display_name", model["name"]),
        model_type=model.get("type", "remote"),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=usage.get("cached_tokens", 0),
        cost=cost,
        duration_seconds=duration_seconds,
    )
