"""
Tests for streaming proxy behaviour and concurrent usage recording.

These cover the fixes for:
  #1 streaming usage is recorded (after the stream completes)
  #2 streaming upstream errors are surfaced, not masked as a 200 stream
  #3 fallback works for streaming requests
  #4 concurrent usage upserts don't lose increments
"""

import json
import threading
from datetime import date

import proxy
from models_config import save_model
from storage import get_usage_db


# ---------------------------------------------------------------------------
# Fake httpx client (routes by host substring in the URL)
# ---------------------------------------------------------------------------

class _FakeStreamResp:
    def __init__(self, status_code, lines, headers=None):
        self.status_code = status_code
        self._lines = lines
        self.headers = headers or {"content-type": "text/event-stream"}
        self._content = ("\n".join(lines)).encode("utf-8")

    def iter_lines(self):
        for ln in self._lines:
            yield ln

    def read(self):
        return self._content


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *a):
        return False


def _make_fake_client(routes, record=None):
    """routes: {host_substring: (status_code, [sse_line, ...])}

    If *record* is a list, each request's parsed JSON body is appended to it.
    """
    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def stream(self, method, url, headers=None, content=None, timeout=None):
            if record is not None:
                record.append(json.loads(content))
            for host, (status, lines) in routes.items():
                if host in url:
                    return _FakeStreamCtx(_FakeStreamResp(status, lines))
            raise AssertionError(f"no fake route for {url}")

        def close(self):
            pass

    return _FakeClient


def _save(name, tag, base_url):
    save_model({
        "name": name, "display_name": name, "provider": "test",
        "type": "remote", "tags": [tag] if tag else [],
        "base_url": base_url, "api_key": "k", "api_model_name": name,
        "input_price_per_million": 1.0, "output_price_per_million": 2.0,
        "cached_price_per_million": 0.0, "enabled": True,
    })


def _usage(name):
    with get_usage_db() as db:
        return db.find_one({"_id": f"{date.today().isoformat()}:{name}"})


# ---------------------------------------------------------------------------
# #1 — streaming usage is recorded
# ---------------------------------------------------------------------------

def test_streaming_records_usage(client, monkeypatch):
    _save("smart-model", "smart", "https://smart.test")
    lines = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        'data: {"usage":{"prompt_tokens":10,"completion_tokens":20}}',
        'data: [DONE]',
        '',
    ]
    monkeypatch.setattr(proxy.httpx, "Client",
                        _make_fake_client({"smart.test": (200, lines)}))

    resp = client.post("/v1/chat/completions",
                       json={"model": "smart-model", "stream": True})
    assert resp.status_code == 200
    assert "hi" in resp.get_data(as_text=True)

    rec = _usage("smart-model")
    assert rec is not None, "streaming usage was not recorded"
    assert rec["input_tokens"] == 10
    assert rec["output_tokens"] == 20
    assert rec["requests"] == 1


# ---------------------------------------------------------------------------
# #2 — streaming upstream errors are surfaced, not masked as a 200
# ---------------------------------------------------------------------------

def test_streaming_error_surfaced(client, monkeypatch):
    _save("solo-model", None, "https://solo.test")  # no tag → no fallback
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_client({
        "solo.test": (400, ['{"error":{"message":"invalid model"}}']),
    }))

    resp = client.post("/v1/chat/completions",
                       json={"model": "solo-model", "stream": True})
    assert resp.status_code == 400
    assert "invalid model" in resp.get_data(as_text=True)
    assert _usage("solo-model") is None  # errors don't record usage


# ---------------------------------------------------------------------------
# #3 — fallback works for streaming requests
# ---------------------------------------------------------------------------

def test_streaming_fallback(client, monkeypatch):
    _save("fast-model", "fast", "https://fast.test")
    _save("smart-model", "smart", "https://smart.test")
    smart_lines = [
        'data: {"choices":[{"delta":{"content":"fallback-ok"}}]}',
        'data: {"usage":{"prompt_tokens":5,"completion_tokens":7}}',
        'data: [DONE]',
        '',
    ]
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_client({
        "fast.test": (500, ['data: {"error":"boom"}', '']),
        "smart.test": (200, smart_lines),
    }))

    resp = client.post("/v1/chat/completions",
                       json={"model": "fast-model", "stream": True})
    assert resp.status_code == 200
    assert "fallback-ok" in resp.get_data(as_text=True)

    # Usage recorded under the fallback model, not the failed primary.
    assert _usage("smart-model") is not None
    assert _usage("fast-model") is None


# ---------------------------------------------------------------------------
# #5 — a fallback-eligible status is passed through when there's no fallback
# ---------------------------------------------------------------------------

def test_retryable_status_passed_through_without_fallback(client, monkeypatch):
    """A 429 with no fallback configured must reach the client as a 429.

    It used to be swallowed and re-emitted as a synthetic 502, which breaks
    the backoff logic in the OpenAI SDK and most agent clients.
    """
    _save("solo-model", None, "https://solo.test")  # no tag → no fallback
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_client({
        "solo.test": (429, ['{"error":{"message":"rate limited"}}']),
    }))

    resp = client.post("/v1/chat/completions",
                       json={"model": "solo-model", "stream": True})
    assert resp.status_code == 429
    assert "rate limited" in resp.get_data(as_text=True)
    assert _usage("solo-model") is None


def test_both_models_failing_surfaces_upstream_status(client, monkeypatch):
    """When primary and fallback both fail, keep the real upstream status."""
    _save("fast-model", "fast", "https://fast.test")
    _save("smart-model", "smart", "https://smart.test")
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_client({
        "fast.test": (500, ['{"error":"primary boom"}']),
        "smart.test": (503, ['{"error":"fallback boom"}']),
    }))

    resp = client.post("/v1/chat/completions",
                       json={"model": "fast-model", "stream": True})
    assert resp.status_code == 503
    assert "fallback boom" in resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# #6 — stream_options is injected so streaming usage is actually reported
# ---------------------------------------------------------------------------

def test_stream_options_injected(client, monkeypatch):
    _save("smart-model", "smart", "https://smart.test")
    sent = []
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_client(
        {"smart.test": (200, ['data: [DONE]', ''])}, record=sent))

    client.post("/v1/chat/completions",
                json={"model": "smart-model", "stream": True})

    assert sent[0]["stream_options"] == {"include_usage": True}


def test_stream_options_not_injected_for_non_streaming(client, monkeypatch):
    """Only streaming requests get the option — it's meaningless otherwise."""
    _save("smart-model", "smart", "https://smart.test")
    sent = []
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_client(
        {"smart.test": (200, ['data: [DONE]', ''])}, record=sent))

    client.post("/v1/chat/completions",
                json={"model": "smart-model", "stream": True,
                      "stream_options": {"include_usage": False}})

    # A client that set it explicitly keeps its own value.
    assert sent[0]["stream_options"] == {"include_usage": False}


def test_stream_options_respects_setting(client, monkeypatch):
    from storage import save_settings

    save_settings({"stream_include_usage": False})
    _save("smart-model", "smart", "https://smart.test")
    sent = []
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_client(
        {"smart.test": (200, ['data: [DONE]', ''])}, record=sent))

    client.post("/v1/chat/completions",
                json={"model": "smart-model", "stream": True})

    assert "stream_options" not in sent[0]


def test_stream_options_retry_on_rejection(client, monkeypatch):
    """A backend that rejects stream_options gets one retry without it."""
    _save("solo-model", None, "https://solo.test")
    sent = []
    responses = [
        (400, ['{"error":{"message":"unknown parameter: stream_options"}}']),
        (200, ['data: {"choices":[{"delta":{"content":"ok"}}]}',
               'data: [DONE]', '']),
    ]

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def stream(self, method, url, headers=None, content=None, timeout=None):
            sent.append(json.loads(content))
            status, lines = responses[min(len(sent) - 1, len(responses) - 1)]
            return _FakeStreamCtx(_FakeStreamResp(status, lines))

        def close(self):
            pass

    monkeypatch.setattr(proxy.httpx, "Client", _FakeClient)

    resp = client.post("/v1/chat/completions",
                       json={"model": "solo-model", "stream": True})

    assert len(sent) == 2, "expected exactly one retry"
    assert "stream_options" in sent[0]
    assert "stream_options" not in sent[1]
    assert resp.status_code == 200
    assert "ok" in resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# #7 — same guarantees on the non-streaming path
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8")
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return json.loads(self.content)


def _make_fake_sync_client(routes, record=None):
    """routes: {host_substring: (status_code, payload_dict)}"""
    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def request(self, method, url, headers=None, content=None, timeout=None):
            if record is not None:
                record.append(json.loads(content))
            for host, (status, payload) in routes.items():
                if host in url:
                    return _FakeResp(status, payload)
            raise AssertionError(f"no fake route for {url}")

        def close(self):
            pass

    return _FakeClient


def test_non_streaming_records_usage(client, monkeypatch):
    _save("solo-model", None, "https://solo.test")
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_sync_client({
        "solo.test": (200, {"choices": [], "usage": {"prompt_tokens": 3,
                                                     "completion_tokens": 4}}),
    }))

    resp = client.post("/v1/chat/completions", json={"model": "solo-model"})
    assert resp.status_code == 200

    rec = _usage("solo-model")
    assert rec["input_tokens"] == 3
    assert rec["output_tokens"] == 4


def test_non_streaming_429_passed_through(client, monkeypatch):
    _save("solo-model", None, "https://solo.test")
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_sync_client({
        "solo.test": (429, {"error": {"message": "slow down"}}),
    }))

    resp = client.post("/v1/chat/completions", json={"model": "solo-model"})
    assert resp.status_code == 429
    assert "slow down" in resp.get_data(as_text=True)


def test_non_streaming_fallback_still_works(client, monkeypatch):
    _save("fast-model", "fast", "https://fast.test")
    _save("smart-model", "smart", "https://smart.test")
    monkeypatch.setattr(proxy.httpx, "Client", _make_fake_sync_client({
        "fast.test": (500, {"error": "boom"}),
        "smart.test": (200, {"choices": [], "usage": {"prompt_tokens": 1,
                                                      "completion_tokens": 2}}),
    }))

    resp = client.post("/v1/chat/completions", json={"model": "fast-model"})
    assert resp.status_code == 200
    assert _usage("smart-model") is not None
    assert _usage("fast-model") is None


def test_connection_error_still_502s(client, monkeypatch):
    """No upstream response at all → synthetic 502 (unchanged behaviour)."""
    _save("solo-model", None, "https://solo.test")

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def request(self, *a, **k):
            raise proxy.httpx.ConnectError("refused")

        def close(self):
            pass

    monkeypatch.setattr(proxy.httpx, "Client", _FakeClient)

    resp = client.post("/v1/chat/completions", json={"model": "solo-model"})
    assert resp.status_code == 502
    assert "Connection error" in resp.get_data(as_text=True)


def test_http_client_is_reused(monkeypatch):
    """The pooled client must be built once, not per request."""
    built = []

    class _FakeClient:
        def __init__(self, *a, **k):
            built.append(1)

        def close(self):
            pass

    monkeypatch.setattr(proxy.httpx, "Client", _FakeClient)
    first = proxy.get_http_client()
    second = proxy.get_http_client()

    assert first is second
    assert len(built) == 1


# ---------------------------------------------------------------------------
# #4 — concurrent usage upserts don't lose increments
# ---------------------------------------------------------------------------

def test_concurrent_usage_no_lost_updates():
    """The pooled lock must hold for the whole read-modify-write block, so
    concurrent upserts of the same record don't lose increments.

    A sleep widens the read->write window to make the race deterministic —
    without serialization this loses the vast majority of increments (an
    unlocked collection yields ~2 instead of n)."""
    import time

    rid = f"{date.today().isoformat()}:cmodel"
    n = 30

    def worker():
        with get_usage_db() as db:
            existing = db.find_one({"_id": rid})
            time.sleep(0.002)  # widen the read->write window
            if existing:
                db.update_one({"_id": rid}, inc={"input_tokens": 1, "requests": 1})
            else:
                db.insert({"_id": rid, "date": date.today().isoformat(),
                           "model_name": "cmodel", "input_tokens": 1, "requests": 1})

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rec = _usage("cmodel")
    assert rec["requests"] == n
    assert rec["input_tokens"] == n


def test_record_usage_accumulates():
    """End-to-end smoke test of the record_usage upsert path."""
    from usage_tracker import record_usage

    for _ in range(5):
        record_usage("smodel", "S", "remote", 1, 2, 0, 0.001, 0.0)

    rec = _usage("smodel")
    assert rec["requests"] == 5
    assert rec["input_tokens"] == 5
    assert rec["output_tokens"] == 10
