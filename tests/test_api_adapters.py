"""Wire-format contract tests; no external credentials or network required."""
import json
from datetime import date

import pytest

import proxy
from api_adapters import TranslationError, prepare_request
from models_config import save_model
from storage import get_usage_db


def make_model(name, fmt, tag=""):
    save_model({"name": name, "display_name": name, "provider": "test", "type": "remote",
                "tags": [tag] if tag else [], "base_url": f"https://{name}.test/v1", "api_key": "secret",
                "api_model_name": f"upstream-{name}", "api_format": fmt,
                "input_price_per_million": 1.0, "output_price_per_million": 2.0,
                "cached_price_per_million": 0.1, "enabled": True})


class FakeResponse:
    def __init__(self, status, data):
        self.status_code = status
        self.content = json.dumps(data).encode()
        self.headers = {"content-type": "application/json"}

    def json(self):
        return json.loads(self.content)

    @property
    def text(self):
        return self.content.decode()


class FakeStreamResponse:
    def __init__(self, status, events):
        self.status_code = status
        self.events = events
        self.headers = {"content-type": "text/event-stream"}

    def iter_lines(self):
        for event, data in self.events:
            yield f"event: {event}"
            yield f"data: {json.dumps(data)}"
            yield ""

    def read(self):
        return b"error"


class StreamContext:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, *args):
        return False


class FakeClient:
    def __init__(self, replies):
        self.replies = replies
        self.sent = []

    def request(self, method, url, headers, content, timeout):
        self.sent.append((url, headers, json.loads(content)))
        return self.replies.pop(0)

    def stream(self, method, url, headers, content, timeout):
        self.sent.append((url, headers, json.loads(content)))
        return StreamContext(self.replies.pop(0))


def responses_payload(output=None):
    return {"id": "resp_1", "status": "completed", "output": output or [
        {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}],
        "usage": {"input_tokens": 12, "output_tokens": 3,
                  "input_tokens_details": {"cached_tokens": 4}}}


def anthropic_payload(content=None):
    return {"id": "msg_1", "content": content or [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 8, "output_tokens": 3,
                                                   "cache_read_input_tokens": 4}}


@pytest.mark.parametrize("fmt,reply,path", [
    ("openai_responses", responses_payload(), "/v1/responses"),
    ("anthropic_messages", anthropic_payload(), "/v1/messages"),
])
def test_nonstreaming_adapter(client, monkeypatch, fmt, reply, path):
    make_model("native", fmt)
    fake = FakeClient([FakeResponse(200, reply)])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "messages": [
        {"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}]})
    assert result.status_code == 200
    assert fake.sent[0][0].endswith(path)
    assert fake.sent[0][2]["model"] == "upstream-native"
    assert fake.sent[0][2].get("stream_options") is None
    if fmt == "anthropic_messages":
        assert fake.sent[0][1]["x-api-key"] == "secret"
        assert fake.sent[0][1]["anthropic-version"] == "2023-06-01"
        assert "authorization" not in fake.sent[0][1]
        assert fake.sent[0][2]["system"] == "Be brief"
        assert fake.sent[0][2]["max_tokens"] == 4096
    else:
        assert fake.sent[0][2]["input"][0]["role"] == "system"
    payload = result.get_json()
    assert payload["choices"][0]["message"]["content"] == "Hello"
    assert payload["usage"]["prompt_tokens"] == 12
    record = get_record("native")
    assert record["input_tokens"] == 12
    assert record["cached_tokens"] == 4


def get_record(name):
    with get_usage_db() as db:
        return db.find_one({"_id": f"{date.today().isoformat()}:{name}"})


def test_responses_tool_round_trip(client, monkeypatch):
    make_model("native", "openai_responses")
    fake = FakeClient([FakeResponse(200, responses_payload([
        {"type": "function_call", "call_id": "call_1", "name": "weather",
         "arguments": '{"city":"Ottawa"}'}]))])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"id": "call_0", "type": "function",
          "function": {"name": "weather", "arguments": '{"city":"Toronto"}'}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "rainy"}],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]})
    assert result.status_code == 200
    assert fake.sent[0][2]["input"][-1] == {"type": "function_call_output", "call_id": "call_0", "output": "rainy"}
    assert fake.sent[0][2]["tools"][0]["name"] == "weather"
    assert result.get_json()["choices"][0]["finish_reason"] == "tool_calls"
    assert result.get_json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"


def test_anthropic_tool_round_trip(client, monkeypatch):
    make_model("native", "anthropic_messages")
    fake = FakeClient([FakeResponse(200, anthropic_payload([
        {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Ottawa"}}]))])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"id": "call_0", "type": "function",
          "function": {"name": "weather", "arguments": '{"city":"Toronto"}'}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "rainy"}],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]})
    assert result.status_code == 200
    assert fake.sent[0][2]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert fake.sent[0][2]["tools"][0]["input_schema"] == {"type": "object"}
    assert result.get_json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"


def test_mixed_format_fallback(client, monkeypatch):
    make_model("primary", "openai_responses", "fast")
    make_model("backup", "anthropic_messages", "smart")
    fake = FakeClient([FakeResponse(503, {"error": "unavailable"}), FakeResponse(200, anthropic_payload())])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "fast", "messages": [
        {"role": "user", "content": "Hi"}]})
    assert result.status_code == 200
    assert fake.sent[0][0].endswith("/v1/responses")
    assert fake.sent[1][0].endswith("/v1/messages")
    assert fake.sent[1][2]["model"] == "upstream-backup"
    assert result.get_json()["model"] == "backup"
    assert get_record("backup")["requests"] == 1


@pytest.mark.parametrize("fmt,events", [
    ("openai_responses", [
        ("response.created", {"response": {"id": "resp_stream"}}),
        ("response.output_text.delta", {"delta": "Hi"}),
        ("response.completed", {"response": responses_payload()})]),
    ("anthropic_messages", [
        ("message_start", {"message": {"id": "msg_stream", "usage": {"input_tokens": 8,
                                                                      "cache_read_input_tokens": 4}}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
        ("message_stop", {})]),
])
def test_streaming_adapter_tracks_usage(client, monkeypatch, fmt, events):
    make_model("native", fmt)
    fake = FakeClient([FakeStreamResponse(200, events)])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "stream": True,
        "messages": [{"role": "user", "content": "Hi"}]})
    assert result.status_code == 200
    chunks = [json.loads(line[6:]) for line in result.get_data(as_text=True).splitlines() if line.startswith("data: {")]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(chunk["choices"] and chunk["choices"][0]["delta"].get("content") == "Hi" for chunk in chunks)
    assert chunks[-1]["usage"]["prompt_tokens"] == 12
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert get_record("native")["input_tokens"] == 12
    assert "stream_options" not in fake.sent[0][2]


def test_invalid_native_option_rejected_before_upstream(client, monkeypatch):
    make_model("native", "anthropic_messages")
    fake = FakeClient([])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "n": 2,
        "messages": [{"role": "user", "content": "Hi"}]})
    assert result.status_code == 400
    assert fake.sent == []


def test_legacy_model_defaults_to_chat(client, monkeypatch):
    make_model("legacy", "chat_completions")
    fake = FakeClient([FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "legacy", "messages": [{"role": "user", "content": "Hi"}]})
    assert result.status_code == 200
    assert fake.sent[0][0].endswith("/v1/chat/completions")


def test_model_form_persists_format_and_clone(client):
    form = {"name": "native", "base_url": "https://native.test/v1",
            "api_model_name": "native-id", "type": "remote", "api_format": "anthropic_messages"}
    result = client.post("/models/new", data=form)
    assert result.status_code == 302
    from models_config import get_model
    assert get_model("native")["api_format"] == "anthropic_messages"
    assert 'value="anthropic_messages" selected' in client.get("/models/native/clone").get_data(as_text=True)
    form["api_format"] = "openai_responses"
    client.post("/models/native/edit", data=form)
    assert get_model("native")["api_format"] == "openai_responses"
    form["api_format"] = "bad"
    assert b"Invalid upstream API format" in client.post("/models/native/edit", data=form).data


def test_unsupported_options_are_rejected(client, monkeypatch):
    make_model("native", "openai_responses")
    fake = FakeClient([])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "messages": [
        {"role": "user", "content": "Hi"}], "logit_bias": {"1": 5}})
    assert result.status_code == 400
    assert "logit_bias" in result.get_json()["error"]
    assert not fake.sent


def test_streaming_tool_deltas(client, monkeypatch):
    make_model("native", "openai_responses")
    events = [("response.created", {"response": {"id": "resp_2"}}),
              ("response.output_item.added", {"output_index": 1, "item": {
                  "type": "function_call", "call_id": "call_2", "name": "weather"}}),
              ("response.function_call_arguments.delta", {"output_index": 1, "delta": '{"city":'}),
              ("response.function_call_arguments.delta", {"output_index": 1, "delta": '"Ottawa"}'}),
              ("response.completed", {"response": responses_payload()})]
    fake = FakeClient([FakeStreamResponse(200, events)])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "stream": True,
        "messages": [{"role": "user", "content": "weather?"}]})
    chunks = [json.loads(line[6:]) for line in result.get_data(as_text=True).splitlines() if line.startswith("data: {")]
    calls = [chunk["choices"][0]["delta"]["tool_calls"][0]
             for chunk in chunks if chunk["choices"] and chunk["choices"][0]["delta"].get("tool_calls")]
    assert calls[0]["id"] == "call_2"
    assert ''.join(c["function"]["arguments"] for c in calls) == '{"city":"Ottawa"}'
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"


def test_truncated_native_stream_does_not_claim_success(client, monkeypatch):
    make_model("native", "anthropic_messages")
    fake = FakeClient([FakeStreamResponse(200, [("message_start", {
        "message": {"id": "msg_truncated", "usage": {"input_tokens": 1}}})])])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "native", "stream": True,
        "messages": [{"role": "user", "content": "Hi"}]})
    text = result.get_data(as_text=True)
    assert "Upstream stream ended without a completion event" in text
    assert "data: [DONE]" not in text
