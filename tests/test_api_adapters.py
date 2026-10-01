"""Wire-format contract tests; no external credentials or network required."""
import json
from datetime import date

import pytest

import proxy
from api_adapters import TranslationError, prepare_request, normalize_response, ENVELOPE_MARKER
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


def test_responses_reasoning_effort_translation():
    model = {"api_format": "openai_responses", "api_model_name": "GPT-6-astra"}
    path, body = prepare_request(model, "/v1/chat/completions", json.dumps({
        "model": "gpt-6-astra", "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low"}).encode())
    assert path == "/v1/responses"
    assert json.loads(body)["reasoning"] == {"effort": "low"}
    with pytest.raises(TranslationError, match="conflicts"):
        prepare_request(model, "/v1/chat/completions", json.dumps({
            "model": "gpt-6-astra", "messages": [], "reasoning_effort": "low",
            "reasoning": {"effort": "high"}}).encode())
    with pytest.raises(TranslationError, match="does not support reasoning"):
        prepare_request({**model, "api_format": "anthropic_messages"}, "/v1/chat/completions",
                        json.dumps({"model": "x", "messages": [], "reasoning_effort": "low"}).encode())


def test_responses_retries_only_explicitly_unsupported_sampling(client, monkeypatch):
    make_model("astra", "openai_responses")
    bad = lambda field: FakeResponse(400, {"error": {
        "type": "invalid_request_error", "param": field,
        "message": f"Unsupported parameter: '{field}' is not supported with this model."}})
    fake = FakeClient([bad("temperature"), bad("top_p"), FakeResponse(200, responses_payload())])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "astra", "temperature": 0.4,
        "top_p": 0.9, "reasoning_effort": "low", "messages": [{"role": "user", "content": "Hi"}]})
    assert result.status_code == 200
    assert ["temperature" in x[2] for x in fake.sent] == [True, False, False]
    assert ["top_p" in x[2] for x in fake.sent] == [True, True, False]
    assert all(x[2]["reasoning"] == {"effort": "low"} for x in fake.sent)
    assert get_record("astra")["requests"] == 1


def test_responses_preserves_unrelated_400(client, monkeypatch):
    make_model("astra", "openai_responses")
    fake = FakeClient([FakeResponse(400, {"error": {
        "type": "invalid_request_error", "param": "temperature",
        "message": "Invalid sampling request"}})])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "astra", "temperature": 0.4,
        "messages": [{"role": "user", "content": "Hi"}]})
    assert result.status_code == 400
    assert len(fake.sent) == 1


def test_responses_stream_retries_unsupported_sampling(client, monkeypatch):
    make_model("astra", "openai_responses")
    bad = FakeStreamResponse(400, [])
    bad.read = lambda: json.dumps({"error": {"type": "invalid_request_error", "param": "temperature",
        "message": "Unsupported parameter: 'temperature' is not supported with this model."}}).encode()
    fake = FakeClient([bad, FakeStreamResponse(200, [
        ("response.created", {"response": {"id": "resp_stream"}}),
        ("response.output_text.delta", {"delta": "Hi"}),
        ("response.completed", {"response": responses_payload()})])])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "astra", "stream": True,
        "temperature": 0.4, "messages": [{"role": "user", "content": "Hi"}]})
    assert result.status_code == 200
    assert "data: [DONE]" in result.get_data(as_text=True)
    assert "temperature" in fake.sent[0][2]
    assert "temperature" not in fake.sent[1][2]


def test_responses_reasoning_item_does_not_break_text_or_usage(client, monkeypatch):
    make_model("astra", "openai_responses")
    payload = responses_payload([
        {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"},
        {"type": "message", "content": [{"type": "output_text", "text": "221"}]},
    ])
    fake = FakeClient([FakeResponse(200, payload)])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model": "astra",
        "reasoning_effort": "low", "messages": [{"role": "user", "content": "13*17?"}]})
    assert result.status_code == 200
    assert result.get_json()["choices"][0]["message"]["content"] == "221"
    assert result.get_json()["usage"]["completion_tokens"] == 3
    assert get_record("astra")["requests"] == 1


def test_responses_reasoning_item_before_tool_calls_round_trip(client, monkeypatch):
    make_model("astra", "openai_responses")
    first = responses_payload([
        {"id": "rs_1", "type": "reasoning", "summary": []},
        {"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": '{"city":"Ottawa"}'},
    ])
    second = responses_payload([
        {"id": "rs_2", "type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "Sunny"}]},
    ])
    fake = FakeClient([FakeResponse(200, first), FakeResponse(200, second)])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    first_resp = client.post("/v1/chat/completions", json={"model": "astra",
        "reasoning_effort": "low", "messages": [{"role": "user", "content": "Weather?"}], "tools": tools})
    assert first_resp.status_code == 200
    choice = first_resp.get_json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call_1"
    next_resp = client.post("/v1/chat/completions", json={"model": "astra", "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "Weather?"}, choice["message"],
                     {"role": "tool", "tool_call_id": call["id"], "content": "Sunny"}], "tools": tools})
    assert next_resp.status_code == 200
    assert next_resp.get_json()["choices"][0]["message"]["content"] == "Sunny"
    assert fake.sent[1][2]["input"][-1]["call_id"] == "call_1"
    assert get_record("astra")["requests"] == 2


def test_responses_stream_ignores_reasoning_deltas_and_emits_text(client, monkeypatch):
    make_model("astra", "openai_responses")
    fake = FakeClient([FakeStreamResponse(200, [
        ("response.created", {"response": {"id": "resp_2"}}),
        ("response.output_item.added", {"output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}),
        ("response.reasoning_summary_text.delta", {"output_index": 0, "delta": "private thinking"}),
        ("response.output_text.delta", {"delta": "221"}),
        ("response.completed", {"response": responses_payload()}),
    ])])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    resp = client.post("/v1/chat/completions", json={"model": "astra", "reasoning_effort": "low",
        "stream": True, "messages": [{"role": "user", "content": "13*17?"}]})
    assert resp.status_code == 200
    data = resp.get_data(as_text=True)
    assert "private thinking" not in data
    assert '"content": "221"' in data
    assert "data: [DONE]" in data


def test_responses_encrypted_reasoning_exact_replay_and_unmodified_client(client, monkeypatch):
    make_model("astra", "openai_responses")
    native_items = [
        {"id":"rs_1", "type":"reasoning", "summary":[], "encrypted_content":"opaque-blob"},
        {"id":"msg_1", "type":"message", "role":"assistant", "status":"completed",
         "phase":"commentary", "content":[{"type":"output_text", "text":"Checking"}]},
        {"id":"fc_1", "type":"function_call", "call_id":"call_1", "name":"calc",
         "arguments":'{"a":2,"b":3}', "status":"completed"},
    ]
    first = responses_payload(native_items)
    fake = FakeClient([FakeResponse(200, first), FakeResponse(200, responses_payload()),
                       FakeResponse(200, responses_payload())])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    user = {"role":"user", "content":"Calculate"}
    initial = client.post("/v1/chat/completions", json={"model":"astra", "messages":[user]})
    assert initial.status_code == 200
    assert fake.sent[0][2]["store"] is False
    msg = initial.get_json()["choices"][0]["message"]
    assert msg["reasoning_details"]["format"] == ENVELOPE_MARKER
    assert msg["reasoning_details"]["blocks"] == native_items
    tool = {"role":"tool", "tool_call_id":"call_1", "content":"6"}
    assert client.post("/v1/chat/completions", json={"model":"astra", "messages":[user,msg,tool]}).status_code == 200
    assert fake.sent[1][2]["input"][1:4] == native_items
    assert fake.sent[1][2]["input"][-1] == {"type":"function_call_output", "call_id":"call_1", "output":"6"}
    legacy = {k:v for k,v in msg.items() if k != "reasoning_details"}
    assert client.post("/v1/chat/completions", json={"model":"astra", "messages":[user,legacy,tool]}).status_code == 200
    assert fake.sent[2][2]["input"][1] == {"role":"assistant", "content":"Checking"}
    assert not any(item.get("type") == "reasoning" for item in fake.sent[2][2]["input"])


def test_anthropic_signed_blocks_replayed_in_order(client, monkeypatch):
    make_model("claude", "anthropic_messages")
    native_blocks = [{"type":"thinking", "thinking":"summary", "signature":"signed-opaque"},
        {"type":"redacted_thinking", "data":"redacted-opaque"},
        {"type":"text", "text":"Calling tool"},
        {"type":"tool_use", "id":"tool_1", "name":"calc", "input":{"a":2,"b":3}}]
    fake = FakeClient([FakeResponse(200, anthropic_payload(native_blocks)),
                       FakeResponse(200, anthropic_payload())])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    first = client.post("/v1/chat/completions", json={"model":"claude",
        "messages":[{"role":"user","content":"calculate"}]})
    assert first.status_code == 200
    msg = first.get_json()["choices"][0]["message"]
    assert msg["reasoning_details"]["blocks"] == native_blocks
    assert client.post("/v1/chat/completions", json={"model":"claude", "messages":[
        {"role":"user","content":"calculate"},msg,
        {"role":"tool","tool_call_id":"tool_1","content":"6"}]}).status_code == 200
    assert fake.sent[1][2]["messages"][1]["content"] == native_blocks


def test_cross_model_envelope_not_replayed(client, monkeypatch):
    make_model("a", "openai_responses")
    make_model("b", "openai_responses")
    opaque = {"type":"reasoning", "encrypted_content":"opaque"}
    fake = FakeClient([FakeResponse(200, responses_payload([opaque,
        {"type":"message", "content":[{"type":"output_text", "text":"hi"}]}]))])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    envelope = {"format":ENVELOPE_MARKER, "provider":"openai_responses",
                "model":"upstream-a", "proxy_model":"a", "blocks":[opaque]}
    res = client.post("/v1/chat/completions", json={"model":"b", "messages":[
        {"role":"assistant", "content":"Hi", "reasoning_details":envelope},
        {"role":"user", "content":"Continue"}]})
    assert res.status_code == 200
    assert not any(item.get("type") == "reasoning" for item in fake.sent[0][2]["input"])


def test_responses_stream_replays_completed_encrypted_output(client, monkeypatch):
    make_model("astra", "openai_responses")
    native_items = [{"id":"rs_7", "type":"reasoning", "summary":[], "encrypted_content":"secret"},
        {"id":"fc_7", "type":"function_call", "call_id":"call_7", "name":"calc",
         "arguments":'{"x":1}'}]
    completed = responses_payload(native_items)
    fake = FakeClient([FakeStreamResponse(200, [
        ("response.created", {"response":{"id":"resp_7"}}),
        ("response.output_item.added", {"output_index":0, "item":{"id":"rs_7", "type":"reasoning"}}),
        ("response.output_item.added", {"output_index":1, "item":{"type":"function_call", "call_id":"call_7", "name":"calc"}}),
        ("response.function_call_arguments.delta", {"output_index":1, "delta":'{"x":1}'}),
        ("response.completed", {"response":completed}),
    ]), FakeResponse(200, responses_payload())])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    response = client.post("/v1/chat/completions", json={"model":"astra", "stream":True,
        "messages":[{"role":"user","content":"calc"}]})
    chunks = [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines()
              if line.startswith("data: {")]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    details = [d["reasoning_details"] for d in deltas if "reasoning_details" in d]
    assert len(details) == 1
    assert details[0]["blocks"] == native_items
    msg = {"role":"assistant", "content":"", "tool_calls":[{"id":"call_7", "type":"function",
           "function":{"name":"calc", "arguments":'{"x":1}'}}], "reasoning_details":details[0]}
    result = client.post("/v1/chat/completions", json={"model":"astra", "messages":[
        {"role":"user","content":"calc"}, msg,
        {"role":"tool","tool_call_id":"call_7","content":"1"}]})
    assert result.status_code == 200
    assert fake.sent[1][2]["input"][1:3] == native_items


def test_anthropic_stream_preserves_signed_thinking_and_tool(client, monkeypatch):
    make_model("claude", "anthropic_messages")
    events = [("message_start", {"message":{"id":"msg_7", "usage":{"input_tokens":1}}}),
        ("content_block_start", {"index":0, "content_block":{"type":"thinking", "thinking":""}}),
        ("content_block_delta", {"index":0, "delta":{"type":"thinking_delta", "thinking":"private"}}),
        ("content_block_delta", {"index":0, "delta":{"type":"signature_delta", "signature":"sig"}}),
        ("content_block_stop", {"index":0}),
        ("content_block_start", {"index":1, "content_block":{"type":"tool_use", "id":"tool_7", "name":"calc", "input":{}}}),
        ("content_block_delta", {"index":1, "delta":{"type":"input_json_delta", "partial_json":'{"x":1}'}}),
        ("content_block_stop", {"index":1}),
        ("message_delta", {"delta":{"stop_reason":"tool_use"}, "usage":{"output_tokens":2}}),
        ("message_stop", {})]
    fake = FakeClient([FakeStreamResponse(200, events), FakeResponse(200, anthropic_payload())])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model":"claude", "stream":True,
        "messages":[{"role":"user","content":"calc"}]})
    chunks = [json.loads(line[6:]) for line in result.get_data(as_text=True).splitlines()
              if line.startswith("data: {")]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    details = [d["reasoning_details"] for d in deltas if "reasoning_details" in d]
    assert len(details) == 1
    assert details[0]["blocks"][0] == {"type":"thinking", "thinking":"private", "signature":"sig"}
    assert details[0]["blocks"][1]["input"] == {"x":1}
    assert client.post("/v1/chat/completions", json={"model":"claude", "messages":[
        {"role":"user","content":"calc"},
        {"role":"assistant","tool_calls":[{"id":"tool_7","type":"function",
            "function":{"name":"calc","arguments":'{"x":1}'}}],"reasoning_details":details[0]},
        {"role":"tool","tool_call_id":"tool_7","content":"1"}]}).status_code == 200
    assert fake.sent[1][2]["messages"][1]["content"] == details[0]["blocks"]


def test_chat_backend_preserves_its_reasoning_fields_unchanged(client, monkeypatch):
    make_model("chat", "chat_completions")
    original = {"reasoning_details": [{"type":"custom", "opaque":"keep"}],
                "reasoning_content":"backend thinking", "content":"answer", "role":"assistant"}
    fake = FakeClient([FakeResponse(200, {"choices":[{"message":original}]})])
    monkeypatch.setattr(proxy, "get_http_client", lambda: fake)
    result = client.post("/v1/chat/completions", json={"model":"chat", "messages":[
        {"role":"user", "content":"hi"},original,{"role":"user", "content":"again"}]})
    assert result.status_code == 200
    assert result.get_json()["choices"][0]["message"] == original
    assert fake.sent[0][2]["messages"][1] == original
