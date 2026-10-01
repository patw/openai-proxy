"""Translate OpenAI chat completions to native Responses / Anthropic Messages.

The public API remains /v1/chat/completions. Fail explicitly for unsupported
features rather than silently discarding context or fabricating tool results.
"""
import json
import time
import uuid


class TranslationError(ValueError):
    pass


def api_format(model):
    return model.get("api_format") or "chat_completions"


def upstream_path(model, incoming_path):
    fmt = api_format(model)
    if incoming_path != "/v1/chat/completions" or fmt == "chat_completions":
        return incoming_path
    if fmt == "openai_responses":
        return "/v1/responses"
    if fmt == "anthropic_messages":
        return "/v1/messages"
    raise TranslationError(f"Unsupported upstream API format: {fmt}")


def _text_parts(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if part.get("type") != "text":
                raise TranslationError("Only text content parts are supported by this upstream format")
            parts.append(part.get("text", ""))
        return "\n".join(parts)
    raise TranslationError("Unsupported message content")


def _json_args(raw):
    try:
        return json.loads(raw or "{}")
    except (ValueError, TypeError):
        raise TranslationError("Tool arguments must be valid JSON")


# A private chat-message extension: no client must understand it to make a
# request, but clients that preserve it can replay native provider state.
# Keep provider/model identity on the envelope to prevent cross-provider replay.
ENVELOPE_MARKER = "openai-proxy/reasoning-v1"


def _replay_blocks(msg, model, provider):
    details = msg.get("reasoning_details")
    if not isinstance(details, dict) or details.get("format") != ENVELOPE_MARKER:
        return None  # Other chat providers own their own reasoning_details.
    if (details.get("provider") != provider or details.get("model") != model["api_model_name"]
            or details.get("proxy_model") != model["name"]):
        return None  # A different model/fallback must not receive opaque state.
    blocks = details.get("blocks")
    if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
        raise TranslationError("Invalid reasoning_details blocks")
    valid_types = {"openai_responses": {"reasoning", "message", "function_call"},
                   "anthropic_messages": {"thinking", "redacted_thinking", "text", "tool_use"}}
    if not all(b.get("type") in valid_types[provider] for b in blocks):
        raise TranslationError("Invalid reasoning_details block type")
    if provider == "anthropic_messages":
        calls = {c.get("id") for c in msg.get("tool_calls", [])}
        replayed_calls = {b.get("id") for b in blocks if b["type"] == "tool_use"}
        if calls != replayed_calls:
            raise TranslationError("reasoning_details tool calls do not match assistant message")
        if not all(b.get("signature") for b in blocks if b["type"] == "thinking"):
            raise TranslationError("reasoning_details thinking block missing signature")
    if provider == "openai_responses":
        # Only replayable encrypted reasoning may be used with store:false.
        # Unencrypted reasoning IDs refer to server-side state unavailable here.
        blocks = [b for b in blocks if b.get("type") != "reasoning" or b.get("encrypted_content")]
        calls = {c.get("id") for c in msg.get("tool_calls", [])}
        replayed_calls = {b.get("call_id") for b in blocks if b["type"] == "function_call"}
        if calls != replayed_calls:
            raise TranslationError("reasoning_details function calls do not match assistant message")
    return blocks


def _envelope(model, provider, blocks):
    return {"format": ENVELOPE_MARKER, "provider": provider,
            "model": model["api_model_name"], "proxy_model": model["name"],
            "blocks": blocks}


_COMMON_CHAT_KEYS = {"model", "messages", "stream", "max_tokens", "max_completion_tokens",
                     "temperature", "top_p", "tools", "tool_choice", "n", "response_format",
                     "stream_options", "reasoning_effort", "reasoning"}


def _validate_options(data, fmt):
    for key in data:
        if key not in _COMMON_CHAT_KEYS:
            raise TranslationError(f"{fmt} does not support chat option: {key}")
    if data.get("n", 1) != 1:
        raise TranslationError("Only n=1 is supported by native upstream formats")
    if "stream_options" in data and data["stream_options"] not in ({"include_usage": True}, {}):
        raise TranslationError("Only stream_options.include_usage is supported")


def _common_fields(data, target):
    for key, dest in (("temperature", "temperature"), ("top_p", "top_p"),
                      ("max_completion_tokens", "max_output_tokens"),
                      ("max_tokens", "max_output_tokens")):
        if key in data:
            target[dest] = data[key]
    if data.get("stream"):
        target["stream"] = True


def _responses_request(data, model):
    result = {"model": data["model"], "input": [], "store": False}
    _common_fields(data, result)
    effort = data.get("reasoning_effort")
    reasoning = data.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict) or set(reasoning) != {"effort"}:
            raise TranslationError("Only reasoning.effort is supported")
        if effort is not None and reasoning["effort"] != effort:
            raise TranslationError("reasoning.effort conflicts with reasoning_effort")
        effort = reasoning["effort"]
    if effort is not None:
        if effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            raise TranslationError("Unsupported reasoning effort")
        result["reasoning"] = {"effort": effort}
    for msg in data.get("messages", []):
        role = msg.get("role")
        if role in ("system", "developer", "user", "assistant"):
            replay = _replay_blocks(msg, model, "openai_responses") if role == "assistant" else None
            if replay is not None:
                # Reuse the complete ordered output of the original response:
                # reasoning + message + function_call items. Regenerating any
                # of these from chat text would lose phase/encrypted state.
                result["input"].extend(replay)
                continue
            if msg.get("content") is not None:
                result["input"].append({"role": role, "content": _text_parts(msg["content"])})
            elif role != "assistant" or not msg.get("tool_calls"):
                raise TranslationError("Message content is required")
            if role == "assistant":
                for call in msg.get("tool_calls", []):
                    if call.get("type") != "function":
                        raise TranslationError("Only function tool calls are supported")
                    fn = call["function"]
                    result["input"].append({"type": "function_call", "call_id": call["id"],
                                            "name": fn["name"], "arguments": fn["arguments"]})
        elif role == "tool":
            result["input"].append({"type": "function_call_output", "call_id": msg["tool_call_id"],
                                    "output": _text_parts(msg.get("content"))})
        else:
            raise TranslationError(f"Unsupported role: {role}")
    if "tools" in data:
        result["tools"] = []
        for tool in data["tools"]:
            if tool.get("type") != "function":
                raise TranslationError("Only function tools are supported")
            fn = tool["function"]
            result["tools"].append({"type": "function", "name": fn["name"],
                                    "description": fn.get("description", ""),
                                    "parameters": fn.get("parameters", {"type": "object"})})
    if "tool_choice" in data:
        choice = data["tool_choice"]
        if isinstance(choice, dict):
            result["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
        elif choice in ("required", "auto", "none"):
            result["tool_choice"] = choice
        else:
            raise TranslationError("Unsupported tool_choice")
    if "response_format" in data:
        fmt = data["response_format"]
        if fmt.get("type") == "json_schema":
            schema = fmt["json_schema"]
            result["text"] = {"format": {"type": "json_schema", "name": schema["name"],
                                          "schema": schema["schema"], "strict": schema.get("strict", False)}}
        elif fmt.get("type") == "json_object":
            result["text"] = {"format": {"type": "json_object"}}
        elif fmt.get("type") != "text":
            raise TranslationError("Unsupported response_format")
    return result


def _anthropic_request(data, model):
    if "reasoning_effort" in data or "reasoning" in data:
        raise TranslationError("Anthropic Messages does not support reasoning options here")
    result = {"model": data["model"], "messages": [],
              "max_tokens": data.get("max_completion_tokens", data.get("max_tokens", 4096))}
    for key in ("temperature", "top_p", "stream"):
        if key in data:
            result[key] = data[key]
    system = []
    for msg in data.get("messages", []):
        role = msg.get("role")
        if role in ("system", "developer"):
            system.append(_text_parts(msg.get("content")))
        elif role in ("user", "assistant"):
            replay = _replay_blocks(msg, model, "anthropic_messages") if role == "assistant" else None
            blocks = list(replay) if replay is not None else []
            if replay is None and msg.get("content") is not None:
                text = _text_parts(msg["content"])
                if text:
                    blocks.append({"type": "text", "text": text})
            for call in ([] if replay is not None else msg.get("tool_calls", [])):
                if role != "assistant" or call.get("type") != "function":
                    raise TranslationError("Only assistant function tool calls are supported")
                fn = call["function"]
                blocks.append({"type": "tool_use", "id": call["id"], "name": fn["name"],
                               "input": _json_args(fn.get("arguments"))})
            if not blocks:
                raise TranslationError("Empty messages are not supported")
            # Anthropic requires alternating turns. Merge consecutive messages of the same role.
            if result["messages"] and result["messages"][-1]["role"] == role:
                result["messages"][-1]["content"].extend(blocks)
            else:
                result["messages"].append({"role": role, "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": msg["tool_call_id"],
                     "content": _text_parts(msg.get("content"))}
            if not result["messages"] or result["messages"][-1]["role"] != "assistant":
                # Multiple tool results may be consecutive; they share a user turn.
                if not (result["messages"] and result["messages"][-1]["role"] == "user"
                        and any(b.get("type") == "tool_result" for b in result["messages"][-1]["content"])):
                    raise TranslationError("Tool result must follow an assistant tool call")
            if result["messages"] and result["messages"][-1]["role"] == "user":
                result["messages"][-1]["content"].append(block)
            else:
                result["messages"].append({"role": "user", "content": [block]})
        else:
            raise TranslationError(f"Unsupported role: {role}")
    if system:
        result["system"] = "\n\n".join(system)
    if "tools" in data:
        result["tools"] = []
        for tool in data["tools"]:
            if tool.get("type") != "function":
                raise TranslationError("Only function tools are supported")
            fn = tool["function"]
            result["tools"].append({"name": fn["name"], "description": fn.get("description", ""),
                                    "input_schema": fn.get("parameters", {"type": "object"})})
    if "tool_choice" in data:
        choice = data["tool_choice"]
        if isinstance(choice, dict):
            result["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
        elif choice in ("required", "auto", "none"):
            result["tool_choice"] = {"required": {"type": "any"}, "auto": {"type": "auto"},
                                     "none": {"type": "none"}}[choice]
        else:
            raise TranslationError("Unsupported tool_choice")
    if "response_format" in data and data["response_format"].get("type") != "text":
        raise TranslationError("Anthropic Messages does not support response_format here")
    return result


def prepare_request(model, path, body):
    """Return upstream path and JSON request bytes, or leave unrelated routes unchanged."""
    fmt = api_format(model)
    if path != "/v1/chat/completions" or fmt == "chat_completions":
        return path, body
    target_path = upstream_path(model, path)
    try:
        data = json.loads(body)
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise TranslationError("messages must be an array")
        _validate_options(data, fmt)
        data["model"] = model["api_model_name"]
        result = (_responses_request(data, model) if fmt == "openai_responses"
                  else _anthropic_request(data, model))
        return target_path, json.dumps(result).encode("utf-8")
    except (KeyError, TypeError, IndexError) as exc:
        raise TranslationError(f"Invalid chat request: {exc}") from exc


def normalize_usage(fmt, usage):
    if not isinstance(usage, dict):
        return None
    if fmt == "anthropic_messages":
        cached = usage.get("cache_read_input_tokens", 0)
        prompt = (usage.get("input_tokens", 0) + cached
                  + usage.get("cache_creation_input_tokens", 0))
        completion = usage.get("output_tokens", 0)
    else:
        prompt = usage.get("input_tokens", 0)
        completion = usage.get("output_tokens", 0)
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached}}


def _chat_payload(model, ident, created, text, calls, finish, usage=None, details=None):
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    if details:
        message["reasoning_details"] = details
    result = {"id": ident or "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
              "created": created or int(time.time()), "model": model["name"],
              "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}]}
    if usage is not None:
        result["usage"] = usage
    return result


def normalize_response(model, data):
    fmt = api_format(model)
    if fmt == "chat_completions":
        return data
    text = []
    calls = []
    if fmt == "openai_responses":
        if data.get("status") not in (None, "completed"):
            raise TranslationError(f"Responses request did not complete: {data.get('status')}")
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text.append(part.get("text", ""))
                    elif part.get("type") == "refusal":
                        text.append(part.get("refusal", ""))
                    else:
                        raise TranslationError(f"Unsupported Responses content part: {part.get('type')}")
            elif item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "type": "function",
                              "function": {"name": item["name"], "arguments": item.get("arguments", "{}")}})
            elif item.get("type") == "reasoning":
                # Native reasoning items may contain provider-only or encrypted
                # state; chat completions has no corresponding output item.
                # Never surface it as user-facing assistant text.
                continue
            else:
                raise TranslationError(f"Unsupported Responses output item: {item.get('type')}")
        finish = "tool_calls" if calls else "stop"
    else:
        for part in data.get("content", []):
            if part.get("type") == "text":
                text.append(part.get("text", ""))
            elif part.get("type") == "tool_use":
                calls.append({"id": part["id"], "type": "function",
                              "function": {"name": part["name"], "arguments": json.dumps(part.get("input", {}))}})
            elif part.get("type") in ("thinking", "redacted_thinking"):
                # Preserve signed native blocks in reasoning_details, not text.
                continue
            else:
                raise TranslationError(f"Unsupported Anthropic content block: {part.get('type')}")
        finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                  "tool_use": "tool_calls"}.get(data.get("stop_reason"), "stop")
    blocks = data.get("output" if fmt == "openai_responses" else "content", [])
    details = _envelope(model, fmt, blocks) if blocks else None
    return _chat_payload(model, data.get("id"), data.get("created_at") if fmt == "openai_responses" else None,
                         "".join(text), calls, finish, normalize_usage(fmt, data.get("usage")), details)


class StreamAdapter:
    """Convert native SSE events into OpenAI chat chunks, tracking tool indices."""
    def __init__(self, model):
        self.model = model
        self.fmt = api_format(model)
        self.ident = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.started = False
        self.finished = False
        self.usage = None
        self.calls = {}
        self.finish = "stop"
        self.received_terminal = False
        self.output_items = {}  # Responses SSE output_index -> completed native item
        self.content_blocks = {}  # Anthropic SSE block index -> completed native block

    def _chunk(self, delta, finish=None, usage=None):
        result = {"id": self.ident, "object": "chat.completion.chunk", "created": self.created,
                  "model": self.model["name"], "choices": [] if usage is not None else
                  [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}
        if usage is not None:
            result["usage"] = usage
        return result

    def _reasoning_delta(self):
        if self.fmt == "openai_responses":
            indices = sorted(self.output_items)
            if not indices:
                return None
            blocks = [self.output_items[i] for i in indices]
        else:
            indices = sorted(self.content_blocks)
            if not indices:
                return None
            blocks = [self.content_blocks[i] for i in indices]
        return {"reasoning_details": _envelope(self.model, self.fmt, blocks)}

    def _start(self):
        if not self.started:
            self.started = True
            return [self._chunk({"role": "assistant", "content": ""})]
        return []

    def feed(self, event, data):
        """Return zero or more chat chunks for one parsed native SSE event."""
        out = []
        if event in ("error", "response.failed"):
            raise TranslationError(f"Upstream stream failed: {data}")
        if self.fmt == "anthropic_messages":
            if event == "message_start":
                self.ident = data.get("message", {}).get("id", self.ident)
                out += self._start()
                self.usage = normalize_usage(self.fmt, data.get("message", {}).get("usage"))
            elif event == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") in ("thinking", "redacted_thinking", "text", "tool_use"):
                    self.content_blocks[data["index"]] = dict(block)
                if block.get("type") == "tool_use":
                    idx = data["index"]
                    self.calls[idx] = block
                    self.finish = "tool_calls"
                    out += self._start()
                    out.append(self._chunk({"tool_calls": [{"index": idx, "id": block["id"],
                        "type": "function", "function": {"name": block["name"], "arguments": ""}}]}))
                elif block.get("type") == "text" and block.get("text"):
                    out += self._start()
                    out.append(self._chunk({"content": block["text"]}))
            elif event == "content_block_delta":
                delta = data.get("delta", {})
                block = self.content_blocks.get(data.get("index"))
                if block:
                    if delta.get("type") == "text_delta" and block["type"] == "text":
                        block["text"] = block.get("text", "") + delta.get("text", "")
                    elif delta.get("type") == "thinking_delta" and block["type"] == "thinking":
                        block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                    elif delta.get("type") == "signature_delta" and block["type"] == "thinking":
                        block["signature"] = block.get("signature", "") + delta.get("signature", "")
                    elif delta.get("type") == "input_json_delta" and block["type"] == "tool_use":
                        block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")
                if delta.get("type") == "text_delta":
                    out += self._start()
                    out.append(self._chunk({"content": delta.get("text", "")}))
                elif delta.get("type") == "input_json_delta":
                    out += self._start()
                    out.append(self._chunk({"tool_calls": [{"index": data["index"],
                        "function": {"arguments": delta.get("partial_json", "")}}]}))
            elif event == "content_block_stop":
                block = self.content_blocks.get(data.get("index"))
                if block and "_partial_json" in block:
                    block["input"] = _json_args(block.pop("_partial_json"))
            elif event == "message_delta":
                self.finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                               "tool_use": "tool_calls"}.get(data.get("delta", {}).get("stop_reason"), self.finish)
                if "usage" in data:
                    prior = self.usage or normalize_usage(self.fmt, {})
                    self.usage = dict(prior)
                    self.usage["completion_tokens"] = data["usage"].get("output_tokens", 0)
                    self.usage["total_tokens"] = self.usage["prompt_tokens"] + self.usage["completion_tokens"]
            elif event == "message_stop":
                self.received_terminal = True
                out += self.complete()
        else:
            if event == "response.created":
                self.ident = data.get("response", {}).get("id", self.ident)
                out += self._start()
            elif event == "response.output_text.delta":
                out += self._start()
                out.append(self._chunk({"content": data.get("delta", "")}))
            elif event == "response.output_item.added":
                item = data.get("item", {})
                if item.get("type") in ("reasoning", "message", "function_call"):
                    self.output_items[data.get("output_index", 0)] = dict(item)
                if item.get("type") == "function_call":
                    idx = data.get("output_index", 0)
                    self.finish = "tool_calls"
                    out += self._start()
                    out.append(self._chunk({"tool_calls": [{"index": idx, "id": item["call_id"],
                        "type": "function", "function": {"name": item["name"], "arguments": ""}}]}))
            elif event == "response.output_item.done":
                item = data.get("item", {})
                if item.get("type") in ("reasoning", "message", "function_call"):
                    self.output_items[data.get("output_index", 0)] = dict(item)
            elif event == "response.function_call_arguments.delta":
                out += self._start()
                out.append(self._chunk({"tool_calls": [{"index": data.get("output_index", 0),
                    "function": {"arguments": data.get("delta", "")}}]}))
            elif event == "response.completed":
                response = data.get("response", {})
                if response.get("status") not in (None, "completed"):
                    raise TranslationError(f"Responses stream did not complete: {response.get('status')}")
                self.usage = normalize_usage(self.fmt, response.get("usage"))
                if isinstance(response.get("output"), list):
                    self.output_items = {i: item for i, item in enumerate(response["output"])}
                self.received_terminal = True
                out += self.complete()
            elif event == "response.incomplete":
                self.finish = "length"
                self.usage = normalize_usage(self.fmt, data.get("response", {}).get("usage"))
                self.received_terminal = True
                out += self.complete()
        return out

    def complete(self):
        if self.finished:
            return []
        self.finished = True
        out = self._start()
        details = self._reasoning_delta()
        if details:
            out.append(self._chunk(details))
        out.append(self._chunk({}, finish=self.finish))
        if self.usage is not None:
            out.append(self._chunk({}, usage=self.usage))
        return out
