"""
Extract token usage from LLM responses and persist daily aggregates.

Handles multiple provider response formats gracefully:
  - Standard OpenAI-compatible  (usage.prompt_tokens / completion_tokens)
  - Google Gemini               (usageMetadata.promptTokenCount / …)
  - Falls back to returning None when usage is unrecognised
"""

import json
from datetime import date
from storage import get_usage_db


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_usage_from_json(data: dict) -> dict | None:
    """
    Given a parsed JSON response body from a chat completion, return:

        {
            "input_tokens": int,
            "output_tokens": int,
            "cached_tokens": int,   # 0 if unknown
        }

    or None if no usage information was found.
    """
    usage = data.get("usage")
    if usage and isinstance(usage, dict):
        return {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": _cached_tokens(usage),
        }

    # Google Gemini format (usageMetadata)
    gm = data.get("usageMetadata")
    if gm and isinstance(gm, dict):
        return {
            "input_tokens":  gm.get("promptTokenCount", 0),
            "output_tokens": gm.get("candidatesTokenCount", 0),
            "cached_tokens": gm.get("cachedContentTokenCount", 0),
        }

    return None


def _cached_tokens(usage: dict) -> int:
    """Try to find a cached-token count in an OpenAI-style usage block."""
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return details.get("cached_tokens", 0)
    # Some providers nest under usage.prompt_cache_hit_tokens, etc.
    for key in ("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens"):
        if key in usage:
            return usage[key]
    return 0


def extract_usage_from_sse_line(data_line: str) -> dict | None:
    """
    Parse an SSE data line (the string between 'data: ' and the newline).
    Returns usage dict or None.
    """
    if not data_line or data_line == "[DONE]":
        return None
    try:
        return extract_usage_from_json(json.loads(data_line))
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def record_usage(
    model_name: str,
    model_display: str,
    model_type: str,        # "remote" or "local"
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    cost: float,
    duration_seconds: float = 0.0,
):
    """Upsert the daily usage record for a model, accumulating values."""
    today = date.today().isoformat()
    record_id = f"{today}:{model_name}"

    with get_usage_db() as db:
        existing = db.find_one({"_id": record_id})
        if existing:
            db.update_one(
                {"_id": record_id},
                inc={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "cost": cost,
                    "requests": 1,
                    "duration_seconds": duration_seconds,
                },
            )
        else:
            db.insert({
                "_id": record_id,
                "date": today,
                "model_name": model_name,
                "model_display": model_display,
                "model_type": model_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "cost": cost,
                "requests": 1,
                "duration_seconds": duration_seconds,
            })
