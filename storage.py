"""
Moofile-backed persistence layer.

Three collections:
  models   — all configured LLM backends (remote + local)
  usage    — per-model, per-day token/cost tracking
  settings — global settings (electricity cost, wattage)
"""

import os
from moofile import Collection

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _open(path: str, **kwargs) -> Collection:
    """Open a moofile collection — creates the data dir if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return Collection(os.path.join(DATA_DIR, path), **kwargs)


# ---------------------------------------------------------------------------
# Models collection
# ---------------------------------------------------------------------------
# Schema:
#   name                     str   unique short identifier (used in "model" field)
#   display_name             str   human-readable label
#   provider                 str   just for display (fireworks, openai, local, …)
#   base_url                 str   backend base URL
#   api_key                  str   authentication key for the backend
#   api_model_name           str   model name to send to the backend
#   type                     str   "remote" or "local"
#   tags                     list  subset of ["fast", "smart", "local"] (max 1)
#   input_price_per_million  float
#   output_price_per_million float
#   cached_price_per_million float
#   enabled                  bool

def get_models_db() -> Collection:
    # NOTE: "tags" is a list field — moofile can't index list values,
    # so we only index "name" and filter tags in Python.
    return _open("models.bson", indexes=["name"])


# ---------------------------------------------------------------------------
# Usage collection
# ---------------------------------------------------------------------------
# Schema:
#   _id              str   "{date}:{model_name}"
#   date             str   ISO date "YYYY-MM-DD"
#   model_name       str   matches models.name
#   model_display    str   display name at time of request
#   input_tokens     int
#   output_tokens    int
#   cached_tokens    int
#   cost             float
#   requests         int
#   duration_seconds float (for local models)

def get_usage_db() -> Collection:
    return _open("usage.bson", indexes=["date", "model_name"])


# ---------------------------------------------------------------------------
# Settings collection (single document)
# ---------------------------------------------------------------------------
# Schema:
#   _id                          str   "global"
#   electricity_cost_per_kwh     float  $ per kWh
#   local_model_max_wattage      int    watts

def get_settings_db() -> Collection:
    return _open("settings.bson")


def get_settings() -> dict:
    """Return the global settings dict, creating defaults if absent."""
    with get_settings_db() as db:
        s = db.find_one({"_id": "global"})
        defaults = {
            "_id": "global",
            "electricity_cost_per_kwh": 0.12,
            "local_model_max_wattage": 300,
            "proxy_timeout_seconds": 120,
        }
        if s is None:
            db.insert(defaults)
            return defaults

        # Backfill any missing keys from the defaults (for older records)
        merged = {**defaults, **s}
        if merged != s:
            set_kwargs = {k: v for k, v in merged.items() if k not in s and k != "_id"}
            if set_kwargs:
                db.update_one({"_id": "global"}, set=set_kwargs)
        return merged


def save_settings(updates: dict) -> dict:
    """Apply updates to the global settings and return the new doc."""
    with get_settings_db() as db:
        existing = db.find_one({"_id": "global"})
        if existing is None:
            existing = {}
            existing["_id"] = "global"
            db.insert(existing)
        # merge
        set_kwargs = {}
        for k, v in updates.items():
            if k != "_id":
                set_kwargs[k] = v
        if set_kwargs:
            db.update_one({"_id": "global"}, set=set_kwargs)
        return db.find_one({"_id": "global"})
