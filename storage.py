"""
Moofile-backed persistence layer.

Three collections:
  models   — all configured LLM backends (remote + local)
  usage    — per-model, per-day token/cost tracking
  settings — global settings (electricity cost, wattage)
"""

import os
import atexit
import threading
from moofile import Collection

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
# moofile Collections load the whole file into an in-memory index on open, so
# re-opening one per request means a full scan every time (and the usage file
# grows unbounded).  Instead we keep one long-lived Collection per file and
# reuse it.
#
# moofile Collections are NOT thread-safe, and the proxy serves requests on
# multiple threads (waitress threads=8), so every access is serialized by a
# per-collection re-entrant lock.  Holding that lock for the whole `with`
# block also makes read-modify-write sequences (e.g. the usage upsert in
# record_usage, or the tag-uniqueness update in save_model) atomic within the
# process, which prevents the lost-increment race that plain per-request
# opens are subject to.
#
# NOTE: because the index is now long-lived, this process will not observe
# writes made by *other* processes (e.g. the rename_model.py / backfill_costs.py
# maintenance scripts) until it is restarted.  Run those with the server stopped.

_pool: dict[str, "_PooledCollection"] = {}
_pool_guard = threading.Lock()


class _PooledCollection:
    """A shared Collection plus its lock, usable as a context manager.

    Entering acquires the lock and returns the underlying Collection;
    exiting releases the lock but keeps the Collection open for reuse.
    """

    def __init__(self, full_path: str, **kwargs):
        self._lock = threading.RLock()
        self._collection = Collection(full_path, **kwargs)

    def __enter__(self) -> Collection:
        self._lock.acquire()
        return self._collection

    def __exit__(self, *exc) -> bool:
        self._lock.release()
        return False

    def close(self) -> None:
        with self._lock:
            self._collection.close()


def _open(path: str, **kwargs) -> "_PooledCollection":
    """Return the pooled collection for *path*, creating it once on first use."""
    os.makedirs(DATA_DIR, exist_ok=True)
    full_path = os.path.join(DATA_DIR, path)
    key = os.path.abspath(full_path)
    with _pool_guard:
        pooled = _pool.get(key)
        if pooled is None:
            pooled = _PooledCollection(full_path, **kwargs)
            _pool[key] = pooled
        return pooled


@atexit.register
def _close_pool() -> None:
    """Flush and close all pooled collections on interpreter shutdown."""
    with _pool_guard:
        for pooled in _pool.values():
            try:
                pooled.close()
            except Exception:
                pass
        _pool.clear()


# ---------------------------------------------------------------------------
# Models collection
# ---------------------------------------------------------------------------
# Schema:
#   name                     str   unique short identifier (used in "model" field)
#   display_name             str   human-readable label
#   provider                 str   just for display (fireworks, openai, local, …)
#   base_url                 str   backend base URL
#   api_key                  str   authentication key for the backend
#   api_key_header           str   optional — header name for auth ("" → Authorization: Bearer, "api-key" → Azure-style)
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
#   input_tokens     int   total prompt tokens sent to the backend
#                           (cached_tokens is a *subset* of this)
#   output_tokens    int   completion tokens
#   cached_tokens    int   subset of input_tokens that hit the prompt cache
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
            "stream_include_usage": True,
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
