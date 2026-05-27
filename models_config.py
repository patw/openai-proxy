"""
CRUD operations for model configurations.

Ensures tag uniqueness: only one model can have "fast", "smart", or "local".
"""

from typing import Optional
from storage import get_models_db

ALL_TAGS = {"fast", "smart", "local"}


def list_models(enabled_only: bool = False) -> list:
    """Return all models, optionally filtered to enabled ones."""
    with get_models_db() as db:
        if enabled_only:
            return db.find({"enabled": True}).sort("name").to_list()
        return db.find({}).sort("name").to_list()


def get_model(name: str) -> Optional[dict]:
    """Look up a model by name (exact match). Returns None if not found."""
    with get_models_db() as db:
        return db.find_one({"name": name})


def get_model_by_tag(tag: str) -> Optional[dict]:
    """Return the model that has the given tag, or None."""
    if tag not in ALL_TAGS:
        return None
    with get_models_db() as db:
        # moofile doesn't support $in on array fields directly,
        # so we scan enabled models and check tags in Python.
        for m in db.find({"enabled": True}).to_list():
            if tag in m.get("tags", []):
                return m
        return None


def save_model(data: dict) -> dict:
    """
    Insert or update a model.  `data` must include "name".
    If a tag (fast/smart/local) is being assigned, it is removed from
    any other model that currently holds it.
    """
    name = data["name"]
    tags = data.get("tags", [])
    # Normalise tags: only keep valid ones, max one
    valid_tags = [t for t in tags if t in ALL_TAGS]
    if len(valid_tags) > 1:
        valid_tags = valid_tags[:1]  # keep only the first
    data["tags"] = valid_tags

    with get_models_db() as db:
        existing = db.find_one({"name": name})

        # ---------- Enforce tag uniqueness ----------
        # If we're *changing* the tag, remove it from the old holder.
        old_tags = set(existing.get("tags", [])) if existing else set()
        new_tag = valid_tags[0] if valid_tags else None
        old_tag = next((t for t in old_tags if t in ALL_TAGS), None)

        if new_tag and new_tag != old_tag:
            # Clear this tag from any other model that holds it
            for m in db.find({"enabled": True}).to_list():
                if m["name"] != name and new_tag in m.get("tags", []):
                    new_tags = [t for t in m["tags"] if t != new_tag]
                    db.update_one({"name": m["name"]}, set={"tags": new_tags})

        # ---------- Insert or update ----------
        if existing:
            set_fields = {k: v for k, v in data.items() if k != "name" and k != "_id"}
            db.update_one({"name": name}, set=set_fields)
        else:
            data.pop("_id", None)  # let moofile generate it
            db.insert(data)

        return db.find_one({"name": name})


def delete_model(name: str) -> bool:
    """Delete a model by name. Returns True if deleted."""
    with get_models_db() as db:
        return db.delete_one({"name": name})


def validate_model_form(form: dict, editing: bool = False) -> list:
    """
    Validate form data for a model. Returns a list of error strings.
    Empty list means valid.
    """
    errors = []
    name = (form.get("name") or "").strip()
    if not name:
        errors.append("Name is required.")
    elif not all(c.isalnum() or c in "-_" for c in name):
        errors.append("Name may only contain letters, numbers, hyphens, and underscores.")

    # Unique name check
    if not editing:
        with get_models_db() as db:
            if db.exists({"name": name}):
                errors.append(f"A model named '{name}' already exists.")

    if not (form.get("base_url") or "").strip():
        errors.append("Base URL is required.")
    if not (form.get("api_model_name") or "").strip():
        errors.append("API model name is required.")

    mtype = form.get("type", "remote")
    if mtype not in ("remote", "local"):
        errors.append("Type must be 'remote' or 'local'.")

    if mtype == "remote":
        for field in ["input_price_per_million", "output_price_per_million"]:
            try:
                val = float(form.get(field, 0))
                if val < 0:
                    errors.append(f"{field} must be >= 0.")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a number.")
        # cached is optional, default 0
        try:
            val = float(form.get("cached_price_per_million", 0))
            if val < 0:
                errors.append("cached_price_per_million must be >= 0.")
        except (ValueError, TypeError):
            errors.append("cached_price_per_million must be a number.")

    return errors


def model_tags_summary() -> dict:
    """Return {tag: model_name_or_None} for fast, smart, local."""
    result = {"fast": None, "smart": None, "local": None}
    with get_models_db() as db:
        for m in db.find({"enabled": True}).to_list():
            for tag in m.get("tags", []):
                if tag in result:
                    result[tag] = m["name"]
    return result
