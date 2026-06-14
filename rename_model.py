#!/usr/bin/env python3
"""
Rename a model and update all associated usage records.

Usage:
    python rename_model.py old_name new_name

Example:
    python rename_model.py gpt-4o-mini gpt-4o-mini-2024-07
"""

import sys
from storage import get_models_db, get_usage_db


def rename_model(old_name: str, new_name: str):
    # Validate names
    if not all(c.isalnum() or c in "-_" for c in new_name):
        print(f"Error: New name may only contain letters, numbers, hyphens, and underscores.")
        sys.exit(1)

    with get_models_db() as db:
        model = db.find_one({"name": old_name})
        if not model:
            print(f"Error: Model '{old_name}' not found.")
            sys.exit(1)

        if db.exists({"name": new_name}):
            print(f"Error: A model named '{new_name}' already exists.")
            sys.exit(1)

        # Update model name
        set_fields = {k: v for k, v in model.items() if k != "_id"}
        set_fields["name"] = new_name
        db.update_one({"name": old_name}, set=set_fields)
        print(f"Updated model '{old_name}' -> '{new_name}'")

    # Update all usage records (delete + reinsert since _id can't be updated in-place)
    with get_usage_db() as db:
        records = db.find({"model_name": old_name}).to_list()
        if not records:
            print("No usage records to update.")
            return

        for r in records:
            new_id = f"{r['date']}:{new_name}"
            db.delete_one({"_id": r["_id"]})
            r["_id"] = new_id
            r["model_name"] = new_name
            db.insert(r)

        print(f"Updated {len(records)} usage record(s).")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} old_name new_name")
        sys.exit(1)

    rename_model(sys.argv[1], sys.argv[2])
    print("Done.")
