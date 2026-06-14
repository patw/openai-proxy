#!/usr/bin/env python3
"""
Rename a model and/or normalize its usage records.

Usage:
    # Rename a model (updates name + usage record keys)
    python rename_model.py old_name new_name

    # Normalize display names for an existing model
    python rename_model.py --fix-display model_name

Examples:
    python rename_model.py gpt-4o-mini gpt-4o-mini-2024-07
    python rename_model.py --fix-display gpt-4o-mini-2024-07

The --fix-display mode collapses all historical display name variants for a
model into its current display_name, cleaning up the Per Model Summary report.
"""

import sys
import argparse
from storage import get_models_db, get_usage_db


def rename_model(old_name: str, new_name: str):
    """Rename a model and update all associated usage record keys."""
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

        set_fields = {k: v for k, v in model.items() if k != "_id"}
        set_fields["name"] = new_name
        db.update_one({"name": old_name}, set=set_fields)
        print(f"Model renamed: '{old_name}' -> '{new_name}'")

    current_display = model.get("display_name", new_name)

    with get_usage_db() as db:
        records = db.find({"model_name": old_name}).to_list()
        if not records:
            print("No usage records to update.")
            return

        old_displays = sorted(set(r.get("model_display", "?") for r in records))
        print(f"Found {len(records)} usage record(s) with display name(s): {old_displays}")

        for r in records:
            new_id = f"{r['date']}:{new_name}"
            db.delete_one({"_id": r["_id"]})
            r["_id"] = new_id
            r["model_name"] = new_name
            db.insert(r)

        print(f"Updated {len(records)} usage record(s).")


def fix_display(model_name: str):
    """Normalize all usage records for a model to its current display_name."""
    with get_models_db() as db:
        model = db.find_one({"name": model_name})
        if not model:
            print(f"Error: Model '{model_name}' not found.")
            sys.exit(1)

    current_display = model.get("display_name", model_name)

    with get_usage_db() as db:
        records = db.find({"model_name": model_name}).to_list()
        if not records:
            print(f"No usage records for '{model_name}'.")
            return

        old_displays = sorted(set(r.get("model_display", "?") for r in records))
        changed = 0

        for r in records:
            old_disp = r.get("model_display")
            if old_disp and old_disp != current_display:
                db.update_one({"_id": r["_id"]}, set={"model_display": current_display})
                changed += 1

        print(f"Model '{model_name}' display_name: '{current_display}'")
        print(f"Usage records: {len(records)} total, {changed} updated from variants: {old_displays}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename a model and/or fix its usage records.")
    parser.add_argument("old_name", nargs="?", help="Current model name (for rename)")
    parser.add_argument("new_name", nargs="?", help="New model name (for rename)")
    parser.add_argument("--fix-display", metavar="MODEL",
                        help="Normalize display names for an existing model")
    args = parser.parse_args()

    if args.fix_display:
        fix_display(args.fix_display)
    elif args.old_name and args.new_name:
        rename_model(args.old_name, args.new_name)
    else:
        parser.print_help()
        sys.exit(1)

    print("Done.")
