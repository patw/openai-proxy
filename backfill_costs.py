#!/usr/bin/env python3
"""
One-time backfill of usage costs.

The bug: cached tokens were double-counted — input_tokens from OpenAI already
includes cached tokens, but the old calculate_cost() priced ALL input_tokens
at the full input rate AND THEN added cached tokens again at the cached rate.

This script reads every usage record, looks up the model's pricing,
recomputes the cost correctly, and updates the database if it changed.
"""

import sys
from storage import get_usage_db, get_models_db, get_settings


def corrected_cost(record: dict, model: dict | None, settings: dict) -> float:
    """Recompute cost using the corrected logic (no double-count of cache)."""
    if record.get("model_type") == "local":
        wattage = settings.get("local_model_max_wattage", 300)
        price_kwh = settings.get("electricity_cost_per_kwh", 0.12)
        hours = record.get("duration_seconds", 0) / 3600.0
        kwh = (wattage / 1000.0) * hours
        return round(kwh * price_kwh, 10)

    if model is None:
        return record["cost"]  # can't recompute, leave as-is

    it = record.get("input_tokens", 0)
    ot = record.get("output_tokens", 0)
    ct = record.get("cached_tokens", 0)

    iprice = model.get("input_price_per_million", 0)
    oprice = model.get("output_price_per_million", 0)
    cprice = model.get("cached_price_per_million", 0)

    non_cached_input = max(it - ct, 0)

    cost = (non_cached_input / 1_000_000) * iprice
    cost += (ot / 1_000_000) * oprice
    cost += (ct / 1_000_000) * cprice
    return round(cost, 10)


def main():
    dry_run = "--dry-run" in sys.argv

    settings = get_settings()

    # Build model lookup: name → doc
    with get_models_db() as mdb:
        models = {m["name"]: m for m in mdb.find({}).to_list()}

    updated = 0
    skipped = 0
    errors = 0

    with get_usage_db() as db:
        records = db.find({}).to_list()

    print(f"Found {len(records)} usage record(s)\n")

    for r in records:
        rid = r["_id"]
        model_name = r.get("model_name", "?")

        old_cost = r.get("cost", 0)
        model = models.get(model_name)

        new_cost = corrected_cost(r, model, settings)
        delta = round(new_cost - old_cost, 10)

        if abs(delta) < 0.0000000001:
            skipped += 1
            continue

        direction = "↓" if delta < 0 else "↑"
        pct = (new_cost / old_cost * 100) if old_cost else 0
        print(f"  {direction} {rid:50s}  ${old_cost:<9.8f} → ${new_cost:<9.8f}  ({'%+.2f' % (delta * 100):s}¢, {pct:.0f}% of original)")

        if model is None and r.get("model_type") != "local":
            print(f"    ⚠ WARNING: model '{model_name}' not found in config — cost unchanged (can't recompute)")
            skipped += 1
            continue

        if dry_run:
            updated += 1
            continue

        try:
            with get_usage_db() as udb:
                udb.update_one({"_id": rid}, set={"cost": new_cost})
            updated += 1
        except Exception as e:
            print(f"    ✗ ERROR updating {rid}: {e}")
            errors += 1

    action = "preview" if dry_run else "update"
    print(f"\nDone.  {updated} would-be updates ({action}), {skipped} skipped (no change), {errors} errors.")


if __name__ == "__main__":
    main()
