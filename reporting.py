"""
Reporting and chart generation for token usage and cost.

Provides aggregation queries against the usage collection and
generates matplotlib bar charts as PNGs.
"""

import io
import base64
from datetime import date, timedelta
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from storage import get_usage_db


def get_daily_usage(days: int = 30) -> list:
    """
    Return daily usage summaries for the last N days.

    Each row: {date, cost, input_tokens (non-cached), output_tokens,
               cached_tokens, total_input_tokens, requests}

    .. note::
        ``input_tokens`` is *non-cached* input (total prompt tokens minus
        cached).  ``total_input_tokens`` holds the raw API value including
        cached.  The reporting UI uses ``input_tokens`` so the billable
        breakdown is immediately clear: fresh input + cached + output.
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with get_usage_db() as db:
        records = db.find({"date": {"$gte": cutoff}}).sort("date").to_list()

    # Aggregate by date
    by_date = defaultdict(lambda: {
        "cost": 0.0, "input_tokens": 0, "output_tokens": 0,
        "cached_tokens": 0, "requests": 0
    })
    for r in records:
        d = r["date"]
        by_date[d]["cost"] += r.get("cost", 0)
        by_date[d]["input_tokens"] += r.get("input_tokens", 0)
        by_date[d]["output_tokens"] += r.get("output_tokens", 0)
        by_date[d]["cached_tokens"] += r.get("cached_tokens", 0)
        by_date[d]["requests"] += r.get("requests", 0)

    result = []
    for dt in sorted(by_date.keys()):
        row = {"date": dt, **by_date[dt]}
        row["cost"] = round(row["cost"], 6)
        # Store total before we overwrite input_tokens
        total = row["input_tokens"]
        # input_tokens in the DB includes cached — expose non-cached instead
        row["input_tokens"] = max(total - row["cached_tokens"], 0)
        row["total_input_tokens"] = total
        result.append(row)
    return result


def get_weekly_summary(days: int = 91) -> list:
    """Aggregate cost by ISO week. Returns [{week, cost, …}, …]."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with get_usage_db() as db:
        records = db.find({"date": {"$gte": cutoff}}).to_list()

    by_week = defaultdict(lambda: {
        "cost": 0.0, "input_tokens": 0, "output_tokens": 0,
        "cached_tokens": 0, "requests": 0
    })
    for r in records:
        d = date.fromisoformat(r["date"])
        week_key = d.strftime("%Y-W%W")
        by_week[week_key]["cost"] += r.get("cost", 0)
        by_week[week_key]["input_tokens"] += r.get("input_tokens", 0)
        by_week[week_key]["output_tokens"] += r.get("output_tokens", 0)
        by_week[week_key]["cached_tokens"] += r.get("cached_tokens", 0)
        by_week[week_key]["requests"] += r.get("requests", 0)

    result = []
    for k, v in sorted(by_week.items()):
        # Same convention as get_daily_usage: expose non-cached input as
        # input_tokens and keep the raw total (incl. cached) separate.
        total = v["input_tokens"]
        v["input_tokens"] = max(total - v["cached_tokens"], 0)
        v["total_input_tokens"] = total
        v["cost"] = round(v["cost"], 6)
        result.append({"week": k, **v})
    return result


def get_monthly_summary(days: int = 365) -> list:
    """Aggregate cost by month. Returns [{month, cost, …}, …]."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with get_usage_db() as db:
        records = db.find({"date": {"$gte": cutoff}}).to_list()

    by_month = defaultdict(lambda: {
        "cost": 0.0, "input_tokens": 0, "output_tokens": 0,
        "cached_tokens": 0, "requests": 0
    })
    for r in records:
        month_key = r["date"][:7]  # "YYYY-MM"
        by_month[month_key]["cost"] += r.get("cost", 0)
        by_month[month_key]["input_tokens"] += r.get("input_tokens", 0)
        by_month[month_key]["output_tokens"] += r.get("output_tokens", 0)
        by_month[month_key]["cached_tokens"] += r.get("cached_tokens", 0)
        by_month[month_key]["requests"] += r.get("requests", 0)

    result = []
    for k, v in sorted(by_month.items()):
        # Same convention as get_daily_usage: expose non-cached input as
        # input_tokens and keep the raw total (incl. cached) separate.
        total = v["input_tokens"]
        v["input_tokens"] = max(total - v["cached_tokens"], 0)
        v["total_input_tokens"] = total
        v["cost"] = round(v["cost"], 6)
        result.append({"month": k, **v})
    return result


def get_per_model_summary(days: int = 30) -> list:
    """Cost breakdown by model for the last N days."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with get_usage_db() as db:
        records = db.find({"date": {"$gte": cutoff}}).to_list()

    by_model = defaultdict(lambda: {
        "cost": 0.0, "input_tokens": 0, "output_tokens": 0,
        "cached_tokens": 0, "requests": 0
    })
    for r in records:
        mn = r.get("model_display", r["model_name"])
        by_model[mn]["cost"] += r.get("cost", 0)
        by_model[mn]["input_tokens"] += r.get("input_tokens", 0)
        by_model[mn]["output_tokens"] += r.get("output_tokens", 0)
        by_model[mn]["cached_tokens"] += r.get("cached_tokens", 0)
        by_model[mn]["requests"] += r.get("requests", 0)

    result = []
    for mn in sorted(by_model.keys(), key=lambda m: by_model[m]["cost"], reverse=True):
        row = {"model": mn, **by_model[mn]}
        total = row["input_tokens"]
        row["input_tokens"] = max(total - row["cached_tokens"], 0)
        row["total_input_tokens"] = total
        row["cost"] = round(row["cost"], 6)
        result.append(row)

    return result


def _enrich_row(row: dict) -> dict:
    """Add derived fields to an aggregated usage row (mutates and returns it).

    - total_tokens     : total input (incl. cached) + output tokens
    - cached_ratio     : fraction of input tokens served from cache (0..1)
    - cost_per_request : average cost per request in the window
    """
    total_input = row.get(
        "total_input_tokens", row["input_tokens"] + row["cached_tokens"]
    )
    row["total_tokens"] = total_input + row["output_tokens"]
    row["cached_ratio"] = (
        round(row["cached_tokens"] / total_input, 6) if total_input else 0.0
    )
    row["cost_per_request"] = (
        round(row["cost"] / row["requests"], 6) if row["requests"] else 0.0
    )
    return row


def get_reports_payload(days: int = 30) -> dict:
    """
    Everything the /reports page shows, as plain data (no charts) for the
    JSON API. ``days`` controls the daily + per-model window; weekly and
    monthly use fixed lookbacks (91 and 365 days) so the growth series stay
    stable as ``days`` changes.
    """
    daily = get_daily_usage(days)
    weekly = get_weekly_summary(91)
    monthly = get_monthly_summary(365)
    per_model = get_per_model_summary(days)

    today = date.today().isoformat()
    week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
    month_start = date.today().replace(day=1).isoformat()

    def _sum_costs(rows, since):
        return round(sum(r["cost"] for r in rows if r.get("date", "") >= since), 6)

    total_cost = sum(r["cost"] for r in per_model)

    for r in daily:
        _enrich_row(r)
    for r in weekly:
        _enrich_row(r)
    for r in monthly:
        _enrich_row(r)
    for r in per_model:
        _enrich_row(r)
        r["pct_of_total_cost"] = round(r["cost"] / total_cost, 6) if total_cost else 0.0

    return {
        "generated": date.today().isoformat(),
        "days": days,
        "totals": {
            "today": _sum_costs(daily, today),
            "week": _sum_costs(daily, week_start),
            "month": _sum_costs(daily, month_start),
        },
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def daily_cost_chart(days: int = 30) -> str:
    """Bar chart of daily cost over the last N days. Returns base64 PNG."""
    data = get_daily_usage(days)
    if not data:
        return ""

    dates = [d["date"] for d in data]
    costs = [d["cost"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, costs, color="#4a90d9", edgecolor="white")
    ax.set_title("Daily Token Spend", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cost ($)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.4f"))
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def monthly_cost_chart(months: int = 12) -> str:
    """Bar chart of monthly cost. Returns base64 PNG."""
    data = get_monthly_summary(months * 31)
    if not data:
        return ""

    months_labels = [d["month"] for d in data]
    costs = [d["cost"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(months_labels, costs, color="#e67e22", edgecolor="white")
    ax.set_title("Monthly Token Spend", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cost ($)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.4f"))
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def model_breakdown_chart(days: int = 30) -> str:
    """Horizontal bar chart of cost per model. Returns base64 PNG."""
    data = get_per_model_summary(days)
    if not data:
        return ""

    models = [d["model"] for d in data]
    costs = [d["cost"] for d in data]

    fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.5)))
    colors = ["#4a90d9", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6",
              "#1abc9c", "#f39c12", "#34495e"]
    ax.barh(models, costs, color=colors[:len(models)], edgecolor="white")
    ax.set_title(f"Cost per Model (last {days} days)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cost ($)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_b64(fig)


def token_volume_chart(days: int = 30) -> str:
    """
    Stacked bar chart: non-cached input + cached input + output tokens per day.

    ``get_daily_usage`` now returns ``input_tokens`` as non-cached input,
    so we stack: non-cached (blue), cached (green), output (orange).
    """
    data = get_daily_usage(days)
    if not data:
        return ""

    dates = [d["date"] for d in data]
    non_cached = [d["input_tokens"] for d in data]    # already non-cached
    caches = [d["cached_tokens"] for d in data]
    outputs = [d["output_tokens"] for d in data]
    # Total input = non-cached + cached (for the bottom of output bar)
    total_input = [non_cached[i] + caches[i] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, non_cached, label="Input (non-cached)", color="#4a90d9", edgecolor="white")
    ax.bar(dates, caches, bottom=non_cached, label="Input (cached)",
           color="#2ecc71", edgecolor="white")
    ax.bar(dates, outputs, bottom=total_input, label="Output tokens",
           color="#e67e22", edgecolor="white")
    ax.set_title("Daily Token Volume (input cached / non-cached / output)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Tokens (millions)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))
    ax.legend()
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)
