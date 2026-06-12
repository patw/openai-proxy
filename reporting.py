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
from moofile import sum as moo_sum, count


def get_daily_usage(days: int = 30) -> list:
    """
    Return daily usage summaries for the last N days.
    Each row: {date, cost, input_tokens, output_tokens, cached_tokens, requests}
    Aggregated across all models.
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

    return [{"week": k, **v} for k, v in sorted(by_week.items())]


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

    return [{"month": k, **v} for k, v in sorted(by_month.items())]


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

    return [{"model": k, **v} for k, v in
            sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True)]


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
    """Stacked bar chart: input vs output tokens per day."""
    data = get_daily_usage(days)
    if not data:
        return ""

    dates = [d["date"] for d in data]
    inputs = [d["input_tokens"] for d in data]
    outputs = [d["output_tokens"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, inputs, label="Input tokens", color="#4a90d9", edgecolor="white")
    ax.bar(dates, outputs, bottom=inputs, label="Output tokens",
           color="#e67e22", edgecolor="white")
    ax.set_title("Daily Token Volume (in / out)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tokens (millions)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))
    ax.legend()
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)
