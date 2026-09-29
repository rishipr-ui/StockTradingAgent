"""Performance metrics and run comparisons for backtest JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def load_run(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSON records from one backtest JSONL file."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def _decision(record: dict[str, Any]) -> dict[str, Any]:
    """Return the nested decision object, tolerating older result formats."""
    value = record.get("decision", {})
    return value if isinstance(value, dict) else {}


def _action(record: dict[str, Any]) -> str:
    """Read the normalized action from a result record."""
    return str(_decision(record).get("action", record.get("action", "HOLD"))).upper()


def _outcome(record: dict[str, Any]) -> float | None:
    """Read a numeric realized or counterfactual outcome."""
    value = record.get("outcome")
    return None if value is None else float(value)


def _equity_curve(records: list[dict[str, Any]]) -> list[float]:
    """Return valid equity observations in file order."""
    return [float(record["equity"]) for record in records if record.get("equity") is not None]


def _max_drawdown(equity: list[float]) -> float:
    """Return maximum peak-to-trough drawdown as a positive fraction."""
    peak = None
    maximum = 0.0
    for value in equity:
        peak = value if peak is None else max(peak, value)
        if peak:
            maximum = max(maximum, (peak - value) / peak)
    return maximum


def _buy_and_hold_return(records: list[dict[str, Any]]) -> float | None:
    """Compute equal-weighted ticker buy-and-hold return from logged prices."""
    by_ticker: dict[str, list[tuple[str, float]]] = {}
    for record in records:
        price = record.get("price")
        if price is None:
            continue
        by_ticker.setdefault(str(record["ticker"]), []).append(
            (str(record.get("date", "")), float(price))
        )
    returns: list[float] = []
    for prices in by_ticker.values():
        prices.sort()
        if len(prices) >= 2 and prices[0][1]:
            returns.append((prices[-1][1] / prices[0][1]) - 1)
    return sum(returns) / len(returns) if returns else None


def compute_run_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute requested performance and memory-behavior metrics for one run."""
    if not records:
        return {
            "total_return": None,
            "max_drawdown": 0.0,
            "win_rate": None,
            "number_of_trades": 0,
            "repeated_mistake_rate": 0.0,
            "losses_avoided": 0,
            "buy_and_hold_return": None,
        }

    equity = _equity_curve(records)
    total_return = (equity[-1] / equity[0]) - 1 if len(equity) >= 2 and equity[0] else None
    trades = [
        record for record in records
        if _action(record) != "HOLD" and _outcome(record) is not None
    ]
    wins = sum(1 for record in trades if (_outcome(record) or 0) > 0)

    prior_losses: set[tuple[str, str, str]] = set()
    repeated_mistakes = 0
    considered_mistakes = 0
    losses_avoided = 0
    for record in records:
        decision = _decision(record)
        key = (
            str(record.get("ticker", "")),
            str(decision.get("setup", "")),
            str(decision.get("market_regime", "")),
        )
        outcome = _outcome(record)
        action = _action(record)
        memories = decision.get("memories_used", [])
        if action != "HOLD" and outcome is not None:
            considered_mistakes += 1
            if key in prior_losses:
                repeated_mistakes += 1
        if action == "HOLD" and memories and outcome is not None and outcome < 0:
            losses_avoided += 1
        if outcome is not None and outcome < -1:
            prior_losses.add(key)

    return {
        "total_return": total_return,
        "max_drawdown": _max_drawdown(equity),
        "win_rate": wins / len(trades) if trades else None,
        "number_of_trades": len(trades),
        "repeated_mistake_rate": (
            repeated_mistakes / considered_mistakes if considered_mistakes else 0.0
        ),
        "losses_avoided": losses_avoided,
        "buy_and_hold_return": _buy_and_hold_return(records),
    }


def summarize_results(results_dir: str | Path = "results") -> dict[str, dict[str, Any]]:
    """Compute metrics for every JSONL run under ``results_dir``."""
    directory = Path(results_dir)
    summary: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        summary[path.stem] = compute_run_metrics(load_run(path))
    return summary


def write_outputs(
    summary: dict[str, dict[str, Any]],
    summary_path: str | Path,
    comparison_path: str | Path,
) -> None:
    """Write the summary JSON and a flat comparison CSV table."""
    summary_file = Path(summary_path)
    comparison_file = Path(comparison_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = [
        "run",
        "total_return",
        "max_drawdown",
        "win_rate",
        "number_of_trades",
        "repeated_mistake_rate",
        "losses_avoided",
        "buy_and_hold_return",
    ]
    with comparison_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run, metrics in summary.items():
            writer.writerow({"run": run, **metrics})


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse metrics CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--summary-json", default="results/metrics_summary.json")
    parser.add_argument("--comparison-csv", default="results/comparison.csv")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Compute and write metrics for all discovered runs."""
    args = _parse_args(argv)
    summary = summarize_results(args.results_dir)
    write_outputs(summary, args.summary_json, args.comparison_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
