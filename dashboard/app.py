"""Streamlit dashboard for reviewing offline bull-bear-desk backtests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_run_metrics, load_run

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RUNS = ["baseline_test", "memory_test"]


def _records_for_run(run_name: str) -> list[dict[str, Any]]:
    """Load one local JSONL run without making network requests."""
    path = RESULTS_DIR / f"{run_name}.jsonl"
    return load_run(path) if path.exists() else []


def _available_runs() -> list[str]:
    """Return preferred test runs followed by any other local JSONL runs."""
    discovered = sorted(path.stem for path in RESULTS_DIR.glob("*.jsonl"))
    preferred = [run for run in DEFAULT_RUNS if run in discovered]
    return preferred + [run for run in discovered if run not in preferred]


def _pct(value: Any) -> str:
    """Format a decimal metric as a percentage."""
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _metric_card(label: str, value: str, accent: str = "#8b9cff") -> None:
    """Render a compact dark-theme metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _comparison_card(
    label: str,
    memory_value: str,
    baseline_value: str,
    accent: str = "#3b82f6",
    key_metric: bool = False,
) -> None:
    """Render a wireframe-style metric card with both test-run values."""
    badge = '<span class="key-badge">KEY METRIC</span>' if key_metric else ""
    st.markdown(
        f"""
        <div class="metric-card comparison-card">
            <div class="metric-label">{label} {badge}</div>
            <div class="comparison-values">
                <span class="metric-value" style="color:{accent}">{memory_value}</span>
                <span class="baseline-value">{baseline_value}</span>
            </div>
            <div class="comparison-legend">
                <span><i class="dot memory-dot"></i>With memory</span>
                <span><i class="dot baseline-dot"></i>No memory</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _records_frame(records: list[dict[str, Any]], ticker: str | None = None) -> pd.DataFrame:
    """Convert records to a sorted replay frame."""
    filtered = [row for row in records if not ticker or row.get("ticker") == ticker]
    frame = pd.DataFrame(filtered)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    return frame


def _equity_figure(
    run_records: dict[str, list[dict[str, Any]]],
    selected_run: str,
) -> go.Figure:
    """Build normalized equity and buy-and-hold curves for available runs."""
    figure = go.Figure()
    colors = {"baseline_test": "#f59e0b", "memory_test": "#60a5fa"}
    for run_name, records in run_records.items():
        rows = [row for row in records if row.get("equity") is not None]
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"])
        curve = frame.groupby("date", as_index=False)["equity"].last()
        curve["normalized"] = curve["equity"] / curve["equity"].iloc[0] * 100
        figure.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["normalized"],
                mode="lines",
                name=run_name,
                line={"color": colors.get(run_name, "#a78bfa"), "width": 2.5},
                visible=True if run_name == selected_run else "legendonly",
            )
        )

        prices = frame.dropna(subset=["price"]).groupby("date", as_index=False)["price"].mean()
        if not prices.empty:
            prices["normalized"] = prices["price"] / prices["price"].iloc[0] * 100
            figure.add_trace(
                go.Scatter(
                    x=prices["date"],
                    y=prices["normalized"],
                    mode="lines",
                    name=f"{run_name} buy & hold",
                    line={"color": "#94a3b8", "dash": "dot"},
                    visible=True if run_name == selected_run else "legendonly",
                )
            )
    figure.update_layout(
        template="plotly_dark",
        height=390,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="Indexed value (start = 100)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
    )
    return figure


def _render_replay(records: list[dict[str, Any]], ticker: str | None) -> None:
    """Render day-by-day decision replay for the selected run."""
    frame = _records_frame(records, ticker)
    if frame.empty:
        st.info("No replay records match the selected ticker.")
        return
    dates = list(frame["date"].dt.strftime("%Y-%m-%d").drop_duplicates())
    index_key = "replay_day_index"
    st.session_state[index_key] = min(st.session_state.get(index_key, 0), len(dates) - 1)
    control_col, slider_col, date_col = st.columns([0.75, 2.5, 1])
    with control_col:
        if st.button("▶ Play", use_container_width=True):
            st.session_state[index_key] = (st.session_state[index_key] + 1) % len(dates)
            st.rerun()
    with slider_col:
        selected_index = st.slider(
            "Replay day",
            min_value=0,
            max_value=len(dates) - 1,
            key=index_key,
            label_visibility="collapsed",
        )
    selected_date = dates[selected_index]
    with date_col:
        st.markdown(f"**{selected_date}**")
    day_rows = frame[frame["date"].dt.strftime("%Y-%m-%d") == selected_date]

    for _, row in day_rows.iterrows():
        decision = row.get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}
        st.markdown(f"### {row.get('ticker', 'Unknown ticker')}")
        content_col, memory_col = st.columns([2.1, 1])
        with content_col:
            bull_col, bear_col = st.columns(2)
            with bull_col:
                st.markdown('<div class="argument-label bull-label">● BULL ARGUMENT</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="argument-card bull-card">{row.get("bull_text") or "No bull argument recorded."}</div>', unsafe_allow_html=True)
            with bear_col:
                st.markdown('<div class="argument-label bear-label">● BEAR ARGUMENT</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="argument-card bear-card">{row.get("bear_text") or "No bear argument recorded."}</div>', unsafe_allow_html=True)

            st.markdown('<div class="verdict-card">', unsafe_allow_html=True)
            verdict_col, reason_col = st.columns([0.65, 2])
            with verdict_col:
                st.caption("JUDGE VERDICT")
                action = decision.get("action", "HOLD")
                st.markdown(f"**{action}** · {_pct(decision.get('confidence'))} confidence")
            with reason_col:
                st.caption("REASONING")
                st.write(decision.get("reasoning", "No reasoning recorded."))
            outcome = row.get("outcome")
            if outcome is not None:
                color = "#34d399" if float(outcome) >= 0 else "#fb7185"
                st.markdown(
                    f'<div class="outcome" style="color:{color}">5-DAY OUTCOME '
                    f'{float(outcome):+.2f}%</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        with memory_col:
            st.markdown("**Recalled memories**")
            st.caption(f"{len(row.get('recalled_memories') or [])} recalled")
            recalled = row.get("recalled_memories") or []
            used = set(decision.get("memories_used") or [])
            if not recalled:
                st.caption("No relevant memory.")
            for memory in recalled:
                is_used = memory in used or any(memory in item for item in used)
                label = "Used by judge" if is_used else "Recalled"
                st.markdown(
                    f'<div class="memory-card {"used-memory" if is_used else ""}">'
                    f'<b>{label}</b><br>{memory}</div>',
                    unsafe_allow_html=True,
                )
        st.divider()


def _render_lessons(records: list[dict[str, Any]]) -> None:
    """Render retained lessons ordered by date."""
    rows = [
        {
            "date": row.get("date"),
            "ticker": row.get("ticker"),
            "setup": (row.get("decision") or {}).get("setup", ""),
            "outcome": row.get("outcome"),
            "winner": row.get("winner_side"),
            "lesson": row.get("lesson"),
        }
        for row in records
        if row.get("lesson")
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("date"), use_container_width=True, hide_index=True)
    else:
        st.info("No retained lessons are present in this run.")


def _render_trust(records: list[dict[str, Any]]) -> None:
    """Show bull and bear correctness by market regime."""
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        decision = row.get("decision") or {}
        regime = decision.get("market_regime", "unknown")
        winner = row.get("winner_side")
        if winner not in {"bull", "bear"}:
            continue
        counts.setdefault(regime, {"bull": 0, "bear": 0})[winner] += 1
    rows = [
        {
            "Market regime": regime,
            "Bull right": values["bull"],
            "Bear right": values["bear"],
            "Bull share": values["bull"] / sum(values.values()),
            "Bear share": values["bear"] / sum(values.values()),
        }
        for regime, values in sorted(counts.items())
    ]
    if rows:
        table = pd.DataFrame(rows)
        table["Bull share"] = table["Bull share"].map(lambda value: f"{value:.1%}")
        table["Bear share"] = table["Bear share"].map(lambda value: f"{value:.1%}")
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No resolved bull/bear outcomes are available.")


def _render_loss_review(records: list[dict[str, Any]]) -> None:
    """Review losing trades and later decisions that recalled their lessons."""
    losses = [row for row in records if row.get("outcome") is not None and float(row["outcome"]) < 0]
    if not losses:
        st.info("No losing trades are present in this run.")
        return
    for loss in losses:
        lesson = loss.get("lesson") or "No lesson retained."
        loss_date = str(loss.get("date", ""))
        later = [
            row for row in records
            if str(row.get("date", "")) > loss_date
            and any(lesson in memory for memory in (row.get("recalled_memories") or []))
        ]
        st.markdown(
            f"**{loss.get('date')} · {loss.get('ticker')} · "
            f"{float(loss['outcome']):+.2f}%**"
        )
        st.write(f"Retained lesson: {lesson}")
        if later:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "date": row.get("date"),
                            "ticker": row.get("ticker"),
                            "decision": (row.get("decision") or {}).get("action", "HOLD"),
                            "outcome": row.get("outcome"),
                        }
                        for row in later
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No later decision explicitly recalled this lesson.")
        st.divider()


def main() -> None:
    """Render the offline backtest dashboard."""
    st.set_page_config(page_title="Bull Bear Desk", page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #080b12; color: #e5e7eb; }
        [data-testid="stSidebar"] { background: #0d111b; }
        .block-container { max-width: 1380px; padding-top: 1.1rem; }
        .metric-card { background: #11151e; border: 1px solid #252b38;
          border-radius: 10px; padding: 14px; min-height: 96px; }
        .comparison-card { min-height: 104px; }
        .metric-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;
          letter-spacing: .06em; }
        .metric-value { font-size: 1.65rem; font-weight: 700; margin-top: 8px; }
        .baseline-value { color: #91a3bf; font-size: 1rem; margin-left: 14px; }
        .comparison-values { display: flex; align-items: baseline; }
        .comparison-legend { color: #64748b; font-size: .68rem; display: flex;
          gap: 14px; margin-top: 8px; }
        .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
          margin-right: 4px; }
        .memory-dot { background: #3b82f6; }
        .baseline-dot { background: #64748b; }
        .key-badge { background: #17355f; color: #93c5fd; border-radius: 4px;
          font-size: .58rem; padding: 3px 5px; margin-left: 5px; }
        .memory-card { background: #111923; border: 1px solid #26324d;
          border-radius: 8px; padding: 10px; margin: 7px 0; font-size: .9rem; }
        .used-memory { border-color: #60a5fa; background: #172b4d; }
        .outcome { font-size: 1.1rem; font-weight: 700; padding: 8px 0; }
        .argument-card { border-radius: 7px; padding: 11px; min-height: 65px;
          font-size: .85rem; line-height: 1.45; }
        .bull-card { background: #09211e; border: 1px solid #12443b; }
        .bear-card { background: #29151d; border: 1px solid #5c2634; }
        .argument-label { font-size: .7rem; font-weight: 700; margin-bottom: 5px; }
        .bull-label { color: #34d399; }
        .bear-label { color: #fb7185; }
        .verdict-card { background: #11151e; border: 1px solid #252b38;
          border-radius: 7px; padding: 9px 12px; margin-top: 10px; }
        .top-disclaimer { background: #211b0b; border: 1px solid #4c3b12;
          color: #fbbf24; border-radius: 16px; padding: 7px 12px;
          text-align: center; font-size: .72rem; margin-top: 11px; }
        .section-gap { height: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    runs = _available_runs()
    if not runs:
        st.info("No results/*.jsonl runs found. Run a backtest before opening the dashboard.")
        return
    preferred_index = runs.index("memory_test") if "memory_test" in runs else 0
    header_col, period_col, ticker_col, disclaimer_col = st.columns([1.8, 1.25, 1.05, 1.6])
    with header_col:
        st.markdown("## 📈 Bull Bear Desk")
        st.caption("MEMORY-AGENT ANALYTICS")
    with period_col:
        st.markdown("**Test period:** Jul–Dec")
    with ticker_col:
        st.markdown("**Scope:** All tickers")
    with disclaimer_col:
        st.markdown('<div class="top-disclaimer">● Simulation only — not financial advice</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("Run controls")
        st.caption("All data is read from local results/*.jsonl files.")
        selected_run = st.selectbox("Selected run", runs, index=preferred_index)
        records = _records_for_run(selected_run)
        tickers = sorted({str(row.get("ticker")) for row in records if row.get("ticker")})
        ticker = st.selectbox("Ticker", ["All tickers"] + tickers)
        st.caption(f"{len(records)} recorded decisions")
        if selected_run in {"memory_test", "baseline_test"} and len(records) < 15:
            st.warning(
                f"{selected_run} is incomplete ({len(records)}/15 tiny-mode decisions)."
            )
    ticker_filter = None if ticker == "All tickers" else ticker

    all_run_records = {run: _records_for_run(run) for run in runs}
    metrics = compute_run_metrics(records)
    baseline_records = all_run_records.get("baseline_test", [])
    memory_records = all_run_records.get("memory_test", [])
    baseline_metrics = compute_run_metrics(baseline_records)
    memory_metrics = compute_run_metrics(memory_records)
    cards = st.columns(5)
    comparison_complete = len(baseline_records) >= 15 and len(memory_records) >= 15
    if comparison_complete:
        card_values = [
            ("Total return", _pct(memory_metrics["total_return"]), _pct(baseline_metrics["total_return"]), "#00d3a7", False),
            ("Max drawdown", _pct(memory_metrics["max_drawdown"]), _pct(baseline_metrics["max_drawdown"]), "#ff5b61", False),
            ("Win rate", _pct(memory_metrics["win_rate"]), _pct(baseline_metrics["win_rate"]), "#00d3a7", False),
            ("Repeated-mistake rate", _pct(memory_metrics["repeated_mistake_rate"]), _pct(baseline_metrics["repeated_mistake_rate"]), "#4d9cff", True),
            ("Mistakes avoided", str(memory_metrics["losses_avoided"]), str(baseline_metrics["losses_avoided"]), "#4d9cff", False),
        ]
    else:
        card_values = [
            ("Total return", _pct(metrics["total_return"]), "selected run", "#00d3a7", False),
            ("Max drawdown", _pct(metrics["max_drawdown"]), "selected run", "#ff5b61", False),
            ("Win rate", _pct(metrics["win_rate"]), "selected run", "#00d3a7", False),
            ("Repeated-mistake rate", _pct(metrics["repeated_mistake_rate"]), "selected run", "#4d9cff", True),
            ("Mistakes avoided", str(metrics["losses_avoided"]), "selected run", "#4d9cff", False),
        ]
        st.info(
            "Memory vs. no-memory cards are hidden until both test runs contain "
            "the full tiny-mode decision set."
        )
    for column, values in zip(cards, card_values):
        with column:
            if comparison_complete:
                _comparison_card(*values)
            else:
                _metric_card(values[0], values[1], values[3])

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    chart_col, trust_col = st.columns([2, 1.08])
    with chart_col:
        st.markdown("### Equity curve")
        st.caption("100 indexed · selected test-period simulation")
        st.plotly_chart(_equity_figure(all_run_records, selected_run), use_container_width=True)
        st.caption(f"Buy & hold baseline · {_pct(metrics['buy_and_hold_return'])}")
    with trust_col:
        st.markdown("### Agent trust by market regime")
        st.caption("Historical argument accuracy")
        _render_trust(memory_records or records)

    replay_tab, lessons_tab, loss_tab = st.tabs(
        ["Replay", "Lessons over time", "Loss review"]
    )
    with replay_tab:
        _render_replay(records, ticker_filter)
    with lessons_tab:
        _render_lessons(records)
    with loss_tab:
        _render_loss_review(records)


if __name__ == "__main__":
    main()
