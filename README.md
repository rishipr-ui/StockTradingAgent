# bull-bear-desk

`bull-bear-desk` is a Python 3.11 research project for testing whether
structured bull/bear debate and long-term Hindsight memory can improve
rule-constrained paper-trading decisions. It downloads daily OHLCV data,
calculates technical indicators, asks separate bull and bear agents for
arguments, and sends both arguments plus recalled lessons to a judge agent.

> **Not financial advice:** This project is an educational simulation and
> research tool. It does not provide investment advice, recommendations, or
> guarantees of future returns. The optional Alpaca integration is explicitly
> configured for paper trading only.

## Architecture

![bull-bear-desk architecture](docs/architecture.png)

The main data flow is:

1. `src/simulator.py` loads and caches daily OHLCV data, exposes only rows up
   to the simulator's current date, applies orders and risk controls, and
   resolves five-day outcomes.
2. `src/indicators.py` calculates RSI(14), MACD(12,26,9), SMA20, SMA50,
   volume-spike ratio, and 20-day volatility from visible data only.
3. `src/agents.py` uses the Groq Python SDK for the bull, bear, and JSON
   validated judge calls. Prompt templates live in `src/prompts.py`.
4. `src/memory.py` wraps the Hindsight Python client for retaining and
   recalling structured trade lessons. `NoMemory` provides the baseline.
5. `src/backtest.py` runs the walk-forward loop and writes one JSON object per
   decision to `results/<run_name>.jsonl`.
6. `src/metrics.py` reads those JSONL files and writes performance summaries
   and a comparison CSV.
7. `dashboard/app.py` reads local results and displays equity curves, metrics,
   replay, lessons, agent trust, and loss review. It does not call live APIs.

Supporting directories:

| Path | Purpose |
| --- | --- |
| `data/` | Cached market data files |
| `results/` | Backtest JSONL logs, metric outputs, and LLM cache |
| `scripts/` | Smoke tests and the optional Alpaca paper-trading script |
| `tests/` | Pytest coverage for simulator, indicators, memory, agents, backtest, and metrics |

## How Hindsight memory is used

### Retain

After a five-trading-day horizon is available, a hindsight-mode backtest
creates a `TradeLesson` and retains it in Hindsight. Each retained lesson
contains:

- `ticker`
- `sector`
- `date`
- `setup`: `oversold_bounce`, `breakout`, `breakdown`, or `mean_reversion`
- `market_regime`: `trending_up`, `trending_down`, `choppy`, or `high_vol`
- `indicators_summary`
- `bull_argument`
- `bear_argument`
- `judge_decision`: `BUY`, `SELL`, or `HOLD`
- `winner_side`: `bull`, `bear`, or `neither`
- `outcome_pct`
- `lesson`

The lesson text is natural language so Hindsight can use semantic retrieval.
Lessons are generated in the format:

> When `[setup]` appears in a `[regime]` market for `[sector]`, `[bull/bear]`
> was right because `[reason]`. Next time: `[action]`.

Training and test runs reuse the same logical Hindsight bank when their names
share a base name, for example `memory_train` and `memory_test`. Baseline runs
use `NoMemory` and do not retain or recall lessons.

### Recall

Before each agent decision in hindsight mode, the backtest performs:

- A similar-lesson query containing the ticker, sector, setup, market regime,
  and current indicator summary.
- A track-record query asking which side has been right for the setup and
  market regime.

The returned lesson texts and winning-side history are supplied to the judge
alongside the current bull and bear arguments. The judge is instructed to
weigh similar setups that previously failed heavily and to say
`no relevant memory` when the recalled context is not useful. The decision
records which recalled lessons it used in `memories_used`.

Hindsight failures are retried and converted to empty results so a memory
service outage does not crash the backtest loop.

## Setup

Use Python 3.11:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create `.env` in the project root. The backtest requires Groq credentials;
hindsight mode additionally requires Hindsight credentials:

```dotenv
GROQ_API_KEY=your-groq-key
GROQ_MODEL=your-groq-model
HINDSIGHT_API_KEY=your-hindsight-key
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io
```

The optional `scripts/live_paper_trade.py` also requires:

```dotenv
ALPACA_API_KEY=your-alpaca-paper-key
ALPACA_SECRET_KEY=your-alpaca-paper-secret
```

Never commit `.env` or real credentials. It is ignored by `.gitignore`.

## Reproducing the four runs

Run the training baseline:

```bash
python -m src.backtest --mode none --start 2024-01-01 --end 2024-06-30 --run baseline_train
```

Run the Hindsight training period:

```bash
python -m src.backtest --mode hindsight --start 2024-01-01 --end 2024-06-30 --run memory_train
```

Run the Hindsight test period. This reuses the logical memory bank populated
by `memory_train`; it does not wipe that bank:

```bash
python -m src.backtest --mode hindsight --start 2024-07-01 --end 2024-12-31 --run memory_test
```

Run the no-memory test baseline:

```bash
python -m src.backtest --mode none --start 2024-07-01 --end 2024-12-31 --run baseline_test
```

For a smaller smoke run, append `--fast`; this uses three tickers and every
second trading day:

```bash
python -m src.backtest --mode hindsight --start 2024-07-01 --end 2024-12-31 --run memory_test --fast
```

Compute metrics for all local runs:

```bash
python -m src.metrics
```

This writes `results/metrics_summary.json` and
`results/comparison.csv`. Launch the local dashboard with:

```bash
streamlit run dashboard/app.py
```

## Results

Each run writes `results/<run_name>.jsonl`. The metrics command computes total
return, maximum drawdown, win rate, trade count, repeated-mistake rate,
losses avoided, and an equal-weighted buy-and-hold baseline.

The current checkout includes completed tiny-mode test runs with 15 decision
records each. These results use the `RELIANCE.NS` ticker and are intended for
dashboard demonstration, not statistical evaluation. The full-period commands
above remain the reproducible research configuration.

| Run | Mode | Period | Total return | Max drawdown | Win rate | Repeated-mistake rate | Losses avoided | Buy-and-hold return |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_train` | No memory | 2024-01-01 to 2024-06-30 | generated locally | generated locally | generated locally | generated locally | generated locally | generated locally |
| `memory_train` | Hindsight | 2024-01-01 to 2024-06-30 | generated locally | generated locally | generated locally | generated locally | generated locally | generated locally |
| `memory_test` | Hindsight | 2024-07-01 to 2024-12-31 | **0.00%** | **0.00%** | **50.00%** | **0.00%** | **4** | **-12.12%** |
| `baseline_test` | No memory | 2024-07-01 to 2024-12-31 | **-10.01%** | **10.01%** | **25.00%** | **0.00%** | **2** | **-12.12%** |

The completed test runs recorded 6 trades with memory and 4 without memory.
The memory run finished flat while the no-memory run lost 10.01%; both
improved on the equal-weighted buy-and-hold baseline of -12.12% in this small
sample. The dashboard compares these local JSONL records and can replay each
decision, recalled lesson, and five-day outcome.

## Limitations

- **LLM frozen:** The model, prompts, and provider behavior can change over
  time; cached responses can also make reruns depend on prior local state.
- **Noisy markets:** Technical indicators and short natural-language arguments
  cannot reliably explain or predict all market behavior.
- **Small sample:** Four runs and a limited ticker universe are not enough to
  establish statistical significance or robustness.
- **Daily data only:** Decisions use daily OHLCV bars; intraday movement,
  spreads, slippage, liquidity, gaps, and execution timing are simplified.
- **Synthetic evaluation:** Five-day outcomes and the counterfactual HOLD
  outcomes are research measures, not live portfolio performance.
- **Data and service dependencies:** Results depend on market-data availability,
  Groq responses, and Hindsight retrieval quality.
- **Paper trading only:** The optional Alpaca script is intended for paper
  trading and submits at most one paper order per invocation.
