# How Hindsight Changed HOLD Decisions After Earlier Losses

The first version of my trading agent did something familiar: it made a plausible decision, logged the outcome, and then forgot the outcome before making the next decision. That was enough to produce a backtest, but not enough to produce a system that could learn from repeated mistakes.

I built this project around a narrower question: can an agent use the outcome of an earlier trade as operational context for a later one, without allowing future information to leak into the decision? The answer depends less on clever prompts than on the boundaries around memory. I had to define exactly what became a lesson, when it became available, how it was retrieved, and how the judge was allowed to use it.

The result is a daily, long-only trading research system with a bull analyst, a bear analyst, a judge, a bounded market simulator, and [Hindsight agent memory](https://vectorize.io/what-is-agent-memory). The interesting part is not that the system can store text. It is that every retained memory is tied to a resolved decision, a setup, a market regime, a sector, and a concrete next action.

## A walk-forward system with memory at the boundary

The repository is organized around a small number of deliberately separate responsibilities:

- `src/simulator.py` loads and caches daily OHLCV data, exposes only data through the current date, applies orders, enforces position and drawdown limits, and resolves five-day outcomes.
- `src/indicators.py` computes RSI, moving averages, MACD, volume ratios, volatility, and a compact prompt summary.
- `src/agents.py` runs the bull, bear, and judge calls with structured validation.
- `src/memory.py` adapts Hindsight’s `retain` and `recall` operations to trade lessons.
- `src/backtest.py` advances the simulation one decision date at a time and writes append-only JSONL records.
- `src/metrics.py` measures both financial behavior and memory behavior.
- `dashboard/app.py` replays decisions from local result files without calling live services.

That separation matters because the simulator, not the agent, owns the future. The judge sees a current indicator summary, current arguments, and any lessons recalled from prior resolved trades. It never receives the future rows used to calculate the outcome.

The simulator makes the visibility rule explicit:

```python
def get_visible_data(self, ticker: str) -> pd.DataFrame:
    """Return only data available through the current simulation date."""
    if ticker not in self.data:
        raise KeyError(f"Unknown ticker: {ticker}")
    visible = self.data[ticker].loc[
        self.data[ticker].index <= self.current_date
    ].copy()
    assert visible.index.max() <= self.current_date
    return visible
```

That assertion is intentionally unglamorous. It protects the most important invariant in the project. If a future close, volume value, or indicator reaches the decision prompt, the backtest is invalid regardless of how realistic the resulting chart looks.

## The real design problem was not retention

Calling Hindsight’s API is straightforward. The harder question was deciding what should be retained.

An unstructured note such as “the trade failed” is nearly useless. It is difficult to retrieve precisely, difficult to inspect in a dashboard, and too vague for a later judge. I instead made the lesson a typed `TradeLesson` with a controlled vocabulary:

```python
class TradeLesson(BaseModel):
    ticker: str
    sector: str
    date: str
    setup: Literal[
        "oversold_bounce",
        "breakout",
        "breakdown",
        "mean_reversion",
    ]
    market_regime: Literal[
        "trending_up",
        "trending_down",
        "choppy",
        "high_vol",
    ]
    judge_decision: Literal["BUY", "SELL", "HOLD"]
    winner_side: Literal["bull", "bear", "neither"]
    outcome_pct: float
    lesson: str
```

The complete model also carries the indicator summary and both arguments. That gives Hindsight enough semantic context to retrieve a similar situation while preserving structured fields for metrics and later analysis.

The timing is just as important as the schema. A lesson is not retained when the order is opened. It is retained only after the five-day horizon has resolved. At that point, the simulator has calculated the realized percentage, and the lesson-generation call is constrained to use that outcome. The lesson writer cannot invent a headline, a price pattern, or a reason that was not supplied.

The prompt enforces a format that is short enough to inspect and specific enough to reuse:

```text
"When [setup] appears in a [regime] market for [sector],
[bull/bear] was right because [reason]. Next time: [action]."
```

That format turned out to be more useful than asking for a general reflection. It forces the memory to answer three operational questions: what setup was present, under what conditions, and what should change next time?

## How recall enters a decision

Before each decision, hindsight mode makes two retrievals. The first asks for similar lessons using the ticker, sector, setup, market regime, and current indicator summary. The second asks for a track record: which side has previously won for this setup and regime?

The adapter keeps those queries explicit:

```python
query = (
    f"Similar trade for ticker {ticker}, sector {sector}, setup {setup}, "
    f"market regime {market_regime}, indicators {indicators_summary}"
)
response = self.client.recall(
    bank_id=self.bank_id,
    query=query,
    max_tokens=max(1, k) * 512,
)
```

The returned lessons are not converted into a hidden score. They are passed to the judge as visible context, alongside the bull and bear arguments. The judge must decide whether the memories are relevant and records the subset it actually used in `memories_used`.

That makes the behavior auditable. I can inspect a JSONL row and answer: what did the bull say, what did the bear say, what did Hindsight return, which memories did the judge cite, what action was taken, and what happened five trading days later?

The prompt also tells the judge how to treat negative history:

```text
If recalled lessons show similar setups failed, weigh that heavily
and say so in reasoning. If no memories are relevant, say
'no relevant memory'.
```

This is a small but important distinction. Retrieval is not decision-making. Hindsight can return relevant text; the judge still has to explain whether the text changes the current decision. That prevents the memory layer from becoming an opaque policy engine.

## The failure mode I cared about: repeating a loss

The most useful evaluation signal in this system is not simply total return. It is whether the agent repeats a mistake after a similar loss.

The metrics module tracks a repeated mistake when the same `(ticker, setup, market_regime)` has previously lost by more than one percent and the agent later takes another non-HOLD trade with that key. It also counts losses avoided when the agent chooses HOLD, memories were used, and the counterfactual five-day outcome was negative.

```python
if action == "HOLD" and memories and outcome is not None and outcome < 0:
    losses_avoided += 1
if outcome is not None and outcome < -1:
    prior_losses.add(key)
```

This is not a claim that every memory-informed HOLD is correct. It is an attempt to measure the behavior the memory system is supposed to influence: recognizing a familiar failure mode and declining to repeat it.

The dashboard makes that behavior inspectable through replay. For a selected date, I can see the two arguments, the judge’s reasoning, the recalled lessons, the selected memories, and the eventual outcome. The loss review view then connects a losing trade to the lesson retained from it and to later decisions where that lesson was recalled.

## Avoiding false confidence in the backtest

There are several places where a superficially reasonable implementation could quietly invalidate the experiment.

First, `resolve_trade` is the only component allowed to inspect future rows. It finds the close five trading days after the decision date and writes the realized return onto the trade. The agent never receives that data while making the original decision.

Second, market data caching had to handle the shape returned by `yfinance`. A multi-index column result can produce a malformed CSV if written directly. The loader now flattens columns before writing, uses a single `Date` index label, and rejects a cache whose parsed index is not a `DatetimeIndex`. A stale cache is deleted and downloaded again rather than being silently interpreted as market data.

Third, the Hindsight bank is shared deliberately between a training run and its corresponding test run, but not between memory and no-memory modes. Run names such as `memory_train` and `memory_test` map to the same logical bank base. This makes the experiment resemble a real deployment: lessons accumulate during the training period and remain available during the later test period.

Finally, the result log is append-and-flush rather than a single write at the end. A long run can be interrupted without losing every preceding decision, and the dashboard can inspect partial output while the run is still progressing.

## What I learned from building it

### 1. Memory needs a contract, not just storage

The useful unit was not “a previous conversation.” It was a resolved trade lesson with a typed setup, normalized regime, sector, outcome, and next action. Without that contract, semantic retrieval would return prose that was interesting but difficult to apply.

### 2. Future-data protection belongs below the agent layer

Prompt instructions such as “do not use future information” are necessary but insufficient. The simulator enforces the rule by construction, and tests prove that visible data ends at the current date. The strongest safety boundary is an API that cannot provide the forbidden rows.

### 3. Retrieval quality depends on normalization

Free-text labels such as “bullish uptrend with thin liquidity” are not stable keys. I normalize them into `trending_up`, `trending_down`, `choppy`, or `high_vol` before storing or querying lessons. The same idea applies to setups and sectors. Controlled vocabulary is not bureaucracy here; it is what makes similar situations actually similar.

### 4. A memory system needs failure metrics

Total return can move for many reasons. Repeated-mistake rate and losses avoided are closer to the mechanism I wanted to test. They ask whether recalled experience changed behavior, not just whether the final equity curve happened to improve.

### 5. The dashboard is part of the experiment

If I cannot replay a decision and understand why the judge acted, I do not trust the aggregate result. Local JSONL logs, explicit memory fields, and a day-by-day replay view make the system debuggable rather than merely measurable.

## The practical boundary

This architecture does not make daily market data predictable, and it does not turn a language model into a trading strategy. It gives an agent a durable, queryable history and places that history inside a controlled walk-forward evaluation.

For engineers building memory-enabled agents, that is the part worth carrying over. Start with the event boundary: what becomes knowable only after an outcome? Define the memory schema before choosing retrieval prompts. Keep the decision-maker separate from the memory service. Then measure the behavior memory was intended to change.

The [Hindsight GitHub repository](https://github.com/vectorize-io/hindsight) and the [Hindsight documentation](https://hindsight.vectorize.io/) describe the underlying retain and recall interface. In this project, those primitives became useful only after I wrapped them in domain-specific timing, schemas, normalization, and evaluation. The hard part was not remembering more. It was remembering the right thing at the right time, and proving that the agent had access to nothing else.
