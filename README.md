# Strategy Lab #1 — Momentum with Adaptive Volatility Control

**Algorithmic Token · ENTER Invest**

> **Primary academic source:**
> Devanathan, Rueter, Boyd, Candès, Hastie, Kochenderfer, Apoorv, Soronow, Zamkovsky (2026)
> *Single-Asset Adaptive Leveraged Volatility Control*
> BlackRock AI Lab & BlackRock Index Services
> [arXiv:2603.01298](https://arxiv.org/abs/2603.01298) · q-fin.PM · March 2026

Experimental Python prototype accompanying the Strategy Lab #1 article published at [Algorithmic Token on Substack](https://algorithmictoken.substack.com/p/momentum-with-volatility-targeting).

---

## What This Is

This module implements time-series momentum combined with a **proportional-control volatility targeting mechanism**, based on the primary academic source above.

The standard open-loop volatility targeting approach — scaling position size as `target_vol / realized_vol` — has three well-documented failure modes: high turnover, leverage spikes during regime transitions, and no mechanism to correct persistent tracking error. The proportional controller in this implementation addresses all three by adding a feedback correction term that closes the gap between realized and target volatility.

This is a **pseudocode-grade prototype**, not production-ready trading infrastructure. It is intended as a learning artefact and a starting point for further development. See the full article for the strategy rationale, failure modes, and backtest assumptions.

---

## Files

```
strategy_lab_01/
├── strategy_lab_01.py   — core implementation
└── README.md            — this file
```

---

## Installation

```bash
pip install numpy pandas yfinance
```

No other dependencies required.

---

## Quick Start

```python
from strategy_lab_01 import run_backtest

results = run_backtest(
    ticker     = "SPY",
    start      = "2005-01-01",
    end        = "2024-12-31",
    target_vol = 0.10,   # 10% annualised vol target
    Kp         = 0.5,    # proportional gain
)

print(results["performance"])
```

Or run directly from the command line for a quick demo:

```bash
python strategy_lab_01.py
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `target_vol` | `0.10` | Annualised volatility target (10%) |
| `Kp` | `0.5` | Proportional gain — controls feedback aggressiveness |
| `lookback` | `252` | Momentum signal lookback in trading days (1 year) |
| `skip` | `21` | Days skipped to avoid short-term reversal (1 month) |
| `vol_window` | `21` | Realized vol estimation window |
| `w_min` | `0.0` | Minimum weight (0 = long-only, no shorting) |
| `w_max` | `1.5` | Maximum leverage cap (1.5 = 150%) |
| `cost_per_trade_bps` | `7.0` | Round-trip transaction cost in basis points |

### On calibrating `Kp`

`Kp` is the single most important parameter in the feedback controller. The BlackRock paper demonstrates that values in the **0.3–0.7 range** are robust across asset classes and time periods. Important rules:

- Calibrate `Kp` on a **holdout period**, never on the test period
- Higher `Kp` = tighter vol tracking but more turnover and cost drag
- Monitor tracking error in live deployment — if it drifts persistently, recalibrate

---

## What the `run_backtest` Function Returns

```python
{
    "prices":           pd.Series,  # raw close prices
    "returns":          pd.Series,  # daily returns
    "weights":          pd.Series,  # daily position weights
    "strategy_returns": pd.Series,  # gross strategy returns
    "net_returns":      pd.Series,  # after transaction costs
    "performance":      pd.DataFrame, # summary metrics table
    "turnover":         pd.Series,  # daily weight changes
}
```

---

## Known Limitations

- **Single asset only** — multi-asset extension is planned for a future Strategy Lab
- **No execution model** — costs are approximated as proportional to weight change; a proper market impact model is needed for realistic sizing
- **`K_p` is not auto-calibrated** — the validation framework for gain selection will be added in a follow-up commit
- **Weekly rebalancing not enforced** — the backtest runs at daily frequency; for realistic cost modelling, weekly rebalancing is recommended (resample weights to weekly before computing costs)

---

## Planned Extensions

- [ ] Multi-asset portfolio with cross-asset vol targeting
- [ ] `Kp` calibration notebook with walk-forward validation
- [ ] Weekly rebalancing flag
- [ ] Benchmark comparison plots
- [ ] Integration with ENTER Invest backtesting engine (Phase 1)

---

## Risk Disclosure

This code is experimental and provided for educational and research purposes only. Past performance of any modelled strategy is not indicative of future results. All algorithmic trading carries significant financial risk, including the potential total loss of capital. Nothing here constitutes financial advice. ENTER Invest does not manage client funds based on strategies described here unless explicitly contracted to do so.

---

*Algorithmic Token is published by ENTER Invest. [algorithmictoken.substack.com](https://algorithmictoken.substack.com)*

