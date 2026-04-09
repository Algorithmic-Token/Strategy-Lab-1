"""
Strategy Lab #1 — Momentum with Adaptive Volatility Control
============================================================
Algorithmic Token · ENTER Invest

Implements time-series momentum combined with a proportional-control
volatility targeting mechanism, based on:

    Devanathan et al. (2026) — "Single-Asset Adaptive Leveraged Volatility Control"
    BlackRock AI Lab · arXiv:2603.01298

This is an experimental prototype. See risk disclosure at the bottom of this file
and in the accompanying Strategy Lab #1 article at Algorithmic Token.

Dependencies: numpy, pandas, yfinance
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Signal and volatility estimators
# ---------------------------------------------------------------------------

def compute_momentum_signal(prices: pd.Series,
                             lookback: int = 252,
                             skip: int = 21) -> pd.Series:
    """
    Time-series momentum signal over [t-lookback, t-skip].

    Normalised by long-run volatility to make signal magnitude comparable
    across assets and time periods.

    Parameters
    ----------
    prices   : pd.Series — daily close prices
    lookback : int       — total lookback in trading days (default 252 = 1 year)
    skip     : int       — recent days to skip to avoid short-term reversal
                           contamination (default 21 = 1 month)

    Returns
    -------
    pd.Series — scaled momentum signal (positive = long bias, negative = short)
    """
    log_returns = np.log(prices).diff()
    raw_signal  = log_returns.rolling(lookback - skip).sum().shift(skip)
    long_vol    = log_returns.rolling(lookback).std() * np.sqrt(252)
    return raw_signal / long_vol


def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Annualised realised volatility over a trailing window.

    Parameters
    ----------
    returns : pd.Series — daily returns
    window  : int       — rolling window in trading days (default 21 = 1 month)

    Returns
    -------
    pd.Series — annualised volatility estimate
    """
    return returns.rolling(window).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Core weighting function — adaptive volatility control
# ---------------------------------------------------------------------------

def adaptive_vol_weight(signal: pd.Series,
                        realized: pd.Series,
                        target_vol: float = 0.10,
                        Kp: float = 0.5,
                        w_min: float = 0.0,
                        w_max: float = 1.5) -> pd.Series:
    """
    Combine a momentum signal with proportional-control volatility targeting.

    The standard open-loop weight (signal × target_vol / realized_vol) is
    augmented by a proportional correction term that closes the gap between
    realized volatility and the target — the core mechanism from Devanathan
    et al. (2026).

    Parameters
    ----------
    signal     : pd.Series — scaled momentum signal
    realized   : pd.Series — annualised realised volatility
    target_vol : float     — annualised volatility target (default 0.10 = 10%)
    Kp         : float     — proportional gain; controls correction aggressiveness
                             (0.3–0.7 is a reasonable starting range)
    w_min      : float     — minimum weight / leverage floor (default 0.0 = no short)
    w_max      : float     — maximum weight / leverage cap   (default 1.5 = 150%)

    Returns
    -------
    pd.Series — final position weights, clipped to [w_min, w_max]
    """
    # Open-loop weight: standard vol-scaling
    w_ol = np.sign(signal) * (target_vol / realized)

    # Tracking error: signed gap between target and current realized vol
    tracking_error = target_vol - realized

    # Proportional correction
    w_correction = Kp * (tracking_error / target_vol)

    # Final weight with leverage constraints
    return np.clip(w_ol + w_correction, w_min, w_max)


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def performance_summary(strategy_returns: pd.Series,
                         benchmark_returns: pd.Series = None) -> pd.DataFrame:
    """
    Compute key performance metrics for a return series.

    Parameters
    ----------
    strategy_returns  : pd.Series — daily strategy returns
    benchmark_returns : pd.Series — optional benchmark for comparison

    Returns
    -------
    pd.DataFrame — metrics table
    """
    def metrics(r, label):
        r = r.dropna()
        ann_return = r.mean() * 252
        ann_vol    = r.std() * np.sqrt(252)
        sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan
        cum        = (1 + r).cumprod()
        drawdown   = (cum / cum.cummax()) - 1
        max_dd     = drawdown.min()
        calmar     = ann_return / abs(max_dd) if max_dd != 0 else np.nan
        return {
            "Label":          label,
            "Ann. Return":    f"{ann_return:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio":   f"{sharpe:.3f}",
            "Max Drawdown":   f"{max_dd:.2%}",
            "Calmar Ratio":   f"{calmar:.3f}",
        }

    rows = [metrics(strategy_returns, "Strategy")]
    if benchmark_returns is not None:
        rows.append(metrics(benchmark_returns, "Benchmark"))

    return pd.DataFrame(rows).set_index("Label")


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest(ticker: str = "SPY",
                 start: str = "2005-01-01",
                 end: str = "2024-12-31",
                 target_vol: float = 0.10,
                 Kp: float = 0.5,
                 lookback: int = 252,
                 skip: int = 21,
                 vol_window: int = 21,
                 w_min: float = 0.0,
                 w_max: float = 1.5,
                 cost_per_trade_bps: float = 7.0) -> dict:
    """
    End-to-end backtest for a single asset.

    Downloads price data via yfinance, computes signals and weights,
    applies a simple per-trade transaction cost, and returns a results dict.

    Parameters
    ----------
    ticker              : str   — Yahoo Finance ticker symbol
    start / end         : str   — backtest period (YYYY-MM-DD)
    target_vol          : float — annualised vol target
    Kp                  : float — proportional gain
    lookback / skip     : int   — momentum signal parameters
    vol_window          : int   — realized vol estimation window
    w_min / w_max       : float — leverage constraints
    cost_per_trade_bps  : float — round-trip transaction cost in basis points
                                  (default 7bps = 5bps commission + 2bps slippage)

    Returns
    -------
    dict with keys: prices, returns, weights, strategy_returns,
                    net_returns, performance, turnover
    """
    # --- Data ---
    raw     = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
    prices  = raw["Close"].squeeze()
    returns = prices.pct_change()

    # --- Signal and weights ---
    signal  = compute_momentum_signal(prices, lookback=lookback, skip=skip)
    rvol    = realized_vol(returns, window=vol_window)
    weights = adaptive_vol_weight(signal, rvol,
                                  target_vol=target_vol, Kp=Kp,
                                  w_min=w_min, w_max=w_max)

    # --- Gross strategy returns (no look-ahead) ---
    strategy_returns = weights.shift(1) * returns

    # --- Transaction costs (proportional to weight change) ---
    turnover         = weights.diff().abs()
    cost_per_day     = turnover * (cost_per_trade_bps / 10_000)
    net_returns      = strategy_returns - cost_per_day

    # --- Performance ---
    perf = performance_summary(net_returns.dropna(), returns.dropna())

    return {
        "prices":            prices,
        "returns":           returns,
        "weights":           weights,
        "strategy_returns":  strategy_returns,
        "net_returns":       net_returns,
        "performance":       perf,
        "turnover":          turnover,
    }


# ---------------------------------------------------------------------------
# Entry point — quick demo run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Strategy Lab #1 — Momentum + Adaptive Volatility Control")
    print("Algorithmic Token · ENTER Invest")
    print("=" * 60)
    print()

    results = run_backtest(
        ticker    = "SPY",
        start     = "2005-01-01",
        end       = "2024-12-31",
        target_vol = 0.10,
        Kp         = 0.5,
    )

    print(results["performance"].to_string())
    print()
    print(f"Mean daily turnover : {results['turnover'].mean():.4f}")
    print(f"Data points         : {results['net_returns'].dropna().shape[0]}")
    print()
    print("NOTE: This is an experimental prototype.")
    print("See risk disclosure in the accompanying article.")


# ---------------------------------------------------------------------------
# Risk Disclosure
# ---------------------------------------------------------------------------
# The strategies and implementations in this file are experimental and
# provided for educational and research purposes only. Past performance
# is not indicative of future results. All algorithmic trading carries
# significant financial risk, including the potential total loss of capital.
# Nothing here constitutes financial advice. ENTER Invest does not manage
# client funds based on strategies described here unless explicitly contracted.
# ---------------------------------------------------------------------------
