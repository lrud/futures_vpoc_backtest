# Phase 1 Feature Engineering Plan for Minute-Level Futures Data

## Overview
This document provides a step-by-step plan for implementing Phase 1 features in your predictive futures price model using 4+ years of minute-level OHLCV data. It covers exactly what you can build from your dataset, what you must skip (for now), and practical NinjaTrader recommendations for live or expanded data needs.

---

## Your Data Columns
```
timestamp | open | high | low | close | volume | contract | session | date | bar_range | bar_return | VIX
```
**Timespan**: 4+ years | **Granularity**: 1-minute bars | **Features per bar**: 12

---

## Phase 1 Features: Directly Computable Now

### Technical Indicators (with only OHLCV required)
- **RSI (14)**
- **MACD (12,26,9)**
- **Stochastic Oscillator (%K 14, %D 3)**
- **Bollinger Bands Position (Normalized within bands)**
- **ATR (14)**

### Volatility & Regime Features
- **Realized Volatility (RV, daily window on minute returns)**
- **Bipower Variation (BPV, jump-robust volatility on minute returns)**
- **Realized Jump Variance (RJV, RV minus BPV)**
- **HAR Features (1, 5, 22 day rolling RV)**
- **GARCH(1,1) Conditional Volatility**
- **Volatility Regime Flag (high/low based on rolling percentiles)**
- **Day-of-Week Indicator (0=Mon ... 4=Fri)**
- **Session Indicator (already present)**

### Macro Feature
- **VIX (already present!)**

#### *Optional (can compute if you add these columns)*
- **Price-to-VWAP deviation** (can be computed from minute data)

---

## Not Available in Phase 1 (Skip for Now)
- OFI (Order Flow Imbalance) — needs tick data with trade direction
- Volume Imbalance — needs separate bid/ask volume data
- Bid-Ask Spread — needs bid/ask prices per bar
- VPIN, market depth, order book imbalance — needs Level 2 data

---

## Calculation Workflow Example (Python with pandas/talib)
```python
import pandas as pd
import numpy as np
import talib # or pandas_ta for technical indicators
from arch import arch_model

df = load_your_minute_data()

# RSI
df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
# MACD
macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = macd
# Stochastic Oscillator
slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3)
df['Stoch_K'] = slowk
# ATR
df['ATR_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
# Bollinger Bands Position
upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
df['BB_Pos'] = (df['close'] - middle) / (upper - lower)

# Daily realized volatility and bipower variation
minute_returns = df['close'].pct_change()
df['bar_return'] = minute_returns

daily_groups = df.groupby('date')
realized_vol = daily_groups['bar_return'].apply(lambda x: np.sqrt(np.sum(np.square(x.dropna()))))
bi_power = daily_groups['bar_return'].apply(lambda x: (np.pi/2) * (1/(len(x)-1)) * np.sum(np.abs(x.dropna().iloc[:-1]) * np.abs(x.dropna().iloc[1:])))
df = df.merge(realized_vol.rename('RealizedVol_Daily'), left_on='date', right_index=True, how='left')
df = df.merge(bi_power.rename('BiPowerVar_Daily'), left_on='date', right_index=True, how='left')
df['RealizedJumpVar'] = np.maximum(0, df['RealizedVol_Daily'] - df['BiPowerVar_Daily'])

df['HAR_1d'] = df['RealizedVol_Daily'].rolling(1).mean()
df['HAR_5d'] = df['RealizedVol_Daily'].rolling(5).mean()
df['HAR_22d'] = df['RealizedVol_Daily'].rolling(22).mean()

# GARCH(1,1)
model = arch_model(df['bar_return'].dropna()*100, vol='Garch', p=1, q=1)
res = model.fit(disp='off')
df['GARCH_Vol'] = np.nan
df.loc[res.conditional_volatility.index, 'GARCH_Vol'] = res.conditional_volatility

# Volatility regime (high=1, low=0)
vol_25 = df['RealizedVol_Daily'].rolling(60).quantile(0.25)
vol_75 = df['RealizedVol_Daily'].rolling(60).quantile(0.75)
df['Vol_Regime'] = (df['RealizedVol_Daily'] > vol_75).astype(int)
# Day-of-week
import pandas as pd
df['DayOfWeek'] = pd.to_datetime(df['date']).dt.dayofweek
```
**Tip:** Always drop (or forward-fill) rows with any NaN before modeling to prevent leakage or fitting errors.

---

## NinjaTrader: Real-Time Extensions (Live Trading Only)
- Bid-ask spread, OFI, and order book depth require real-time data and live script:
    - Use `OnMarketData()` in NinjaScript for streaming best bid/ask/volume.
    - Use `OnMarketDepth()` for order book (Level 2) but only works in live trading, not historical testing.
- Backtest: Use your existing OHLCV + VIX columns (above features only).
- Live trading: Log bid/ask fields from OnMarketData() for later full feature model.
- If you want true order flow features, you must either:
    - Collect and store live bid/ask/volume for several weeks/months
    - Buy historical tick/bid-ask data for your contracts (see previous recommendations)

---

## How to Extend Later
- When you acquire tick or bid/ask data, add OFI, volume imbalance, and spread features to your model.
- For Level 2 (order book) features, build separate pipelines using OnMarketDepth() output.
- For advanced macro/sentiment/cross-market features (Phase 3), integrate using public APIs, VIX (already present), or third-party feeds.

---

## Summary Table
| Feature                | Status      |
|------------------------|-------------|
| RSI, MACD, ATR, Stoch  | Ready       |
| Bollinger Bands Pos    | Ready       |
| Realized Vol, BPV, RJV | Ready       |
| HAR features           | Ready       |
| GARCH vol              | Ready       |
| Volatility regime      | Ready       |
| VIX, session, DOW      | Ready       |
| OFI, volume imbalance  | Not in data |
| Bid-Ask spread         | Not in data |
| Order book depth       | Not in data |


---

**This implementation plan is optimized for your current dataset. As your live scripts or data acquisition improve, layer in new features by priority.**
