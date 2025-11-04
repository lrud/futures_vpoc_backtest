# Phase 1 Feature Implementation Plan
## Enhancing Existing Codebase with 22+ Sophisticated Features

### Executive Summary
This plan outlines how to enhance our existing `src/ml/feature_engineering_robust.py` with 22+ Phase 1 features from our research plan. We'll modify existing files rather than write new scripts, maintaining code consistency and leveraging our proven stable infrastructure.

---

## Current Codebase Analysis

### **Primary Files to Enhance**

1. **`src/ml/feature_engineering_robust.py`** ⭐ **Main Target**
   - Current: 10 basic features (close_change_pct, vwap, price_range, etc.)
   - Target: 22+ sophisticated features (RSI, MACD, Bollinger Bands, etc.)

2. **`src/ml/train_enhanced_robust.py`**
   - Uses feature_engineering_robust.py for data preparation
   - Will automatically benefit from enhanced features

3. **`src/core/data.py`**
   - Data loading and preprocessing utilities
   - May need minor adjustments for new features

### **Supporting Files** (No changes needed)
- `src/ml/model_robust.py` - Model architecture (ready for enhanced features)
- `src/config/conservative_training_config.py` - Training configurations
- Training scripts - Will use enhanced features automatically

---

## Phase 1 Feature Enhancement Strategy

### **Current Features (10)**
```python
ENHANCED_FEATURES = [
    'close_change_pct',      # Basic price change
    'vwap',                  # Volume weighted average price
    'price_range',           # High-low range
    'price_mom_3d',          # 3-day momentum
    'price_mom_5d',          # 5-day momentum
    'close_to_vwap_15',      # VWAP deviation 15min
    'close_to_vwap_60',      # VWAP deviation 60min
    'volume_change_1',        # Volume change
    'vwap_15',              # 15-min VWAP
    'vwap_60'               # 60-min VWAP
]
```

### **Phase 1 Enhanced Features (22+)**
```python
PHASE1_FEATURES = [
    # === TECHNICAL INDICATORS (5) ===
    'rsi_14',               # Relative Strength Index
    'macd_line',            # MACD line
    'macd_signal',          # MACD signal line
    'macd_histogram',       # MACD histogram
    'stoch_k',              # Stochastic %K
    'stoch_d',              # Stochastic %D
    'atr_14',               # Average True Range
    'bb_position',          # Bollinger Band position

    # === ADVANCED VOLATILITY FEATURES (8) ===
    'realized_vol_daily',   # Daily realized volatility
    'bipower_var_daily',    # Bipower variation (jump-robust)
    'realized_jump_var',    # Realized jump variance
    'har_1d',              # HAR 1-day volatility
    'har_5d',              # HAR 5-day volatility
    'har_22d',             # HAR 22-day volatility
    'garch_vol',           # GARCH(1,1) conditional volatility
    'vol_regime',          # Volatility regime flag

    # === TIME-BASED FEATURES (3) ===
    'day_of_week',         # Day of week (0-4)
    'session_indicator',   # Session type (already present)
    'time_of_day',         # Intraday time features

    # === MACRO FEATURES (1) ===
    'vix',                 # VIX (already present)

    # === ENHANCED MOMENTUM FEATURES (5) ===
    'close_change_pct',     # Keep existing basic features
    'vwap',
    'price_range',
    'price_mom_3d',
    'price_mom_5d',
    'close_to_vwap_15',
    'close_to_vwap_60',
    'volume_change_1',
    'vwap_15',
    'vwap_60'
]
```

---

## Implementation Plan

### **Step 1: Technical Dependencies**
Add required libraries to existing imports:
```python
# Add to feature_engineering_robust.py imports
import talib  # or pandas_ta as alternative
from arch import arch_model  # For GARCH modeling
```

### **Step 2: Feature Calculation Functions**
Add new methods to `RobustFeatureEngineer` class:

#### **Technical Indicators**
```python
def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI, MACD, Stochastic, ATR, Bollinger Bands"""
    # RSI
    df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(
        df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['macd_line'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist

    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(
        df['high'].values, df['low'].values, df['close'].values,
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    df['stoch_k'] = slowk
    df['stoch_d'] = slowd

    # ATR
    df['atr_14'] = talib.ATR(
        df['high'].values, df['low'].values, df['close'].values, timeperiod=14
    )

    # Bollinger Bands Position
    upper, middle, lower = talib.BBANDS(
        df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df['bb_position'] = (df['close'].values - middle) / (upper - lower)

    return df
```

#### **Advanced Volatility Features**
```python
def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate HAR, GARCH, bipower variation, jump variance"""
    # Daily realized volatility from minute returns
    df['bar_return'] = df['close'].pct_change()

    # Group by date for daily calculations
    daily_groups = df.groupby('date')

    # Realized Volatility
    realized_vol = daily_groups['bar_return'].apply(
        lambda x: np.sqrt(np.sum(np.square(x.dropna())))
    )

    # Bipower Variation (jump-robust)
    bipower_var = daily_groups['bar_return'].apply(
        lambda x: (np.pi/2) * (1/(len(x)-1)) *
                 np.sum(np.abs(x.dropna().iloc[:-1]) * np.abs(x.dropna().iloc[1:]))
    )

    # Merge back to main dataframe
    df = df.merge(realized_vol.rename('realized_vol_daily'),
                   left_on='date', right_index=True, how='left')
    df = df.merge(bipower_var.rename('bipower_var_daily'),
                   left_on='date', right_index=True, how='left')

    # Realized Jump Variance
    df['realized_jump_var'] = np.maximum(0,
        df['realized_vol_daily'] - df['bipower_var_daily'])

    # HAR Features
    df['har_1d'] = df['realized_vol_daily'].rolling(1).mean()
    df['har_5d'] = df['realized_vol_daily'].rolling(5).mean()
    df['har_22d'] = df['realized_vol_daily'].rolling(22).mean()

    # GARCH(1,1) - sample calculation (optimize for performance)
    # Note: This is computationally intensive, consider caching
    try:
        model = arch_model(df['bar_return'].dropna()*100,
                          vol='Garch', p=1, q=1, disp='off')
        res = model.fit(disp='off')
        df.loc[res.conditional_volatility.index, 'garch_vol'] = res.conditional_volatility
    except:
        df['garch_vol'] = np.nan  # Fallback if GARCH fails

    # Volatility Regime
    vol_25 = df['realized_vol_daily'].rolling(60).quantile(0.25)
    vol_75 = df['realized_vol_daily'].rolling(60).quantile(0.75)
    df['vol_regime'] = (df['realized_vol_daily'] > vol_75).astype(int)

    return df
```

#### **Time-Based Features**
```python
def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate day-of-week, session, time-of-day features"""
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Day of week (0=Monday, 4=Friday)
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Time of day (normalized 0-1)
    df['time_of_day'] = df['timestamp'].dt.hour / 24.0 + df['timestamp'].dt.minute / 1440.0

    return df
```

### **Step 3: Integration Points**

#### **Modify `create_features()` Method**
```python
def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature creation with Phase 1 indicators"""
    logger.info("🔧 Creating enhanced Phase 1 features...")

    # Existing basic features (keep for backward compatibility)
    df = self._create_basic_features(df)

    # NEW: Technical indicators
    df = self._calculate_technical_indicators(df)

    # NEW: Advanced volatility features
    df = self._calculate_volatility_features(df)

    # NEW: Time-based features
    df = self._calculate_time_features(df)

    # Feature selection - keep most predictive
    phase1_features = [
        'rsi_14', 'macd_line', 'macd_signal', 'stoch_k', 'atr_14',
        'bb_position', 'realized_vol_daily', 'har_5d', 'vol_regime',
        'day_of_week', 'vix', 'close_change_pct', 'vwap', 'price_range'
    ]

    # Ensure all required features exist
    missing_features = [f for f in phase1_features if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    logger.info(f"✅ Created {len(phase1_features)} Phase 1 features")
    return df
```

#### **Update Feature List**
```python
# Replace existing ENHANCED_FEATURES
PHASE1_FEATURES = [
    'rsi_14', 'macd_line', 'macd_signal', 'stoch_k', 'atr_14',
    'bb_position', 'realized_vol_daily', 'har_5d', 'vol_regime',
    'day_of_week', 'vix', 'close_change_pct', 'vwap', 'price_range'
]  # Start with 14 most predictive, expand to 22+ as needed
```

---

## Testing Strategy

### **Step 4: Gradual Rollout**
1. **Test Technical Indicators Only**
   - Add RSI, MACD, ATR first
   - Validate with ultra-conservative model

2. **Add Volatility Features**
   - HAR, realized volatility
   - Monitor performance impact

3. **Add Time Features**
   - Day-of-week, session effects
   - Full feature set evaluation

### **Step 5: Validation Commands**
```bash
# Test with enhanced features
export PYTHONPATH=/workspace && python src/ml/train_enhanced_robust.py \
  --data DATA/MERGED/merged_es_vix_test.csv \
  --output_dir TRAINING/PHASE1_FEATURES_TEST \
  --epochs 20 \
  --batch_size 16 \
  --learning_rate 0.00001 \
  --hidden_dims 32 16 \
  --dropout_rate 0.4 \
  --weight_decay 0.01 \
  --verbose
```

---

## Expected Performance Impact

### **Before Enhancement**
- **Features**: 10 basic features
- **Accuracy**: 49.43% (random baseline)
- **Model**: Ultra-conservative 1,013 parameters

### **After Phase 1 Enhancement**
- **Features**: 22+ sophisticated features
- **Target Accuracy**: >55% (meaningful predictive power)
- **Model**: Same stable architecture with enhanced inputs

### **Success Metrics**
- ✅ **Training Stability**: Maintain <5% NaN rate
- ✅ **Accuracy Improvement**: >5% absolute improvement
- ✅ **Feature Validation**: Each feature contributes predictive value
- ✅ **Backward Compatibility**: Existing training scripts work unchanged

---

## Implementation Timeline

### **Week 1: Technical Indicators**
- Day 1-2: Add talib dependency and indicator functions
- Day 3-4: Integration testing with basic indicators
- Day 5: Validate RSI, MACD, ATR performance

### **Week 2: Volatility Features**
- Day 1-3: Implement HAR and realized volatility
- Day 4-5: Add GARCH and regime detection
- Week 2: Performance validation

### **Week 3: Integration & Testing**
- Day 1-2: Time-based features
- Day 3-4: Full feature set testing
- Day 5: Performance benchmarking

---

## Risk Mitigation

### **Computational Performance**
- GARCH modeling is intensive → implement caching
- Large feature sets → feature importance analysis
- Memory usage → chunked processing maintained

### **Numerical Stability**
- All features inherit existing robust scaling
- NaN handling from existing codebase
- Gradient clipping maintains training stability

### **Backward Compatibility**
- Existing training scripts unchanged
- Feature selection allows gradual rollout
- Conservative configuration remains available

---

**Bottom Line**: This plan enhances our proven stable codebase with 22+ sophisticated features while maintaining training stability and backward compatibility. The modular approach allows gradual testing and validation of each feature category.