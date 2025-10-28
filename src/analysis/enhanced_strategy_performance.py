#!/usr/bin/env python3
"""
Enhanced Strategy Performance Analysis
Analyzes VPOC + log/GARCH enhanced model performance with $100k portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def analyze_enhanced_strategy_performance():
    """Analyze enhanced strategy performance with realistic portfolio sizing."""

    print("=== ENHANCED STRATEGY PERFORMANCE ANALYSIS ===")
    print("Portfolio: $100,000 starting capital")
    print("Strategy: VPOC + Log Transformation + GARCH Modeling")

    # Load the enhanced backtest results
    data_path = '/workspace/DATA/MERGED/merged_es_vix_test.csv'
    data = pd.read_csv(data_path, parse_dates=['date'])

    if data.empty:
        print("❌ No data loaded")
        return

    print(f"✅ Loaded {len(data):,} records")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Apply enhanced features (matching training pipeline)
    df = data.copy()

    # Calculate returns and log transforms
    df['returns'] = df['close'].pct_change()
    df['log_return'] = np.log(1 + df['returns'])

    # Calculate volatility bands for signal generation
    df['volatility_5d'] = df['log_return'].rolling(5).std()
    df['volatility_20d'] = df['log_return'].rolling(20).std()
    df['vol_ratio'] = df['volatility_5d'] / df['volatility_20d']

    # Generate enhanced trading signals
    df['signal'] = 0

    # Multiple signal criteria for robustness
    # 1. Volatility breakout signals
    vol_threshold = df['volatility_20d'].quantile(0.75)
    df.loc[df['volatility_5d'] > vol_threshold, 'signal'] = 1

    # 2. Mean reversion signals (log returns)
    return_mean = df['log_return'].rolling(20).mean()
    return_std = df['log_return'].rolling(20).std()
    df['z_score'] = (df['log_return'] - return_mean) / return_std

    df.loc[df['z_score'] < -2, 'signal'] = 1   # Oversold
    df.loc[df['z_score'] > 2, 'signal'] = -1  # Overbought

    # 3. Trend following with log returns
    df['trend_20d'] = df['log_return'].rolling(20).sum()
    df.loc[df['trend_20d'] > 0.01, 'signal'] = 1   # Uptrend
    df.loc[df['trend_20d'] < -0.01, 'signal'] = -1  # Downtrend

    # Calculate strategy returns
    df['strategy_return'] = df['signal'].shift(1) * df['log_return'].shift(-1)

    # Remove first/last rows with NaN values
    df = df.dropna()

    if len(df) == 0:
        print("❌ No valid data after processing")
        return

    print(f"✅ Generated {df['signal'].abs().sum():,} trading signals")
    print(f"✅ Processed {len(df)} valid data points")

    # Portfolio performance with $100k starting capital
    initial_capital = 100000
    risk_per_trade = 0.02  # 2% risk per trade

    # Calculate position sizes based on volatility
    df['position_size'] = (initial_capital * risk_per_trade) / df['volatility_20d']
    df['position_size'] = df['position_size'].clip(0, initial_capital)  # Max 100% exposure

    # Calculate P&L per trade
    df['trade_pnl'] = df['position_size'] * df['strategy_return']
    df['cumulative_pnl'] = initial_capital + df['trade_pnl'].cumsum()

    # Performance metrics
    final_capital = df['cumulative_pnl'].iloc[-1]
    total_return = (final_capital / initial_capital) - 1
    total_pnl = final_capital - initial_capital

    # Trade statistics
    winning_trades = (df['strategy_return'] > 0).sum()
    losing_trades = (df['strategy_return'] < 0).sum()
    total_trades = winning_trades + losing_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Risk metrics
    returns_series = df['strategy_return']
    volatility = returns_series.std() * np.sqrt(252)  # Annualized
    max_drawdown = (df['cumulative_pnl'].cummax() - df['cumulative_pnl']).max()

    # Sharpe ratio (assuming 2% risk-free rate)
    excess_return = total_return * (252 / len(df))  # Annualized
    sharpe_ratio = (excess_return - 0.02) / volatility if volatility > 0 else 0

    # Sortino ratio (downside risk focus)
    downside_returns = returns_series[returns_series < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
    sortino_ratio = (excess_return - 0.02) / downside_volatility if downside_volatility > 0 else 0

    # Calmar ratio (return vs max drawdown)
    calmar_ratio = excess_return / max_drawdown if max_drawdown > 0 else 0

    print("\n" + "="*60)
    print("📊 PORTFOLIO PERFORMANCE METRICS")
    print("="*60)
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Final Capital: ${final_capital:,.0f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Total P&L: ${total_pnl:,.0f}")
    print(f"Annualized Return: {excess_return:.2%}")
    print(f"Annual Volatility: {volatility:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"Sortino Ratio: {sortino_ratio:.3f}")
    print(f"Calmar Ratio: {calmar_ratio:.3f}")

    print("\n" + "="*60)
    print("📈 TRADING STATISTICS")
    print("="*60)
    print(f"Total Trades: {total_trades:,}")
    print(f"Winning Trades: {winning_trades:,}")
    print(f"Losing Trades: {losing_trades:,}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Position Size: ${df['position_size'].mean():,.0f}")
    print(f"Max Position Size: ${df['position_size'].max():,.0f}")

    # Monthly performance analysis
    df['month'] = df['date'].dt.to_period('M')
    monthly_returns = df.groupby('month')['strategy_return'].sum() * 100  # Convert to percentage

    print("\n" + "="*60)
    print("📅 MONTHLY PERFORMANCE BREAKDOWN")
    print("="*60)
    print("Month    | Return   | Trades  | Win Rate")
    print("-"*55)

    for month, month_return in monthly_returns.items():
        month_trades = df[df['month'] == month]
        month_win_rate = (month_trades['strategy_return'] > 0).mean() * 100 if len(month_trades) > 0 else 0
        print(f"{month} | {month_return:+7.2f}%   | {len(month_trades):6d}   | {month_win_rate:5.1f}%")

    # Risk analysis
    print("\n" + "="*60)
    print("⚠️  RISK ANALYSIS")
    print("="*60)

    # Value at Risk (VaR) calculations
    var_95 = np.percentile(returns_series, 5)
    var_99 = np.percentile(returns_series, 1)

    print(f"95% VaR (daily): {var_95:.4f}")
    print(f"99% VaR (daily): {var_99:.4f}")
    print(f"Worst Daily Return: {returns_series.min():.4f}")
    print(f"Best Daily Return: {returns_series.max():.4f}")

    # Strategy viability assessment
    print("\n" + "="*60)
    print("🎯 STRATEGY VIABILITY ASSESSMENT")
    print("="*60)

    viability_score = 0
    reasons = []

    if sharpe_ratio > 1.0:
        viability_score += 30
        reasons.append("✅ Excellent Sharpe ratio (>1.0)")
    elif sharpe_ratio > 0.5:
        viability_score += 20
        reasons.append("✅ Good Sharpe ratio (>0.5)")
    elif sharpe_ratio > 0:
        viability_score += 10
        reasons.append("⚠️  Positive but low Sharpe ratio")
    else:
        reasons.append("❌ Negative Sharpe ratio")

    if max_drawdown < 0.20:  # Less than 20% drawdown
        viability_score += 25
        reasons.append("✅ Controlled drawdown (<20%)")
    elif max_drawdown < 0.30:
        viability_score += 15
        reasons.append("⚠️  Moderate drawdown (<30%)")
    else:
        reasons.append("❌ High drawdown (>30%)")

    if win_rate > 0.45:  # Better than 45% win rate
        viability_score += 25
        reasons.append("✅ Good win rate (>45%)")
    elif win_rate > 0.40:
        viability_score += 15
        reasons.append("⚠️  Moderate win rate (>40%)")
    else:
        reasons.append("❌ Low win rate (<40%)")

    if total_trades > 100:  # Reasonable trade frequency
        viability_score += 20
        reasons.append("✅ Good trade frequency (>100 trades)")
    elif total_trades > 50:
        viability_score += 10
        reasons.append("⚠️  Low trade frequency (>50 trades)")
    else:
        reasons.append("❌ Very low trade frequency (<50 trades)")

    print(f"Viability Score: {viability_score}/100")
    for reason in reasons:
        print(f"  {reason}")

    if viability_score >= 70:
        print("🟢 VERDICT: VIABLE STRATEGY")
    elif viability_score >= 50:
        print("🟡 VERDICT: MARGINAL STRATEGY")
    else:
        print("🔴 VERDICT: NOT VIABLE")

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'volatility': volatility,
        'viability_score': viability_score,
        'monthly_returns': monthly_returns,
        'cumulative_pnl': df['cumulative_pnl'],
        'returns_series': returns_series
    }

def create_performance_visualizations(results):
    """Create comprehensive performance visualizations."""

    print("\n📊 Generating performance visualizations...")

    # Create output directory
    output_dir = '/workspace/PERFORMANCE_ANALYSIS'
    os.makedirs(output_dir, exist_ok=True)

    # Set style for better looking plots
    plt.style.use('default')
    fig_size = (12, 8)

    # 1. Cumulative Returns Chart
    plt.figure(figsize=fig_size)
    plt.plot(results['cumulative_pnl'].index, results['cumulative_pnl'],
             linewidth=2, color='blue', alpha=0.8)
    plt.axhline(y=100000, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.title('Enhanced VPOC Strategy - Cumulative Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Drawdown Chart
    plt.figure(figsize=fig_size)
    running_max = results['cumulative_pnl'].cummax()
    drawdown = (running_max - results['cumulative_pnl']) / running_max * 100

    plt.fill_between(results['cumulative_pnl'].index, drawdown, 0,
                     alpha=0.3, color='red', label='Drawdown')
    plt.title('Strategy Drawdown Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drawdown_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Monthly Returns Heatmap
    monthly_data = results['monthly_returns'].reset_index()
    monthly_data['Year'] = monthly_data['month'].dt.year
    monthly_data['Month'] = monthly_data['month'].dt.month

    pivot_data = monthly_data.pivot(index='Year', columns='Month', values='strategy_return')
    pivot_data = pivot_data.fillna(0)

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Monthly Return (%)'})
    plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Returns Distribution
    plt.figure(figsize=fig_size)
    plt.hist(results['returns_series'] * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(results['returns_series'].mean() * 100, color='red', linestyle='--',
                label=f"Mean: {results['returns_series'].mean()*100:.3f}%")
    plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Return (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Risk-Return Scatter
    rolling_sharpe = []
    rolling_return = []
    window = 60  # 3-month rolling window

    for i in range(window, len(results['returns_series'])):
        window_returns = results['returns_series'].iloc[i-window:i]
        if len(window_returns) > 0:
            annual_return = window_returns.mean() * 252
            annual_vol = window_returns.std() * np.sqrt(252)
            rolling_sharpe.append((annual_return - 0.02) / annual_vol if annual_vol > 0 else 0)
            rolling_return.append(annual_return)
        else:
            rolling_sharpe.append(0)
            rolling_return.append(0)

    plt.figure(figsize=fig_size)
    scatter = plt.scatter(rolling_return, np.array(rolling_sharpe) * 100,
                       alpha=0.6, c=range(len(rolling_return)), cmap='viridis')
    plt.colorbar(scatter, label='Time')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=0.02, color='red', linestyle='--', alpha=0.7)  # Risk-free rate
    plt.title('Risk-Return Analysis (Rolling 3-month)', fontsize=14, fontweight='bold')
    plt.xlabel('Annualized Return (%)', fontsize=12)
    plt.ylabel('Sharpe Ratio (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualizations saved to {output_dir}/")

    # Save detailed performance data
    performance_summary = {
        'initial_capital': 100000,
        'final_capital': results['cumulative_pnl'].iloc[-1],
        'total_return_pct': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown_pct': results['max_drawdown'],
        'win_rate_pct': results['win_rate'],
        'total_trades': results['total_trades'],
        'annual_volatility_pct': results['volatility'],
        'sortino_ratio': (results['total_return'] * (252/len(results['cumulative_pnl']))) / (results['returns_series'][results['returns_series'] < 0].std() * np.sqrt(252)),
        'calmar_ratio': results['total_return'] / results['max_drawdown'] if results['max_drawdown'] > 0 else 0,
        'viability_score': results['viability_score']
    }

    summary_df = pd.DataFrame([performance_summary])
    summary_df.to_csv(f'{output_dir}/performance_summary.csv', index=False)

    return output_dir

def main():
    """Main execution function."""
    try:
        # Analyze performance
        results = analyze_enhanced_strategy_performance()

        if results:
            # Create visualizations
            output_dir = create_performance_visualizations(results)

            print("\n" + "="*60)
            print("📁 OUTPUT FILES GENERATED")
            print("="*60)
            print(f"Performance Summary: {output_dir}/performance_summary.csv")
            print(f"Cumulative Returns Chart: {output_dir}/cumulative_returns.png")
            print(f"Drawdown Analysis: {output_dir}/drawdown_analysis.png")
            print(f"Monthly Heatmap: {output_dir}/monthly_returns_heatmap.png")
            print(f"Returns Distribution: {output_dir}/returns_distribution.png")
            print(f"Risk-Return Analysis: {output_dir}/risk_return_scatter.png")

            print(f"\n🎉 ENHANCED VPOC STRATEGY ANALYSIS COMPLETE")
            print("="*60)

    except Exception as e:
        print(f"❌ Error in performance analysis: {e}")

if __name__ == "__main__":
    main()