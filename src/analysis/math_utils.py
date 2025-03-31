"""
Mathematical analysis utilities for trading strategies.
Provides statistical validation, Monte Carlo simulations, and other mathematical tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import os
from datetime import datetime
import scipy.stats as stats

from src.utils.logging import get_logger

logger = get_logger(__name__)

def validate_trend(values: List[float], dates: List, lookback: int = 20) -> Dict:
    """
    Validate if there's a statistically significant trend.
    
    Parameters:
    -----------
    values: List[float]
        List of numerical values to analyze
    dates: List
        Corresponding dates
    lookback: int
        Number of periods to look back
        
    Returns:
    --------
    Dict
        Trend validation results
    """
    # Ensure we have enough data
    if len(values) < lookback:
        logger.warning(f"Not enough data for trend validation: {len(values)} < {lookback}")
        return {
            'valid_trend': False,
            'p_value': 1.0,
            'run_test_p': 1.0,
            'direction': 'neutral',
            'consecutive_up': 0,
            'consecutive_down': 0,
            'confidence': 0,
            'slope': 0,
            'r_squared': 0
        }
        
    # Use recent data based on lookback
    recent_values = values[-lookback:]
    recent_dates = dates[-lookback:]
    
    # Calculate consecutive moves
    diffs = np.diff(recent_values)
    current_up_streak = 0
    current_down_streak = 0
    max_up_streak = 0
    max_down_streak = 0
    
    for diff in diffs:
        if diff > 0:
            current_up_streak += 1
            current_down_streak = 0
            max_up_streak = max(max_up_streak, current_up_streak)
        elif diff < 0:
            current_down_streak += 1
            current_up_streak = 0
            max_down_streak = max(max_down_streak, current_down_streak)
            
    # Linear regression for trend strength
    x = np.arange(len(recent_values))
    y = np.array(recent_values)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    direction = 'up' if slope > 0 else 'down'
    
    # Runs test for randomness
    median_value = np.median(recent_values)
    runs = [1 if v > median_value else 0 for v in recent_values]
    runs_count = 1
    
    for i in range(1, len(runs)):
        if runs[i] != runs[i-1]:
            runs_count += 1
            
    # Calculate expected runs and standard deviation
    n1 = runs.count(1)
    n2 = runs.count(0)
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1 if (n1 + n2) > 0 else 0
    std_runs_denominator = ((n1 + n2)**2 * (n1 + n2 - 1))
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                       std_runs_denominator) if std_runs_denominator > 0 else 0
                       
    # Run test p-value
    run_test_p = 2 * (1 - stats.norm.cdf(abs((runs_count - expected_runs) / std_runs))) if std_runs > 0 else 1.0
    
    # Trend validation criteria
    is_valid_trend = (
        (p_value < 0.1) and          # More lenient p-value threshold
        (r_squared > 0.2) and        # Lower R-squared requirement
        (                            # Reduced consecutive moves requirement
            (direction == 'up' and max_up_streak >= 2) or
            (direction == 'down' and max_down_streak >= 2)
        )
    )
    
    # Alternative validation if basic conditions are met
    if not is_valid_trend and ((max_up_streak >= 3) or (max_down_streak >= 3)):
        is_valid_trend = True  # Valid if we have at least 3 consecutive moves
        
    # Calculate confidence
    if is_valid_trend:
        trend_strength = min(r_squared * 100, 100)  # R-squared as percentage
        statistical_significance = (1 - min(p_value, 0.5) * 2) * 100  # Convert p-value to confidence
        streak_factor = (max(max_up_streak, max_down_streak) / 5) * 100  # Streak factor
        confidence = (trend_strength + statistical_significance + streak_factor) / 3  # Weighted average
    else:
        # Even without validation, assign some confidence based on direction strength
        consecutive_factor = max(max_up_streak, max_down_streak) * 10  # 10% per consecutive move
        confidence = min(consecutive_factor, 50)  # Cap at 50% if not fully validated
        
    return {
        'valid_trend': is_valid_trend,
        'p_value': p_value,
        'run_test_p': run_test_p,
        'direction': direction,
        'consecutive_up': max_up_streak,
        'consecutive_down': max_down_streak,
        'confidence': confidence,
        'slope': slope,
        'r_squared': r_squared
    }

def monte_carlo_simulation(trade_results: List[Dict], 
                          initial_capital: float = 100000, 
                          iterations: int = 1000) -> Tuple[Dict, List[Dict]]:
    """
    Perform Monte Carlo simulation on trade results.
    
    Parameters:
    -----------
    trade_results: List[Dict]
        List of trade result dictionaries with 'profit' key
    initial_capital: float
        Initial capital amount
    iterations: int
        Number of Monte Carlo iterations
        
    Returns:
    --------
    Tuple[Dict, List[Dict]]
        Statistics and raw simulation results
    """
    if not trade_results:
        logger.warning("No trade results provided for Monte Carlo simulation")
        return {}, []
        
    # Extract profits from trades
    profits = [trade['profit'] for trade in trade_results]
    
    # Run simulations
    simulation_results = []
    
    for i in range(iterations):
        # Shuffle profits randomly
        np.random.shuffle(profits)
        
        # Calculate equity curve
        equity = [initial_capital]
        for profit in profits:
            equity.append(equity[-1] + profit)
            
        # Calculate returns and drawdowns
        final_equity = equity[-1]
        return_pct = (final_equity - initial_capital) / initial_capital * 100
        
        # Calculate drawdown
        peak = initial_capital
        drawdown = 0
        
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            drawdown = max(drawdown, dd)
            
        # Store result
        simulation_results.append({
            'final_equity': final_equity,
            'return_pct': return_pct,
            'max_drawdown': drawdown,
            'equity_curve': equity
        })
    
    # Analyze results
    final_equities = [r['final_equity'] for r in simulation_results]
    returns_pct = [r['return_pct'] for r in simulation_results]
    drawdowns = [r['max_drawdown'] for r in simulation_results]
    
    # Calculate confidence intervals
    ci_level = 0.95
    lower_ci_idx = int((1 - ci_level) / 2 * iterations)
    upper_ci_idx = int((1 - (1 - ci_level) / 2) * iterations)
    
    sorted_returns = sorted(returns_pct)
    sorted_drawdowns = sorted(drawdowns)
    
    # Compile statistics
    results = {
        'mean_return': np.mean(returns_pct),
        'median_return': np.median(returns_pct),
        'mean_drawdown': np.mean(drawdowns),
        'median_drawdown': np.median(drawdowns),
        'worst_drawdown': max(drawdowns),
        'best_return': max(returns_pct),
        'worst_return': min(returns_pct),
        'return_ci_lower': sorted_returns[lower_ci_idx],
        'return_ci_upper': sorted_returns[upper_ci_idx],
        'drawdown_ci_lower': sorted_drawdowns[upper_ci_idx],
        'drawdown_ci_upper': sorted_drawdowns[lower_ci_idx],
        'probability_positive': sum(1 for r in returns_pct if r > 0) / iterations * 100
    }
    
    return results, simulation_results

def plot_monte_carlo_results(simulation_results: List[Dict], 
                           initial_capital: float,
                           ci_level: float = 0.95,
                           output_path: Optional[str] = None) -> None:
    """
    Plot Monte Carlo simulation results.
    
    Parameters:
    -----------
    simulation_results: List[Dict]
        Results from monte_carlo_simulation
    initial_capital: float
        Initial capital amount
    ci_level: float
        Confidence interval level (0-1)
    output_path: Optional[str]
        Path to save the visualization
    """
    if not simulation_results:
        logger.warning("No simulation results to plot")
        return
        
    # Extract equity curves
    curves = [r['equity_curve'] for r in simulation_results]
    
    # Find shortest curve length (in case of varying trade counts)
    min_length = min(len(c) for c in curves)
    
    # Trim all curves to shortest length
    trimmed_curves = [c[:min_length] for c in curves]
    
    # Convert to numpy array for easier analysis
    equity_array = np.array(trimmed_curves)
    
    # Calculate statistics for each point in time
    median_curve = np.median(equity_array, axis=0)
    lower_ci_idx = int((1 - ci_level) / 2 * len(curves))
    upper_ci_idx = int((1 - (1 - ci_level) / 2) * len(curves))
    
    # Sort each time point to find percentiles
    sorted_equities = np.sort(equity_array, axis=0)
    lower_ci = sorted_equities[lower_ci_idx, :]
    upper_ci = sorted_equities[upper_ci_idx, :]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot sample of individual paths (semi-transparent)
    sample_size = min(100, len(curves))
    indices = np.random.choice(len(curves), sample_size, replace=False)
    
    for idx in indices:
        plt.plot(curves[idx], color='gray', alpha=0.1)
    
    # Plot confidence intervals and median
    plt.plot(median_curve, color='blue', linewidth=2, label='Median Path')
    plt.plot(lower_ci, color='red', linewidth=1, linestyle='--', 
            label=f'Lower {ci_level*100:.0f}% CI')
    plt.plot(upper_ci, color='green', linewidth=1, linestyle='--',
            label=f'Upper {ci_level*100:.0f}% CI')
    
    # Add initial capital reference
    plt.axhline(y=initial_capital, color='black', linestyle='-', alpha=0.5, 
               label=f'Initial Capital (${initial_capital:,.0f})')
    
    # Labels and title
    plt.title('Monte Carlo Simulation of Trading Strategy', fontsize=16)
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Account Equity', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Monte Carlo visualization saved to {output_path}")
    else:
        plt.show()

def calculate_bayesian_probabilities(values: List[float], 
                                   prior_up: float = 0.5,
                                   lookback: int = 20) -> Dict[str, float]:
    """
    Calculate Bayesian probabilities for market direction.
    
    Parameters:
    -----------
    values: List[float]
        List of price or other values
    prior_up: float
        Prior probability of upward move (0-1)
    lookback: int
        Number of periods to consider
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with probability estimates
    """
    if len(values) < lookback:
        logger.warning(f"Not enough data for Bayesian analysis: {len(values)} < {lookback}")
        return {
            'probability_up': prior_up,
            'probability_down': 1 - prior_up,
            'confidence': 0.5
        }
        
    # Get recent values and calculate changes
    recent_values = values[-lookback:]
    changes = np.diff(recent_values)
    
    # Count ups and downs
    up_count = sum(1 for change in changes if change > 0)
    down_count = sum(1 for change in changes if change < 0)
    equal_count = sum(1 for change in changes if change == 0)
    
    # Calculate total observations (excluding equals)
    total_obs = up_count + down_count
    
    if total_obs == 0:
        return {
            'probability_up': prior_up,
            'probability_down': 1 - prior_up,
            'confidence': 0.5
        }
    
    # Calculate likelihood factors
    up_pct = up_count / total_obs
    down_pct = down_count / total_obs
    
    # Apply Bayes' rule
    # P(Up|Data) = P(Data|Up) * P(Up) / P(Data)
    # where P(Data) = P(Data|Up) * P(Up) + P(Data|Down) * P(Down)
    
    # Likelihood * Prior
    up_posterior = up_pct * prior_up
    down_posterior = down_pct * (1 - prior_up)
    
    # Normalize
    total_posterior = up_posterior + down_posterior
    
    if total_posterior == 0:
        probability_up = prior_up
        probability_down = 1 - prior_up
    else:
        probability_up = up_posterior / total_posterior
        probability_down = down_posterior / total_posterior
    
    # Calculate confidence based on sample size
    confidence = min(1.0, total_obs / lookback)
    
    return {
        'probability_up': probability_up,
        'probability_down': probability_down,
        'confidence': confidence,
        'up_count': up_count,
        'down_count': down_count,
        'equal_count': equal_count
    }