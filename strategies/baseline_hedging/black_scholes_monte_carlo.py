# %%
# Monte Carlo Delta Hedging Simulation
# Combines historical backtesting with Monte Carlo path generation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import yfinance as yf
import scipy.stats as stats
from datetime import datetime

# SET TRANSACTION COSTS:
TRANSACTION_COST_HIST = 0.001
TRANSACTION_COST_MC = 0.001


# %%
# =========================================================================
# SECTION 1: HELPER FUNCTIONS
# =========================================================================

def black_scholes_delta(row, risk_free_rate=0.01):
    """Calculate Black-Scholes delta for an option"""
    S = row['spx_close']
    K = row['strike_price']
    T = (pd.to_datetime(row['exdate']) - pd.to_datetime(row['date'])).days / 365.0
    sigma = row['impl_volatility']
    cp_flag = row['cp_flag']
    
    if not all(np.isfinite([S, K, T, sigma])) or T <= 0 or sigma <= 0:
        return np.nan
    
    d1 = (np.log(S / K) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if cp_flag == 'C':
        return stats.norm.cdf(d1)
    elif cp_flag == 'P':
        return stats.norm.cdf(d1) - 1
    else:
        return np.nan

def black_scholes_call_price(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)  # Intrinsic value at expiry
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def black_scholes_delta_direct(S, K, T, r, sigma, option_type='C'):
    """Calculate delta directly from price and parameters"""
    if T <= 0:
        return 1.0 if S > K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'C':
        return stats.norm.cdf(d1)
    else:  # Put
        return stats.norm.cdf(d1) - 1

# %%
# =========================================================================
# SECTION 2: HISTORICAL BACKTESTING FUNCTION
# =========================================================================

def run_delta_hedge_historical(options_df, position_option=1, contract_size=100, 
                               transaction_cost_pct=0.001):
    """
    Run delta hedging simulation on historical data
    Returns: dictionary with performance metrics
    """
    cash = 0
    underlying_pos = 0
    results = []
    hedging_errors = []
    initial_value = None
    skipped_rows = 0
    processed_rows = 0
    
    for idx, row in options_df.iterrows():
        date = row['date']
        spx_price = row['spx_close']
        delta = row['delta']
        mid_price = row['mid_price']
        
        # Validation
        if not (np.isfinite(spx_price) and np.isfinite(delta) and np.isfinite(mid_price)):
            skipped_rows += 1
            continue
        
        # Delta hedge
        target_underlying = -position_option * delta * contract_size
        trade = target_underlying - underlying_pos
        trade_notional = abs(trade) * spx_price
        tx_cost = trade_notional * transaction_cost_pct
        
        cash -= trade * spx_price + tx_cost
        underlying_pos = target_underlying
        
        # Mark-to-market
        option_value = mid_price * position_option * contract_size
        underlying_value = underlying_pos * spx_price
        total_value = cash + option_value + underlying_value
        
        if not np.isfinite(total_value):
            skipped_rows += 1
            continue
        
        # Calculate hedging error
        if initial_value is None:
            initial_value = total_value
            hedging_error = 0.0
        else:
            hedging_error = total_value - initial_value
        
        hedging_errors.append(hedging_error)
        processed_rows += 1
        
        results.append({
            'date': date,
            'total_value': total_value,
            'hedging_error': hedging_error,
            'tx_cost': tx_cost
        })
    
    results_df = pd.DataFrame(results)
    hedging_errors_clean = [e for e in hedging_errors if np.isfinite(e)]
    
    # Calculate metrics
    if len(hedging_errors_clean) > 1:
        mean_error = np.nanmean(hedging_errors_clean)
        variance_error = np.nanvar(hedging_errors_clean)
        std_error = np.sqrt(variance_error)
        rmshe = np.sqrt(np.nanmean(np.array(hedging_errors_clean)**2))
        
        # CVaR
        alpha = 0.05
        threshold = np.nanpercentile(hedging_errors_clean, alpha * 100)
        tail_errors = [e for e in hedging_errors_clean if e <= threshold]
        cvar = np.nanmean(tail_errors) if len(tail_errors) > 0 else np.nan
        
        # P&L and costs
        total_costs = results_df['tx_cost'].sum()
        finite_values = results_df['total_value'].dropna()
        final_pnl = finite_values.iloc[-1] - finite_values.iloc[0] if len(finite_values) >= 2 else np.nan
        
        return {
            'observations': len(hedging_errors_clean),
            'skipped': skipped_rows,
            'mean_error': mean_error,
            'std_error': std_error,
            'variance': variance_error,
            'rmshe': rmshe,
            'min_error': np.nanmin(hedging_errors_clean),
            'max_error': np.nanmax(hedging_errors_clean),
            'cvar': cvar,
            'tail_obs': len(tail_errors),
            'total_pnl': final_pnl,
            'total_costs': total_costs,
            'cost_ratio': final_pnl / total_costs if total_costs > 0 else np.nan
        }
    else:
        return None

# %%
# =========================================================================
# SECTION 3: MONTE CARLO SIMULATION FUNCTIONS
# =========================================================================

def simulate_price_paths(S0, mu, sigma, T, dt, n_paths, seed=42):
    """
    Simulate stock price paths using geometric Brownian motion
    
    Parameters:
    - S0: initial stock price
    - mu: drift (expected return)
    - sigma: volatility
    - T: time horizon (years)
    - dt: time step (e.g., 1/252 for daily)
    - n_paths: number of simulation paths
    - seed: random seed for reproducibility
    
    Returns:
    - paths: array of shape (n_paths, n_steps+1) containing price paths
    """
    np.random.seed(seed)
    
    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return paths

def run_mc_delta_hedging(price_path, strike, time_to_expiry, risk_free_rate, 
                        volatility, contract_size=100, transaction_cost_pct=0.001):
    """
    Run delta hedging simulation on a single Monte Carlo price path
    
    Parameters:
    - price_path: array of simulated prices
    - strike: option strike price
    - time_to_expiry: total time to expiry (years)
    - risk_free_rate: risk-free rate
    - volatility: implied/model volatility
    - contract_size: shares per contract
    - transaction_cost_pct: transaction cost as percentage
    
    Returns:
    - dict with hedging performance metrics
    """
    cash = 0
    underlying_pos = 0
    hedging_errors = []
    tx_costs = []
    initial_value = None
    
    n_steps = len(price_path)
    dt = time_to_expiry / n_steps
    
    for i, S in enumerate(price_path):
        T_remaining = time_to_expiry - i * dt
        
        if T_remaining <= 0:
            # At expiry
            option_value = max(S - strike, 0) * contract_size
            underlying_value = underlying_pos * S
            total_value = cash + option_value + underlying_value
            
            if initial_value is not None:
                hedging_errors.append(total_value - initial_value)
            break
        
        # Calculate Black-Scholes delta and option price
        delta = black_scholes_delta_direct(S, strike, T_remaining, risk_free_rate, volatility, 'C')
        call_price = black_scholes_call_price(S, strike, T_remaining, risk_free_rate, volatility)
        
        # Delta hedge
        target_underlying = -1 * delta * contract_size
        trade = target_underlying - underlying_pos
        tx_cost = abs(trade) * S * transaction_cost_pct
        
        cash -= trade * S + tx_cost
        underlying_pos = target_underlying
        tx_costs.append(tx_cost)
        
        # Mark-to-market
        option_value = call_price * contract_size
        underlying_value = underlying_pos * S
        total_value = cash + option_value + underlying_value
        
        if initial_value is None:
            initial_value = total_value
            hedging_error = 0.0
        else:
            hedging_error = total_value - initial_value
        
        hedging_errors.append(hedging_error)
    
    # Calculate metrics
    if len(hedging_errors) > 1:
        return {
            'final_pnl': hedging_errors[-1],
            'mean_error': np.mean(hedging_errors),
            'std_error': np.std(hedging_errors),
            'rmshe': np.sqrt(np.mean(np.array(hedging_errors)**2)),
            'max_drawdown': np.min(hedging_errors),
            'max_error': np.max(hedging_errors),
            'total_costs': np.sum(tx_costs),
            'n_rebalances': len(hedging_errors)
        }
    else:
        return None

# %%
# =========================================================================
# SECTION 4: LOAD AND PREPARE HISTORICAL DATA
# =========================================================================

print("="*80)
print("PART 1: HISTORICAL BACKTESTING")
print("="*80)

# Load options dataset
options_full = pl.scan_parquet("../../data/options_dataset.parquet")

# Define date range
start_date = datetime(2007, 1, 1)
end_date = datetime(2008, 12, 31)

# Download SPX data
print("\nDownloading SPX historical data...")
spx_df = yf.download("^SPX", start=start_date, end=end_date)
if isinstance(spx_df.columns, pd.MultiIndex):
    spx_df.columns = spx_df.columns.get_level_values(0)
spx_df = spx_df[['Close']].reset_index()
spx_df.columns = ['date', 'spx_close']

initial_spx = float(spx_df['spx_close'].iloc[0])
print(f"Initial SPX: {initial_spx:.2f}")

# Check and fix strike prices
strikes_available = options_full.select("strike_price").unique().collect()["strike_price"].to_list()

print("\nChecking strike prices...")
print(f"Sample strikes: {sorted(strikes_available)[:10]}")

if min(strikes_available) > 10000:
    print("WARNING: Strikes appear to be scaled by 1000x")
    print("Applying correction: dividing all strikes by 1000")
    
    options_full = options_full.with_columns(
        (pl.col("strike_price") / 1000).alias("strike_price")
    )
    
    strikes_available = options_full.select("strike_price").unique().collect()["strike_price"].to_list()
    print(f"Corrected strikes: {sorted(strikes_available)[:10]}")

# Find ATM strikes
atm_strikes = [s for s in strikes_available if abs(s - initial_spx) / initial_spx < 0.05]
print(f"ATM strikes (within 5%): {len(atm_strikes)} strikes found")

# Filter for ATM call options
if len(atm_strikes) > 0:
    print("\nFiltering for ATM call options...")
    
    options_atm = options_full.filter(
        (pl.col("strike_price").is_in(atm_strikes)) & 
        (pl.col("cp_flag") == "C")
    ).collect().to_pandas()
    
    print(f"After strike/call filtering: {len(options_atm)} rows")
    
    # Filter by date
    options_atm['date'] = pd.to_datetime(options_atm['date'])
    options_atm = options_atm[
        (options_atm['date'] >= start_date) &
        (options_atm['date'] <= end_date)
    ]
    
    print(f"After date filtering: {len(options_atm)} rows")
    
    # Get one option for testing
    unique_optionids = options_atm['optionid'].unique()
    print(f"Found {len(unique_optionids)} unique ATM call options")
    
    if len(unique_optionids) > 0:
        selected_optionid = unique_optionids[0]
        print(f"\nSelected optionid {selected_optionid} for historical backtest")
        
        # Filter for this option
        option_data = options_atm[options_atm['optionid'] == selected_optionid].copy()
        option_data = option_data.merge(spx_df, on='date', how='left')
        option_data = option_data.sort_values('date').reset_index(drop=True)
        
        # Calculate delta and mid price
        option_data['impl_volatility'] = option_data['impl_volatility'].fillna(method='ffill').fillna(method='bfill')
        option_data['delta'] = option_data.apply(black_scholes_delta, axis=1)
        option_data['mid_price'] = (option_data['best_bid'] + option_data['best_offer']) / 2
        
        # Get option details for Monte Carlo
        strike_hist = option_data['strike_price'].iloc[0]
        initial_price_hist = option_data['spx_close'].iloc[0]
        time_to_expiry_hist = (pd.to_datetime(option_data['exdate'].iloc[0]) - 
                               pd.to_datetime(option_data['date'].iloc[0])).days / 365.0
        vol_hist = option_data['impl_volatility'].mean()
        
        print(f"\nOption Details:")
        print(f"  Strike: {strike_hist:.2f}")
        print(f"  Initial SPX: {initial_price_hist:.2f}")
        print(f"  Time to Expiry: {time_to_expiry_hist:.2f} years")
        print(f"  Avg Impl Vol: {vol_hist:.2%}")
        print(f"  Observations: {len(option_data)}")
        
        # Run historical backtest
        print("\nRunning historical delta hedging...")
        hist_metrics = run_delta_hedge_historical(option_data, transaction_cost_pct=TRANSACTION_COST_HIST)
        
        if hist_metrics:
            print("\nHistorical Backtest Results:")
            print(f"  Observations: {hist_metrics['observations']}")
            print(f"  Final P&L: ${hist_metrics['total_pnl']:,.2f}")
            print(f"  Mean Error: ${hist_metrics['mean_error']:,.2f}")
            print(f"  RMSHE: ${hist_metrics['rmshe']:,.2f}")
            print(f"  Transaction Costs: ${hist_metrics['total_costs']:,.2f}")

# %%
# =========================================================================
# SECTION 5: MONTE CARLO SIMULATION
# =========================================================================

print("\n" + "="*80)
print("PART 2: MONTE CARLO SIMULATION")
print("="*80)

# Monte Carlo parameters
S0_mc = initial_price_hist  # Start from historical initial price
K_mc = strike_hist  # Use same strike as historical
T_mc = time_to_expiry_hist  # Same maturity
r_mc = 0.01  # 1% risk-free rate
sigma_mc = vol_hist  # Use historical average vol
mu_mc = 0.10  # 10% expected return (can adjust)

n_paths = 1000  # Number of Monte Carlo paths
dt_mc = 1/252  # Daily time steps

print(f"\nMonte Carlo Parameters:")
print(f"  Initial Price (S0): ${S0_mc:.2f}")
print(f"  Strike (K): ${K_mc:.2f}")
print(f"  Time to Expiry (T): {T_mc:.2f} years")
print(f"  Volatility (σ): {sigma_mc:.2%}")
print(f"  Drift (μ): {mu_mc:.2%}")
print(f"  Risk-free rate (r): {r_mc:.2%}")
print(f"  Number of paths: {n_paths}")
print(f"  Time step: {dt_mc} (daily)")

# Generate price paths
print("\nGenerating Monte Carlo price paths...")
price_paths = simulate_price_paths(S0_mc, mu_mc, sigma_mc, T_mc, dt_mc, n_paths)
print(f"Generated {n_paths} paths with {price_paths.shape[1]} time steps each")

# Run delta hedging on each path
print("\nRunning delta hedging on all Monte Carlo paths...")
mc_results = []

for path_idx in range(n_paths):
    metrics = run_mc_delta_hedging(
        price_paths[path_idx], 
        strike=K_mc,
        time_to_expiry=T_mc,
        risk_free_rate=r_mc,
        volatility=sigma_mc,
        transaction_cost_pct=TRANSACTION_COST_MC
    )
    
    if metrics:
        mc_results.append(metrics)
    
    if (path_idx + 1) % 100 == 0:
        print(f"  Completed {path_idx + 1}/{n_paths} paths...")

mc_results_df = pd.DataFrame(mc_results)

# %%
# =========================================================================
# SECTION 6: PAYOFF ANALYSIS
# =========================================================================
def calculate_option_payoff(final_stock_price, strike, option_type='C', premium_paid=0):
    """
    Calculate option payoff at expiration
    
    Parameters:
    - final_stock_price: stock price at expiry (S_T)
    - strike: strike price (K)
    - option_type: 'C' for call, 'P' for put
    - premium_paid: initial option premium (for profit calculation)
    
    Returns:
    - payoff: intrinsic value at expiry
    - profit: payoff minus premium paid
    """
    if option_type == 'C':
        payoff = max(final_stock_price - strike, 0)
    else:  # Put
        payoff = max(strike - final_stock_price, 0)
    
    profit = payoff - premium_paid
    
    return payoff, profit

# Example for your historical option
strike = option_data['strike_price'].iloc[0]
initial_premium = option_data['mid_price'].iloc[0]
final_spx = option_data['spx_close'].iloc[-1]

payoff, profit = calculate_option_payoff(final_spx, strike, 'C', initial_premium)

print(f"\nOption Payoff Analysis:")
print(f"  Strike: ${strike:.2f}")
print(f"  Initial Premium: ${initial_premium:.2f}")
print(f"  Final SPX Price: ${final_spx:.2f}")
print(f"  Payoff at Expiry: ${payoff:.2f}")
print(f"  Profit/Loss: ${profit:.2f}")
print(f"  Status: {'ITM' if payoff > 0 else 'OTM'}")

def plot_option_payoff_diagram(strike, premium, option_type='C', 
                                spot_price=None, title="Option Payoff Diagram"):
    """
    Create classic option payoff diagram (hockey stick chart)
    """
    # Generate range of stock prices around strike
    price_range = np.linspace(strike * 0.7, strike * 1.3, 100)
    
    payoffs = []
    profits = []
    
    for S in price_range:
        if option_type == 'C':
            payoff = max(S - strike, 0)
        else:  # Put
            payoff = max(strike - S, 0)
        
        profit = payoff - premium
        
        payoffs.append(payoff)
        profits.append(profit)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Payoff diagram (intrinsic value only)
    ax1.plot(price_range, payoffs, linewidth=2, color='blue', label='Payoff')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=strike, color='red', linestyle='--', linewidth=1, label=f'Strike = ${strike:.0f}')
    if spot_price:
        ax1.axvline(x=spot_price, color='green', linestyle='--', linewidth=1, 
                    label=f'Current = ${spot_price:.0f}')
    ax1.fill_between(price_range, 0, payoffs, alpha=0.3)
    ax1.set_xlabel('Stock Price at Expiry ($)')
    ax1.set_ylabel('Payoff ($)')
    ax1.set_title(f'{option_type}all Option Payoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Profit/Loss diagram (includes premium)
    ax2.plot(price_range, profits, linewidth=2, color='green', label='Profit/Loss')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=strike, color='red', linestyle='--', linewidth=1, label=f'Strike = ${strike:.0f}')
    if spot_price:
        ax2.axvline(x=spot_price, color='green', linestyle='--', linewidth=1, 
                    label=f'Current = ${spot_price:.0f}')
    
    # Shade profit/loss regions
    ax2.fill_between(price_range, 0, profits, where=(np.array(profits) >= 0), 
                      alpha=0.3, color='green', label='Profit')
    ax2.fill_between(price_range, 0, profits, where=(np.array(profits) < 0), 
                      alpha=0.3, color='red', label='Loss')
    
    ax2.set_xlabel('Stock Price at Expiry ($)')
    ax2.set_ylabel('Profit/Loss ($)')
    ax2.set_title(f'{option_type}all Option P&L (Premium = ${premium:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Generate for your option
fig = plot_option_payoff_diagram(
    strike=strike, 
    premium=initial_premium, 
    option_type='C',
    spot_price=initial_spx
)
plt.savefig('option_payoff_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

# Add after generating Monte Carlo paths in your existing code

print("\n" + "="*80)
print("MONTE CARLO OPTION PAYOFF ANALYSIS")
print("="*80)

# Calculate payoff at expiry for each path
final_prices = price_paths[:, -1]  # Last price in each path
initial_premium = black_scholes_call_price(S0_mc, K_mc, T_mc, r_mc, sigma_mc)

payoffs = []
profits = []

for final_price in final_prices:
    payoff = max(final_price - K_mc, 0)
    profit = payoff - initial_premium
    
    payoffs.append(payoff)
    profits.append(profit)

payoffs = np.array(payoffs)
profits = np.array(profits)

print(f"\nOption Premium Paid: ${initial_premium:.2f}")
print(f"\nPayoff Distribution:")
print(f"  Mean Payoff: ${payoffs.mean():.2f}")
print(f"  Std Dev: ${payoffs.std():.2f}")
print(f"  Median: ${np.median(payoffs):.2f}")
print(f"  Max: ${payoffs.max():.2f}")
print(f"  Probability ITM: {(payoffs > 0).sum() / len(payoffs) * 100:.1f}%")
print(f"  Probability OTM: {(payoffs == 0).sum() / len(payoffs) * 100:.1f}%")

print(f"\nProfit/Loss Distribution:")
print(f"  Mean Profit: ${profits.mean():.2f}")
print(f"  Std Dev: ${profits.std():.2f}")
print(f"  Median: ${np.median(profits):.2f}")
print(f"  Probability Profitable: {(profits > 0).sum() / len(profits) * 100:.1f}%")
print(f"  Expected Return: {(profits.mean() / initial_premium) * 100:.2f}%")

# Plot payoff distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Payoff histogram
axes[0].hist(payoffs, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(payoffs.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean = ${payoffs.mean():.2f}')
axes[0].set_xlabel('Payoff at Expiry ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Option Payoffs')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Profit/Loss histogram
axes[1].hist(profits, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1].axvline(profits.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean = ${profits.mean():.2f}')
axes[1].set_xlabel('Profit/Loss ($)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Option P&L')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Payoff vs Final Stock Price
axes[2].scatter(final_prices, payoffs, alpha=0.3, s=10)
axes[2].axvline(K_mc, color='red', linestyle='--', linewidth=2, label=f'Strike = ${K_mc:.0f}')
axes[2].plot([K_mc, final_prices.max()], [0, final_prices.max() - K_mc], 
             'r-', linewidth=2, label='Theoretical Payoff')
axes[2].set_xlabel('Final Stock Price ($)')
axes[2].set_ylabel('Option Payoff ($)')
axes[2].set_title('Payoff vs Final Stock Price')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_payoff_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

def compare_hedged_vs_unhedged(mc_results_df, payoffs, profits):
    """
    Compare hedged portfolio P&L vs unhedged option P&L
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Distribution comparison
    axes[0].hist(profits, bins=50, alpha=0.5, label='Unhedged Option P&L', 
                 edgecolor='black', color='blue')
    axes[0].hist(mc_results_df['final_pnl'], bins=50, alpha=0.5, 
                 label='Delta-Hedged P&L', edgecolor='black', color='orange')
    axes[0].axvline(profits.mean(), color='blue', linestyle='--', linewidth=2)
    axes[0].axvline(mc_results_df['final_pnl'].mean(), color='orange', 
                    linestyle='--', linewidth=2)
    axes[0].set_xlabel('P&L ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Hedged vs Unhedged P&L Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Statistics comparison
    comparison_data = {
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Sharpe Ratio (approx)'],
        'Unhedged': [
            profits.mean(),
            profits.std(),
            profits.min(),
            profits.max(),
            profits.mean() / profits.std() if profits.std() > 0 else 0
        ],
        'Delta-Hedged': [
            mc_results_df['final_pnl'].mean(),
            mc_results_df['final_pnl'].std(),
            mc_results_df['final_pnl'].min(),
            mc_results_df['final_pnl'].max(),
            mc_results_df['final_pnl'].mean() / mc_results_df['final_pnl'].std() 
            if mc_results_df['final_pnl'].std() > 0 else 0
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Plot as bar chart
    x = np.arange(len(comp_df))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, comp_df['Unhedged'], width, 
                        label='Unhedged', alpha=0.7, color='blue')
    bars2 = axes[1].bar(x + width/2, comp_df['Delta-Hedged'], width, 
                        label='Delta-Hedged', alpha=0.7, color='orange')
    
    axes[1].set_ylabel('Value')
    axes[1].set_title('Performance Metrics Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comp_df['Metric'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    print("\nHedged vs Unhedged Comparison:")
    print(comp_df.to_string(index=False))
    print(f"\nRisk Reduction: {(1 - mc_results_df['final_pnl'].std() / profits.std()) * 100:.1f}%")
    
    return fig

# Run comparison
compare_hedged_vs_unhedged(mc_results_df, payoffs, profits)
plt.savefig('hedged_vs_unhedged_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
# =========================================================================
# SECTION 6: MONTE CARLO RESULTS ANALYSIS
# =========================================================================

print("\n" + "="*80)
print("MONTE CARLO DELTA HEDGING RESULTS")
print("="*80)
print(f"Successfully completed: {len(mc_results_df)}/{n_paths} paths")

print(f"\nFinal P&L Distribution:")
print(f"  Mean: ${mc_results_df['final_pnl'].mean():,.2f}")
print(f"  Std Dev: ${mc_results_df['final_pnl'].std():,.2f}")
print(f"  Median: ${mc_results_df['final_pnl'].median():,.2f}")
print(f"  Min: ${mc_results_df['final_pnl'].min():,.2f}")
print(f"  Max: ${mc_results_df['final_pnl'].max():,.2f}")
print(f"  5th percentile: ${mc_results_df['final_pnl'].quantile(0.05):,.2f}")
print(f"  95th percentile: ${mc_results_df['final_pnl'].quantile(0.95):,.2f}")

print(f"\nRMSHE Distribution:")
print(f"  Mean: ${mc_results_df['rmshe'].mean():,.2f}")
print(f"  Std Dev: ${mc_results_df['rmshe'].std():,.2f}")
print(f"  Median: ${mc_results_df['rmshe'].median():,.2f}")

print(f"\nTransaction Costs:")
print(f"  Mean: ${mc_results_df['total_costs'].mean():,.2f}")
print(f"  Total across all paths: ${mc_results_df['total_costs'].sum():,.2f}")

print(f"\nRebalancing Frequency:")
print(f"  Mean rebalances per path: {mc_results_df['n_rebalances'].mean():.1f}")

# %%
# =========================================================================
# SECTION 7: COMPARISON AND VISUALIZATION
# =========================================================================

print("\n" + "="*80)
print("COMPARISON: HISTORICAL vs MONTE CARLO")
print("="*80)

if hist_metrics:
    print("\nHistorical (Single Realized Path):")
    print(f"  Final P&L: ${hist_metrics['total_pnl']:,.2f}")
    print(f"  RMSHE: ${hist_metrics['rmshe']:,.2f}")
    print(f"  Transaction Costs: ${hist_metrics['total_costs']:,.2f}")
    
    print("\nMonte Carlo (Average Across Simulated Paths):")
    print(f"  Mean Final P&L: ${mc_results_df['final_pnl'].mean():,.2f}")
    print(f"  Mean RMSHE: ${mc_results_df['rmshe'].mean():,.2f}")
    print(f"  Mean Transaction Costs: ${mc_results_df['total_costs'].mean():,.2f}")
    
    print("\nKey Insight:")
    hist_pnl = hist_metrics['total_pnl']
    mc_pnl_percentile = (mc_results_df['final_pnl'] < hist_pnl).sum() / len(mc_results_df) * 100
    print(f"  The historical P&L of ${hist_pnl:,.2f} falls at the {mc_pnl_percentile:.1f}th percentile")
    print(f"  of the Monte Carlo distribution.")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Sample price paths
axes[0, 0].plot(price_paths[:10].T, alpha=0.5)
axes[0, 0].axhline(y=K_mc, color='r', linestyle='--', label=f'Strike = {K_mc:.0f}')
axes[0, 0].set_xlabel('Time Steps')
axes[0, 0].set_ylabel('Stock Price')
axes[0, 0].set_title('Sample Monte Carlo Price Paths (10 paths shown)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Final P&L distribution
axes[0, 1].hist(mc_results_df['final_pnl'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(mc_results_df['final_pnl'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = ${mc_results_df["final_pnl"].mean():,.0f}')
if hist_metrics:
    axes[0, 1].axvline(hist_metrics['total_pnl'], color='green', linestyle='--', 
                       linewidth=2, label=f'Historical = ${hist_metrics["total_pnl"]:,.0f}')
axes[0, 1].set_xlabel('Final P&L ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Final P&L')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. RMSHE distribution
axes[0, 2].hist(mc_results_df['rmshe'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 2].axvline(mc_results_df['rmshe'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = ${mc_results_df["rmshe"].mean():,.0f}')
if hist_metrics:
    axes[0, 2].axvline(hist_metrics['rmshe'], color='green', linestyle='--', 
                       linewidth=2, label=f'Historical = ${hist_metrics["rmshe"]:,.0f}')
axes[0, 2].set_xlabel('RMSHE ($)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Distribution of RMSHE')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Transaction costs distribution
axes[1, 0].hist(mc_results_df['total_costs'], bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1, 0].axvline(mc_results_df['total_costs'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = ${mc_results_df["total_costs"].mean():,.0f}')
axes[1, 0].set_xlabel('Total Transaction Costs ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Transaction Costs')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Scatter: P&L vs RMSHE
axes[1, 1].scatter(mc_results_df['rmshe'], mc_results_df['final_pnl'], alpha=0.3)
axes[1, 1].set_xlabel('RMSHE ($)')
axes[1, 1].set_ylabel('Final P&L ($)')
axes[1, 1].set_title('P&L vs Hedging Error')
axes[1, 1].grid(True, alpha=0.3)

# 6. Q-Q plot for normality check
from scipy import stats as sp_stats
sp_stats.probplot(mc_results_df['final_pnl'], dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('Q-Q Plot: Final P&L Normality Check')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_delta_hedging_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'monte_carlo_delta_hedging_analysis.png'")
plt.show()

# %%
# =========================================================================
# SECTION 8: SAVE RESULTS
# =========================================================================

# Save Monte Carlo results
mc_results_df.to_csv('monte_carlo_hedging_results.csv', index=False)
print("\nMonte Carlo results saved to 'monte_carlo_hedging_results.csv'")

# Save summary statistics
summary_stats = {
    'Metric': ['Mean P&L', 'Std Dev P&L', 'Median P&L', '5th Percentile P&L', 
               '95th Percentile P&L', 'Mean RMSHE', 'Mean Costs'],
    'Monte Carlo': [
        mc_results_df['final_pnl'].mean(),
        mc_results_df['final_pnl'].std(),
        mc_results_df['final_pnl'].median(),
        mc_results_df['final_pnl'].quantile(0.05),
        mc_results_df['final_pnl'].quantile(0.95),
        mc_results_df['rmshe'].mean(),
        mc_results_df['total_costs'].mean()
    ]
}

if hist_metrics:
    summary_stats['Historical'] = [
        hist_metrics['total_pnl'],
        np.nan,  # Only one realization
        hist_metrics['total_pnl'],
        np.nan,
        np.nan,
        hist_metrics['rmshe'],
        hist_metrics['total_costs']
    ]

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('hedging_comparison_summary.csv', index=False)
print("Summary comparison saved to 'hedging_comparison_summary.csv'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
