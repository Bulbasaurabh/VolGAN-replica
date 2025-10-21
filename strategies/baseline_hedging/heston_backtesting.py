import pandas as pd
import numpy as np
import polars as pl
import scipy.stats as stats
from datetime import datetime
from scipy.integrate import quad
from scipy.optimize import minimize


# -------------------------
# 1. Define hedging function
# -------------------------
def run_delta_hedge(options_df, position_option=1, contract_size=100, transaction_cost_pct=0.001):
    """
    Run delta hedging simulation on a single option
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

# -------------------------
# 2. Prepare data and select multiple options
# -------------------------

# Load full dataset
options_full = pl.scan_parquet("../../data/options_dataset.parquet")

# Define your date range and SPX reference
start_date = datetime(2000, 1, 1)
end_date = datetime(2000, 12, 31)

# Download SPX data
import yfinance as yf
spx_df = yf.download("^SPX", start=start_date, end=end_date)
if isinstance(spx_df.columns, pd.MultiIndex):
    spx_df.columns = spx_df.columns.get_level_values(0)
spx_df = spx_df[['Close']].reset_index()
spx_df.columns = ['date', 'spx_close']

# Get initial SPX for ATM strike finding
initial_spx = float(spx_df['spx_close'].iloc[0])
print(f"Initial SPX: {initial_spx:.2f}")


# Find ATM strikes (within 5%)
strikes_available = options_full.select("strike_price").unique().collect()["strike_price"].to_list()
# Check if strikes need scaling
print("\nChecking strike prices...")
print(f"Sample strikes: {sorted(strikes_available)[:10]}")
print(f"Initial SPX: {initial_spx:.2f}")

# If all strikes are > 10,000, they likely need scaling
if min(strikes_available) > 10000:
    print("WARNING: Strikes appear to be scaled by 1000x")
    print("Applying correction: dividing all strikes by 1000")
    
    # Apply scaling to the full dataset BEFORE filtering
    options_full = options_full.with_columns(
        (pl.col("strike_price") / 1000).alias("strike_price")
    )
    
    # Recalculate strikes_available
    strikes_available = options_full.select("strike_price").unique().collect()["strike_price"].to_list()
    print(f"Corrected strikes: {sorted(strikes_available)[:10]}")

atm_strikes = [s for s in strikes_available if abs(s - initial_spx) / initial_spx < 0.05]
print(f"ATM strikes: {sorted(atm_strikes)}")


# Filter for ATM call options 
if len(atm_strikes) > 0:
    print("\nFiltering for ATM call options...")
    
    # Filter only by strike and call/put in Polars
    options_atm = options_full.filter(
        (pl.col("strike_price").is_in(atm_strikes)) & 
        (pl.col("cp_flag") == "C")
    ).collect().to_pandas()
    
    print(f"After strike/call filtering: {len(options_atm)} rows")
    
    # Convert date column to datetime
    options_atm['date'] = pd.to_datetime(options_atm['date'])
    
    # NOW filter by date range using pandas
    options_atm = options_atm[
        (options_atm['date'] >= start_date) &
        (options_atm['date'] <= end_date)
    ]
    
    print(f"After date filtering: {len(options_atm)} rows")
    
    # Get unique option IDs
    unique_optionids = options_atm['optionid'].unique()
    print(f"Found {len(unique_optionids)} unique ATM call options")
    
    # Select multiple options (e.g., first 5-10)
    selected_optionids = unique_optionids[:10]  # Adjust number as needed
else:
    print("No ATM options found!")
    selected_optionids = []

print(f"\nPreparing to test {len(selected_optionids)} options...")


# -------------------------
# 3. Run backtests for each option
# -------------------------

backtest_results = []

for optionid in selected_optionids:
    print(f"\n{'='*60}")
    print(f"Testing optionid: {optionid}")
    print('='*60)
    
    # Filter for this specific option
    option_data = options_atm[options_atm['optionid'] == optionid].copy()
    
    # Skip if insufficient data
    if len(option_data) < 20:
        print(f"Skipping - insufficient data ({len(option_data)} rows)")
        continue
    
    # Merge with SPX data
    option_data = option_data.merge(spx_df, on='date', how='left')
    option_data['date'] = pd.to_datetime(option_data['date'])
    option_data = option_data.sort_values('date').reset_index(drop=True)
    
    # Fill missing impl_volatility
    option_data['impl_volatility'] = option_data['impl_volatility'].fillna(method='ffill').fillna(method='bfill')
    

def heston_characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, tau, r):
    """
    Heston characteristic function for option pricing
    
    Parameters:
    - phi: frequency parameter
    - S0: current stock price
    - v0: current variance
    - kappa: mean reversion speed
    - theta: long-term variance
    - sigma: volatility of volatility
    - rho: correlation between price and volatility
    - tau: time to maturity
    - r: risk-free rate
    """
    # Complex number calculations
    d = np.sqrt((rho * sigma * phi * 1j - kappa)**2 + sigma**2 * (phi * 1j + phi**2))
    g = (kappa - rho * sigma * phi * 1j - d) / (kappa - rho * sigma * phi * 1j + d)
    
    C = r * phi * 1j * tau + (kappa * theta) / (sigma**2) * \
        ((kappa - rho * sigma * phi * 1j - d) * tau - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    
    D = (kappa - rho * sigma * phi * 1j - d) / (sigma**2) * \
        ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
    
    return np.exp(C + D * v0 + 1j * phi * np.log(S0))

def heston_call_price(S0, K, v0, kappa, theta, sigma, rho, tau, r):
    """
    Calculate European call option price using Heston model
    """
    # Integration for probability P1
    def integrand_P1(phi):
        numerator = np.exp(-1j * phi * np.log(K)) * heston_characteristic_function(
            phi - 1j, S0, v0, kappa, theta, sigma, rho, tau, r)
        denominator = 1j * phi * heston_characteristic_function(
            -1j, S0, v0, kappa, theta, sigma, rho, tau, r)
        return np.real(numerator / denominator)
    
    # Integration for probability P2
    def integrand_P2(phi):
        numerator = np.exp(-1j * phi * np.log(K)) * heston_characteristic_function(
            phi, S0, v0, kappa, theta, sigma, rho, tau, r)
        denominator = 1j * phi
        return np.real(numerator / denominator)
    
    P1 = 0.5 + (1/np.pi) * quad(integrand_P1, 0, 100)[0]
    P2 = 0.5 + (1/np.pi) * quad(integrand_P2, 0, 100)[0]
    
    # Call price
    call_price = S0 * P1 - K * np.exp(-r * tau) * P2
    return call_price

def heston_delta(row, v0_initial=0.04, kappa=2.0, theta=0.04, sigma_vol=0.3, rho=-0.7, risk_free_rate=0.01):
    """
    Calculate delta using Heston model via finite difference
    
    Heston parameters:
    - v0: initial variance (squared volatility)
    - kappa: mean reversion speed
    - theta: long-term variance
    - sigma_vol: volatility of volatility
    - rho: correlation between asset and volatility (-1 to 1)
    """
    S = row['spx_close']
    K = row['strike_price']
    T = (pd.to_datetime(row['exdate']) - pd.to_datetime(row['date'])).days / 365.0
    cp_flag = row['cp_flag']
    
    # Check for invalid inputs
    if not all(np.isfinite([S, K, T])) or T <= 0:
        return np.nan
    
    # Estimate v0 from implied volatility if available
    if np.isfinite(row['impl_volatility']) and row['impl_volatility'] > 0:
        v0 = row['impl_volatility']**2  # Convert vol to variance
    else:
        v0 = v0_initial
    
    # Calculate delta using finite difference (bump and reprice)
    epsilon = 0.01  # 1 cent bump
    
    try:
        if cp_flag == 'C':
            price_up = heston_call_price(S + epsilon, K, v0, kappa, theta, sigma_vol, rho, T, risk_free_rate)
            price_down = heston_call_price(S - epsilon, K, v0, kappa, theta, sigma_vol, rho, T, risk_free_rate)
        elif cp_flag == 'P':
            # Put-call parity: Put delta = Call delta - 1
            price_up = heston_call_price(S + epsilon, K, v0, kappa, theta, sigma_vol, rho, T, risk_free_rate)
            price_down = heston_call_price(S - epsilon, K, v0, kappa, theta, sigma_vol, rho, T, risk_free_rate)
            call_delta = (price_up - price_down) / (2 * epsilon)
            return call_delta - 1  # Put delta
        else:
            return np.nan
        
        delta = (price_up - price_down) / (2 * epsilon)
        return delta
    
    except:
        return np.nan

# NEW: Heston delta
print("Calculating Heston deltas (this may take longer)...")
option_data['delta'] = option_data.apply(heston_delta, axis=1)
option_data['mid_price'] = (option_data['best_bid'] + option_data['best_offer']) / 2

# Get option details
strike = option_data['strike_price'].iloc[0]
expiry = option_data['exdate'].iloc[0]
initial_spx_price = option_data['spx_close'].iloc[0]
moneyness = initial_spx_price / strike

print(f"Strike: {strike}, Expiry: {expiry}")
print(f"Initial SPX: {initial_spx_price:.2f}, Moneyness: {moneyness:.3f}")
print(f"Total observations: {len(option_data)}")

# DAILY hedging
print("\nRunning DAILY hedging...")
daily_metrics = run_delta_hedge(option_data)

# WEEKLY hedging
print("Running WEEKLY hedging...")
option_data['day_of_week'] = option_data['date'].dt.dayofweek
option_data_weekly = option_data[option_data['day_of_week'] == 0].reset_index(drop=True)

if len(option_data_weekly) < 5:
    print(f"Insufficient weekly data ({len(option_data_weekly)} rows), skipping weekly test")
    weekly_metrics = None
else:
    weekly_metrics = run_delta_hedge(option_data_weekly)

# Store results
if daily_metrics or weekly_metrics:
    backtest_results.append({
        'optionid': optionid,
        'strike': strike,
        'expiry': expiry,
        'moneyness': moneyness,
        'total_days': len(option_data),
        'daily': daily_metrics,
        'weekly': weekly_metrics
    })

# -------------------------
# 4. Aggregate and compare results
# -------------------------

print("\n" + "="*80)
print("AGGREGATE BACKTEST RESULTS")
print("="*80)

# Create comparison DataFrame
comparison_data = []

for result in backtest_results:
    daily = result['daily']
    weekly = result['weekly']
    
    row = {
        'optionid': result['optionid'],
        'strike': result['strike'],
        'moneyness': result['moneyness'],
        'daily_obs': daily['observations'] if daily else np.nan,
        'daily_pnl': daily['total_pnl'] if daily else np.nan,
        'daily_costs': daily['total_costs'] if daily else np.nan,
        'daily_rmshe': daily['rmshe'] if daily else np.nan,
        'weekly_obs': weekly['observations'] if weekly else np.nan,
        'weekly_pnl': weekly['total_pnl'] if weekly else np.nan,
        'weekly_costs': weekly['total_costs'] if weekly else np.nan,
        'weekly_rmshe': weekly['rmshe'] if weekly else np.nan,
    }
    
    # Calculate improvements
    if daily and weekly:
        row['cost_savings'] = daily['total_costs'] - weekly['total_costs']
        row['cost_savings_pct'] = (row['cost_savings'] / daily['total_costs']) * 100
        row['pnl_diff'] = weekly['total_pnl'] - daily['total_pnl']
        row['rmshe_diff'] = weekly['rmshe'] - daily['rmshe']
    
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# Display summary
print("\nComparison Summary:")
print(comparison_df.to_string())

# Calculate averages
print("\n" + "="*80)
print("AVERAGE METRICS ACROSS ALL OPTIONS")
print("="*80)

print("\nDaily Hedging Averages:")
print(f"  Mean P&L: ${comparison_df['daily_pnl'].mean():,.2f}")
print(f"  Mean Costs: ${comparison_df['daily_costs'].mean():,.2f}")
print(f"  Mean RMSHE: ${comparison_df['daily_rmshe'].mean():,.2f}")

print("\nWeekly Hedging Averages:")
print(f"  Mean P&L: ${comparison_df['weekly_pnl'].mean():,.2f}")
print(f"  Mean Costs: ${comparison_df['weekly_costs'].mean():,.2f}")
print(f"  Mean RMSHE: ${comparison_df['weekly_rmshe'].mean():,.2f}")

print("\nCost Savings (Daily → Weekly):")
print(f"  Mean Cost Savings: ${comparison_df['cost_savings'].mean():,.2f}")
print(f"  Mean Cost Savings %: {comparison_df['cost_savings_pct'].mean():.1f}%")
print(f"  Mean P&L Difference: ${comparison_df['pnl_diff'].mean():,.2f}")
print(f"  Mean RMSHE Increase: ${comparison_df['rmshe_diff'].mean():,.2f}")

# Statistical test: Is weekly consistently better/worse?
pnl_improvements = comparison_df['pnl_diff'].dropna()
if len(pnl_improvements) > 2:
    t_stat, p_value = stats.ttest_1samp(pnl_improvements, 0)
    print(f"\nStatistical test (P&L difference):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.3f}")
    if p_value < 0.05:
        direction = "better" if pnl_improvements.mean() > 0 else "worse"
        print(f"  Result: Weekly hedging is statistically significantly {direction}")
    else:
        print(f"  Result: No significant difference between daily and weekly")

# Save results
comparison_df.to_csv('heston_hedging_backtest_results.csv', index=False)
print("\nResults saved to 'heston_hedging_backtest_results.csv'")
