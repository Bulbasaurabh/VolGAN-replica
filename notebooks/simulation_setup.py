import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import yfinance as yf
from scipy.stats import norm

# ===========================
# Configuration
# ===========================
selected_option_id = 10010807
simulation_model = 'black_scholes'  # 'black_scholes' or 'heston'
n_simulations = 1000  # Number of Monte Carlo paths
transaction_cost_pct = 0.001  # 0.1% per trade

# ===========================
# 1. Load Dataset
# ===========================
options_df = pl.scan_parquet("data/options_dataset.parquet")
options_df = options_df.filter(pl.col("optionid") == selected_option_id).collect()
options_df = options_df.to_pandas()

# Preprocess
options_df['date'] = pd.to_datetime(options_df['date'], errors='coerce')
options_df["mid_price"] = (options_df["best_bid"] + options_df["best_offer"]) / 2

# ===========================
# 2. Load SPX Historical Data
# ===========================
ticker = "^SPX"
start_date = options_df["date"].min()
end_date = options_df["date"].max()

spx_df = yf.download(ticker, start=start_date, end=end_date)
spx_df = spx_df[['Close']].reset_index().rename(columns={'Date': 'date', 'Close': 'spx_close'})
spx_df.columns = [col[0] if isinstance(col, tuple) else col for col in spx_df.columns]

options_df = options_df.merge(spx_df, on='date', how='left')

# ===========================
# 3. Extract Option Parameters
# ===========================
S0 = options_df['spx_close'].iloc[0]
K = options_df['strike_price'].iloc[0] / 1000  # Convert to actual strike
T_total = (pd.to_datetime(options_df['exdate'].iloc[0]) - options_df['date'].iloc[0]).days / 365
r = 0.05  # Risk-free rate (5%)
option_type = options_df['cp_flag'].iloc[0]  # 'C' for call

# Calculate historical volatility from SPX data
returns = options_df['spx_close'].pct_change().dropna()
historical_vol = returns.std() * np.sqrt(252)

print(f"Initial Stock Price (S0): {S0:.2f}")
print(f"Strike Price (K): {K:.2f}")
print(f"Time to Maturity (T): {T_total:.4f} years")
print(f"Historical Volatility: {historical_vol:.4f}")
print(f"Risk-free Rate: {r:.4f}")

# ===========================
# 4. Black-Scholes Functions
# ===========================
def black_scholes_price(S, K, T, r, sigma, option_type='C'):
    """Calculate Black-Scholes option price"""
    if T <= 0:
        return max(S - K, 0) if option_type == 'C' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_delta(S, K, T, r, sigma, option_type='C'):
    """Calculate Black-Scholes delta"""
    if T <= 0:
        return 1.0 if S > K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'C':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

# ===========================
# 5. Simulation Functions
# ===========================
def simulate_black_scholes(S0, T, r, sigma, n_steps, n_sims):
    """Simulate stock paths under Black-Scholes (constant volatility)"""
    dt = T / n_steps
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return paths

def simulate_heston(S0, v0, T, r, kappa, theta, sigma_v, rho, n_steps, n_sims):
    """
    Simulate stock paths under Heston stochastic volatility model
    
    Parameters:
    - S0: Initial stock price
    - v0: Initial variance
    - T: Time to maturity
    - r: Risk-free rate
    - kappa: Mean reversion speed
    - theta: Long-term variance
    - sigma_v: Volatility of volatility
    - rho: Correlation between stock and volatility
    - n_steps: Number of time steps
    - n_sims: Number of simulations
    """
    dt = T / n_steps
    paths = np.zeros((n_sims, n_steps + 1))
    variance = np.zeros((n_sims, n_steps + 1))
    
    paths[:, 0] = S0
    variance[:, 0] = v0
    
    for t in range(1, n_steps + 1):
        # Correlated random variables
        Z1 = np.random.standard_normal(n_sims)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_sims)
        
        # Update variance (with full truncation to ensure non-negativity)
        variance[:, t] = np.maximum(
            variance[:, t-1] + kappa * (theta - variance[:, t-1]) * dt + 
            sigma_v * np.sqrt(np.maximum(variance[:, t-1], 0)) * np.sqrt(dt) * Z2,
            0
        )
        
        # Update stock price
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5 * variance[:, t-1]) * dt + 
            np.sqrt(np.maximum(variance[:, t-1], 0)) * np.sqrt(dt) * Z1
        )
    
    return paths, variance

# ===========================
# 6. Delta-Hedge Simulation
# ===========================
def delta_hedge_simulation(stock_paths, T, K, r, sigma, option_type, tx_cost):
    """
    Perform delta-hedging simulation on given stock paths
    """
    n_sims, n_steps = stock_paths.shape
    dt = T / (n_steps - 1)
    
    portfolio_values = np.zeros((n_sims, n_steps))
    
    for sim in range(n_sims):
        cash = 0
        underlying_pos = 0
        
        for t in range(n_steps):
            S = stock_paths[sim, t]
            T_remaining = T - t * dt
            
            # Calculate option value and delta
            option_value = black_scholes_price(S, K, T_remaining, r, sigma, option_type)
            delta = black_scholes_delta(S, K, T_remaining, r, sigma, option_type)
            
            # Delta hedge (short 1 option contract)
            target_underlying = -delta * 100  # 100 shares per contract
            trade = target_underlying - underlying_pos
            trade_cost = abs(trade) * S * tx_cost
            
            cash -= trade * S + trade_cost
            underlying_pos = target_underlying
            
            # Mark-to-market
            portfolio_values[sim, t] = cash + option_value * 100 + underlying_pos * S
    
    return portfolio_values

# ===========================
# 7. Run Simulation
# ===========================
n_steps = len(options_df)
dates = options_df['date'].values

print(f"\nRunning {n_simulations} simulations using {simulation_model.upper()} model...")

if simulation_model == 'black_scholes':
    sigma = historical_vol
    stock_paths = simulate_black_scholes(S0, T_total, r, sigma, n_steps-1, n_simulations)
    
elif simulation_model == 'heston':
    # Heston parameters (calibrated values - adjust as needed)
    v0 = historical_vol**2  # Initial variance
    kappa = 2.0  # Mean reversion speed
    theta = historical_vol**2  # Long-term variance
    sigma_v = 0.3  # Volatility of volatility
    rho = -0.7  # Correlation (typically negative)
    
    stock_paths, variance_paths = simulate_heston(
        S0, v0, T_total, r, kappa, theta, sigma_v, rho, n_steps-1, n_simulations
    )
    sigma = historical_vol  # Use for delta calculation

# Run delta-hedging simulation
portfolio_values = delta_hedge_simulation(
    stock_paths, T_total, K, r, sigma, option_type, transaction_cost_pct
)

# ===========================
# 8. Calculate Statistics
# ===========================
mean_pnl = portfolio_values[:, -1].mean()
std_pnl = portfolio_values[:, -1].std()
percentile_5 = np.percentile(portfolio_values[:, -1], 5)
percentile_95 = np.percentile(portfolio_values[:, -1], 95)

print(f"\n{'='*50}")
print(f"Delta-Hedging Results ({simulation_model.upper()} Model)")
print(f"{'='*50}")
print(f"Mean Final P&L: ${mean_pnl:,.2f}")
print(f"Std Dev of P&L: ${std_pnl:,.2f}")
print(f"5th Percentile: ${percentile_5:,.2f}")
print(f"95th Percentile: ${percentile_95:,.2f}")
print(f"{'='*50}\n")

# ===========================
# 9. Plot Results
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Stock price paths (sample)
ax = axes[0, 0]
n_display = min(100, n_simulations)
for i in range(n_display):
    ax.plot(dates, stock_paths[i, :], alpha=0.1, color='blue')
ax.plot(dates, stock_paths.mean(axis=0), color='red', linewidth=2, label='Mean Path')
ax.plot(dates, options_df['spx_close'].values, color='black', linewidth=2, 
        linestyle='--', label='Historical Path')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.set_title(f'Simulated Stock Paths ({simulation_model.upper()})')
ax.legend()
ax.grid(True)

# Plot 2: Portfolio value paths
ax = axes[0, 1]
for i in range(n_display):
    ax.plot(dates, portfolio_values[i, :], alpha=0.1, color='green')
ax.plot(dates, portfolio_values.mean(axis=0), color='red', linewidth=2, label='Mean P&L')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('Delta-Hedged Portfolio Values')
ax.legend()
ax.grid(True)

# Plot 3: Distribution of final P&L
ax = axes[1, 0]
ax.hist(portfolio_values[:, -1], bins=50, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(mean_pnl, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_pnl:,.0f}')
ax.axvline(percentile_5, color='orange', linestyle='--', linewidth=2, label=f'5th: ${percentile_5:,.0f}')
ax.axvline(percentile_95, color='orange', linestyle='--', linewidth=2, label=f'95th: ${percentile_95:,.0f}')
ax.set_xlabel('Final P&L ($)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Final P&L')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Volatility comparison (if Heston)
ax = axes[1, 1]
if simulation_model == 'heston':
    realized_vol = np.sqrt(variance_paths)
    for i in range(n_display):
        ax.plot(dates, realized_vol[i, :], alpha=0.1, color='orange')
    ax.plot(dates, realized_vol.mean(axis=0), color='red', linewidth=2, label='Mean Volatility')
    ax.axhline(y=np.sqrt(theta), color='black', linestyle='--', linewidth=2, label='Long-term Vol')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.set_title('Stochastic Volatility Paths (Heston)')
    ax.legend()
else:
    ax.text(0.5, 0.5, f'Constant Volatility\nσ = {sigma:.4f}', 
            ha='center', va='center', transform=ax.transAxes, fontsize=16)
    ax.set_title('Black-Scholes: Constant Volatility')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()