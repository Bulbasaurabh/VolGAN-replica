import pandas as pd
import numpy as np
import yfinance as yf

# === 1. Load OptionMetrics data ===
spxoptions_df = pd.read_parquet("data/options_dataset.parquet")
print("✅ Loaded OptionMetrics data")
# Ensure dates are datetime
spxoptions_df['date_dt'] = pd.to_datetime(spxoptions_df['date'], errors='coerce')
spxoptions_df['exdate_dt'] = pd.to_datetime(spxoptions_df['exdate'], errors='coerce')

# Drop rows with missing dates
spxoptions_df = spxoptions_df.dropna(subset=['date_dt', 'exdate_dt'])

# Compute days to expiry and tau (years)
spxoptions_df['days_to_exp'] = (spxoptions_df['exdate_dt'] - spxoptions_df['date_dt']).dt.days
spxoptions_df = spxoptions_df[spxoptions_df['days_to_exp'] > 0]  # keep positive maturities
spxoptions_df['tau'] = spxoptions_df['days_to_exp'] / 365.0

# === 2. Get SPX spot prices from Yahoo Finance ===
startdate = spxoptions_df['date_dt'].min()
enddate = spxoptions_df['date_dt'].max()

ticker_symbol = "^GSPC"
spxspot_df = yf.download(ticker_symbol, start=startdate, end=enddate, progress=False)
spxspot_df['date_dt'] = pd.to_datetime(spxspot_df.index)

# Align SPX prices to observation dates
observation_dates = np.sort(spxoptions_df['date_dt'].unique())
spot_prices = np.zeros(len(observation_dates))

for i, date in enumerate(observation_dates):
    match = spxspot_df.loc[spxspot_df['date_dt'] == date, 'Close']
    if not match.empty:
        spot_prices[i] = match.iloc[0]
    else:
        # fallback to nearest available date
        nearest_idx = (spxspot_df['date_dt'] - date).abs().idxmin()
        spot_prices[i] = spxspot_df.loc[nearest_idx, 'Close']

# === 3. Compute log returns ===
prices_prev = np.roll(spot_prices, 1)
prices_prev[0] = spot_prices[0]  # first day fallback
log_rtn = np.log(spot_prices) - np.log(prices_prev)

# === 4. Build spx.csv rows ===
rows = []
for i, date in enumerate(observation_dates):
    # convert numpy.datetime64 → pandas Timestamp
    date_ts = pd.Timestamp(date)
    
    # Pick all days-to-expiry for this date
    days_list = spxoptions_df.loc[spxoptions_df['date_dt'] == date_ts, 'days_to_exp'].unique()
    if len(days_list) == 0:
        continue  # skip if no valid expiries
    for d in days_list:
        rows.append({
            'date': date_ts.strftime("%Y-%m-%d"),
            'SPX': spot_prices[i],
            'log_rtn': log_rtn[i],
            'days': d
        })

spx_csv = pd.DataFrame(rows)

# === 5. Save CSV ===
spx_csv.to_csv("data/spx.csv", index=False)
print(f"Saved spx.csv with shape {spx_csv.shape}")
print(spx_csv.head(10))
