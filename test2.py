import pandas as pd
import numpy as np
from notebooks.VolGAN.datacleaning import optionemtricsdata_transform

# --- Load OptionMetrics dataset ---
spxoptions_df = pd.read_parquet("data/options_dataset.parquet")
spxoptions_df['days_to_maturity'] = (
    pd.to_datetime(spxoptions_df['exdate']) - pd.to_datetime(spxoptions_df['date'])
).dt.days
print("✅ Loaded OptionMetrics data")

# --- Transform using your custom function ---
spxoptions_df, spxspot_df, spot, observation_dates = optionemtricsdata_transform(
    spxoptions_df,
    startdate="2000-01-01",
    enddate="2023-08-31"
)
print("✅ OptionMetrics transformation complete")

# --- Extract unique option maturities (in days) ---
option_days = spxoptions_df['days_to_maturity'].unique()
option_days = np.sort(option_days)
print("✅ Unique maturities (days):", option_days)

# --- Create SPX DataFrame with spot prices and log returns ---
spx_df = pd.DataFrame({
    "date": observation_dates,
    "SPX": spot
})
spx_df["log_rtn"] = np.log(spx_df["SPX"] / spx_df["SPX"].shift(1))
spx_df["days"] = np.arange(len(spx_df))  # cumulative day index for VolGAN

# --- Save SPX.csv ---
spx_df.to_csv("data/SPX.csv", index=False)
print("✅ Saved SPX.csv with spot, log returns, and cumulative days:")
print(spx_df.head())
