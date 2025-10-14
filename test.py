import pandas as pd
import numpy as np
from scipy.interpolate import griddata
print("im at step 1", pd.__version__)
# === 1. Load your Parquet data (already done) ===
options_df = pd.read_parquet("data/options_dataset.parquet")
print("im at step 2")
# === 2. Basic cleaning ===
options_df = options_df.dropna(subset=["impl_volatility", "strike_price"])
options_df["date"] = pd.to_datetime(options_df["date"])
options_df["exdate"] = pd.to_datetime(options_df["exdate"])
print("im at step 3")
# === 3. Compute time-to-maturity (τ in years) and moneyness ===
# Need underlying price per date — here we approximate using ATM strike midpoint
spot_estimate = (
    options_df.groupby("date")["strike_price"]
    .median()
    .rename("spot")
)
options_df = options_df.merge(spot_estimate, on="date")
options_df["moneyness"] = options_df["strike_price"] / options_df["spot"]
options_df["tau"] = (options_df["exdate"] - options_df["date"]).dt.days / 365.0
print("im at step 4")
# === 4. Define fixed (m, τ) grid ===
m_grid = np.linspace(0.8, 1.2, 10)     # 0.8 → 1.2
tau_grid = np.linspace(0.05, 0.5, 8) # 0.05y (~18d) → 0.5y (~6mo)
M, T = np.meshgrid(m_grid, tau_grid)
grid_points = np.column_stack([M.ravel(), T.ravel()])
print("im at step 5")
# === 5. Interpolate IV surface for each trading date ===
surfaces = []
dates = []
for date, group in options_df.groupby("date"):
    print(date)
    pts = group[["moneyness", "tau"]].to_numpy()
    vols = group["impl_volatility"].to_numpy()

    # Skip days with too few points
    if len(vols) < 20:
        continue

    # Interpolate implied vols to fixed grid
    grid_vols = griddata(pts, vols, grid_points, method="linear", fill_value=np.nan)
    # Fill remaining NaNs with nearest
    if np.isnan(grid_vols).any():
        grid_vols = griddata(pts, vols, grid_points, method="nearest", fill_value=np.nan)

    surfaces.append(grid_vols)
    dates.append(date)

# Convert to DataFrame (each row = flattened surface)
surf_df = pd.DataFrame(surfaces, index=dates)
surf_df.index.name = "date"
print("im at step 6")
# === 6. Save to CSV (VolGAN expects CSV of flattened surfaces) ===
output_dir = "data"
surf_df.to_csv(f"{output_dir}/surfacesTransform.csv", index=True)
print(f"Saved surfacesTransform.csv with shape {surf_df.shape}")
print("im at step 7")

