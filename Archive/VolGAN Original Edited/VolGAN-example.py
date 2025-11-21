import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import VolGAN

plt.rcParams['figure.figsize'] = [15.75, 9.385]
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams.update({'font.size': 12})

datapath = "../../data/SPX.csv"
surfacepath = "../../data/surfacesTransform.csv"
surfaces_transform, prices, prices_prev, log_rtn, m, tau, ms, taus, dates_dt = VolGAN.SPXData(datapath, surfacepath)

n = min(len(dates_dt), len(log_rtn))
dates_aligned = dates_dt[:n]
log_rtn_aligned = log_rtn[:n]

# --- 21-day rolling realised volatility ---
window = 21
log_rtn_sq = log_rtn_aligned**2
cumsum = np.cumsum(log_rtn_sq)
cumsum_window = cumsum[window:] - cumsum[:-window]
realised_vol_t = np.sqrt(252 / window) * np.sqrt(cumsum_window)

# --- Dates for plotting realised volatility ---
dates_rv = dates_aligned[window:]
plt.figure(figsize=(12,5))
plt.plot(dates_rv, realised_vol_t)
plt.title("21-Day Realised Volatility")
plt.xlabel("Date")
plt.ylabel("Realised Volatility")
plt.show()

# --- Shifted data for t, t-1, t-2 ---
dates_t = dates_aligned[window+1:]
log_rtn_t = log_rtn_aligned[window+1:]
log_rtn_tm1 = log_rtn_aligned[window: -1]
log_rtn_tm2 = log_rtn_aligned[window-1: -2]

surfaces_transform = np.array(surfaces_transform, dtype=float)
if np.any(surfaces_transform <= 0):
    surfaces_transform[surfaces_transform <= 0] = 1e-8

# --- Log implied vol surfaces ---
log_iv_t = np.log(surfaces_transform[window+1:])
log_iv_tm1 = np.log(surfaces_transform[window: -1])
log_iv_inc_t = log_iv_t - log_iv_tm1

# --- VolGAN parameters ---
tr = 0.85
noise_dim = 32
hidden_dim = 16
device = 'cpu'
n_epochs = 10000
n_grad = 25
val = True

# --- Train VolGAN ---
gen, gen_opt, disc, disc_opt, true_train, true_val, true_test, condition_train, condition_val, condition_test, dates_t, m, tau, ms, taus = VolGAN.VolGAN(
    datapath, surfacepath, tr, noise_dim=noise_dim, hidden_dim=hidden_dim,
    n_epochs=n_epochs, n_grad=n_grad, lrg=0.0001, lrd=0.0001, batch_size=100, device=device
)

# --- Prepare tensors for arbitrage calculation ---
n_test = true_test.shape[0]
B = 100
dtm = tau * 365  # tau already loaded

surface_size = surfaces_transform[0].shape[0]  # Should now be 80
print("surface_size:", surface_size)
print("ms.shape:", ms.shape)
print("taus.shape:", taus.shape)
print("ms.flatten().shape:", ms.flatten().shape)
print("taus.flatten().shape:", taus.flatten().shape)
# Prepare moneyness and tau tensors matching the surface vector
m_tensor = torch.tensor(ms.flatten(), dtype=torch.float, device=device)      # shape [80]
tau_tensor = torch.tensor(taus.flatten(), dtype=torch.float, device=device)  # shape [80]

m_expanded = m_tensor.unsqueeze(0).expand(n_test, surface_size)
tau_expanded = tau_tensor.unsqueeze(0).expand(n_test, surface_size)

mP_t, mP_k, mPb_K = VolGAN.penalty_mutau_tensor(m, dtm, device)
Pks_t_test = mP_k.unsqueeze(0).repeat(n_test, 1, 1)
Pkbs_t_test = mPb_K.unsqueeze(0).repeat(n_test, 1, 1)

# Storage
fk = torch.empty((B, n_test, surface_size), device=device)
fk_ent = np.zeros((B, n_test, surface_size))
fk_inc = np.zeros((B, n_test, surface_size))
ret_u = np.zeros((B, n_test))
tots_test = np.zeros((n_test, B))

gen.eval()
with torch.no_grad():
    for l in tqdm(range(B)):
        noise = torch.randn((n_test, noise_dim), device=device, dtype=torch.float)
        fake = gen(noise, condition_test[:, :])

        surface_past_test = condition_test[:, 3:]
        fake_surface = torch.exp(fake[:, 1:] + surface_past_test)  # shape [n_test, surface_size]

        fk_ent[l, :, :] = fake_surface.cpu().numpy()
        fk_inc[l, :, :] = fake[:, 1:].cpu().numpy()
        ret_u[l, :] = fake[:, 0].cpu().numpy()
        fk[l, :, :] = fake_surface

        BS = VolGAN.smallBS_tensor(m_expanded, tau_expanded, fk[l, :, :], 0)
        _, _, _, tot = VolGAN.arbitrage_penalty_tensor(
        BS, mP_t, mP_k, mPb_K, lk=10, lt=8
        )
        tots_test[:, l] = tot.cpu().numpy()

# --- Check arbitrage penalties ---
print(
    "Simulated arbitrage penalties:",
    np.mean(tots_test),
    np.std(np.mean(tots_test, axis=1)),
    np.median(np.mean(tots_test, axis=1))
)