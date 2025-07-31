import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration
np.random.seed(42)
n_samples = 5000

# ──────────────────────────────────────────────────────────────────────────────
# 2) Sample predictors uniformly (no ordering in x’s themselves)
x1 = np.random.uniform(0, 200, size=n_samples)
x2 = np.random.uniform(0, 50, size=n_samples)
x3 = np.random.uniform(0, 150, size=n_samples)
x4 = np.random.uniform(0, 100, size=n_samples)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Define clean transforms
f1 = 0.5  * x1
f2 = 1.2  * np.sqrt(x2)
f3 = 2.0  * np.log1p(x3)
f4 = -0.8 * x4

# ──────────────────────────────────────────────────────────────────────────────
#4) Inject piecewise constant “bumps” in each monotonic feature

def piecewise_offsets(x, n_bins=20, scale=50):
    """
    Bin x into n_bins uniform-width bins (no edge bins overflow),
    then assign each bin a random offset.
    """
    # create n_bins+1 edges, then take the interior n_bins-1 edges for digitize
    edges = np.linspace(np.min(x), np.max(x), n_bins+1)[1:-1]
    bins = np.digitize(x, edges)           # bins in 0..(n_bins-1)
    offsets = np.random.normal(0, scale, size=n_bins)
    return offsets[bins]

off1 = piecewise_offsets(x1, n_bins=20, scale=50)
off2 = piecewise_offsets(x2, n_bins=20, scale=30)
off3 = piecewise_offsets(x3, n_bins=20, scale=20)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Build noisy target
noise = np.random.normal(0, 10, size=n_samples)

total_number_trips = (
    f1 + f2 + f3 + f4    # underlying monotonic signal
    + off1 + off2 + off3 # local non‑monotonic bumps
    + noise              # random noise
)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Assemble & save
df_synth = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4,
    'total_number_trips': total_number_trips
})
df_synth.to_csv('synthetic_monotonic_trips_new.csv', index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 7) Sanity check
print(df_synth.head())
print("\nCorrelations with total_number_trips:")
print(df_synth.corr()['total_number_trips'].drop('total_number_trips'))
