import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

# --- Parameters ---
grid_size = 40  # 40x40 grid
P_min, P_max = 7.0, 17.0
base_cost = 6.0
amp = 0.0
freq = 0.03
linear_slope = 0.05
shock_magnitude = 1.0
shock_start = 120
shock_duration = 5

default_shock_magnitude = 4.5  # Use this for the targeted shock
shock_fraction = 0.1  # Fraction of cells to shock (for reference)

price_jump = 2.0
stress_transfer = 0.5 * price_jump
imitation_rate = 0.05
mutation_strength = 0.13
min_margin = 0.5
threshold = 3.0  # constant for all stalls
timesteps = 350

# --- New constants for overpricing penalty and upward pressure ---
beta_overprice = 0.15
margin_slack = 2.0
gamma_align = 0.35
max_push = 0.5


num_patches = 4
np.random.seed(42)

# --- Cost function ---
def linear_cost(t):
    return linear_slope * t if t < 70 else linear_slope * (70)
def cost_function(t):
    return (base_cost
            + amp * np.sin(2 * np.pi * freq * t)
            + linear_cost(t))

def random_patches_mask(grid_size, num_patches, patch_size):
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for _ in range(num_patches):
        i = np.random.randint(0, grid_size - patch_size + 1)
        j = np.random.randint(0, grid_size - patch_size + 1)
        mask[i:i+patch_size, j:j+patch_size] = True
    return mask

# Calculate patch size: previous patch size ~ sqrt(grid_size*grid_size*shock_fraction)
prev_patch_area = int(grid_size * grid_size * shock_fraction)
prev_patch_size = int(np.sqrt(prev_patch_area))
patch_size = max(1, int(prev_patch_size / 3))  # 1/3 the previous patch size, at least 1

shock_mask = random_patches_mask(grid_size, num_patches, patch_size)

def targeted_shock(t, mask, start=shock_start, magnitude=default_shock_magnitude, duration=shock_duration):
    if start <= t < start + duration:
        return magnitude * mask
    else:
        return np.zeros_like(mask, dtype=float)

# --- Initialization ---
P = np.random.uniform(P_min, P_max, size=(grid_size, grid_size))
S = np.zeros((grid_size, grid_size))
T = np.full((grid_size, grid_size), threshold)

# --- Data recording ---
P_history = []
avg_price_history = []
high_price_history = []
low_price_history = []
cost_history = []
avalanche_size_history = []
cumulative_avalanche_count = 0
cumulative_avalanche_history = []

# --- Helper: 4-neighbor (von Neumann) ---
def get_neighbors(i, j, n):
    return [((i-1)%n, j), ((i+1)%n, j), (i, (j-1)%n), (i, (j+1)%n)]

# --- Simulation loop ---
for t in range(timesteps):
    base_C = cost_function(t)
    C = base_C + targeted_shock(t, shock_mask)
    cost_history.append(base_C)  # For plotting, keep the base cost
    avalanche_size = 0
    topple_mask = np.zeros_like(S, dtype=bool)

    # 1. Accumulate stress (underpricing)
    S += np.maximum(0, C - P)
    # 1b. Accumulate stress (overpricing)
    S += beta_overprice * np.maximum(0, P - (C + margin_slack))

    # 2. Toppling (cascades)
    to_check = list(zip(*np.where(S >= T)))
    while to_check:
        new_to_check = []
        for i, j in to_check:
            if S[i, j] >= T[i, j]:
                avalanche_size += 1
                if P[i, j] > C[i, j] + margin_slack:
                    P[i, j] -= price_jump  # downward jump for overpriced
                else:
                    P[i, j] += price_jump  # upward jump for underpriced/normal
                S[i, j] = 0
                topple_mask[i, j] = True
                for ni, nj in get_neighbors(i, j, grid_size):
                    S[ni, nj] += stress_transfer
                    if S[ni, nj] >= T[ni, nj] and not topple_mask[ni, nj]:
                        new_to_check.append((ni, nj))
        to_check = new_to_check

    cumulative_avalanche_count += avalanche_size
    avalanche_size_history.append(avalanche_size)
    cumulative_avalanche_history.append(cumulative_avalanche_count)

    # 2b. Soft upward pressure during cost increases
    P += gamma_align * np.clip(C - P, 0, max_push)


    # 3. Ising alignment
    P_new = P.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = get_neighbors(i, j, grid_size)
            peer_avg = np.mean([P[ni, nj] for ni, nj in neighbors])
            P_new[i, j] += imitation_rate * (peer_avg - P[i, j])
    P = P_new

    # 4. Mutation
    P += mutation_strength * np.random.randn(grid_size, grid_size)

    # 5. Clamp prices
    P = np.maximum(P, C + min_margin)
    P = np.clip(P, P_min, P_max)

    # 6. Record metrics
    avg_price_history.append(P.mean())
    high_price_history.append(P.max())
    low_price_history.append(P.min())
    if t % 10 == 0:
        P_history.append(P.copy())

# --- Animation with improved subplot layout ---
frames = len(P_history)
timesteps_shown = [i*10 for i in range(frames)]

# --- Heatmap binning setup ---
num_bins = 50
price_bins = np.linspace(P_min, P_max, num_bins + 1)
price_dist_over_time = np.zeros((num_bins, frames))
prices = P_history[0].flatten()
hist, _ = np.histogram(prices, bins=price_bins)
price_dist_over_time[:, 0] = hist / hist.sum()

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.3)

# Left: Price grid (spans both rows)
ax_grid = fig.add_subplot(gs[:, 0])
cax = ax_grid.imshow(P_history[0], cmap='plasma', vmin=P_min, vmax=P_max)
cbar = fig.colorbar(cax, ax=ax_grid, label='Price', fraction=0.046, pad=0.04)
title = ax_grid.set_title('Prices at t=0')
annotation = ax_grid.text(0.02, 0.98, '', color='white', fontsize=12, ha='left', va='top', transform=ax_grid.transAxes, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
ax_grid.axis('off')

# Top-right: Time series
ax_ts = fig.add_subplot(gs[0, 1])
line_avg, = ax_ts.plot(avg_price_history, label='Average Price', color='blue')
line_high, = ax_ts.plot(high_price_history, label='Highest Price', color='green', linestyle=':')
line_low, = ax_ts.plot(low_price_history, label='Lowest Price', color='purple', linestyle=':')
line_cost, = ax_ts.plot(cost_history, label='Cost', color='red', linestyle='--')
vline = ax_ts.axvline(0, color='black', linestyle='--', alpha=0.7)
ax_ts.set_xlim(0, timesteps)
ax_ts.set_ylim(min(P_min, min(cost_history)), max(P_max, max(cost_history)))
ax_ts.set_xlabel('Time')
ax_ts.set_ylabel('RM')
ax_ts.legend()
ax_ts.set_title('Prices and Cost Over Time')

# Bottom-right: Heatmap
ax_heat = fig.add_subplot(gs[1, 1])
bin_centers = 0.5 * (price_bins[:-1] + price_bins[1:])
heatmap = ax_heat.imshow(
    price_dist_over_time,
    aspect='auto',
    origin='lower',
    extent=[0, frames-1, price_bins[0], price_bins[-1]],
    cmap='plasma',
    vmin=0, vmax=1
)
ax_heat.set_xlabel('Frame (every 10 steps)')
ax_heat.set_ylabel('Price Strategy (RM)')
ax_heat.set_title('Strategy Distribution Over Time')
cbar_heat = fig.colorbar(heatmap, ax=ax_heat, label='Population Share', fraction=0.046, pad=0.04)


def update(frame_idx):
    cax.set_data(P_history[frame_idx])
    t = timesteps_shown[frame_idx]
    title.set_text(f'Prices at t={t}')
    avg_price = avg_price_history[t]
    avalanche = avalanche_size_history[t]
    annotation.set_text(f'Avg Price: {avg_price:.2f}\nAvalanche Size: {avalanche}')
    vline.set_xdata([t, t])
    # Update heatmap data up to current frame
    for f in range(frame_idx+1):
        prices = P_history[f].flatten()
        hist, _ = np.histogram(prices, bins=price_bins)
        price_dist_over_time[:, f] = hist / hist.sum()
    heatmap.set_data(price_dist_over_time)
    return cax, title, annotation, vline, heatmap

ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
plt.tight_layout()
plt.show()
# To save: ani.save('hawker_sandpile_ising_anim.mp4', fps=5) 