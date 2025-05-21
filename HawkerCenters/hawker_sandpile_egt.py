import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import random

# --- Parameters ---
grid_size = 40  # 40x40 grid
P_min, P_max = 7.0, 20.0
base_cost = 6.5
amp = 0.1
freq = 0.03
linear_slope = 0.01
shock_magnitude = 2.0
shock_start = 120
shock_duration = 8

delta_p = 0.1       # Price jump per avalanche
phi = 0.1            # Stress transfer fraction to neighbors
eta = 0.1             # Imitation rate (probabilistic filter)
mutation_strength = 0.1

np.random.seed(42)
random.seed(42)

# --- Cost function ---
def linear_cost(t):
    return linear_slope * t if t < 100 else linear_slope * (100)
def cost_function(t):
    return (base_cost
            + amp * np.sin(2 * np.pi * freq * t)
            + linear_cost(t))

def shock(t, start=shock_start, magnitude=shock_magnitude, duration=shock_duration):
    return magnitude if start <= t < start + duration else 0.0

#EGT Logistic demand imitation
def logistic_imitation(p_i, p_j, Pi_i, Pi_j, eta):
    delta = Pi_j - Pi_i
    return 1 / (1 + np.exp(-eta * delta))

def logistic_demand(P, P_avg, lam=1.0):
    return 1 / (1 + np.exp(lam * (P - P_avg)))



def effective_payoff(price, cost, gamma=0.01, pref=10):
    margin = price - cost
    demand = 1 - gamma * (price - pref)**2
    return margin * max(demand, 0)


# --- Initialization ---
price = np.random.uniform(P_min, P_max, size=(grid_size, grid_size))
cost = np.zeros((grid_size, grid_size))
stress = np.zeros((grid_size, grid_size))
# Thresholds can be fixed or sampled per agent
threshold = np.random.normal(loc=3.0, scale=0.2, size=(grid_size, grid_size))

# --- Data recording ---
price_history = []
avg_price_history = []
high_price_history = []
low_price_history = []
cost_history = []
avalanche_size_history = []

# --- Helper: 4-neighbor (von Neumann) ---
def get_neighbors(i, j, n):
    return [((i-1)%n, j), ((i+1)%n, j), (i, (j-1)%n), (i, (j+1)%n)]

# --- Simulation loop ---
timesteps = 500
for t in range(timesteps):
    # Update cost for all stalls
    C = cost_function(t) + shock(t)
    cost[:, :] = C
    avalanche_set = set()
    # 1. Stress accumulation
    stress += np.maximum(0, cost - price)
    # 2. Avalanche rule
    to_check = list(zip(*np.where(stress > threshold)))
    avalanche_size = 0
    while to_check:
        new_to_check = []
        for i, j in to_check:
            if stress[i, j] > threshold[i, j]:
                price[i, j] += delta_p
                stress[i, j] = 0
                avalanche_set.add((i, j))
                avalanche_size += 1
                # Optional: spread stress to neighbors
                for ni, nj in get_neighbors(i, j, grid_size):
                    stress[ni, nj] += phi * delta_p
                    if stress[ni, nj] > threshold[ni, nj] and (ni, nj) not in avalanche_set:
                        new_to_check.append((ni, nj))
        to_check = new_to_check
    # 3. Local EGT imitation
    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = get_neighbors(i, j, grid_size)
            ni, nj = random.choice(neighbors)
            payoff_i = price[i, j] - cost[i, j]
            payoff_j = price[ni, nj] - cost[ni, nj]
            dp = payoff_j - payoff_i
            P_avg = np.mean(price)  # or smoother historical average if needed

            demand_i = logistic_demand(price[i, j], P_avg, lam=2.0)
            demand_j = logistic_demand(price[ni, nj], P_avg, lam=2.0)

            fitness_i = demand_i * (price[i, j] - cost[i, j])
            fitness_j = demand_j * (price[ni, nj] - cost[ni, nj])
            dp = fitness_j - fitness_i

            if np.random.rand() < 1 / (1 + np.exp(-eta * dp)):
                price[i, j] = price[ni, nj] + np.random.normal(0, mutation_strength)


    # 4. Clamp prices
    price = np.clip(price, P_min, P_max)
    # 5. Record metrics
    price_history.append(price.copy())
    avg_price_history.append(price.mean())
    high_price_history.append(price.max())
    low_price_history.append(price.min())
    cost_history.append(C)
    avalanche_size_history.append(avalanche_size)

# --- Animation and Visualization ---
frames = len(price_history)
timesteps_shown = list(range(frames))

# --- Heatmap binning setup ---
num_bins = 50
price_bins = np.linspace(P_min, P_max, num_bins + 1)
price_dist_over_time = np.zeros((num_bins, frames))
for f in range(frames):
    prices = price_history[f].flatten()
    hist, _ = np.histogram(prices, bins=price_bins)
    price_dist_over_time[:, f] = hist / hist.sum()

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.3)

# Left: Price grid (spans both rows)
ax_grid = fig.add_subplot(gs[:, 0])
cax = ax_grid.imshow(price_history[0], cmap='plasma', vmin=P_min, vmax=P_max)
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
ax_heat.set_xlabel('Time')
ax_heat.set_ylabel('Price Strategy (RM)')
ax_heat.set_title('Strategy Distribution Over Time')
cbar_heat = fig.colorbar(heatmap, ax=ax_heat, label='Population Share', fraction=0.046, pad=0.04)


def update(frame_idx):
    cax.set_data(price_history[frame_idx])
    t = timesteps_shown[frame_idx]
    title.set_text(f'Prices at t={t}')
    avg_price = avg_price_history[t]
    avalanche = avalanche_size_history[t]
    annotation.set_text(f'Avg Price: {avg_price:.2f}\nAvalanche Size: {avalanche}')
    vline.set_xdata([t, t])
    # Update heatmap data up to current frame
    heatmap.set_data(price_dist_over_time)
    return cax, title, annotation, vline, heatmap

ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
plt.tight_layout()
plt.show()
# To save: ani.save('hawker_sandpile_egt_anim.mp4', fps=5) 