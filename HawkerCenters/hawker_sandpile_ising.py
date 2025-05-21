import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
grid_size = 40  # 40x40 grid
P_min, P_max = 7.0, 20.0
base_cost = 6.5
amp = 0.5
freq = 0.03
linear_slope = 0.1
shock_magnitude = 0.0
shock_start = 120
shock_duration = 8

price_jump = 1.0
stress_transfer = 0.5 * price_jump
imitation_rate = 0.1
mutation_strength = 0.05
min_margin = 0.5
threshold = 3.0  # constant for all stalls
timesteps = 500

np.random.seed(42)

# --- Cost function ---
def shock(t, start=shock_start, magnitude=shock_magnitude, duration=shock_duration):
    return magnitude if start <= t < start + duration else 0.0

def cost_function(t):
    return (base_cost
            + amp * np.sin(2 * np.pi * freq * t)
            + linear_slope * t
            + shock(t))

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
    C = cost_function(t)
    cost_history.append(C)
    avalanche_size = 0
    topple_mask = np.zeros_like(S, dtype=bool)

    # 1. Accumulate stress
    S += np.maximum(0, C - P)

    # 2. Toppling (cascades)
    to_check = list(zip(*np.where(S >= T)))
    while to_check:
        new_to_check = []
        for i, j in to_check:
            if S[i, j] >= T[i, j]:
                avalanche_size += 1
                P[i, j] += price_jump
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

# --- Animation with subplot for price evolution ---
frames = len(P_history)
timesteps_shown = [i*10 for i in range(frames)]

fig, (ax_grid, ax_ts) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})

# Price grid
cax = ax_grid.imshow(P_history[0], cmap='plasma', vmin=P_min, vmax=P_max)
cbar = fig.colorbar(cax, ax=ax_grid, label='Price')
title = ax_grid.set_title('Prices at t=0')
annotation = ax_grid.text(0.02, 0.98, '', color='white', fontsize=12, ha='left', va='top', transform=ax_grid.transAxes, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
ax_grid.axis('off')

# Time series
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


def update(frame_idx):
    cax.set_data(P_history[frame_idx])
    t = timesteps_shown[frame_idx]
    title.set_text(f'Prices at t={t}')
    avg_price = avg_price_history[t]
    avalanche = avalanche_size_history[t]
    annotation.set_text(f'Avg Price: {avg_price:.2f}\nAvalanche Size: {avalanche}')
    vline.set_xdata([t, t])
    return cax, title, annotation, vline

ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
plt.tight_layout()
plt.show()
# To save: ani.save('hawker_sandpile_ising_anim.mp4', fps=5) 