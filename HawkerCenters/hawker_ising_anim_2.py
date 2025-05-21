import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
grid_size = 40  # 20x20 grid
P_min, P_max = 7.0, 15.0
R_max = 1.0
kappa = 0.5  # Return curve shape parameter (toggle)
J = 0.9     # Ising coupling strength (toggle)
mutation_noise = 0.2  # Mutation noise (toggle)
h = 0.0  # External field (toggle), tunes towards global average
timesteps = 200
Amp = 0
Freq = 0.01
Mag = 0
Mag_Linear = 3

np.random.seed(42)

# --- Cost function ---
def base_cost_function(t, base_cost=6.5, amplitude=Amp, frequency=Freq):
    return base_cost + amplitude * np.sin(2 * np.pi * frequency * t)
def shock(t, start=100, magnitude=Mag, duration=10):
    return magnitude if start <= t < start + duration else 0.0
def linear_cost_function(t, start=0, magnitude=Mag_Linear, duration=100):
    return magnitude * (t - start) / duration if start <= t < start + duration else magnitude
def cost_function(t):
    return base_cost_function(t) + linear_cost_function(t)

# --- Initialization ---
P = np.random.uniform(P_min, P_max, size=(grid_size, grid_size))
P_history = [P.copy()]
avg_price_history = [P.mean()]
std_price_history = [P.std()]
cost_history = [cost_function(0)]
high_price_history = [P.max()]
low_price_history = [P.min()]

# --- Helper: Moore neighbors with periodic boundary ---
def get_neighbors(P, i, j):
    n = P.shape[0]
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni = (i + di) % n
            nj = (j + dj) % n
            neighbors.append(P[ni, nj])
    return np.array(neighbors)

# --- Simulation loop (asynchronous update) ---
frames = []
for t in range(1, timesteps + 1):
    C = cost_function(t)
    cost_history.append(C)
    indices = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    np.random.shuffle(indices)
    for i, j in indices:
        P_ij = P[i, j]
        neighbors = get_neighbors(P, i, j)
        R_P = R_max * (1 - np.exp(-kappa * (P_ij - P_min)))
        V_P = R_P / P_ij
        R_prime = R_max * kappa * np.exp(-kappa * (P_ij - P_min))
        dVdP = (R_prime * P_ij - R_P) / (P_ij ** 2)
        imitation_penalty = np.sum(2 * J * (P_ij - neighbors))
        P_target = P.mean()
        external_field = 2 * h * (P_ij - P_target)
        grad = dVdP * (P_ij - C) + V_P - imitation_penalty - external_field
        dP = 0.1 * grad + mutation_noise * np.random.randn()
        P[i, j] = np.clip(P_ij + dP, P_min, P_max)
        if P[i, j] - C < 0:
            P[i, j] = np.random.uniform(P_min, P_max)
    P_history.append(P.copy())
    avg_price_history.append(P.mean())
    std_price_history.append(P.std())
    high_price_history.append(P.max())
    low_price_history.append(P.min())
    frames.append(P.copy())

# --- Animation with subplot for average price and cost ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.2]})
cax = ax1.imshow(frames[0], cmap='plasma', vmin=P_min, vmax=P_max)
cbar = fig.colorbar(cax, ax=ax1, label='Price')
title = ax1.set_title('Prices at t=0')
ax1.axis('off')

line_price, = ax2.plot([], [], label='Average Price', color='blue')
line_cost, = ax2.plot([], [], label='Cost', color='red', linestyle='--')
line_high, = ax2.plot([], [], label='Highest Price', color='green', linestyle=':')
line_low, = ax2.plot([], [], label='Lowest Price', color='purple', linestyle=':')
ax2.set_xlim(0, timesteps)
ax2.set_ylim(min(P_min, min(cost_history)), max(P_max, max(cost_history)))
ax2.set_xlabel('Time')
ax2.set_ylabel('RM')
ax2.legend()
ax2.set_title('Average, High, Low Price and Cost vs Time')


def update(frame_idx):
    cax.set_data(frames[frame_idx])
    title.set_text(f'Prices at t={frame_idx}')
    line_price.set_data(np.arange(frame_idx+1), avg_price_history[:frame_idx+1])
    line_cost.set_data(np.arange(frame_idx+1), cost_history[:frame_idx+1])
    line_high.set_data(np.arange(frame_idx+1), high_price_history[:frame_idx+1])
    line_low.set_data(np.arange(frame_idx+1), low_price_history[:frame_idx+1])
    return cax, title, line_price, line_cost, line_high, line_low

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
plt.tight_layout()
plt.show()
# To save: ani.save('hawker_ising_anim.mp4', fps=20)

# --- Heatmap of price distribution over time ---
# Parameters for binning
num_bins = 50
price_bins = np.linspace(P_min, P_max, num_bins + 1)

# Build a 2D histogram: rows=price bins, cols=timesteps
price_dist_over_time = np.zeros((num_bins, len(P_history)))

for t, P_snapshot in enumerate(P_history):
    # Flatten grid to 1D
    prices = P_snapshot.flatten()
    # Histogram: count agents in each price bin
    hist, _ = np.histogram(prices, bins=price_bins)
    price_dist_over_time[:, t] = hist / hist.sum()  # Normalize to population share

# Y-axis: bin centers
bin_centers = 0.5 * (price_bins[:-1] + price_bins[1:])

fig, ax = plt.subplots(figsize=(14, 4))
cax = ax.imshow(
    price_dist_over_time,
    aspect='auto',
    origin='lower',
    extent=[0, len(P_history)-1, price_bins[0], price_bins[-1]],
    cmap='plasma',
    vmin=0, vmax=1
)
ax.set_xlabel('Time')
ax.set_ylabel('Price Strategy (RM)')
ax.set_title('Strategy Distribution Over Time')
cbar = fig.colorbar(cax, ax=ax, label='Population Share')
plt.tight_layout()
plt.show() 