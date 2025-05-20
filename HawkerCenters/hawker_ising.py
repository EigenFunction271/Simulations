import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
grid_size = 20  # 20x20 grid
P_min, P_max = 7.0, 15.0
R_max = 1.0
kappa = 0.5  # Return curve shape parameter (toggle)
J = 0.8      # Ising coupling strength (toggle)
mutation_noise = 0.2  # Mutation noise (toggle)
h = 0.0  # External field (toggle), tunes towards global average
timesteps = 200
Amp = 1
Freq = 0.01
Mag = 10

np.random.seed(42)

# --- Cost function ---
def base_cost_function(t, base_cost=6.5, amplitude=Amp, frequency=Freq):
    return base_cost + amplitude * np.sin(2 * np.pi * frequency * t)
def shock(t, start=100, magnitude=Mag, duration=10):
    return magnitude if start <= t < start + duration else 0.0
def linear_cost_function(t, start=100, magnitude=Mag, duration=10):
    return magnitude * (t - start) / duration if start <= t < start + duration else 0.0
def cost_function(t):
    return base_cost_function(t) + linear_cost_function(t)

# --- Initialization ---
P = np.random.uniform(P_min, P_max, size=(grid_size, grid_size))
P_history = [P.copy()]
avg_price_history = [P.mean()]
std_price_history = [P.std()]
cost_history = [cost_function(0)]

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
for t in range(1, timesteps + 1):
    C = cost_function(t)
    cost_history.append(C)
    # Flatten grid indices and shuffle for random update order
    indices = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    np.random.shuffle(indices)
    for i, j in indices:
        P_ij = P[i, j]
        neighbors = get_neighbors(P, i, j)
        # 1. Return and value
        R_P = R_max * (1 - np.exp(-kappa * (P_ij - P_min)))
        V_P = R_P / P_ij
        # 2. Derivative of value
        R_prime = R_max * kappa * np.exp(-kappa * (P_ij - P_min))
        dVdP = (R_prime * P_ij - R_P) / (P_ij ** 2)
        # 3. Fitness gradient
        imitation_penalty = np.sum(2 * J * (P_ij - neighbors))
        P_target = P.mean()  # or set to a fixed value if you want
        external_field = 2 * h * (P_ij - P_target)
        grad = dVdP * (P_ij - C) + V_P - imitation_penalty - external_field
        # 4. Update with mutation
        dP = 0.1 * grad + mutation_noise * np.random.randn()
        # 5. Clamp to allowed range
        P[i, j] = np.clip(P_ij + dP, P_min, P_max)
        # 6. Death and new entrant: if profit margin < 0, replace with new random price
        if P[i, j] - C < 0:
            P[i, j] = np.random.uniform(P_min, P_max)
    P_history.append(P.copy())
    avg_price_history.append(P.mean())
    std_price_history.append(P.std())

# --- Visualization ---
P_history = np.array(P_history)
plt.figure(figsize=(16, 10))

# Show heatmaps at selected times
times_to_show = [0, timesteps//4, timesteps//2, 3*timesteps//4, timesteps]
for idx, t_show in enumerate(times_to_show):
    plt.subplot(2, 3, idx+1)
    plt.imshow(P_history[t_show], cmap='plasma', vmin=P_min, vmax=P_max)
    plt.title(f'Prices at t={t_show}')
    plt.colorbar(label='Price')
    plt.axis('off')

# Plot average price and std over time
plt.subplot(2, 3, 6)
plt.plot(avg_price_history, label='Average Price')
plt.plot(std_price_history, label='Price Std')
plt.plot(cost_history, label='Cost', linestyle='--')
plt.xlabel('Time')
plt.ylabel('RM')
plt.legend()
plt.title('Average Price, Std, and Cost vs Time')
plt.tight_layout()
plt.show()
