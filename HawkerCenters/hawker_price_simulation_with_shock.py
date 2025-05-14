
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function

# --- Parameters ---
n_strategies = 30
P_min, P_max = 7.0, 10.0
prices = np.linspace(P_min, P_max, n_strategies)
lambda_competition = 5
R = 1.0  # Total demand
timesteps = 200
epsilon_noise = 0.01

# --- Cost function components ---
def base_cost_function(t, base_cost=6.5, amplitude=0.3, frequency=0.05):
    return base_cost + amplitude * np.sin(2 * np.pi * frequency * t)

def shock(t, start=100, magnitude=0.5, duration=10):
    return magnitude if start <= t < start + duration else 0.0

def cost_function(t):
    return base_cost_function(t) + shock(t)

# Initialize population distribution (uniform)
x = np.ones(n_strategies) / n_strategies
x_history = [x.copy()]
price_history = [np.sum(x * prices)]
cost_history = [cost_function(0)]

# Simulation loop
for t in range(1, timesteps + 1):
    C = cost_function(t)
    cost_history.append(C)

    P_bar = np.sum(x * prices)
    
    # Demand share based on logistic response
    demand = expit(-lambda_competition * (prices - P_bar)) * R
    
    # Fitness = demand * profit margin
    profit = demand * (prices - C)
    avg_profit = np.sum(x * profit)
    
    # Replicator dynamics with mutation
    dx = x * (profit - avg_profit)
    dx += epsilon_noise * np.random.normal(0, 1, size=n_strategies) * x
    x += dx
    x = np.clip(x, 0, 1)
    x /= np.sum(x)
    
    x_history.append(x.copy())
    price_history.append(np.sum(x * prices))

# Convert history to arrays
x_history = np.array(x_history)
price_history = np.array(price_history)
cost_history = np.array(cost_history)

# Plot results
plt.figure(figsize=(14, 6))

# Strategy distribution heatmap
plt.subplot(1, 2, 1)
plt.imshow(x_history.T, aspect='auto', cmap='plasma', extent=[0, timesteps, P_min, P_max])
plt.colorbar(label='Population Share')
plt.axvline(100, color='white', linestyle='--', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Price Strategy (RM)')
plt.title('Strategy Distribution Over Time (Shock at t=100)')

# Average price vs time with cost overlay
plt.subplot(1, 2, 2)
plt.plot(price_history, label='Average Price')
plt.plot(cost_history, label='Cost Function', linestyle='--')
plt.axvline(100, color='black', linestyle='--', linewidth=1, label='Shock Start')
plt.xlabel('Time')
plt.ylabel('RM')
plt.title('Average Price and Cost Over Time')
plt.legend()

plt.tight_layout()
plt.show()
