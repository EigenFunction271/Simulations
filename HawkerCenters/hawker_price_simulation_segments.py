import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function

# --- Parameters ---
n_strategies = 100
P_min, P_max = 7.0, 25.0  # Initial price range
prices = np.linspace(P_min, P_max, n_strategies)
R = 1.0  # Total demand
factor = 1
segments = [
    {"weight": 0.5, "lambda": 6*factor},  # Budget buyers
    {"weight": 0.35, "lambda": 1*factor},  # Quality seekers
    {"weight": 0.15, "lambda": 0.3*factor},  # Loyal locals
]
timesteps = 200
epsilon_noise = 0.25 #controls the magnitude of random fluctuations in the population share of each strategy.
min_profit_margin = 0.5  # Minimum profit margin above cost
survival_threshold = -0.1  # Fitness threshold for survival
entry_rate = 0.07  # Probability of new strategy entry per timestep
max_strategies = n_strategies*1.25  # Maximum number of strategies allowed
Amp = 1
Freq = 0.05
Mag = 5

# Time-varying cost function
def base_cost_function(t, base_cost=6.5, amplitude=Amp, frequency=Freq):
    return base_cost + amplitude * np.sin(2 * np.pi * frequency * t)

def shock(t, start=100, magnitude=Mag, duration=10):
    return magnitude if start <= t < start + duration else 0.0

def cost_function(t):
    return base_cost_function(t) + shock(t)

# Initialize population distribution (uniform)
x = np.ones(n_strategies) / n_strategies
x_history = [x.copy()]
price_history = [np.sum(x * prices)]
cost_history = [cost_function(0)]
min_price_history = [cost_function(0) + min_profit_margin]  # Track minimum viable price
active_strategies_history = [n_strategies]  # Track number of active strategies

# Simulation loop
for t in range(1, timesteps + 1):
    C = cost_function(t)
    cost_history.append(C)
    min_viable_price = C + min_profit_margin
    min_price_history.append(min_viable_price)
    
    # Enforce minimum price constraint
    below_min = prices < min_viable_price
    if np.any(below_min):
        prices[below_min] = min_viable_price
    
    P_bar = np.sum(x * prices)
    
    # --- Updated profit calculation: sum over segments ---
    profit = np.zeros_like(prices)
    for seg in segments:
        demand_k = expit(-seg["lambda"] * (prices - P_bar)) * R
        profit += seg["weight"] * demand_k * (prices - C)
    avg_profit = np.sum(x * profit)
    
    # Survival check - remove strategies below threshold
    surviving = profit >= survival_threshold
    if not np.all(surviving):
        x[~surviving] = 0
        if np.sum(x) > 0:
            x /= np.sum(x)
    
    # Entry of new strategies
    if np.random.random() < entry_rate and len(prices) < max_strategies:
        new_price = np.random.uniform(P_min, P_max)
        new_share = 0.01
        prices = np.append(prices, new_price)
        x = np.append(x, new_share)
        x /= np.sum(x)
        P_bar = np.sum(x * prices)
        profit = np.zeros_like(prices)
        for seg in segments:
            demand_k = expit(-seg["lambda"] * (prices - P_bar)) * R
            profit += seg["weight"] * demand_k * (prices - C)
        avg_profit = np.sum(x * profit)
    
    # Replicator dynamics with mutation
    dx = x * (profit - avg_profit)
    dx += epsilon_noise * np.random.normal(0, 1, size=len(prices)) * x
    x += dx
    x = np.clip(x, 0, 1)
    if np.sum(x) > 0:
        x /= np.sum(x)
    
    # Improved price strategy evolution
    price_adjustment = epsilon_noise * (profit - avg_profit) * 0.1
    max_adjustment = 0.2
    price_adjustment = np.clip(price_adjustment, -max_adjustment, max_adjustment)
    prices += price_adjustment
    prices = np.maximum(prices, min_viable_price)
    
    # Store history
    x_history.append(x.copy())
    price_history.append(np.sum(x * prices))
    active_strategies_history.append(np.sum(x > 0.001))

# Convert history to arrays
max_len = max(len(x_array) for x_array in x_history)
padded_x_history = []
for x_array in x_history:
    padded = np.zeros(max_len)
    padded[:len(x_array)] = x_array
    padded_x_history.append(padded)

x_history = np.array(padded_x_history)
price_history = np.array(price_history)
cost_history = np.array(cost_history)
min_price_history = np.array(min_price_history)
active_strategies_history = np.array(active_strategies_history)

# Plot results
plt.figure(figsize=(14, 12))

plt.subplot(2, 1, 1)
plt.imshow(x_history.T, aspect='auto', cmap='plasma', 
           extent=[0, timesteps, min(prices), max(prices)])
plt.colorbar(label='Population Share')
plt.xlabel('Time')
plt.ylabel('Price Strategy (RM)')
plt.title('Strategy Distribution Over Time')
plt.grid(alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(price_history, label='Average Price', color='blue')
plt.plot(cost_history, label='Cost', color='red', linestyle='--')
plt.plot(min_price_history, label=f'Minimum Viable Price (Cost + {min_profit_margin} RM)', color='orange', linestyle='-.')
plt.fill_between(range(len(cost_history)), cost_history, min_price_history, color='red', alpha=0.1, label='Unprofitable Zone')
plt.xlabel('Time')
plt.ylabel('RM')
plt.title('Average Price and Cost Over Time')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout(pad=3.0)
plt.show()

final_avg_price = price_history[-1]
final_cost = cost_history[-1]
final_margin = final_avg_price - final_cost
final_margin_pct = (final_margin / final_cost) * 100
final_active_strategies = active_strategies_history[-1]

print(f"Final Statistics:")
print(f"Average Price: RM {final_avg_price:.2f}")
print(f"Cost: RM {final_cost:.2f}")
print(f"Profit Margin: RM {final_margin:.2f} ({final_margin_pct:.1f}%)")
print(f"Active Strategies: {final_active_strategies}") 