import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function

# --- Parameters ---
n_strategies = 50
P_min, P_max = 7.0, 15.0  # Initial price range
prices = np.linspace(P_min, P_max, n_strategies)
lambda_competition = 5
R = 1.0  # Total demand
timesteps = 500
epsilon_noise = 0.1
min_profit_margin = 0.5  # Minimum profit margin above cost
survival_threshold = -0.1  # Fitness threshold for survival
entry_rate = 0.05  # Probability of new strategy entry per timestep
max_strategies = 100  # Maximum number of strategies allowed

# Time-varying cost function
def base_cost_function(t, base_cost=6.5, amplitude=0.1, frequency=0.05):
    return base_cost + amplitude * np.sin(2 * np.pi * frequency * t) + min(t, 200)*0.01
    #return base_cost + t * 0.01

def shock(t, start=100, magnitude=0, duration=10):
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
        # Adjust prices below minimum to the minimum viable price
        prices[below_min] = min_viable_price
    
    P_bar = np.sum(x * prices)
    
    # Demand share based on logistic response
    demand = expit(-lambda_competition * (prices - P_bar)) * R
    
    # Fitness = demand * profit margin
    profit = demand * (prices - C)
    avg_profit = np.sum(x * profit)
    
    # Survival check - remove strategies below threshold
    surviving = profit >= survival_threshold
    if not np.all(surviving):
        # Remove strategies below threshold
        x[~surviving] = 0
        if np.sum(x) > 0:
            x /= np.sum(x)  # Renormalize remaining strategies
    
    # Entry of new strategies
    if np.random.random() < entry_rate and len(prices) < max_strategies:
        # Generate new strategy
        new_price = np.random.uniform(P_min, P_max)
        new_share = 0.01  # Small initial share
        
        # Add new strategy
        prices = np.append(prices, new_price)
        x = np.append(x, new_share)
        x /= np.sum(x)  # Renormalize
    
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
    
    # Ensure prices remain within reasonable bounds and above minimum
    prices = np.maximum(prices, min_viable_price)
    prices = np.minimum(prices, P_max * 1.5)
    
    x_history.append(x.copy())
    price_history.append(np.sum(x * prices))
    active_strategies_history.append(np.sum(x > 0.001))  # Count strategies with non-negligible share

# Convert history to arrays
x_history = np.array(x_history)
price_history = np.array(price_history)
cost_history = np.array(cost_history)
min_price_history = np.array(min_price_history)
active_strategies_history = np.array(active_strategies_history)

# Plot results
plt.figure(figsize=(14, 12))

# Strategy distribution heatmap
plt.subplot(3, 1, 1)
plt.imshow(x_history.T, aspect='auto', cmap='plasma', extent=[0, timesteps, min(prices), max(prices)])
plt.colorbar(label='Population Share')
plt.xlabel('Time')
plt.ylabel('Price Strategy (RM)')
plt.title('Strategy Distribution Over Time')
plt.grid(alpha=0.3)

# Average price vs time with cost overlay
plt.subplot(3, 1, 2)
plt.plot(price_history, label='Average Price', color='blue')
plt.plot(cost_history, label='Cost', color='red', linestyle='--')
plt.plot(min_price_history, label=f'Minimum Viable Price (Cost + {min_profit_margin} RM)', color='orange', linestyle='-.')
plt.fill_between(range(len(cost_history)), cost_history, min_price_history, color='red', alpha=0.1, label='Unprofitable Zone')
plt.xlabel('Time')
plt.ylabel('RM')
plt.title('Average Price and Cost Over Time')
plt.legend()
plt.grid(alpha=0.3)

# Active strategies plot
plt.subplot(3, 1, 3)
plt.plot(active_strategies_history, label='Active Strategies', color='green')
plt.xlabel('Time')
plt.ylabel('Number of Strategies')
plt.title('Number of Active Strategies Over Time')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Print some statistics at the end
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