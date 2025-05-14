import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networks import (generate_random_chat_network, generate_erdos_renyi_network,
                     generate_watts_strogatz_network, generate_barabasi_albert_network,
                     simulate_chat)

def run_simulation(N, connectivity, num_trials=10):
    """
    Run multiple trials of the chat simulation for each network type and return average results.
    """
    networks = {
        'Erdős-Rényi': generate_erdos_renyi_network,
        'Watts-Strogatz': lambda N, p: generate_watts_strogatz_network(N, k=4, p=p),
        'Barabási-Albert': lambda N, m: generate_barabasi_albert_network(N, m=2),
        'Random Chat': lambda N, p: (generate_random_chat_network(N, p), None)
    }
    
    results = {}
    
    for name, generator in networks.items():
        total_messages = 0
        for _ in range(num_trials):
            if name == 'Random Chat':
                P, _ = generator(N, connectivity)
            else:
                P, _ = generator(N, connectivity)
            _, history = simulate_chat(P, max_steps=100)  # Increased max_steps but will terminate early if no messages
            total_messages += history[-1]  # Use the final number of messages
        
        results[name] = total_messages / num_trials
    
    return results

def main():
    # Parameters
    N = 15  # Number of participants
    connectivity_values = np.linspace(0, 1, 20)  # Range of connectivity values from 0 to 1
    num_trials = 15  # Number of trials for each point
    
    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Slightly smaller figure
    axes = axes.flatten()  # Convert to 1D array for easier iteration
    
    # Add a central title
    fig.suptitle(f'Message Spread vs Connectivity\nN={N}, {num_trials} trials', fontsize=14, y=0.95)
    
    # Store results for each network type
    results = {name: [] for name in ['Erdős-Rényi', 'Watts-Strogatz', 'Barabási-Albert', 'Random Chat']}
    
    # Run simulations for each connectivity value
    for i, connectivity in enumerate(connectivity_values):
        print(f"Running simulations for connectivity {connectivity:.2f} ({i+1}/{len(connectivity_values)})")
        trial_results = run_simulation(N, connectivity, num_trials)
        
        for name in results:
            results[name].append(trial_results[name])
    
    # Plot results for each network type
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        # Use smooth lines without markers
        ax.plot(connectivity_values, data, linestyle='-', linewidth=2, label='Simulation')
        
        # Add theoretical curve for Erdős-Rényi
        if name == 'Erdős-Rényi':
            theoretical = N * (1 + (N-1) * connectivity_values)
            ax.plot(connectivity_values, theoretical, '--', color='gray', linewidth=2, label='Theoretical')
        
        ax.set_xlabel('Connectivity')
        ax.set_ylabel('Average Number of Messages')
        ax.set_title(f'{name} Network')  # Simplified title
        ax.grid(False)  # Remove gridlines
        ax.legend()
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Increase spacing between subplots
    plt.show()
    
    # Print final statistics
    print("\nFinal Statistics:")
    for name in results:
        print(f"\n{name}:")
        print(f"Maximum messages: {max(results[name]):.1f}")
        print(f"Minimum messages: {min(results[name]):.1f}")
        print(f"Average messages: {np.mean(results[name]):.1f}")

if __name__ == "__main__":
    main() 