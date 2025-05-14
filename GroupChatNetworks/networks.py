import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')  # Set the backend explicitly
import matplotlib.pyplot as plt
import random

def generate_random_chat_network(N, connectivity=0.3):
    """
    Generates a random network where each participant has a probability of responding to another.
    """
    P = np.random.rand(N, N) * connectivity  # Random response probabilities scaled by connectivity
    np.fill_diagonal(P, 0)  # No self-replies
    return P

def generate_erdos_renyi_network(N, p):
    """
    Generates an Erdős–Rényi random network where each edge exists with probability p.
    """
    G = nx.erdos_renyi_graph(N, p, directed=True)
    P = nx.to_numpy_array(G) * np.random.rand(N, N)
    np.fill_diagonal(P, 0)
    return P, G

def generate_watts_strogatz_network(N, k, p):
    """
    Generates a Watts-Strogatz small-world network with N nodes, each connected to k nearest neighbors,
    and rewiring probability p.
    """
    G = nx.watts_strogatz_graph(N, k, p)
    P = nx.to_numpy_array(G) * np.random.rand(N, N)
    np.fill_diagonal(P, 0)
    return P, G

def generate_barabasi_albert_network(N, m):
    """
    Generates a Barabási–Albert scale-free network where new nodes attach to m existing nodes.
    """
    G = nx.barabasi_albert_graph(N, m)  # Create undirected graph first
    G = G.to_directed()  # Convert to directed graph
    P = nx.to_numpy_array(G) * np.random.rand(N, N)
    np.fill_diagonal(P, 0)
    return P, G

def simulate_chat(P, max_steps=20):
    """
    Simulates the spread of a conversation in a group chat based on the reply probability matrix P.
    Rules:
    1. Each node has a random probability of sending a message (input vector)
    2. Only nodes connected to message senders can initiate new messages
    3. Response probabilities are weighted by edge weights
    """
    N = P.shape[0]
    
    # Initialize random input vector for message probabilities
    message_probs = np.random.rand(N)
    
    # Start with nodes that exceed their random threshold
    active_messages = [i for i in range(N) if random.random() < message_probs[i]]
    message_counts = np.zeros(N)
    for i in active_messages:
        message_counts[i] += 1
    
    total_messages = len(active_messages)
    history = [total_messages]  # Stores message spread over time
    
    for step in range(max_steps):
        new_active = []
        # For each active message sender
        for sender in active_messages:
            # Check all potential responders
            for receiver in range(N):
                if sender != receiver and P[sender, receiver] > 0:  # If there's a connection
                    # Generate random number for this potential responder
                    if random.random() < P[sender, receiver]:
                        new_active.append(receiver)
                        message_counts[receiver] += 1
                        total_messages += 1
        
        active_messages = list(set(new_active))  # Remove duplicates
        history.append(total_messages)
        if not active_messages:
            break  # Stop if no one responds
    
    return message_counts, history

def compute_largest_eigenvalue(P):
    """
    Computes the largest eigenvalue of the reply probability matrix P.
    """
    eigenvalues = np.linalg.eigvals(P)
    return max(abs(eigenvalues))

# Parameters
N = 15  # Number of participants
connectivity = 0.5  # Probability scaling factor

# Generate all network types
networks = {
    'Erdős-Rényi': generate_erdos_renyi_network(N, connectivity),
    'Watts-Strogatz': generate_watts_strogatz_network(N, k=4, p=0.1),
    'Barabási-Albert': generate_barabasi_albert_network(N, m=2),
    'Random Chat': (generate_random_chat_network(N, connectivity), None)
}

# Print network statistics
print(f"\nNetwork Statistics (N={N}, connectivity={connectivity}):")
for name, (P, G) in networks.items():
    if G is not None:
        avg_degree = sum(dict(G.degree()).values()) / N
        density = nx.density(G)
        print(f"\n{name} Network:")
        print(f"Average degree: {avg_degree:.2f}")
        print(f"Network density: {density:.3f}")
    else:
        # For random chat network, calculate average response probability
        avg_prob = np.mean(P[P > 0])
        print(f"\n{name} Network:")
        print(f"Average response probability: {avg_prob:.3f}")

# Create a figure with subplots
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2)

# Plot network structures and message spread for each network type
for idx, (name, (P, G)) in enumerate(networks.items()):
    # Simulate the chat
    message_counts, history = simulate_chat(P, max_steps=20)
    
    # Compute eigenvalue
    largest_eigenvalue = compute_largest_eigenvalue(P)
    
    # Plot network structure
    ax1 = fig.add_subplot(gs[idx])
    if G is not None:  # For networks that return a graph object
        nx.draw(G, ax=ax1, with_labels=True, node_color='lightblue', 
                edge_color='gray', node_size=300)
    else:  # For random chat network
        # Create a directed graph from the probability matrix
        G = nx.DiGraph()
        for i in range(N):
            for j in range(N):
                if P[i,j] > 0:
                    G.add_edge(i, j)
        nx.draw(G, ax=ax1, with_labels=True, node_color='lightblue', 
                edge_color='gray', node_size=300)
    
    ax1.set_title(f'{name} Network\nλ = {largest_eigenvalue:.3f}')

# Add a separate plot for message spread comparison
plt.figure(figsize=(10, 6))
for name, (P, _) in networks.items():
    message_counts, history = simulate_chat(P, max_steps=20)
    plt.plot(history, marker='o', linestyle='-', label=name)

plt.xlabel('Time Step')
plt.ylabel('Total Messages Sent')
plt.title('Message Growth Comparison Across Networks')
plt.grid(True)
plt.legend()
plt.show()

# Show individual message counts for each network
for name, (P, _) in networks.items():
    message_counts, _ = simulate_chat(P, max_steps=20)
    print(f"\nMessages sent by each participant in {name} network:", message_counts)

