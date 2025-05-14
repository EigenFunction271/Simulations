import numpy as np
import networkx as nx
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
    G = nx.barabasi_albert_graph(N, m, directed=True)
    P = nx.to_numpy_array(G) * np.random.rand(N, N)
    np.fill_diagonal(P, 0)
    return P, G

def simulate_chat(P, initiator=0, max_steps=20):
    """
    Simulates the spread of a conversation in a group chat based on the reply probability matrix P.
    """
    N = P.shape[0]
    active_messages = [initiator]  # Start with one person's message
    message_counts = np.zeros(N)
    message_counts[initiator] = 1
    
    total_messages = 1
    history = []  # Stores message spread over time
    
    for step in range(max_steps):
        new_active = []
        for sender in active_messages:
            for receiver in range(N):
                if sender != receiver and random.random() < P[sender, receiver]:
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
N = 10  # Number of participants
connectivity = 0.4  # Probability scaling factor
P, G = generate_erdos_renyi_network(N, connectivity)

# Simulate the chat
message_counts, history = simulate_chat(P, initiator=0)

# Compute eigenvalue
largest_eigenvalue = compute_largest_eigenvalue(P)
print(f"Largest eigenvalue of P: {largest_eigenvalue}")

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(history, marker='o', linestyle='-', color='b')
plt.xlabel('Time Step')
plt.ylabel('Total Messages Sent')
plt.title('Group Chat Message Growth Over Time')
plt.grid()
plt.show()

# Visualize the network
plt.figure(figsize=(6,6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
plt.title('Generated Chat Network')
plt.show()

# Show individual message counts
print("Messages sent by each participant:", message_counts)

