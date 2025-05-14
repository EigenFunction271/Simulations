import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Set the backend explicitly

# Create figure and axis explicitly
fig, ax = plt.subplots()

# Generate Erdős-Rényi graph
n = 20
p = 0.15
G = nx.erdos_renyi_graph(n, p)

# Visualize using circular layout
pos = nx.circular_layout(G)
nx.draw(G, pos, ax=ax, node_size=200, with_labels=True)
ax.set_title(f"G(n={n}, p={p}) Example")
plt.show()