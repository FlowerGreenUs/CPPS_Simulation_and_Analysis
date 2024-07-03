import networkx as nx
import matplotlib.pyplot as plt

# Create a heterogeneous graph
G = nx.Graph()

# Add nodes with different types
G.add_node(1, type='transformer', attribute='temperature')
G.add_node(2, type='substation', attribute='voltage')
G.add_node(3, type='line', attribute='current')

# Add edges with different types
G.add_edge(1, 2, type='connected')
G.add_edge(2, 3, type='connected')

# Visualize the graph
pos = nx.spring_layout(G)
node_colors = ['blue' if G.nodes[n]['type'] == 'transformer' else 'green' if G.nodes[n]['type'] == 'substation' else 'red' for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Blues)
plt.title('Heterogeneous Graph of CPPS')
plt.show()

# Simple node embedding
def simple_node_embedding(G):
    embedding = {}
    for node in G.nodes():
        embedding[node] = np.random.rand(5)  # Random embedding vector of size 5
    return embedding

embeddings = simple_node_embedding(G)
print("Node Embeddings:")
for node, embed in embeddings.items():
    print(f"Node {node}: {embed}")