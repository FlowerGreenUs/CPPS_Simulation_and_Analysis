import matplotlib.pyplot as plt

def visualize_graph(G, embeddings):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', linewidths=1, font_size=15)

    # Plot embeddings
    for i, (node, (x, y)) in enumerate(pos.items()):
        plt.text(x, y, str(i), color='red', fontsize=12, ha='center', va='center')
        plt.scatter(embeddings[i, 0], embeddings[i, 1], c='blue', label='Node Embedding')
    plt.title('Graph and Embeddings')
    plt.show()

visualize_graph(G, embeddings)
