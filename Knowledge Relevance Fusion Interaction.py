import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create a heterogeneous graph
G = nx.Graph()

# Add nodes with different types
G.add_node(1, type='transformer', attribute='temperature')
G.add_node(2, type='substation', attribute='voltage')
G.add_node(3, type='line', attribute='current')

# Add edges with different types
G.add_edge(1, 2, type='connected')
G.add_edge(2, 3, type='connected')

# Node features (random for example)
x = torch.tensor([[1, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1]], dtype=torch.float)

# Edge index
edge_index = torch.tensor([[0, 1, 2],
                           [1, 2, 0]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

# Initialize model, optimizer, and loss function
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training the GCN model
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, torch.tensor([0, 1, 0]))  # Example labels
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train(data)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Get the embeddings
model.eval()
embeddings = model(data).detach().numpy()
print("Node Embeddings:")
print(embeddings)