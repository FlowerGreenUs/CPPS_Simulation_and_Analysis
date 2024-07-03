import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个空的无向图
G = nx.Graph()

# 添加节点和边，表示电网中的组成部分和它们之间的关系
G.add_node('Transformer 1', type='transformer', attribute='temperature')
G.add_node('Substation A', type='substation', attribute='voltage')
G.add_node('Line 1', type='line', attribute='current')

G.add_edge('Transformer 1', 'Substation A', type='connected')
G.add_edge('Substation A', 'Line 1', type='connected')

# 定义节点特征和边特征
node_features = {
    'Transformer 1': [1.0, 0.0, 0.0, 0.0, 1.0],  # 5维特征向量
    'Substation A': [0.0, 1.0, 0.0, 1.0, 0.0],
    'Line 1': [0.0, 0.0, 1.0, 0.0, 1.0]
}

edge_features = {
    ('Transformer 1', 'Substation A'): [1.0],  # 单个边的特征向量
    ('Substation A', 'Line 1'): [1.0]
}

# 将节点特征和边特征转换为PyTorch张量
x = torch.tensor([node_features[node] for node in G.nodes()], dtype=torch.float)
edge_index = torch.tensor([[G.nodes().index(edge[0]), G.nodes().index(edge[1])] for edge in G.edges()], dtype=torch.long)
edge_attr = torch.tensor([edge_features[edge] for edge in G.edges()], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr.t().contiguous())

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

# 初始化模型、优化器和损失函数
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 训练GCN模型
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, torch.tensor([0, 1, 0]))  # 示例标签
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train(data)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

# 获取节点的嵌入向量
model.eval()
embeddings = model(data).detach().numpy()
print("Node Embeddings:")
print(embeddings)

# 可视化图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', font_color='black')
edge_labels = nx.get_edge_attributes(G, 'type')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('Electric Grid Knowledge Representation with Node Features and Edge Features')
plt.show()