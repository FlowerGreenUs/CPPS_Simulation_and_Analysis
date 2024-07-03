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

# 可视化图
pos = nx.spring_layout(G)  # 设置节点的布局
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', font_color='black')  # 绘制图形
edge_labels = nx.get_edge_attributes(G, 'type')  # 获取边的类型属性
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # 绘制边的标签

plt.title('Electric Grid Knowledge Representation')
plt.show()