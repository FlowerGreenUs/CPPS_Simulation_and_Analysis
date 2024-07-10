import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集
with open('transformer_data.json', 'r') as f:
    data = json.load(f)

temperature = np.array(data['temperature'])
electromagnetic_field = np.array(data['electromagnetic_field'])
internal_structure = np.array(data['internal_structure'])
parameters = data['parameters']
environment_conditions = data['environment_conditions']

# 打印设备参数和环境条件
print(f"Voltage: {parameters['voltage']} V")
print(f"Current: {parameters['current']} A")
print(f"Power: {parameters['power']} W")
print(f"Environment Temperature: {environment_conditions['temperature']} °C")
print(f"Environment Humidity: {environment_conditions['humidity']} %")

# 综合可视化
fig, axs = plt.subplots(4, 1, figsize=(12, 32))

# 温度分布可视化
sns.heatmap(temperature, annot=True, cmap='coolwarm', ax=axs[0])
axs[0].set_title('Transformer Temperature Distribution')
axs[0].set_xlabel('X Axis')
axs[0].set_ylabel('Y Axis')

# 电磁场可视化
sns.heatmap(electromagnetic_field, annot=True, cmap='viridis', ax=axs[1])
axs[1].set_title('Transformer Electromagnetic Field')
axs[1].set_xlabel('X Axis')
axs[1].set_ylabel('Y Axis')

# 内部结构状态可视化
sns.heatmap(internal_structure, annot=True, cmap='Greys', ax=axs[2])
axs[2].set_title('Transformer Internal Structure')
axs[2].set_xlabel('X Axis')
axs[2].set_ylabel('Y Axis')

# 环境条件可视化
env_data = np.array([[environment_conditions['temperature'], environment_conditions['humidity']]])
sns.heatmap(env_data, annot=True, cmap='YlGnBu', ax=axs[3], cbar=False, xticklabels=['Temperature (°C)', 'Humidity (%)'])
axs[3].set_title('Environment Conditions')
axs[3].set_xticklabels(['Temperature (°C)', 'Humidity (%)'])
axs[3].set_yticklabels(['Environment'], rotation=0)

plt.tight_layout()
plt.show()
