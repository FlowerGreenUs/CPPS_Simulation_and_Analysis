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
maintenance_records = data['maintenance_records']
historical_data = data['historical_data']

# 打印设备参数和环境条件
print(f"Voltage: {parameters['voltage']} V")
print(f"Current: {parameters['current']} A")
print(f"Power: {parameters['power']} W")
print(f"Load: {parameters['load']} %")
print(f"Voltage Stability: {parameters['voltage_stability']} %")
print(f"Environment Temperature: {environment_conditions['temperature']} °C")
print(f"Environment Humidity: {environment_conditions['humidity']} %")

print("\nMaintenance Records:")
for record in maintenance_records:
    print(f"Date: {record['date']}, Action: {record['action']}, Status: {record['status']}")

# 综合可视化
fig, axs = plt.subplots(5, 1, figsize=(14, 40))

# 温度分布可视化
sns.heatmap(temperature, annot=True, cmap='coolwarm', ax=axs[0])
axs[0].set_title('Transformer Temperature Distribution')
axs[0].set_xlabel('X Axis')
axs[0].set_ylabel('Y Axis')

# 电磁场可视化
sns.heatmap(electromagnetic_field,
