import numpy as np
import matplotlib.pyplot as plt

# Function to calculate entropy
def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities))

# Example probabilities of events in the system
probabilities = np.array([0.2, 0.3, 0.5])
entropy = calculate_entropy(probabilities)
print(f"Entropy of the system: {entropy:.4f}")

# Function to adjust node states
def adjust_node_states(initial_states, desired_states, adjustment_coefficient):
    return initial_states + adjustment_coefficient * (desired_states - initial_states)

# Example initial and desired states
initial_states = np.array([1.0, 2.0, 3.0])
desired_states = np.array([1.5, 2.5, 3.5])
adjustment_coefficient = 0.1
adjusted_states = adjust_node_states(initial_states, desired_states, adjustment_coefficient)
print(f"Adjusted states of nodes: {adjusted_states}")

# Plotting the initial and adjusted states
plt.plot(initial_states, label='Initial States')
plt.plot(adjusted_states, label='Adjusted States')
plt.legend()
plt.xlabel('Node Index')
plt.ylabel('State Value')
plt.title('Node State Adjustment')
plt.show()