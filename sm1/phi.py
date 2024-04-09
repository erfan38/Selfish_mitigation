import numpy as np
import matplotlib.pyplot as plt

# Define the range for alpha
alpha_range = np.linspace(0.1, 0.5, 5)

# Define time range
time_range = np.linspace(0, 5, 100)  # Increased the number of points for smoother curve

# Create a figure and axis object for plotting
plt.figure(figsize=(10, 6))

# Plot phi(t) for each alpha
for alpha in alpha_range:
    phi_values = 10 / (1 + alpha * time_range)
    plt.plot(time_range, phi_values, label=f'alpha = {alpha:.2f}')

# Add labels and legend
plt.xlabel('Time (t)')
plt.ylabel('Phi(t)')
plt.title('Plot of Phi(t)')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
