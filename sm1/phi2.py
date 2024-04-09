##########   affecting alpha and gamma in the formula #####################
# import numpy as np
# import matplotlib.pyplot as plt

# # Define alpha and gamma values
# alpha_values = [0.2, 0.3, 0.4]
# gamma_values = [0, 0.5, 1]

# # Define time values
# t_values = np.arange(0, 10, 1)

# # Plot phi(t) for each combination of alpha and gamma
# plt.figure(figsize=(10, 6))
# for alpha in alpha_values:
#     for gamma in gamma_values:
#         lam = alpha + (1 - alpha) * gamma
#         phi_t = np.exp(-lam * t_values)
#         label = f'alpha={alpha}, gamma={gamma}'
#         plt.plot(t_values, phi_t, label=label)

# plt.title('Plot of phi(t)')
# plt.xlabel('Time (t)')
# plt.ylabel('phi(t)')
# plt.legend()
# plt.grid(True)
# plt.show()
###############################alpha and gamma is not considered in the formula######################
# import numpy as np
# import matplotlib.pyplot as plt

# # Define lambda value for exponential decay
# lambda_value = 0.5  # Adjust as needed

# # Define time values
# t_values = np.arange(0, 11)

# # Calculate phi(t) for each time value
# phi_t_values = np.exp(-lambda_value * t_values)

# # Plot phi(t)
# plt.figure(figsize=(8, 6))
# plt.plot(t_values, phi_t_values, label='phi(t)')
# plt.title('Plot of phi(t)')
# plt.xlabel('Time (t)')
# plt.ylabel('phi(t)')
# plt.legend()
# plt.grid(True)
# plt.show()
############################################3
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
C = 1  # Constant cost
sigma = 0.9  # Discount factor
r = 10  # Reward

# Define time values
t_values = np.arange(0, 11)  # Time steps from 1 to 10

# Calculate utility for each time step
utility_values = []
for t in t_values:
    utility = -C * np.sum(sigma ** np.arange(t)) + (sigma ** t) * r
    utility_values.append(utility)

# Plot utility over time
plt.figure(figsize=(8, 6))
plt.plot(t_values, utility_values, marker='o')
plt.title('Utility over Time')
plt.xlabel('Time (t)')
plt.ylabel('Utility (U)')
plt.grid(True)
plt.show()
