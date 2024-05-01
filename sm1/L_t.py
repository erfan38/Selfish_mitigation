# import numpy as np
# import matplotlib.pyplot as plt

# def L(t, Ci, sigma):
#     return Ci * (1 - sigma**t) / (1 - sigma)

# # Define parameters
# Ci = 100  # Example value for Ci
# sigma = 0.9  # Example value for sigma

# # Generate time values
# t_values = np.arange(0, 10, 0.1)

# # Calculate L(t) for each time value
# L_values = L(t_values, Ci, sigma)

# # Plot
# plt.plot(t_values, L_values)
# plt.xlabel('Time (t)')
# plt.ylabel('L(t)')
# plt.title('Plot of L(t)')
# plt.grid(True)
# plt.show()
##################reverse ################33
# import numpy as np
# import matplotlib.pyplot as plt

# def reverse_L(L, Ci, sigma):
#     return np.log(1 - L * (1 - sigma) / -Ci) / np.log(sigma)

# # Define parameters
# Ci = 10  # Example value for Ci
# sigma = 0.9  # Example value for sigma

# # Generate L values
# L_values = np.linspace(-100, 100, 1000)

# # Calculate corresponding t values using the reversed formula
# t_values = reverse_L(L_values, Ci, sigma)

# # Plot
# plt.plot(L_values, t_values)
# plt.xlabel('L')
# plt.ylabel('Time (t)')
# plt.title('Reversed Formula: Plot of Time (t) vs L')
# plt.grid(True)
# plt.show()

##################### Punishment function #################3
# import numpy as np
# import matplotlib.pyplot as plt

# # Define the parameters
# P_0 = 10
# sigma = 0.9

# # Define the function
# def L(t):
#     return P_0 * (1 - sigma**t) / (1 - sigma)

# # Generate t values
# t_values = np.linspace(0, 10, 100)

# # Calculate L(t) values
# L_values = L(t_values)

# # Plot
# plt.figure(figsize=(8, 6))
# plt.plot(t_values, L_values, label='L(t)', color='blue')
# plt.title('Plot of L(t)')
# plt.xlabel('t')
# plt.ylabel('L(t)')
# plt.grid(True)
# plt.legend()
# plt.show()

##################phi function only with t and lambda ###########3
import numpy as np
import matplotlib.pyplot as plt

def phi_t(t, lmbda):
    return np.exp(-lmbda * t)

# Define the range of t
t_values = np.linspace(0, 5, 100)

# Define the value of lambda
lmbda = 0.9  

# Calculate phi(t) for each value of t
phi_values = 100 * phi_t(t_values, lmbda)

# Plot phi(t)
plt.plot(t_values, phi_values)
plt.xlabel('t')
plt.ylabel('phi(t)')
plt.title('Plot of phi(t) = e^(-lambda * t)')
plt.grid(True)
plt.show()
