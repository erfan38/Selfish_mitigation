import numpy as np
import matplotlib.pyplot as plt

# Define constants
C = 1
P = 1
sigma = 0.2
r = 100  # interest rate

# Define range of t values
t_values = np.arange(0, 11)

# Calculate L(t) using summation
L_sum = []
for t in t_values:
    L_t_sum = -C - P * np.sum(sigma ** np.arange(t + 1))
    L_sum.append(L_t_sum)

# Calculate U(t)
U = []
for t in t_values:
    U_t = -C - P * np.sum(sigma ** np.arange(t + 1)) + (sigma ** t) * r
    U.append(U_t)

# Plotting L(t) and U(t)
plt.figure(figsize=(10, 6))
plt.plot(t_values, L_sum, label=r'$L(t) = -C -P \sum \sigma ^{i}$', marker='o')
plt.plot(t_values, U, label=r'$U(t) = -C -P \sum \sigma ^{i} + \sigma ^ {t} r$', marker='s')
plt.xlabel('t')
plt.ylabel('Value')
plt.title('Comparison of $L(t)$ and $U(t)$')
plt.legend()
plt.grid(True)
plt.show()
