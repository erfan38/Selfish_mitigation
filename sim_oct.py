import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
r = 10.0    # Reward for mining a block
C_a = 4.0   # Cost for the attacker
C_d = 3.0   # Cost for the defender
P = 5.0     # Penalty for concealing
phi_t = 0.9 # Decreasing factor of reward over time (e.g., 90% of original reward)

# Payoff functions
def pi_R(x):
    """Payoff for using the Reveal strategy."""
    return x * (phi_t * r - C_d) + (1 - x) * (-C_a)

def pi_C(x):
    """Payoff for using the Conceal strategy."""
    return x * (phi_t * r - C_d) + (1 - x) * (-C_a - P)

def average_payoff(x):
    """Average payoff across the population."""
    return x * pi_R(x) + (1 - x) * pi_C(x)

# Replicator equation: dx/dt = x * (pi_R - avg_pi)
def replicator_dynamics(x, t):
    pi_R_x = pi_R(x)
    avg_pi = average_payoff(x)
    dxdt = x * (pi_R_x - avg_pi)
    return dxdt

# Initial proportion of miners using the Reveal strategy
x0 = 0.1

# Time points for simulation
time_points = np.linspace(0, 50, 500)

# Solve the replicator dynamics equation using odeint
x_trajectory = odeint(replicator_dynamics, x0, time_points)

# Plot the results
plt.plot(time_points, x_trajectory, label="Proportion of Reveal Strategy")
plt.xlabel('Time')
plt.ylabel('Proportion of Miners Revealing (x)')
plt.title('Evolution of Proportion of Miners Revealing')
plt.grid(True)
plt.legend()
plt.show()
