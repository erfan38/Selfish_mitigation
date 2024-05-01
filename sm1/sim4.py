import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r = 150  # reward (increased for positive utility)
C_a = 10  # cost for attacker (increased for positive utility)
C_d = 15  # cost for defender (increased for positive utility)
P_0 = 5  # initial punishment (reduced for positive utility)
rep = 0.5  # reputation penalty (reduced for positive utility)

# Define function phi(t)
def phi(t, lmbda):
    return np.exp(-lmbda * t)

def phi_inverse(phi, lmbda):
    return -np.log(phi) / lmbda

# Function to calculate payoff for a given action combination
def calculate_payoff(action_attacker, action_defender, t, gamma_attacker, gamma_defender):
    if action_attacker == 0 and action_defender == 0:  # Both reveal
        return phi(t, gamma_attacker) * r - C_a + rep, phi(t, gamma_attacker) * r - C_d + rep
    elif action_attacker == 0 and action_defender == 1:  # Attacker reveals, defender conceals
        return -C_a - phi_inverse(phi(t, gamma_defender), gamma_defender) * P_0 - rep, phi(t, gamma_attacker) * r - C_d + rep
    elif action_attacker == 1 and action_defender == 0:  # Attacker conceals, defender reveals
        return phi(t, gamma_attacker) * r - C_a + rep, -C_d - phi(t, gamma_defender) * phi_inverse(phi(t, gamma_attacker), gamma_attacker) * P_0 - rep
    else:  # Both conceal
        return -C_a - phi_inverse(phi(t, gamma_defender), gamma_defender) * P_0 - rep, -C_d - phi(t, gamma_defender) * phi_inverse(phi(t, gamma_attacker), gamma_attacker) * P_0 - rep

# Strategies for attacker and defender
def attacker_strategy(attacker_history, defender_history, t):
    if t % 2 == 0:
        return 1 if defender_history and defender_history[-1] == 0 else 0  # Conceal if defender revealed in the last round
    else:
        return 0 if defender_history and defender_history[-1] == 1 else 1  # Reveal if defender concealed in the last round

def defender_strategy(attacker_history, defender_history, t):
    if t == 0:
        return 1  # Revealing
    else:
        return 0  # Concealing

# Simulation
num_rounds = 100  # Number of rounds to simulate
t_values = np.arange(11)  # Rounds from 0 to 10
gamma_attacker = 0.5  # Choose a specific value for gamma_attacker
gamma_defender = 1 - gamma_attacker  # Calculate gamma_defender based on gamma_attacker

attacker_utilities = []
defender_utilities = []

for t in t_values:
    total_attacker_utility = []
    total_defender_utility = []

    for _ in range(num_rounds):
        # Determine actions for attacker and defender based on strategies and current round
        attacker_action = attacker_strategy([], [], t)
        defender_action = defender_strategy([], [], t)

        # Calculate payoff for each player
        attacker_payoff, defender_payoff = calculate_payoff(attacker_action, defender_action, t, gamma_attacker, gamma_defender)

        # Calculate utility for each player
        total_attacker_utility.append(attacker_payoff)
        total_defender_utility.append(defender_payoff)

    # Store the average utility for each player
    avg_attacker_utility = np.mean(total_attacker_utility)
    avg_defender_utility = np.mean(total_defender_utility)

    attacker_utilities.append(avg_attacker_utility)
    defender_utilities.append(avg_defender_utility)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(t_values, attacker_utilities, label='Attacker Utility', marker='o')
plt.plot(t_values, defender_utilities, label='Defender Utility', marker='o')
plt.xlabel('Round (t)')
plt.ylabel('Average Utility')
plt.title('Average Utility vs. Round (t)')
plt.legend()
plt.grid(True)
plt.show()
