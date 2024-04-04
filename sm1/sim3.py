import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r = 100  # reward
C_a = 8  # cost for attacker
C_d = 10  # cost for defender
P_0 = 10  # initial punishment
rep = 1  # reputation penalty

# Define function phi(t)
def phi(t, alpha, gamma):  # Add gamma as a parameter
    return 1 / (1 + alpha * t + gamma)  # Include gamma in the calculation

def phi_inverse(phi, alpha):
    return 1 / (alpha * phi) - 1 / alpha

# Function to calculate payoff for a given action combination
def calculate_payoff(action_attacker, action_defender, t, alpha_attacker, alpha_defender, gamma_attacker, gamma_defender):  # Add gamma as parameters
    payoff_matrix = np.array([
        [phi(t, alpha_attacker, gamma_attacker) * r - C_a + rep, phi(t, alpha_attacker, gamma_attacker) * r - C_d + rep, phi(t, alpha_attacker, gamma_attacker) * r - C_d + rep],
        [-C_a - phi(t, alpha_attacker, gamma_attacker) * phi_inverse(phi(t, alpha_defender, gamma_defender), alpha_defender) * P_0 - rep, phi(t, alpha_attacker, gamma_attacker) * r - C_d + rep, -rep],
        [-rep, -C_d - phi(t, alpha_defender, gamma_defender) * phi_inverse(phi(t, alpha_attacker, gamma_attacker), alpha_attacker) * P_0 - rep, -rep]
    ])
    return payoff_matrix[action_attacker, action_defender]

# Strategies for attacker and defender
def attacker_strategy(attacker_history, defender_history, t):
    if t % 5 == 0:
        return 1 if defender_history and defender_history[-1] == 0 else 0  # Conceal if defender revealed in the last round
    elif t % 5 == 1:
        return 0 if defender_history and defender_history[-1] == 1 else 1  # Reveal if defender concealed in the last round
    elif t % 5 == 2:
        return 0 if defender_history and defender_history[-1] == 1 else 1  # Reveal if defender concealed in the last round
    elif t % 5 == 3:
        return 0  # Reveal while the defender chooses quit
    else:
        return 1 if defender_history and defender_history[-1] == 0 else 0  # Quit if defender revealed in the last round

def defender_strategy(attacker_history, defender_history, t):
    if t % 5 == 0:
        return 0  # Revealing
    elif t % 5 == 1:
        return 1  # Concealing
    elif t % 5 == 2:
        return 1  # Concealing
    elif t % 5 == 3:
        return 2  # Quit
    else:
        return 0  # Revealing
   
# Simulation
num_rounds = 100  # Number of rounds to simulate
num_alpha_values = 100
alpha_values = np.linspace(0.01, 0.5, num_alpha_values)  # Hash rate between 0 and 0.5
attacker_utilities = []
defender_utilities = []

for alpha in alpha_values:
    total_attacker_utility = 0
    total_defender_utility = 0

    for t in range(num_rounds):
        # Initialize history for each round
        attacker_history = []
        defender_history = []

        for t_round in range(t):
            # Determine actions for attacker and defender based on strategies and history
            attacker_action = attacker_strategy(attacker_history, defender_history, t_round)
            defender_action = defender_strategy(attacker_history, defender_history, t_round)

            # Update history
            attacker_history.append(attacker_action)
            defender_history.append(defender_action)

        # Determine actions for attacker and defender based on strategies and history for the current round
        attacker_action = attacker_strategy(attacker_history, defender_history, t)
        defender_action = defender_strategy(attacker_history, defender_history, t)

        # Update history
        attacker_history.append(attacker_action)
        defender_history.append(defender_action)

        # Calculate payoff for each player
        attacker_payoff = calculate_payoff(attacker_action, defender_action, t, alpha, alpha, 0, 0)  # Set gamma to 0 for both attacker and defender
        defender_payoff = calculate_payoff(defender_action, attacker_action, t, alpha, alpha, 0, 0)  # Set gamma to 0 for both attacker and defender

        # Calculate utility for each player
        attacker_utility = attacker_payoff
        defender_utility = defender_payoff
        total_attacker_utility += attacker_utility
        total_defender_utility += defender_utility

    # Calculate average utility for each player
    avg_attacker_utility = total_attacker_utility / num_rounds
    avg_defender_utility = total_defender_utility / num_rounds

    attacker_utilities.append(avg_attacker_utility)
    defender_utilities.append(avg_defender_utility)

# Plotting
plt.plot(alpha_values, attacker_utilities, label='Attacker Utility')
plt.plot(alpha_values, defender_utilities, label='Defender Utility')
plt.xlabel('Hash Rate Alpha')
plt.ylabel('Average Utility')
plt.title('Average Utility vs. Hash Rate Alpha')
plt.legend()
plt.grid(True)
plt.show()
