import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r = 100  # reward
C_a = 10  # cost for attacker
C_d = 10  # cost for defender
P_0 = 10  # initial punishment
rep = 1  # reputation penalty

# Define function phi(t)
def phi(t, alpha):
    return 1 / (1 + alpha * t)

# Function to calculate payoff for a given action combination
def calculate_payoff(action_attacker, action_defender, t, alpha_attacker, alpha_defender):
    payoff_matrix = np.array([
        [phi(t, alpha_attacker) * r - C_a + rep, phi(t, alpha_attacker) * r - C_d + rep, phi(t, alpha_attacker) * r - C_d + rep],
        [-C_a - phi(t, alpha_attacker) * P_0 - rep, phi(t, alpha_attacker) * r - C_d + rep, -rep],
        [-rep, -C_d - phi(t, alpha_defender) * P_0 - rep, -rep]
    ])
    return payoff_matrix[action_attacker, action_defender]

# Strategies for attacker and defender
def attacker_strategy(attacker_history, defender_history, t):
    if t == 0:
        return 0  # Initial strategy
    else:
        # Example: Attacker adjusts strategy based on the last action of the defender
        if defender_history[-1] == 0:  # If the defender chose to reveal last round
            return 1  # Attacker conceals
        else:
            return 0  # Attacker reveals

def defender_strategy(attacker_history, defender_history, t):
    if t == 0:
        return 0  # Initial strategy
    else:
        # Example: Defender adjusts strategy based on the last action of the attacker
        if attacker_history[-1] == 0:  # If the attacker chose to reveal last round
            return 1  # Defender conceals
        else:
            return 0  # Defender reveals

# Simulation
num_rounds = 100  # Number of rounds to simulate
num_alpha_values = 100
alpha_values = np.linspace(0, 0.5, num_alpha_values)  # Hash rate between 0 and 0.5
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
        attacker_payoff = calculate_payoff(attacker_action, defender_action, t, alpha, alpha)
        defender_payoff = calculate_payoff(defender_action, attacker_action, t, alpha, alpha)

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