import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r = 100  # reward
C_a = 8  # cost for attacker
C_d = 10  # cost for defender
P_0 = 10  # initial punishment
rep = 1  # reputation penalty

# Define function phi(t)
def phi(t, alpha, gamma):  # Include gamma in the function
    return 1 / (1 + alpha * t + gamma)

def phi_inverse(phi, alpha):
    return 1 / (alpha * phi) - 1 / alpha

# Function to calculate utility for a given action
def calculate_utility(action, t, alpha, gamma):
    if action == 0:  # Honest action
        return phi(t, alpha, gamma) * r - C_a + rep
    elif action == 1:  # Selfish action
        return -C_a - phi(t, alpha, gamma) * phi_inverse(phi(t, alpha, gamma), alpha)
    elif action == 2:  # Quit action
        return -rep

# Strategies for attacker and defender
def attacker_strategy(attacker_history, defender_history, t, alpha_attacker, alpha_defender, gamma):
    # Check if the attacker has a lead of at least 2 blocks
    if t >= 2 and attacker_history[t - 1] == 1 and attacker_history[t - 2] == 1:
        return 0  # Conceal the block if the attacker has a lead of at least 2 blocks
    
    # Check if the attacker has a lead of 1 block
    if t >= 1 and attacker_history[t - 1] == 1:
        return 0 if defender_history and defender_history[-1] == 0 else 1  # Reveal if defender concealed or quit in the last round
    
    # Check if the attacker is in the fork competition state
    if t >= 1 and defender_history and defender_history[-1] == 2:
        return 0 if np.random.rand() < gamma else 1  # Conceal with probability gamma if in fork competition state
    
    # Default strategy
    return 1 if defender_history and defender_history[-1] == 0 else 0  # Quit if defender revealed in the last round

def defender_strategy(attacker_history, defender_history, t, alpha_attacker, alpha_defender, gamma):
    # Check if the defender is in the same height state
    if t >= 1 and defender_history[-1] == 0:
        return 0 if attacker_history and attacker_history[-1] == 1 else 2  # Quit if attacker revealed in the last round
    
    # Check if the defender is in the selfish miner leads by 1 block state
    if t >= 1 and defender_history[-1] == 1:
        return 1 if attacker_history and attacker_history[-1] == 0 else 0  # Reveal if attacker concealed in the last round
    
    # Check if the defender is in the fork competition state
    if t >= 1 and defender_history[-1] == 2:
        return 0 if np.random.rand() < (1 - gamma) else 1  # Conceal with probability (1 - gamma) if in fork competition state
    
    # Default strategy
    return 1 if attacker_history and attacker_history[-1] == 0 else 0  # Conceal if attacker revealed in the last round

# Simulation
num_rounds = 100  # Number of rounds to simulate
num_alpha_values = 100
alpha_values = np.linspace(0.05, 0.95, num_alpha_values)  # Attacker's hash rate between 0.05 and 0.95
gamma_values = [0, 0.5, 1]  # Specific choices for gamma
attacker_utilities = {0: [], 0.5: [], 1: []}
defender_utilities = []

for gamma in gamma_values:
    for alpha_attacker in alpha_values:
        alpha_defender = 1 - alpha_attacker
        total_attacker_utility = 0
        total_defender_utility = 0
        attacker_history = []
        defender_history = []

        for t in range(num_rounds):
            # Determine actions for attacker and defender based on strategies and history
            attacker_action = attacker_strategy(attacker_history, defender_history, t, alpha_attacker, alpha_defender, gamma)
            defender_action = defender_strategy(attacker_history, defender_history, t, alpha_attacker, alpha_defender, gamma)

            # Update history
            attacker_history.append(attacker_action)
            defender_history.append(defender_action)

            # Calculate utility for each player
            attacker_utility = calculate_utility(attacker_action, t, alpha_attacker, gamma)
            defender_utility = calculate_utility(defender_action, t, alpha_defender, gamma)
            total_attacker_utility += attacker_utility
            total_defender_utility += defender_utility

        # Calculate average utility for each player
        avg_attacker_utility = total_attacker_utility / num_rounds
        avg_defender_utility = total_defender_utility / num_rounds

        attacker_utilities[gamma].append(avg_attacker_utility)
        if gamma == 0:
            defender_utilities.append(avg_defender_utility)  # Only append defender utility once for gamma = 0

# Plotting
plt.figure(figsize=(10, 6))
for gamma, utility in attacker_utilities.items():
    plt.plot(alpha_values, utility, label=f'Attacker Utility (Gamma={gamma})')
plt.plot(alpha_values, defender_utilities, label='Defender Utility', color='black', linestyle='--')
plt.xlabel('Hash rate')
plt.ylabel('Average Utility')
plt.title('Average Utility vs. Hash rate')
plt.legend()
plt.grid(True)
plt.show()

# Display history of the attacker and the defender
print("Attacker History:", attacker_history)
print("Defender History:", defender_history)
