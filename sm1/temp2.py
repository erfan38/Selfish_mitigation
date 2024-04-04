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

def attacker_strategy(attacker_history, defender_history, t, alpha_attacker, alpha_defender, gamma):
    if t == 0:
        return 0  # Start with honest mining
    else:
        # Attacker adjusts strategy based on the defender's previous action
        if defender_history[-1] == 0:  # If the defender revealed last round
            return 0  # Attacker chooses to reveal (honest mining)
        elif defender_history[-1] == 1:  # If the defender concealed last round
            return 1  # Attacker chooses to reveal (honest mining)
        elif defender_history[-1] == 2:  # If the defender quit last round
            return 1  # Attacker chooses to conceal (selfish mining)

def defender_strategy(attacker_history, defender_history, t, alpha_attacker, alpha_defender, gamma):
    if t == 0:
        return 0  # Start with honest mining
    else:
        # Defender adjusts strategy based on the attacker's previous action
        if attacker_history[-1] == 0:  # If the attacker revealed last round
            return 0  # Defender chooses to reveal (honest mining)
        elif attacker_history[-1] == 1:  # If the attacker concealed last round
            return 1  # Defender chooses to conceal (selfish mining)
        elif attacker_history[-1] == 2:  # If the attacker quit last round
            return 0  # Defender chooses to reveal (honest mining)

# Simulation
num_rounds = 100  # Number of rounds to simulate
num_alpha_values = 100
alpha_values = np.linspace(0.1, 0.33, num_alpha_values)  # Attacker's hash rate between 0.1 and 0.33
gamma_values = [0, 0.5, 1]  # Specific choices for gamma
attacker_utilities = {0: [], 0.5: [], 1: []}
defender_utilities = []
attacker_history = []
defender_history = []
for gamma in gamma_values:

    for alpha_attacker in alpha_values:
        alpha_defender = 1 - alpha_attacker
        total_attacker_utility = 0
        total_defender_utility = 0

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

        # Display history of the attacker and the defender
        print("********************************************")
        print("alpha is:", alpha_attacker)
        print("gamma is:", gamma)
        print(" honest miners win block is:", defender_history.count(0))
        print(" selfish miners win block is:", defender_history.count(1))
        print(" total mined block is:", num_rounds)
        print(" total stale block is:", defender_history.count(2))
        print(" honest miner revenue is:", defender_history.count(0) * r)
        print(" selfish miner revenue is:", defender_history.count(1) * r)
        print(" honest miner expected reward is:", r * (1 - gamma))
        print(" selfish miner expected reward is:", r * gamma)
        print("********************************************")    

print("Attacker History:", attacker_history)
print("Defender History:", defender_history)

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