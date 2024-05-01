import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r = 150  # reward (increased for positive utility)
C_a = 10  # cost for attacker (increased for positive utility)
C_d = 15  # cost for defender (increased for positive utility)
P_0 = 5  # initial punishment (reduced for positive utility)
rep = 0.5  # reputation penalty (reduced for positive utility)
gamma_attacker = 0.5  # Choose a specific value for gamma_attacker
gamma_defender = 1 - gamma_attacker  # Calculate gamma_defender based on gamma_attacker

# Define function phi(t)
def phi(t, lmbda):
    return np.exp(-lmbda * t)

# Function to calculate payoff for a given action combination
def calculate_payoff(action_attacker, action_defender, t):
    attacker_punishment = -C_a - phi(t, gamma_defender) * P_0 - rep
    defender_punishment = -C_d - phi(t, gamma_attacker) * P_0 - rep
    
    if action_attacker == 0 and action_defender == 0:  # Both reveal
        return phi(t, gamma_attacker) * r - C_a + rep, phi(t, gamma_attacker) * r - C_d + rep
    elif action_attacker == 0 and action_defender == 1:  # Attacker reveals, defender conceals
        return attacker_punishment, phi(t, gamma_attacker) * r - C_d + rep
    elif action_attacker == 1 and action_defender == 0:  # Attacker conceals, defender reveals
        return phi(t, gamma_attacker) * r - C_a + rep, defender_punishment
    else:  # Both conceal
        return attacker_punishment, defender_punishment

# Backward Induction
def backward_induction(num_rounds):
    # Initialize matrices to store payoffs
    attacker_payoffs = np.zeros((num_rounds + 1, 2))  # Rows: t, Columns: action
    defender_payoffs = np.zeros((num_rounds + 1, 2))  # Rows: t, Columns: action
    attacker_utilities = np.zeros(num_rounds + 1)
    defender_utilities = np.zeros(num_rounds + 1)

    # Calculate payoffs for the last round (t = num_rounds)
    for a_attacker in range(2):
        for a_defender in range(2):
            attacker_payoffs[num_rounds, a_attacker], defender_payoffs[num_rounds, a_defender] = calculate_payoff(a_attacker, a_defender, num_rounds)

    # Backward induction
    for t in range(num_rounds - 1, -1, -1):
        # Calculate payoffs for each possible action combination at time t
        for a_attacker in range(2):
            for a_defender in range(2):
                attacker_payoffs[t, a_attacker], defender_payoffs[t, a_defender] = calculate_payoff(a_attacker, a_defender, t)

        # Update the payoffs for each player based on the future expected payoffs
        for a_attacker in range(2):
            for a_defender in range(2):
                expected_attacker_payoff = attacker_payoffs[t + 1, a_attacker]
                expected_defender_payoff = defender_payoffs[t + 1, a_defender]
                attacker_payoffs[t, a_attacker] += expected_attacker_payoff
                defender_payoffs[t, a_defender] += expected_defender_payoff

        # Calculate utilities for each player at time t
        attacker_utilities[t] = max(attacker_payoffs[t])
        defender_utilities[t] = max(defender_payoffs[t])

    # Determine the Markov Perfect Equilibrium strategies
    attacker_strategy = np.argmax(attacker_payoffs, axis=1)
    defender_strategy = np.argmax(defender_payoffs, axis=1)

    return attacker_utilities, defender_utilities, attacker_strategy, defender_strategy

# Calculate MPE strategies and utilities
num_rounds = 10
attacker_utilities, defender_utilities, attacker_strategy, defender_strategy = backward_induction(num_rounds)

# Plot utilities over time
t_values = np.arange(num_rounds + 1)
plt.plot(t_values, attacker_utilities, label='Attacker Utility')
plt.plot(t_values, defender_utilities, label='Defender Utility')
plt.xlabel('Time (t)')
plt.ylabel('Utility')
plt.title('Utility Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Print MPE strategies
print("Markov Perfect Equilibrium Strategies:")
print("Attacker:", attacker_strategy)
print("Defender:", defender_strategy)
