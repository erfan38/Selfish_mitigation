import numpy as np

# Define states
states = ['S0', 'S1', 'S0_prime', 'S2', 'S3', 'S_n']

# Define actions for each player
actions_attacker = ['Reveal', 'Conceal']
actions_defender = ['Reveal', 'Conceal']

# Define parameters
r = 1  # Reward for mining a block
C_a = 0.5  # Cost of mining for the attacker
C_d = 0.5  # Cost of mining for the defender
P = 0.1  # Punishment for concealing a block
gamma = 0.5  # Proportion of honest miners working on the selfish miner's block

# Define alpha and gamma
alpha = 0.3  # Transition probability parameter

# Define transition probabilities based on the specified conditions
transition_probs = {
    'S0': {
        'Reveal': {'S0': 1 - alpha, 'S1': alpha, 'S0_prime': 0.0, 'S2': 0.0, 'S3': 0.0, 'S_n': 0.0},
        'Conceal': {'S0': alpha, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 0.0, 'S3': 0.0, 'S_n': 0.0}
    },
    'S1': {
        'Reveal': {'S0': 0.0, 'S1': 1 - alpha, 'S0_prime': 0.0, 'S2': alpha, 'S3': 0.0, 'S_n': 0.0},
        'Conceal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 1.0, 'S3': 0.0, 'S_n': 0.0}
    },
    'S0_prime': {
        'Reveal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 1 - gamma, 'S2': 0.0, 'S3': 0.0, 'S_n': 0.0},
        'Conceal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': gamma, 'S2': 0.0, 'S3': 0.0, 'S_n': 0.0}
    },
    'S2': {
        'Reveal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 1 - alpha, 'S3': alpha, 'S_n': 0.0},
        'Conceal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 0.0, 'S3': 1.0, 'S_n': 0.0}
    },
    'S3': {
        'Reveal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 0.0, 'S3': 1 - alpha, 'S_n': alpha},
        'Conceal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 0.0, 'S3': 0.0, 'S_n': 1.0}
    },
    'S_n': {
        'Reveal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 0.0, 'S3': 0.0, 'S_n': 1.0},
        'Conceal': {'S0': 0.0, 'S1': 0.0, 'S0_prime': 0.0, 'S2': 0.0, 'S3': 0.0, 'S_n': 0.0}
    }
}

# Initialize payoff matrix
payoff_matrix = {}

# Populate payoff matrix
for attacker_action in actions_attacker:
    for defender_action in actions_defender:
        payoff_matrix[(attacker_action, defender_action)] = {}

        for i, state in enumerate(states):
            if state == 'S0':
                if attacker_action == 'Reveal' and defender_action == 'Conceal':
                    payoff_matrix[(attacker_action, defender_action)][i] = (r - C_a, r - C_d)
                elif attacker_action == 'Reveal' and defender_action == 'Reveal':
                    payoff_matrix[(attacker_action, defender_action)][i] = (r - C_a, r - C_d)
                elif attacker_action == 'Conceal' and defender_action == 'Conceal':
                    payoff_matrix[(attacker_action, defender_action)][i] = (-C_a - P, -C_d - P)
                else:
                    payoff_matrix[(attacker_action, defender_action)][i] = (-C_a - P, r - C_d)
            else:
                payoff_matrix[(attacker_action, defender_action)][i] = (0, 0)

# Initialize strategy profile
strategy_attacker = {i: np.random.choice(actions_attacker) for i in range(len(states))}
strategy_defender = {i: np.random.choice(actions_defender) for i in range(len(states))}

# Iterative algorithm (similar to previous code, adapted for the provided scenario)
max_iter = 100
epsilon = 1e-6
for _ in range(max_iter):
    new_strategy_attacker = {}
    new_strategy_defender = {}

    for i, state in enumerate(states):
        expected_payoffs_attacker = {}
        expected_payoffs_defender = {}

        for attacker_action in actions_attacker:
            for defender_action in actions_defender:
                expected_payoffs_attacker[(attacker_action, defender_action)] = sum(
                    transition_probs[state][attacker_action][next_state] *
                    payoff_matrix[(attacker_action, defender_action)][next_state]
                    for next_state in states
                )

                expected_payoffs_defender[(attacker_action, defender_action)] = sum(
                    transition_probs[state][defender_action][next_state] *
                    payoff_matrix[(attacker_action, defender_action)][next_state]
                    for next_state in states
                )

        new_strategy_attacker[i] = max(expected_payoffs_attacker, key=expected_payoffs_attacker.get)
        new_strategy_defender[i] = max(expected_payoffs_defender, key=expected_payoffs_defender.get)

    if (new_strategy_attacker == strategy_attacker) and (new_strategy_defender == strategy_defender):
        break
    else:
        strategy_attacker = new_strategy_attacker
        strategy_defender = new_strategy_defender

# Output strategy profile
print("Markov-Perfect Equilibrium:")
print("Attacker Strategy:", strategy_attacker)
print("Defender Strategy:", strategy_defender)
