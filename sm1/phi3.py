import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_transition_probs(alpha, gamma):
    transition_probs = np.zeros((4, 4))
    
    # Transition from state S0 to S1
    transition_probs[0, 1] = alpha
    
    # Transition from state S1 to S0'
    transition_probs[1, 2] = 1 - alpha
    
    # Transition from state S0' to S0
    transition_probs[2, 0] = gamma * (1 - alpha)
    
    # Transition from state S0' to S0'
    transition_probs[2, 2] = (1 - gamma) * (1 - alpha)
    
    # Transition from state S0 to S0
    transition_probs[0, 0] = (1 - alpha)
    
    # Transition from state S1 to S2
    transition_probs[1, 3] = alpha
    
    # Transition from state S2 to S2
    transition_probs[3, 3] = alpha
    
    # Transition from state S2 to S1
    transition_probs[3, 1] = 1 - alpha
    
    return transition_probs

def compute_utility(alpha, gamma, r, C, t_max):
    transition_probs = compute_transition_probs(alpha, gamma)
    
    # Initialize utility values
    utility_values = np.zeros(t_max + 3)  # Increase the size by 3
    
    # Compute utility for each state
    for t in range(t_max):
        # Utility for state S0
        utility_values[t] = (-C * np.sum((1 - alpha) ** np.arange(t)) + (1 - alpha) ** t * r)
        
        # Utility for state S1
        utility_values[t + 1] = (-C * np.sum((1 - alpha) ** np.arange(t + 1)) + (1 - alpha) ** (t + 1) * r)
        
        # Utility for state S0'
        utility_values[t + 2] = (-C * np.sum((1 - alpha) ** np.arange(t + 1)) + (1 - alpha) ** (t + 2) * r)
        
        # Utility for state S2
        if t + 3 < t_max + 3:  # Check if index is within bounds
            utility_values[t + 3] = (-C * np.sum((1 - alpha) ** np.arange(t + 1)) + (1 - alpha) ** (t + 3) * r)
        
    return utility_values

def display_utility_table(utility_values, alpha, gamma):
    df = pd.DataFrame({'Time (t)': np.arange(len(utility_values)), 'Utility (U(t))': utility_values})
    df['Alpha'] = alpha
    df['Gamma'] = gamma
    return df

# Define parameters
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
gamma_values = [0, 0.5, 1]
r = 10  # Reward
C = 1  # Constant cost
t_max = 10  # Maximum time step

# Create plot
plt.figure(figsize=(12, 8))

# Compute and plot utility values for each combination of alpha and gamma
for alpha in alpha_values:
    for gamma in gamma_values:
        utility_values = compute_utility(alpha, gamma, r, C, t_max)
        plt.plot(range(len(utility_values)), utility_values, label=f'Alpha={alpha}, Gamma={gamma}')

# Add labels and legend to the plot
plt.xlabel('Time (t)')
plt.ylabel('Utility (U(t))')
plt.title('Utility Curves for Different Alpha and Gamma Values')
plt.legend()
plt.grid(True)
plt.show()

# Display utility tables
for alpha in alpha_values:
    for gamma in gamma_values:
        utility_values = compute_utility(alpha, gamma, r, C, t_max)
        table = display_utility_table(utility_values, alpha, gamma)
        print(f"Utility Table for Alpha={alpha}, Gamma={gamma}:")
        print(table)
        print("\n")
