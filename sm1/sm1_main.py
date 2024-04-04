from sm1_strategy import SelfishMiningOne
#from sim2 import alpha_values
import numpy as np
iteration_number = 10000
num_rounds = 100  # Number of rounds to simulate
num_alpha_values = 100
# # Define parameters
# r = 100  # reward
# C_a = 10  # cost for attacker
# C_d = 10  # cost for defender
# P_0 = 10  # initial punishment
# rep = 1  # reputation penalty

selfish_mining_one = SelfishMiningOne(False)
#selfish_mining_one.alpha = 0.5             #we removed these two lines as now the main program is iterating over 'alpha_values' defined in sim2.py
#selfish_mining_one.gamma = 1
selfish_mining_one.print_input_statistic()

alpha_values = np.linspace(0, 0.5, num_alpha_values)  # Hash rate between 0 and 0.5

for alpha in alpha_values:
    selfish_mining_one.alpha = alpha
    selfish_mining_one.start_simulate(iteration_number)
    selfish_mining_one.print_final_result()


selfish_mining_one.visualize_data(iteration_number)
