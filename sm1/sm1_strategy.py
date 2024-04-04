# # first strategy by using Sirer Aritcla published in 2014 
# # α: attacker’s relative hashrate,  γ: the communication capabilities of the attacker
import sys
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import csv
#from sim2 import calculate_payoff, phi
# Define parameters
r = 100  # reward
C_a = 10  # cost for attacker
C_d = 10  # cost for defender
P_0 = 10  # initial punishment
rep = 1  # reputation penalty

def phi(t, alpha):
    return 1 / (1 + alpha * t)
def phi_inverse(phi, alpha):
    return 1 / (alpha * phi) - 1 / alpha

# Function to calculate payoff for a given action combination
# def calculate_payoff(action_attacker, action_defender, t, alpha_attacker, alpha_defender):
#     payoff_matrix = np.array([
#         [phi(t, alpha_attacker) * r - C_a + rep, phi(t, alpha_attacker) * r - C_d + rep, phi(t, alpha_attacker) * r - C_d + rep],
#         [-C_a - phi(t, alpha_attacker) * P_0 - rep, phi(t, alpha_attacker) * r - C_d + rep, -rep],
#         [-rep, -C_d - phi(t, alpha_defender) * P_0 - rep, -rep]
#     ])
#     return payoff_matrix[action_attacker, action_defender]
#####################################################################################################
# def calculate_payoff(action_attacker, action_defender, t, alpha_attacker, alpha_defender):
#     payoff_matrix = np.array([
#         [phi(t, alpha_attacker) * r - C_a + rep, phi(t, alpha_attacker) * r - C_d + rep, phi(t, alpha_attacker) * r - C_d + rep],
#         [-C_a - phi(t, alpha_attacker) * phi_inverse(phi(t, alpha_defender), alpha_defender) * P_0 - rep, phi(t, alpha_attacker) * r - C_d + rep, -rep],
#         [-rep, -C_d - phi(t, alpha_defender) * phi_inverse(phi(t, alpha_attacker), alpha_attacker) * P_0 - rep, -rep]
#     ])
#     return payoff_matrix[action_attacker, action_defender]
#####################################################################################################
def calculate_payoff(action_attacker, action_defender, t, alpha_attacker, alpha_defender):
    if action_attacker == "Selfish":  # Attacker chooses Selfish behavior
        attacker_payoff = -C_a - phi_inverse(t, alpha_attacker) * P_0 - rep
    elif action_attacker == "Honest":   # Attacker chooses Honest behavior
        attacker_payoff = phi(t, alpha_attacker) * r - C_a + rep
    else:  # Attacker chooses Quit
        attacker_payoff = -rep

    if action_defender == "Selfish":  # Defender chooses Selfish behavior
        defender_payoff = -C_d - phi_inverse(t, alpha_defender) * P_0 - rep
    elif action_defender == "Honest":  # Defender chooses Honest behavior
        defender_payoff = phi(t, alpha_defender) * r - C_d + rep
    else:  # Defender chooses Quit
        defender_payoff = -rep

    return attacker_payoff, defender_payoff

class SelfishMiningOne:
    def __init__(self, show_log=False): #I changed it to true
        self._alpha = 0
        self._gamma = 0
        self.__selfish_history = []     
        self.__honest_history = []      
        self.__history = []
        self.__show_log = show_log
        self.__honest_miners_win = 0  # Add this line to initialize the attribute


        random.seed(None)

        self.__public_chain_length = 0
        self.__private_chain_length = 0
        self.__delta = 0

        self.__selfish_miners_win_block = 0
        self.__honest_miners_win_block = 0

        self.__selfish_miner_revenue = 0
        self.__honest_miner_revenue = 0

        self.__total_mined_block = 0
        self.__total_stale_block = 0

        self.__iteration_number = 0

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0 or value > 0.5:
            raise Exception("invalid value for alpha!")
        self._alpha = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value < 0 or value > 1:
            raise Exception("invalid value for gamma!")
        self._gamma = value

    @property
    def revenue(self):
        return self.__selfish_miner_revenue

    @property
    def stale_block(self):
        return self.__total_stale_block

    def print_input_statistic(self):
        print('alpha is : {}'.format(self._alpha))
        print('gamma is : {}'.format(self._gamma))

    def start_simulate(self, iteration):
        self.log('start simulating')
        self.__iteration_number = iteration

        for i in range(iteration):
            self.log("found a new block")

            random_number = random.random()

            self.calculating_delta()

            # Mining Process
            if random_number < self._alpha:
                self.__selfish_history.append("Selfish")
                self.start_selfish_mining()
            else:
                self.__honest_history.append("Honest")
                self.start_honest_mining()
            
            # Calculate payoffs
            payoff_selfish = calculate_payoff(0, 0, i+1, self._alpha, self._alpha)
            payoff_honest = calculate_payoff(0, 0, i+1, 1 - self._alpha, self._alpha)
            self.log(f"Payoff for Selfish: {payoff_selfish}, Payoff for Honest: {payoff_honest}")

            # Record relevant information for each iteration
            iteration_info = {
                "iteration": i+1,
                "public_chain_length": self.__public_chain_length,
                "private_chain_length": self.__private_chain_length,
                "delta": self.__delta,
                "selfish_miners_win_block": self.__selfish_miners_win_block,
                "honest_miners_win_block": self.__honest_miners_win_block
            }
            self.__history.append(iteration_info)
        
        self.calculating_output()

    def start_selfish_mining(self):
        self.log('starting selfish mining!')
        self.__private_chain_length += 1

        if self.__delta == 0 and self.__private_chain_length == 2:
            self.__selfish_miners_win_block += 2
            self.__private_chain_length = 0
            self.__public_chain_length = 0

    def start_honest_mining(self):
        self.log('starting honest mining!')
        self.__public_chain_length += 1

        if self.__delta == 0 and self.__private_chain_length == 0:
            self.__honest_miners_win_block += 1
            self.__private_chain_length = 0
            self.__public_chain_length = 0
        elif self.__delta == 0 and self.__private_chain_length == 1:
            gamma_random = random.random()
            if gamma_random < self._gamma:
                self.__selfish_miners_win_block += 1
                self.__honest_miners_win_block += 1
                self.__private_chain_length = 0
                self.__public_chain_length = 0
            else:
                self.__honest_miners_win_block += 2
                self.__private_chain_length = 0
                self.__public_chain_length = 0
        elif self.__delta == 2:
            self.__selfish_miners_win_block += self.__private_chain_length
            self.__public_chain_length = 0
            self.__private_chain_length = 0

    def calculating_delta(self):
        self.__delta = self.__private_chain_length - self.__public_chain_length
        self.log('delta is : {}'.format(self.__delta))

    def calculating_output(self):
        self.__total_mined_block = self.__honest_miners_win_block + self.__selfish_miners_win_block
        self.__total_stale_block = self.__iteration_number - self.__total_mined_block

        self.__honest_miner_revenue = float(self.__honest_miners_win_block / self.__total_mined_block) * 100
        self.__selfish_miner_revenue = float(self.__selfish_miners_win_block / self.__total_mined_block) * 100

    def print_final_result(self):
        print('********************************************')
        self.print_input_statistic()
        print('honest miners win block is : {}'.format(self.__honest_miners_win_block))
        print('selfish miners win block is : {}'.format(self.__selfish_miners_win_block))
        print('total mined block is : {}'.format(self.__total_mined_block))
        print('total stale block is : {}'.format(self.__total_stale_block))
        print('honest miner revenue is : {}'.format(self.__honest_miner_revenue))
        print('selfish miner revenue is : {}'.format(self.__selfish_miner_revenue))
        print('honest miner expected reward is : {}'.format((1 - self._alpha) * self.__iteration_number * 100 / self.__iteration_number))
        print('selfish miner expected reward is : {}'.format((self._alpha) * self.__iteration_number * 100 / self.__iteration_number))
        print("Selfish Miner History: ", self.__selfish_history)
        print("Honest Miner History: ", self.__honest_history)
        print('********************Results!************************')

    def log(self, log_message):
        if self.__show_log:
            print(log_message)

    def reset(self):
        random.seed(None)
        self.__public_chain_length = 0
        self.__private_chain_length = 0
        self.__delta = 0
        self.__selfish_miners_win_block = 0
        self.__honest_miners_win=0 #add this line to reset the attribute

    def visualize_data(self, iteration_number):
        alpha_values = [x / 100 for x in range(51) if x % 5 == 0]
        selfish_revenue_value_0 = []
        selfish_revenue_value_0_5 = []
        selfish_revenue_value_1 = []
        honest_revenue_value = []

        for alpha in alpha_values:
            self.alpha = alpha
            self.gamma = 0
            self.start_simulate(iteration_number)
            selfish_revenue_value_0.append(self.__selfish_miner_revenue)
            self.reset()

        for alpha in alpha_values:
            self.alpha = alpha
            self.gamma = 0.5
            self.start_simulate(iteration_number)
            selfish_revenue_value_0_5.append(self.__selfish_miner_revenue)
            self.reset()

        for alpha in alpha_values:
            self.alpha = alpha
            self.gamma = 1
            self.start_simulate(iteration_number)
            selfish_revenue_value_1.append(self.__selfish_miner_revenue)
            self.reset()

        for alpha in alpha_values:
            honest_revenue_value.append(alpha * 100)

        plt.plot(
            alpha_values, selfish_revenue_value_0, color='r', label='gamma = 0')
        plt.plot(
            alpha_values, selfish_revenue_value_0_5, color='y', label='gamma = 0.5')
        plt.plot(
            alpha_values, selfish_revenue_value_1, color='g', label='gamma = 1')

        plt.plot(alpha_values, honest_revenue_value,
                 color='k', label='honest mining')

        plt.title('Selfish Mining')
        plt.xlabel('Pool size')
        plt.ylabel('Relative Revenue')

        plt.legend(loc="upper left")

        plt.show()
    
    def save_strategy_history(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Iteration', 'Selfish Strategy', 'Honest Strategy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(self.__selfish_history)):
                writer.writerow({'Iteration': i+1, 'Selfish Strategy': self.__selfish_history[i], 'Honest Strategy': self.__honest_history[i]})
def main():
    iteration_number = 100
    num_alpha_values = 100

    selfish_mining_one = SelfishMiningOne(False) #i changed it to true
    selfish_mining_one.print_input_statistic()

    alpha_values = np.linspace(0, 0.5, num_alpha_values)

    for alpha in alpha_values:
        selfish_mining_one.alpha = alpha
        selfish_mining_one.start_simulate(iteration_number)
        selfish_mining_one.print_final_result()

    selfish_mining_one.save_strategy_history('strategy_history.csv')
    #selfish_mining_one.visualize_data(iteration_number)

if __name__ == "__main__":
    main()





# import sys
# import os
# sys.path.insert(0, os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..')))

# import random  # NOQA
# import numpy as np # for array operations
# from matplotlib import pyplot as plt  # NOQA
# from sim2 import calculate_payoff, phi

# class SelfishMiningOne:
#     def __init__(self, show_log=False):
#         self._alpha = 0
#         self._gamma = 0
#         self.__selfish_history = []     # Added a history for keeping the strategies
#         self.__honest_history = []      # Added a history...
#         self.__history = []
#         self.__show_log = show_log

#         random.seed(None)

#         self.__public_chain_length = 0
#         self.__private_chain_length = 0
#         self.__delta = 0

#         self.__selfish_miners_win_block = 0
#         self.__honest_miners_win_block = 0

#         self.__selfish_miner_revenue = 0
#         self.__honest_miner_revenue = 0

#         self.__total_mined_block = 0
#         self.__total_stale_block = 0

#         self.__iteration_number = 0

#     @property
#     def alpha(self):
#         return self._alpha

#     @alpha.setter
#     def alpha(self, value):
#         if value < 0 or value > 0.5:
#             raise Exception("invalid value for alpha!")

#         self._alpha = value

#         return

#     @property
#     def gamma(self):
#         return self._gamma

#     @gamma.setter
#     def gamma(self, value):
#         if value < 0 or value > 1:
#             raise Exception("invalid value for gamma!")

#         self._gamma = value

#         return

#     @property
#     def revenue(self):
#         return self.__selfish_miner_revenue

#     @property
#     def stale_block(self):
#         return self.__total_stale_block

#     def print_input_statistic(self):
#         print('alpha is : {}'.format(self._alpha))
#         print('gamma is : {}'.format(self._gamma))

#         return

#     def start_simulate(self, iteration):
#         self.log('start simulating')

#         self.__iteration_number = iteration

#         for i in range(iteration):
#             self.log("found a new block")

#             random_number = random.random()

#             self.calculating_delta()

#             # Mining Process
#             if random_number < self._alpha:
#                 self.start_selfish_mining()
#             else:
#                 self.start_honest_mining()
#             if random_number <self.alpha:
#                 self.__selfish_history.append("Selfish")
#                 self.start_honest_mining()
#             else:
#                 self.__honest_history.append("Honest")
#                 self.start_honest_mining()
#                         # Record relevant information for each iteration
#             iteration_info = {
#                 "iteration": i+1,
#                 "public_chain_length": self.__public_chain_length,
#                 "private_chain_length": self.__private_chain_length,
#                 "delta": self.__delta,
#                 "selfish_miners_win_block": self.__selfish_miners_win_block,
#                 "honest_miners_win_block": self.__honest_miners_win_block
#             }
#             self.__history.append(iteration_info)
#         self.calculating_output()

#         return

#     def start_selfish_mining(self):
#         self.log('starting selfish mining!')

#         self.__private_chain_length += 1

#         if self.__delta == 0 and self.__private_chain_length == 2:
#             #print('1')
#             self.__selfish_miners_win_block += 2
#             self.__private_chain_length = 0
#             self.__public_chain_length = 0

#         return

#     def start_honest_mining(self):
#         self.log('starting honest mining!')

#         self.__public_chain_length += 1

#         if self.__delta == 0 and self.__private_chain_length == 0:
#             #print('2')
#             self.__honest_miners_win_block += 1
#             self.__private_chain_length = 0
#             self.__public_chain_length = 0

#         elif self.__delta == 0 and self.__private_chain_length == 1:
#             #print('3')
#             gamma_random = random.random()

#             if gamma_random < self.gamma:
#                 self.__selfish_miners_win_block += 1
#                 self.__honest_miners_win_block += 1
#                 self.__private_chain_length = 0
#                 self.__public_chain_length = 0
#             else:
#                 self.__honest_miners_win_block += 2
#                 self.__private_chain_length = 0
#                 self.__public_chain_length = 0

#         elif self.__delta == 1:
#             #print('4')
#             # nothing to do...another state define result of this state
#             return

#         elif self.__delta == 2:
#             #print('5')
#             self.__selfish_miners_win_block += self.__private_chain_length
#             self.__public_chain_length = 0
#             self.__private_chain_length = 0

#         elif self.__delta > 2:
#             #print('6')
#             # nothing to do...another state define result of this state
#             return

#         return

#     def calculating_delta(self):
#         self.__delta = self.__private_chain_length - self.__public_chain_length
#         self.log('delta is : {}'.format(self.__delta))

#         return

#     def calculating_output(self):
#         self.__total_mined_block = self.__honest_miners_win_block + \
#             self.__selfish_miners_win_block
#         self.__total_stale_block = self.__iteration_number - self.__total_mined_block

#         self.__honest_miner_revenue = float(
#             self.__honest_miners_win_block / self.__total_mined_block) * 100
#         self.__selfish_miner_revenue = float(
#             self.__selfish_miners_win_block / self.__total_mined_block) * 100

#         return

#     def print_final_result(self):

#         print('********************************************')

#         self.print_input_statistic()

#         print('honest miners win block is : {}'.format(
#             self.__honest_miners_win_block))
#         print('selfish miners win block is : {}'.format(
#             self.__selfish_miners_win_block))

#         print('total mined block is : {}'.format(self.__total_mined_block))
#         print('total stale block is : {}'.format(self.__total_stale_block))

#         print('honest miner revenue is : {}'.format(
#             self.__honest_miner_revenue))
#         print('selfish miner revenue is : {}'.format(
#             self.__selfish_miner_revenue))

#         print('honest miner expected reward is : {}'.format(
#             (1 - self.alpha) * self.__iteration_number * 100 / self.__iteration_number))
#         print('selfish miner expected reward is : {}'.format(
#             (self.alpha) * self.__iteration_number * 100 / self.__iteration_number))
#         print("Selfish Miner History: ", self.__selfish_history)
#         print("Honest Miner History: ", self.__honest_history)
#         print('********************Results!************************')

#         return

#     def log(self, log_message):
#         if self.__show_log:
#             print(log_message)

#         return

#     def reset(self):
#         random.seed(None)

#         self.__public_chain_length = 0
#         self.__private_chain_length = 0
#         self.__delta = 0

#         self.__selfish_miners_win_block = 0
#         self.__honest_miners_win_block = 0

#         self.__selfish_miner_revenue = 0
#         self.__honest_miner_revenue = 0

#         self.__total_mined_block = 0
#         self.__total_stale_block = 0

#         self.__iteration_number = 0

#         return

#     def visualize_data(self, iteration_number):
#         alpha_values = [x / 100 for x in range(51) if x % 5 == 0]
#         selfish_revenue_value_0 = []
#         selfish_revenue_value_0_5 = []
#         selfish_revenue_value_1 = []
#         honest_revenue_value = []

#         for alpha in alpha_values:
#             self.alpha = alpha
#             self.gamma = 0
#             self.start_simulate(iteration_number)
#             self.print_final_result()
#             selfish_revenue_value_0.append(self.__selfish_miner_revenue)

#             self.reset()

#         for alpha in alpha_values:
#             self.alpha = alpha
#             self.gamma = 0.5
#             self.start_simulate(iteration_number)
#             self.print_final_result()
#             selfish_revenue_value_0_5.append(self.__selfish_miner_revenue)

#             self.reset()

#         for alpha in alpha_values:
#             self.alpha = alpha
#             self.gamma = 1
#             self.start_simulate(iteration_number)
#             self.print_final_result()
#             selfish_revenue_value_1.append(self.__selfish_miner_revenue)

#             self.reset()

#         for alpha in alpha_values:
#             honest_revenue_value.append(alpha * 100)

#         plt.plot(
#             alpha_values, selfish_revenue_value_0, color='r', label='gamma = 0')
#         plt.plot(
#             alpha_values, selfish_revenue_value_0_5, color='y', label='gamma = 0.5')
#         plt.plot(
#             alpha_values, selfish_revenue_value_1, color='g', label='gamma = 1')

#         plt.plot(alpha_values, honest_revenue_value,
#                  color='k', label='honest mining')

#         plt.title('Selfish Mining')
#         plt.xlabel('Pool size')
#         plt.ylabel('Relative Revenue')

#         plt.legend(loc="upper left")

#         plt.show()
