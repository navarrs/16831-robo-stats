import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from policy import *
from game import *

def plot(game, policies, policy_names):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    markers = ["+", ".", "x", 'o']

    for k in range(len(policies)):
        game.resetGame()
        reward, action, regret = game.play(policies[k])

        print("{} Reward {:.2f}".format(policy_names[k], reward.sum()))

        ax1.plot(regret, label=policy_names[k])
        ax1.set_ylabel('regret')
        ax1.set_title('regret over time')
        ax1.legend()

        ax2.plot(action, 
                 markers[k % len(markers)], 
                 markersize=2, 
                 label=policy_names[k])
        ax2.set_ylabel('actions')
        ax2.set_title('actions over time')

        plt.xlabel('trials')

    plt.legend()
    plt.show()


#### ---------------------------------------------------------------------- ####
#### q3.3
# actions = 2
# policies = [policyEXP3()]
# policy_names = ['policyEXP3']
# game = gameConstant()
# plot(game, policies, policy_names)
#### ---------------------------------------------------------------------- ####
#### q3.4
# actions = 10
# T = 10000
# policies = [policyEXP3()]
# policy_names = ['policyEXP3']
# game = gameGaussian(actions, T)
# plot(game, policies, policy_names)
#### ---------------------------------------------------------------------- ####
# q4.3
# actions = 2
# policies = [policyUCB()]
# policy_names = ['policyUCB']
# game = gameConstant()
# plot(game, policies, policy_names)
#### ---------------------------------------------------------------------- ####
#### q4.4
# actions = 2
# T = 10000
# policies = [policyUCB()]
# policy_names = ['policyUCB']
# game = gameConstant(T)
# plot(game, policies, policy_names)

# game = gameGaussian(actions, T)
# plot(game, policies, policy_names)
#### ---------------------------------------------------------------------- ####
#### q4.5 

#### ---------------------------------------------------------------------- ####
# actions = 2
# T = 1000
# game = gameGaussian(actions, T)
# # g0 = game.tabR[:, 0]
# policies = [policyUCB(actions=2)]
# policy_names = ['policyUCB']