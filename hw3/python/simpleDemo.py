import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from policy import *
from game import *

def plot(game, policies, plot_confidence=False):
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for policy in policies:
        if "UCB" in policy.name() and plot_confidence:
            plt.close()
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            break

    markers = ["+", ".", "x", 'o']

    for k in range(len(policies)):
        game.resetGame()
        reward, action, regret = game.play(policies[k])

        print("{} Reward {:.2f}".format(policies[k].name(), reward.sum()))

        ax2.plot(regret, label=policies[k].name())
        ax2.set_ylabel('regret')
        ax2.set_title('regret over time')
        ax2.legend()

        ax1.plot(action, 
                 markers[k % len(markers)], 
                 markersize=1, 
                 label=policies[k].name())
        ax1.set_ylabel('actions')
        ax1.set_title('actions over time')
        ax1.legend()
        
        if "UCB" in policies[k].name() and plot_confidence:
            conf = policies[k].getConfidence()
            for i in range(conf.shape[1]):
                ax3.plot(conf[:, i], label=f"action: {i}")
            ax3.set_ylabel('confidence')
            ax3.set_title('confidence over time')

        plt.xlabel('trials')

    plt.legend()
    plt.show()

#### ---------------------------------------------------------------------- ####
#### q2.6 - Test the constant game 
# policies = [policyRandom(), policyConstant()]
# game = gameConstant()
# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q3.1.1 - Test GWM and compare with Random 
# policies = [policyRandom(), policyGWM(), policyConstant()]
# game = gameConstant()
# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q3.3 - Test EXP3 on constant game 
# actions = 2
# policies = [policyEXP3()]
# # policies = [policyRandom(), policyGWM(), policyConstant(), policyEXP3()]
# game = gameConstant()
# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q3.4.2 - Test EXP3 on the Gaussian game
# actions = 10
# T = 10000
# policies = [policyEXP3()]
# game = gameGaussian(actions, T)
# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q4.3 - Test UCB
# actions = 2
# policies = [policyUCB()]
# game = gameConstant()

# g0 = np.zeros(shape=actions, dtype=np.float)
# for i in range(actions):
#     g0[i] = game.reward(i)
# print(f"Initial reward: {g0}")
# policies[0].firstPull(g0)

# plot(game, policies, True)

#### ---------------------------------------------------------------------- ####
#### q4.4 - UCB vs EXP3
# actions = 2
# T = 1000 # change to 10000 for the second question
# policies = [policyUCB(), policyEXP3()]
# game = gameGaussian(actions, T)

# # initial pull 
# g0 = np.zeros(shape=actions, dtype=np.float)
# for i in range(actions):
#     g0[i] = game.reward(i)
# print(f"Initial reward: {g0}")
# policies[0].firstPull(g0)

# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q4.5 - adversarial 
# actions = 2
# T = 1000
# policies = [policyUCB(), policyEXP3()]
# game = gameAdverserial()
# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q5.2 - University Website Latency Dataset
# table = scipy.io.loadmat("../data/univLatencies.mat")['univ_latencies']
# policies = [policyUCB(), policyEXP3(), policyRandom()]
# game = gameLookupTable(table, True)
# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q5.3
# table = scipy.io.loadmat("../data/plannerPerformance.mat")['planner_performance']
# policies = [policyUCB(), policyEXP3(), policyRandom()]
# game = gameLookupTable(table, True)
# plot(game, policies)

#### ---------------------------------------------------------------------- ####
#### q6.1
tuniv = scipy.io.loadmat("../data/univLatencies.mat")['univ_latencies']
n1, t1 = tuniv.shape
tplan = scipy.io.loadmat("../data/plannerPerformance.mat")['planner_performance']
n2, t2 = tplan.shape
table = np.ones(shape=(n1 + n2, t1 + t2 + 1), dtype=np.float)

table[:n1, 0] = 0.0
table[n1:, 0] = 1.0

table[:n1, 1:t1+1] = tuniv
table[n1:, t1+1:] = tplan

nranges = [[0, n1], [n1, -1]]

np.random.shuffle(table)

states = table[:, 0]
table = table[:, 1:]

policies = [policyUCBContext(), policyUCB()]
game = gameLookupTable(table, isLoss=True, states=states, nranges=nranges)
plot(game, policies)
