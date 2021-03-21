import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from policy import *
from game import *

game = gameConstant()

policies = [policyRandom(), policyConstant(), policyGWM()]
policy_names = ['policyRandom', 'policyConstant', 'policyGWM']

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

for k in range(len(policies)):
    game.resetGame()
    reward, action, regret = game.play(policies[k])
    
    print("{} Reward {:.2f}".format(policy_names[k], reward.sum()))
    
    ax1.plot(regret, label=policy_names[k])
    ax1.set_ylabel('regret')
    ax1.set_title('regret over time')
    ax1.legend() 
    
    ax2.plot(action, '.', markersize=k+1, label=policy_names[k])
    ax2.set_ylabel('actions')
    ax2.set_title('actions over time')
    
    plt.xlabel('trials')
    
plt.legend()
plt.show()
