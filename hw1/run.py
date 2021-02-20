import matplotlib.pyplot as plt
import numpy as np
import os

from wma import WeightedMajorityAlgorithm
from rwma import RandomizedWeightedMajorityAlgorithm

from environment import (
    StochasticWorld, DeterministicWorld, AdversarialWorld,
    OddLoseExpert, OptimisticExpert, NegativeExpert
)

SUPPORTED_WORLDS = ["stochastic", "deterministic", "adversarial"]
SUPPORTED_ALGORITHMS = ["wma", "rwma"]
if not os.path.exists("out"):
    os.makedirs("out")

def plot(x, y1, y1_label, y2=None, y2_label=None, 
         out_file="out.png", title = 'plot', x_axis = 'x', y_axis = 'y'):
    colors  = ['salmon', 'limegreen', 'royalblue', 'mediumpurple']
    icolors = ['red', 'green', 'blue', 'purple']

    for n in range(len(y1[1])):
        # interpolate to smooth out
        poly = np.polyfit(x, y1[:, n], 10)
        poly_y = np.poly1d(poly)(x)
        plt.plot(x, poly_y, color=icolors[n], linewidth=0.5)
        plt.plot(x, y1[:, n], color=colors[n], label=y1_label[n], linewidth=3)
    
    if not y2 is None:
        for n in range(len(y2[1])):
            poly = np.polyfit(x, y2[:, n], 10)
            poly_y = np.poly1d(poly)(x)
            plt.plot(x, poly_y, color=icolors[-1+n], linewidth=0.5)
            plt.plot(x, y2[:, n], color=colors[-1+n], label=y2_label[n])
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()
    
    plt.savefig(out_file)
    print(f"Saved to file {out_file}")
    plt.show()
    plt.close()
    
def run(args):
    # Hypothesis
    H = [
        OptimisticExpert("optimistic"), 
        NegativeExpert("negative"), 
        OddLoseExpert("odd")
    ]
    print("Hypothesis created with experts:")
    expert_names = []
    for h in H:
        expert_names.append(h.get_name())
        print("*"*5, f"expert: {h.get_name()}")
    
    if args.world == "stochastic":
        world = StochasticWorld(args.world)
    elif args.world == "deterministic":
        world = DeterministicWorld(args.world)
        world.set_counter()
    elif args.world == "adversarial":
        world = AdversarialWorld(args.world)
        world.set_strategy(args.algo)
    
    if args.algo == "wma":
        algo = WeightedMajorityAlgorithm(H, world, args.T, args.eta)
    else:
        algo = RandomizedWeightedMajorityAlgorithm(H, world, args.T, args.eta)
    print(f"Initialized algorithm: {args.algo}")
    
    print(f"Start...")
    algo.run()
    
    if args.plot:
        print(f"Plotting...")
        t = np.arange(args.T)
        expert_losses = algo.get_experts_loss()
        learner_loss = algo.get_learners_loss()
        regret = algo.get_regret()
        
        plot(t, 
            y1=expert_losses, y1_label=expert_names, 
            y2=learner_loss, y2_label=["learner"],
            out_file=f"out/loss_{args.algo}_{args.world}.png",
            title=f"algo: {args.algo} -- world: {args.world} -- loss vs. time", 
            x_axis='time', y_axis='loss')
        
        plot(t, 
            y1=regret, y1_label=["regret"], 
            out_file=f"out/regret_{args.algo}_{args.world}.png",
            title=f"algo: {args.algo} -- world: {args.world} -- regret vs. time", 
            x_axis='time', y_axis='regret')
    
    print(f"Done...")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PWEA')
    parser.add_argument('--world', type=str, default='stochastic', 
                        help='world: [stochastic, deterministic, adversarial]')
    parser.add_argument('--algo', type=str, default='wma', 
                        help='algorithm: [wma, rwma]')
    parser.add_argument('--T', type=int, default=100, 
                        help='time steps')
    parser.add_argument('--eta', type=float, default=0.5, 
                        help='penalty value')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    
    assert args.world in SUPPORTED_WORLDS, \
        f"Error. world: {args.world} not in supported worlds: {SUPPORTED_WORLDS}"
    
    assert args.algo in SUPPORTED_ALGORITHMS, \
        f"Error. algorithm: {args.algo} not in supported algorithms: {SUPPORTED_ALGORITHMS}"
    
    run(args)