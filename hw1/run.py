from wma import WeightedMajorityAlgorithm
from world import (
    StochasticWorld,
    OddLoseExpert,
    OptimisticExpert,
    NegativeExpert
)

SUPPORTED_WORLDS = ["stochastic", "deterministic", "adversarial"]
SUPPORTED_ALGORITHMS = ["wma", "rwma"]


def run(args):
    # Hypothesis
    H = [
        OptimisticExpert("optimistic"), 
        NegativeExpert("negative"), 
        OddLoseExpert("odd")
    ]
    print("Hypothesis created with experts:")
    for h in H:
        print("*"*5, f"expert: {h.get_name()}")
    
    if args.world == "stochastic":
        world = StochasticWorld(args.world, labels=[-1, 1])
    elif args.world == "deterministic":
        pass
    else:
        pass
    
    if args.algo == "wma":
        algo = WeightedMajorityAlgorithm(H, world, args.T, args.eta)
    else:
        pass
    print(f"Initialized algorithm: {args.algo}")
    
    
    print(f"Start...")
    algo.run()
    print(f"Done...")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PWEA')
    parser.add_argument('--world', type=str, default='stochastic', 
                        help='world: [stochastic, deterministic, adversarial]')
    parser.add_argument('--algo', type=str, default='wma', 
                        help='algorithm: [wma, rwma]')
    parser.add_argument('--T', type=int, default=10, 
                        help='time steps')
    parser.add_argument('--eta', type=float, default=0.5, 
                        help='penalty value')
    args = parser.parse_args()
    
    assert args.world in SUPPORTED_WORLDS, \
        f"Error. world: {args.world} not in supported worlds: {SUPPORTED_WORLDS}"
    
    assert args.algo in SUPPORTED_ALGORITHMS, \
        f"Error. algorithm: {args.algo} not in supported algorithms: {SUPPORTED_ALGORITHMS}"
    
    run(args)