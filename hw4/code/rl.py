#!/usr/bin/python
# 16-831 Fall 2019
# Project 4
# RL questions:
# Fill in the various functions in this file for Q3.2 on the project.

import numpy as np
import gridworld


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
    Q3.2.1
    This implements value iteration for learning a policy given an environment.

    Inputs:
      env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
        The environment to perform value iteration on.
        Must have data members: nS, nA, and P
      gamma: float
        Discount factor, must be in range [0, 1)
      max_iterations: int
        The maximum number of iterations to run before stopping.
      tol: float
        Tolerance used for stopping criterion based on convergence.
        If the values are changing by less than tol, you should exit.

    Output:
      (numpy.ndarray, iteration)
      value_function:  Optimal value function
      iteration: number of iterations it took to converge.
  """
  A = env.nA
  S = env.nS
  
  def maxs(v1, v2):
    return np.max(np.abs(v1 - v2))
  
  def updateQ(Q, V, s):
    for a in range(A):
      cs, r, done, P = env.step(a)
      T = env.generateTransitionMatrices()
      v = 0.0
      for ns in range(s, S):
        v += T[s][a][ns] * V[ns]
      Q[s, a] = r + gamma * v
    return Q
  
  niter = 0

  # initialize the value functions
  Vt = np.random.rand(S)
  Vs = np.random.rand(S)
  
  Q = np.zeros(shape=(S, A), dtype=np.float32)
  
  while maxs(Vs, Vt) >= tol and niter < max_iterations:
    Vt = np.copy(Vs)
    
    # go over all states
    for s in range(S):
      Q = updateQ(Q, Vt, s)
      pi = np.argmax(Q[s, :])
      Vs[s] = Q[s, pi]
    
    if (niter+1) % 10 == 0:
      print(f"\nstep: [{niter+1}/{max_iterations}]")
      print(f"\t-- delta: {maxs(Vs, Vt)}")
      # print(f"\t-- value: {Vs}")
      
    niter += 1
      
  return Vs, niter


def policy_from_value_function(env, value_function, gamma):
    """
    Q3.2.1/Q3.2.2
    This generates a policy given a value function.
    Useful for generating a policy given an optimal value function from value
    iteration.

    Inputs:
      env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
        The environment to perform value iteration on.
        Must have data members: nS, nA, and P
      value_function: numpy.ndarray
        Optimal value function array of length nS
      gamma: float
        Discount factor, must be in range [0, 1)

    Output:
      numpy.ndarray
      policy: Array of integers where each element is the optimal action to take
        from the state corresponding to that index.
    """

    ## YOUR CODE HERE ##
    raise NotImplementedError()


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """
    Q3.2.2: BONUS
    This implements policy iteration for learning a policy given an environment.

    You should potentially implement two functions "evaluate_policy" and 
    "improve_policy" which are called as subroutines for this.

    Inputs:
      env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
        The environment to perform value iteration on.
        Must have data members: nS, nA, and P
      gamma: float
        Discount factor, must be in range [0, 1)
      max_iterations: int
        The maximum number of iterations to run before stopping.
      tol: float
        Tolerance used for stopping criterion based on convergence.
        If the values are changing by less than tol, you should exit.

    Output:
      (numpy.ndarray, iteration)
      value_function:  Optimal value function
      iteration: number of iterations it took to converge.
    """

    ## BONUS QUESTION ##
    ## YOUR CODE HERE ##
    raise NotImplementedError()


def td_zero(env, gamma, policy, alpha):
    """
    Q3.2.2
    This implements TD(0) for calculating the value function given a policy.

    Inputs:
      env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
        The environment to perform value iteration on.
        Must have data members: nS, nA, and P
      gamma: float
        Discount factor, must be in range [0, 1)
      policy: numpy.ndarray
        Array of integers where each element is the optimal action to take
        from the state corresponding to that index.
      alpha: float
        Learning rate/step size for the temporal difference update.

    Output:
      numpy.ndarray
      value_function:  Policy value function
    """

    ## YOUR CODE HERE ##
    raise NotImplementedError()


def n_step_td(env, gamma, policy, alpha, n):
    """
    Q3.2.4: BONUS
    This implements n-step TD for calculating the value function given a policy.

    Inputs:
      env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
        The environment to perform value iteration on.
        Must have data members: nS, nA, and P
      gamma: float
        Discount factor, must be in range [0, 1)
      policy: numpy.ndarray
        Array of integers where each element is the optimal action to take
        from the state corresponding to that index.
      n: int
        Number of future steps for calculating the return from a state.
      alpha: float
        Learning rate/step size for the temporal difference update.

    Output:
      numpy.ndarray
      value_function:  Policy value function
    """

    ## BONUS QUESTION ##
    ## YOUR CODE HERE ##
    raise NotImplementedError()


if __name__ == "__main__":
    env = gridworld.GridWorld(map_name='8x8')

    # Play around with these values if you want!
    gamma = 0.9
    alpha = 0.05
    n = 10

    # Q3.2.1
    V_vi, n_iter = value_iteration(env, gamma)
    # print(f"value iteration converged after {n_iter} steps with value:\n{V_vi}")
    print(f"value iteration converged after {n_iter} steps")
    # policy = policy_from_value_function(env, Vs, gamma)

    # Q3.2.2: BONUS
    # V_pi, n_iter = policy_iteration(env, gamma)

    # Q3.2.3
    # V_td = td_zero(env, gamma, policy, alpha)

    # Q3.2.4: BONUS
    # V_ntd = n_step_td(env, gamma, policy, alpha, n)
