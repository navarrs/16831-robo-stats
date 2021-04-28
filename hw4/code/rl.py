#!/usr/bin/python
# 16-831 Fall 2019
# Project 4
# RL questions:
# Fill in the various functions in this file for Q3.2 on the project.

import numpy as np
import gridworld
import matplotlib.pyplot as plt
import math

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
  P = env.P
  n_iter = 0
  delta = np.finfo(np.float32).max
  
  def updateV(V, s):
    max_v = np.finfo(np.float32).min
    
    for a in range(A):
      v = 0.0
      # P[state][action] = (prob, state, reward, done)
      transition = P[s][a]
      for p, ns, r, done in transition:
        v += p * (r + gamma * V[ns])   
      max_v = np.maximum(max_v, v)

    V[s] = max_v
    return V
  
  V = np.random.rand(S)
  while n_iter < max_iterations and delta > tol:
    delta = 0.0
    
    # loop over all states
    for s in range(S):
      v = V[s]
      V = updateV(V, s)
      delta = np.maximum(delta, np.abs(v - V[s]))
    
    n_iter += 1
    if n_iter % 10 == 0:
      print(f"step: [{n_iter}/{max_iterations}]")
      print(f"\t--delta: {delta}")
  
  return V, n_iter


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
    A = env.nA
    S = env.nS
    P = env.P
    policy = np.zeros(shape=(S, 1), dtype=np.float32)
    
    # def Qat(s, a):
    #   cs, r, done, P = env.step(a)
    #   T = env.generateTransitionMatrices()
    #   v = 0.0
    #   for ns in range(s, S):
    #     v += T[s, a, ns] * (r + gamma * value_function[ns])
    #   return v
      
    
    # for s in range(S):
    #   q = np.zeros(shape=(4, 1), dtype=np.float32)
    #   for a in range(A):
    #     q[a] = Qat(s, a)
    #   policy[s] = np.argmax(q)
    
    for s in range(S):
      max_va = np.iinfo(np.int).min
      max_ac = policy[s]
      for a in range(A):
        transition = P[s][a]
        v = 0.0
        for p, ns, r, done in transition:
          v += p * (r + gamma * value_function[ns]) 
        
        if max_va < v:  
          max_va = v
          max_ac = a
          
      policy[s] = max_ac
    return policy


def policy_iteration(env, gamma, max_iterations=int(1e4), tol=1e-3):
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
  A = env.nA
  S = env.nS
  P = env.P
  
  def updateV(V, s):
    max_v = np.finfo(np.float32).min
    
    for a in range(A):
      v = 0.0
      # P[state][action] = (prob, state, reward, done)
      transition = P[s][a]
      for p, ns, r, done in transition:
        v += p * (r + gamma * V[ns])   
      max_v = np.maximum(max_v, v)

    V[s] = max_v
    return V

  def evaluate_policy():
    V = np.random.rand(S)
    n_iter_eval = 0
    delta = np.finfo(np.float32).max
    while n_iter_eval < max_iterations and delta > tol:
      delta = 0.0
      
      # loop over all states
      for s in range(S):
        v = V[s]
        V = updateV(V, s)
        delta = np.maximum(delta, np.abs(v - V[s]))
      n_iter_eval += 1
    
    return V
  
  def improve_policy(V, policy):
    policy_stable = True
    
    for s in range(S):
      old_action = np.copy(policy[s])
      
      max_v = np.finfo(np.float32).min
      max_action = old_action
      for a in range(A):
        v = 0.0
        # P[state][action] = (prob, state, reward, done)
        transition = P[s][a]
        for p, ns, r, done in transition:
          v += p * (r + gamma * V[ns]) 
        
        if max_v < v:
          max_v = v 
          max_action = a
      
      policy[s] = max_action
      if policy[s] == old_action:
        policy_stable = False 
        
    return policy, policy_stable
  
  
  policy = np.random.randint(0, 4, (S, 1))
  policy_stable = False
  n_iter = 0
  while not policy_stable and n_iter < max_iterations:
    if n_iter % 500 == 0:
      print(f"** step: [{n_iter}/{max_iterations}]")
      
    V = evaluate_policy()
    policy, policy_stable = improve_policy(V, policy)
    
    n_iter += 1
    
  return V, n_iter


def td_zero(env, gamma, policy, alpha, neps=int(1e4)):
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
  S = env.nS
  V = np.random.rand(S)
  V[-1] = 0.0
    
  # for each episode
  for e in range(neps):
      
    # initialize S
    state = env.reset()
      
    # for each step on episode
    done = False 
    while not done:
      # get action given by policy
      action = int(policy[state][0])
      # take action, observe reward and next state
      next_state, reward, done, _ = env.step(action)
      # update value 
      V[state] += alpha * (reward + gamma * V[next_state] - V[state])
      # update state
      state = next_state
        
    if e % 1000 == 0:
      print(f"num episodes: {e}/{neps}")
        
  return V

def n_step_td(env, gamma, policy, alpha, n, neps=int(1e4)):
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
  V = np.random.rand(env.nS)
  
  # for each episode
  for e in range(neps):
    # initialize S
    T = np.iinfo(np.int).max
    S = [env.reset()]
    R = [0]
    
    t, tau = 0, 0
    while True:
      if t < T:
        # take action given by policy
        action = int(policy[S[t]][0])
        # observe and store next reward as and next state
        next_state, reward, done, _ = env.step(action)
        R.append(reward)
        S.append(next_state)
        # if next state is terminal 
        if done:
          T = t + 1
      
      # tau - time whose state's estimate is being updated
      tau = t - n + 1
      if tau >= 0:
        G = 0.0
        # from Sutton's book
        h = np.minimum(tau+n, T)
        l = tau # + 1
        for i in range(l, h):
          G += gamma**(i-tau) * R[i]
          # G += (gamma**(i-tau-1)) * R[i]
        if tau + n < T:
          G += gamma**n * V[S[tau+n]]
        V[S[tau]] += alpha * (G - V[S[tau]])
      
      if tau == T - 1:
        break
        
      t += 1
    
    if e % 1000 == 0:
      print(f"num episodes: {e}/{neps}")
    
  return V

def plot(X, title, text=None, dec=2):
  ax = plt.gca()
  im = ax.imshow(X)
  cbar = ax.figure.colorbar(im, ax=ax)
  ax.set_title(title)
  
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if not text is None:
        v = text[int(X[i, j])]
      else:
        v = X[i, j]
        factor = 10.0 * dec
        v = math.trunc(v * factor) / factor
      ax.text(j, i, v, ha="center", va="center", color="w")
  
  plt.savefig(f"{title}.png")
  # plt.show()
  plt.close()

if __name__ == "__main__":
  mapname = "8x8"
  env = gridworld.GridWorld(map_name=mapname)
  gw, gh = int(mapname[0]), int(mapname[-1])

  # Play around with these values if you want!
  gamma = 0.9
  alpha = 0.05
  n = 4
  action_names = ['L', 'D', 'R', 'U']
  
  # Q3.2.1
  print(f"\n** q3.2.1 value iteration")
  V_vi, n_iter = value_iteration(env, gamma)
  plot(V_vi.reshape(gw, gh), title='value_iteration')
  print(f"value iteration converged after {n_iter} steps")
  policy = policy_from_value_function(env, V_vi, gamma)
  plot(policy.reshape(gw, gh), title='policy_from_value_iteration', text=action_names)

  # Q3.2.2: BONUS
  print(f"\n** q3.2.2 policy iteration")
  V_pi, n_iter = policy_iteration(env, gamma)
  print(f"policy iteration converged after {n_iter} steps")
  plot(V_pi.reshape(gw, gh), title='policy_iteration')
  ppolicy = policy_from_value_function(env, V_pi, gamma)
  plot(ppolicy.reshape(gw, gh), title='policy_from_policy_iteration', text=action_names)

  # Q3.2.3
  print(f"\n** q3.2.3 TD 0")
  V_td = td_zero(env, gamma, policy, alpha)
  plot(V_td.reshape(gw, gh), title='td_zero')
  
  # Q3.2.4: BONUS
  print(f"\n** q3.2.4 TD n")
  for n in range(1, 11):
    print(f"using n: {n}")
    V_ntd = n_step_td(env, gamma, policy, alpha, n)
    plot(V_ntd.reshape(gw, gh), title=f'td_{n}')