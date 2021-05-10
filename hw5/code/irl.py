#!/usr/bin/python
# 16-831 Fall 2019
# Project 4
# IRL questions:
# Fill in the various functions in this file for Q3.3 on the project.

import numpy as np
import cvxopt as cvx

import gridworld
import rl

def irl_lp(policy, T_probs, discount, R_max, l1):
  """
  Solves the linear program formulation for finite discrete state IRL.

  Inputs:
    policy: np.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    T_probs: np.ndarray
      nS x nA x nS matrix where:
      T_probs[s][a] is a probability distribution over states of transitioning
      from state s using action a.
      Can be generated using env.generateTransitionMatrices.
    gamma: float
      Discount factor, must be in range [0, 1)
    R_max: float
      Maximum reward allowed.
    l1: float
      L1 regularization penalty.

  Output:
    np.ndarray
    R: Array of rewards for each state.
  """

  T_probs = np.asarray(T_probs)
  nS, nA, _ = T_probs.shape
  
  def P(s, a):
    # Computes:
    #   (Paopt(s) - Pa(s))(I - gamma * Paopt)^{-1}
    b = np.zeros(shape=(1, nS))
    a_opt = int(policy[s])
    b[0] = (T_probs[s, a_opt] - T_probs[s][a])
    return b @ np.linalg.inv(np.identity(nS) - discount * T_probs[:, a_opt])

  ## YOUR CODE HERE ##
  # Create c, G and h in the standard form for cvxopt.
  # Look at the documentation of cvxopt.solvers.lp for further details
  # 
  # Linear system:
  # | -P &   0  &  0 || R |   |   0  |
  # | -P &   0  &  I |||R|| = |   0  |
  # |  0 &   0  & -I || b |   |   0  |
  # | -I &  -I  &  0 |        |   0  |
  # |  I &  -I  &  0 |        |   0  |
  # | -I &   0  &  0 |        | Rmax |
  # |  I &   0  &  0 |        | Rmax |
  
  # create each row for P 
  p_stack = []
  i_stack = []
  for s in range(nS):
    a_opt = int(policy[s])
    for a in range(nA):
      if a == a_opt:
        continue
      p_stack.append(P(s, a))
      i = np.zeros(shape=(1, nS))
      i[0, s] = 1
      i_stack.append(i)
      
  P = np.vstack(p_stack)
  I = np.vstack(i_stack)
  Z = np.zeros_like(I)
  InS = np.eye(nS)
  ZnS = np.zeros_like(InS)
  
  # create G matrix
  gcol1 = np.vstack((-P, -P,  ZnS, -InS,  InS, -InS, InS))
  gcol2 = np.vstack(( Z,  Z,  ZnS, -InS, -InS,  ZnS, ZnS))
  gcol3 = np.vstack(( I,  Z, -InS,  ZnS,  ZnS,  ZnS, ZnS))
  
  assert gcol1.shape == gcol2.shape == gcol3.shape, \
    f"size mismatch. gcol1: {gcol1.shape} col2: {gcol2.shape} col3: {gcol3.shape}"
  # print(f"col shapes: {gcol1.shape}")
  G = np.hstack((gcol1, gcol2, gcol3))
  # print(f"G shape: {G.shape}")
  
  # create h
  rowsG = G.shape[0]
  h_zeros = np.zeros(shape=(rowsG - 2*nS, 1))
  h_rmax  = R_max * np.ones(shape=(2*nS, 1))
  h = np.vstack((h_zeros, h_rmax))
  # print(f"h shape: {h.shape}")
  
  # create c
  ccol1 = np.zeros(shape=(nS, 1))
  ccol3 = -np.ones(shape=(nS, 1))
  ccol2 = -l1 * ccol3
  c = np.vstack((ccol1, ccol2, ccol3))

  # Don't do this all at once. Create portions of the vectors and matrices for
  # different parts of the objective and constraints and concatenate them
  # together using something like np.r_, np.c_, np.vstack and np.hstack.
  # raise NotImplementedError()

  # You shouldn't need to touch this part.
  c = cvx.matrix(c)
  G = cvx.matrix(G)
  h = cvx.matrix(h)
  sol = cvx.solvers.lp(c, G, h)

  R = np.asarray(sol["x"][:nS]).squeeze()

  return R

if __name__ == "__main__":
  env = gridworld.GridWorld(map_name='8x8')

  # Generate policy from Q3.2.1
  gamma = 0.9
  Vs, n_iter = rl.value_iteration(env, gamma)
  policy = rl.policy_from_value_function(env, Vs, gamma)

  T = env.generateTransitionMatrices()
  print(T.shape, policy.shape)

  # Q3.3.5
  # Set R_max and l1 as you want.
  R_max = 1
  mapname = "8x8"
  gw, gh = int(mapname[0]), int(mapname[-1])
  l1_range = np.arange(0, 1.1, 0.1)
  for l1 in l1_range:
    print(f"running: rmax={R_max} l1={l1}")
    R = irl_lp(policy, T, gamma, R_max, l1)
  
    # You can test out your R by re-running VI with your new rewards as follows:
    env_irl = gridworld.GridWorld(map_name=mapname, R=R)
    Vs_irl, n_iter_irl = rl.value_iteration(env_irl, gamma)
    policy_irl = rl.policy_from_value_function(env_irl, Vs_irl, gamma)
    rl.plot(Vs_irl.reshape(gw, gh), title='irl_policy_l1={:.2f}'.format(l1))