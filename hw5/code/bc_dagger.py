#!/usr/bin/env python
# coding: utf-8

# # 16831: Homework 5 - Behavior Cloning, DAGGER
#
# You will implement this assignment right here in this Jupyter notebook. ote that all cells modify the same global state, so imported packages as well as functions and variables declared in one cell will be accessible in other cells.
#
# You will want to run each cell in this notebook by clicking the "Run' button in the tool bar on top of the notebook (or using [ctrl -> enter]. Look for ``WRITE CODE HERE'' to identify places where you need to write some code. Each section involves writing 3 - 10 lines of code.
#
# When you're done, copy plots genetated by your code into your Latex writeup. Submite the notebook file in your code submission to Gradescope
#

# # Preliminaries
# In these first few cells, you will implement some compoments that will be used for all problems.

#
# ### Setup: Import Dependencies

# In[ ]:
from collections import OrderedDict
import gym
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import random


# ### Make the policy model
# We'll use the same architecture for each of the problems. By implementing a function that creates the model here, you won't need to implement it again for each problem.

# In[ ]:


import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, nS, nA):
        super(Policy, self).__init__()
        self.nS = nS
        self.nA = nA

        # WRITE CODE HERE
        # Add layers to the model:
        self.model = nn.Sequential(
            # a fully connected layer with 10 units
            nn.Linear(in_features=self.nS, out_features=10),
            # a tanh activation
            nn.Tanh(),
            # another fully connected layer with 2 units (the number of actions)
            nn.Linear(in_features=10, out_features=self.nA),
            # a softmax activation (so the output is a proper distribution)
            nn.Softmax()
        )
        # We expect the model to have four weight variables (a kernel and bias for
        # both layers)
        assert len(list(self.model.parameters())) == 4, 'Model should have 4 weights.'

    def forward(self, state):
        return self.model(state)

    def predict(self, state):
        pred = self.model(torch.FloatTensor(
            state).to(device)).detach().cpu().numpy()
        return pred

    def act(self, state):
        with torch.no_grad():
            pred = self.predict(state)
            action = np.argmax(pred, axis=1)
            return action


# ### Test the model
# To confirm that the model is correct, we'll use it to solve a binary classification problem. The target function $f: \mathbb{R}^4 \rightarrow {0, 1}$ indicates whether the sum of the vector coordinates is positive:
# $$f(x) = \delta \left(\sum_{i=1}^4 x_i > 0 \right)$$

# In[ ]:


device = torch.device('cuda')
nS, nA = 4, 2
policy = Policy(nS, nA).to(device)  # define policy
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

N = 2000
X = np.random.normal(size=(N, nS))  # some random data
is_positive = np.sum(X, axis=1) > 0  # A simple binary function
Y = np.sum(X, axis=1) > 0

criterion = nn.CrossEntropyLoss()
X = torch.FloatTensor(X).to(device)
Y = torch.LongTensor(Y).to(device)
batch_size = 256
for epoch in range(100):
    idxes = np.random.permutation(N)
    losses = []
    acces = []
    for i in range(N // batch_size):
        idx = idxes[i*batch_size: (i+1) * batch_size]
        x, y = X[idx], Y[idx]
        pred_y = policy(x)
        loss = criterion(pred_y, y)
        acc = torch.sum(torch.argmax(pred_y, dim=1) == y) / float(y.shape[0])
        # acc = np.true_divide(torch.sum(torch.argmax(pred_y, dim=1) == y).cpu(), y.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        acces.append(acc.item())
    if epoch % 10 == 0:
        print('epoch {}, loss {:.3f}, accuracy {:.2f}'.format(
            epoch, np.mean(np.array(losses)), np.mean(np.array(acces))))


# ### Interacting with the Gym
# Implement the function below for gathering an episode (a "rollout"). The environment we will use will implement the OpenAI Gym interface. For documentation, please see the link below:
# http://gym.openai.com/docs/#environments

# In[ ]:


def action_to_one_hot(env, action):
    action_vec = np.zeros(env.action_space.n)
    action_vec[action] = 1
    return action_vec


def generate_episode(env, policy):
    """Collects one rollout from the policy in an environment. The environment
    should implement the OpenAI Gym interface. A rollout ends when done=True. The
    number of states and actions should be the same, so you should not include
    the final state when done=True.

    Args:
    env: an OpenAI Gym environment.
    policy: a keras model
    Returns:
    states: a list of states visited by the agent.
    actions: a list of actions taken by the agent. 
    rewards: the reward received by the agent at each step.
    """
    done = False
    state = env.reset()

    states = []
    actions = []
    rewards = []
    while not done:
        # WRITE CODE HERE
        states.append(state)
        state = np.asarray(state).reshape(1, len(state))
        
        action = policy.act(state)[0]
        # action_one_hot = action_to_one_hot(env, action)
        # actions.append(action_one_hot)
        actions.append(action)
        
        state, reward, done, info = env.step(action)
        rewards.append(reward)

    return np.array(states), np.array(actions), np.array(rewards)


# ### Test the data collection
# Run the following cell and make sure you see "Test passed!"

# In[ ]:


# Create the environment.
env = gym.make('CartPole-v0')
nS = np.prod(env.observation_space.shape)
nA = env.action_space.n
print(f'Number of states={nS} actions={nA}')

policy = Policy(nS, nA).to(device)
states, actions, rewards = generate_episode(env, policy)
assert len(states) == len(
    actions), 'Number of states and actions should be equal.'
assert len(actions) == len(
    rewards), 'Number of actions and rewards should be equal.'
print('Test passed!')


# Behavior Cloning and DAGGER

# Implementing Behavior Cloning and DAGGER
# To implement behavior cloning and DAGGER, fill in the missing blocks of code below. The provided code loads an expert model upon creation of the `Imitation` class. The function `generate_behavior_cloning_data()` fills in `self._train_states` and `self._train_actions` with states and actions from a single episode. Later, when implementing DAGGER, you will finish implementing `generate_dagger_data()`.

# In[ ]:


class CartpoleExpertAgent():
    def act(self, states):
        if len(states.shape) == 1:
            position, velocity, angle, angle_velocity = states
            action = int(3. * angle + angle_velocity > 0.)
        else:
            position, velocity, angle, angle_velocity = states[:,
                                                               0], states[:, 1], states[:, 2], states[:, 3]
            action = (3. * angle + angle_velocity > 0.).astype(np.int)
        return action


class Imitation():
    def __init__(self, env, num_timesteps):
        self.env = env
        self.expert = CartpoleExpertAgent()
        self.num_timesteps = num_timesteps
        self.policy = Policy(nS, nA).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
    def generate_behavior_cloning_data(self):
        self._train_states = []
        self._train_actions = []
        while len(self._train_states) < self.num_timesteps:
            states, actions, rewards = generate_episode(self.env, self.expert)
            self._train_states.extend(states)
            self._train_actions.extend(actions)

        self._train_states = np.array(self._train_states)
        self._train_actions = np.array(self._train_actions)

    def generate_dagger_data(self):
        # WRITE CODE HERE
        # You should collect states and actions from the student policy
        # (self.policy), and then relabel the actions using the expert policy.
        # This method does not return anything.
        self._train_states = []
        self._train_actions = []
        while len(self._train_states) < self.num_timesteps:
            states, actions, rewards = generate_episode(self.env, self.policy)
            for a in range(actions.shape[0]):
                state = states[a]
                state = np.asarray(state).reshape(1, len(state))
                action = self.expert.act(state)[0]
                actions[a] = action
            
            self._train_states.extend(states)
            self._train_actions.extend(actions)
            
        self._train_states = np.array(self._train_states)
        self._train_actions = np.array(self._train_actions)

    def train(self, num_epochs=200):
        """Trains the model on training data generated by the expert policy.
        Args:
          env: The environment to run the expert policy on.
          num_epochs: number of epochs to train on the data generated by the expert.
        Return:
          loss: (float) final loss of the trained policy.
          acc: (float) final accuracy of the trained policy
        """
        # WRITE CODE HERE
        X = self._train_states
        Y = self._train_actions
        criterion = nn.CrossEntropyLoss()
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)
        batch_size = 256
        N = X.shape[0]
        for epoch in range(num_epochs):
            idxes = np.random.permutation(N)
            losses = []
            acces = []
            for i in range(N // batch_size):
                idx = idxes[i*batch_size: (i+1) * batch_size]
                x, y = X[idx].to(device), Y[idx].to(device)
                pred_y = self.policy(x)
                loss = criterion(pred_y, y)
                acc = torch.sum(torch.argmax(pred_y, dim=1) == y) / float(y.shape[0])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                acces.append(acc.item())
        loss = losses[-1]
        acc = acces[-1]
        return loss, acc

    def evaluate(self, policy, n_episodes=50):
        rewards = []
        for i in range(n_episodes):
            _, _, r = generate_episode(self.env, policy)
            rewards.append(sum(r))
        r_mean = np.mean(rewards)
        return r_mean


# ### Experiment: Student vs Expert
# In the next two cells, you will compare the performance of the expert policy
# to the imitation policies obtained via behavior cloning and DAGGER.

# In[ ]:


# Uncomment one of the two lines below to select whether to run behavior
# cloning or dagger
# mode = 'behavior cloning'
mode = 'dagger'

# Leave this fixed for now. You will experiment with changing it later.
num_timesteps = 20000
num_iterations = 100  # Number of training iterations. Use a small number
# (e.g., 10) for debugging, and then try a larger number
# (e.g., 100).

# Create the environment.
env = gym.make('CartPole-v0')
im = Imitation(env, num_timesteps)
expert_reward = im.evaluate(im.expert)
print('Expert reward: %.2f' % expert_reward)

def run(im, num_iterations, mode):
    loss_vec = []
    acc_vec = []
    imitation_reward_vec = []
    for t in range(num_iterations):
        if mode == 'behavior cloning':
            im.generate_behavior_cloning_data()
        elif mode == 'dagger':
            im.generate_dagger_data()
        else:
            raise ValueError('Unknown mode: %s' % mode)
        loss, acc = im.train(num_epochs=1)
        imitation_reward = im.evaluate(im.policy)
        loss_vec.append(loss)
        acc_vec.append(acc)
        imitation_reward_vec.append(imitation_reward)
        print('(%d) loss = %.3f; accuracy = %.3f; reward = %.3f' %
            (t, loss, acc, imitation_reward))
    
    return loss_vec, acc_vec, imitation_reward_vec

loss_vec, acc_vec, imitation_reward_vec = run(im, num_iterations, mode)

# ### Plot the results
# After saving your plots by running `plt.savefig(FILENAME)`, you can download them by navigating to the `Files` tab on the left, and then right-clicking on each filename and selecting `Download`.

# In[ ]:

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title('Reward')
plt.plot(imitation_reward_vec, label='imitation')
plt.hlines(expert_reward, 0, len(imitation_reward_vec), label='expert')
plt.xlabel('iterations')
plt.ylabel('return')
plt.legend()
plt.ylim([0, None])

plt.subplot(132)
plt.title('Loss')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')

plt.subplot(133)
plt.title('Accuracy')
plt.plot(acc_vec)
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.tight_layout()
plt.savefig(f'student_vs_expert_mode={mode}_iter={num_iterations}.png', dpi=300)
plt.show()


# ### Experiment: How much expert data is needed?
# This question studies how the amount of expert data effects the performance. You will run the same experiment as above, each time varying the number of expert episodes collected at each iteration. Use values of 500, 1000, 5000, and 20000. You can keep the number of iterations fixed at 100.

# In[ ]:


random_seeds = 5
# Dictionary mapping number of expert trajectories to a list of rewards.
# Each is the result of running with a different random seed.
all_timesteps = [500, 1000, 5000, 20000]
num_iterations = 100
reward_data, accuracy_data, loss_data = OrderedDict({}), OrderedDict({}), OrderedDict({})
for num_timesteps in all_timesteps:
    reward_data[num_timesteps] = []
    accuracy_data[num_timesteps] = []
    loss_data[num_timesteps] = []
    
for num_timesteps in all_timesteps:
    for t in range(random_seeds):
        
        im = Imitation(env, num_timesteps)
        expert_reward = im.evaluate(im.expert)
        print('Expert reward: %.2f' % expert_reward)
        
        print('num_timesteps: %s; seed: %d' % (num_timesteps, t))
        # WRITE CODE HERE
        # Hint: The code here should be nearly identical to code after the
        # "Student vs Expert" cell. Feel free to copy and paste.
        loss_vec, acc_vec, imitation_reward_vec = run(im, num_iterations, mode)
        
        reward_data[num_timesteps].append(imitation_reward_vec)
        accuracy_data[num_timesteps].append(acc_vec)
        loss_data[num_timesteps].append(loss_vec)


# Plot the reward, loss, and accuracy for each, remembering to label each line.

# # In[ ]:


keys = all_timesteps
plt.figure(figsize=(16, 4))
for (index, (data, name)) in enumerate(zip([reward_data, accuracy_data, loss_data],
                                           ['reward', 'accuracy', 'loss'])):
    plt.subplot(1, 3, index + 1)
    data = np.array(list(data.values())).reshape(len(all_timesteps), random_seeds, num_iterations)
    for i, num_timesteps in enumerate(all_timesteps):
        mean = np.mean(data[i, :, :], axis=0)
        std = np.std(data[i, :, :], axis=0)

#         plt.plot(np.array(range(num_iterations)) * num_timesteps, mean, label="num_steps: {}".format(num_timesteps))
        plt.plot(np.array(range(num_iterations)), mean, label="num_steps: {}".format(num_timesteps))
        plt.fill_between(range(num_iterations), mean-std, mean+std, alpha=0.2)
    plt.xlabel('number of expert trajectories', fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.legend()
plt.savefig('expert_data_%s.png' % mode, dpi=300)
plt.show()


# # You're Done!
