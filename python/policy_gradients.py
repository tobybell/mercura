import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess
import numpy as np
import random
from torch.autograd import Variable
import sys


class Policy(nn.Module):
    def __init__(self, state_size, action_size,):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 12)
        self.fc2 = nn.Linear(12,12)
        self.fc5 = nn.Linear(12, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc5(x), dim=0)


policy_history = torch.Tensor([])
reward_episode = []
reward_history = []
loss_history = []
policy = Policy(6, 3)
optimizer = optim.Adam(policy.parameters(), lr=.001)
environment = subprocess.Popen(['../c/interactive'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
last_radius = 42241095.67708342

def select_action(state, eps = 0):
    probs = policy(torch.Tensor(state))
    c = torch.distributions.Categorical(probs)
    action = c.sample()
    prob = c.log_prob(action).unsqueeze(0)
    global policy_history
    if policy_history.size(0) != 0:
        policy_history = torch.cat([policy_history, prob])
    else:
        policy_history = prob
    return action 



def update_policy():
    global policy_history
    R = 0
    rewards = []
    for reward in reward_episode[::-1]:
        R = reward + 0.99 * R
        rewards.append(R)
    rewards = list(reversed(rewards))
    
    loss = -torch.sum(torch.mul(policy_history, torch.Tensor(rewards)), -1)
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policy_history = torch.Tensor([])
    reward_episode.clear()


def get_state(s):
    s = s.decode("utf-8").strip('{}\n')
    pv = list(map(float, s.split(',')))
    return np.array(pv)


def take_action(action, timestep=60.0):
    s = 'a ' + str(action) + ' ' + str(timestep) + '\n'
    environment.stdin.write(s.encode('utf-8'))
    environment.stdin.flush()
    out = environment.stdout.readline()
    pv = get_state(out)
    p = pv[:3]
    global last_radius
    r = np.abs(last_radius - 2e7) - np.abs(np.linalg.norm(p) - 2e7)
    last_radius = np.linalg.norm(p)
    #r = -np.abs(np.linalg.norm(p) - 2e7)
    return pv, r


def env_reset(*pv):
    global last_radius
    last_radius = np.linalg.norm(pv[:3])
    s = 'r ' + ' '.join(map(str, pv))
    environment.stdin.write(s.encode('utf-8'))
    environment.stdin.flush()

def env_reset_rand():
    raise NotImplementedError('Random reset not implemented')

def env_reset_geo():
    env_reset(42241095.67708342, 0, 0, 0.017776962751035255, 3071.8591633446, 0)

def env_reset_leo():
    env_reset(7255000.0, 0, 0, 0, 7412.2520611297, 0)

def env_reset_desired_orbit():
    env_reset(42241095.67708342, 0, 0, 0.017776962751035255, 3071.8591633446, 0)

def train(epochs=1000):
    s = environment.stdout.readline()
    env_reset_geo()
    state = get_state(s)
    print(0, state)
    alternate = False
    for epoch in range(epochs):

        #state = starting state
        
        for i in range(0, 1000):
            action = 0.1 * select_action(state).item() - 0.1
            state, reward = take_action(action)
            reward_episode.append(reward)
        print(epoch)
        update_policy()
        if epoch % 50 == 0:
            if epoch % 100 == 0:
                env_reset_geo()
            else:
                env_reset_leo()
    torch.save(policy.state_dict(), 'trained.model')

def test():
    policy.load_state_dict(torch.load('trained.model'))
    env_reset_leo()
    f = open('trajectory.json', 'w')
    s = environment.stdout.readline()
    state = get_state(s)
    f.write('[')

    num_iters = 30000
    for i in range(0,num_iters):
        action = 0.1 * select_action(state).item() - 0.1
        #action = 0.01
        if i > 2500:
            action = 0
        state, reward = take_action(action)
        print(reward)

        if i % 100 == 0:
            velocity = state[3:]
            position = state[:3]
            print(np.linalg.norm(position), np.linalg.norm(velocity))

        f.write('[')
        f.write(','.join(map(str, state[:3])))
        f.write(',' + str(i*60))
        f.write(',' + str(action))
        f.write(']')
        if i < num_iters-1:
            f.write(',\n')
    
    f.write(']')
    f.close()


if __name__ == '__main__':
    if '--train' in sys.argv:
        train()
    elif '--test' in sys.argv:
        test()
