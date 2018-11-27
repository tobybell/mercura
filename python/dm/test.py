#!/usr/bin/env python

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import TransitionModel, InverseTransitionModel
from environment import Environment


itm = InverseTransitionModel(6, 3)
env = Environment()

itm.load_state_dict(torch.load('itm.model'))
itm.eval()


TARGET = torch.tensor([2e7, 0.0, 0.0, 0.0, 0.0, 0.0])


def select_action(state):
    with torch.no_grad():
        a = itm(state.unsqueeze(0), TARGET.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=a[0])
        return dist.sample().item()


trajectory = []
s = env.reset_rand()
for i in range(20000):
  a = select_action(s)
  ns = env.step(a, 60)
  trajectory.append(list(s.numpy()[:3]) + [i * 60, a, 0])
  s = ns


def save_trajectory(data, path):
    with open(path, 'w') as f:
        f.write('[')
        f.write(',\n'.join(map(lambda line: '[' + ','.join(map(str, line)) + ']', data)))
        f.write(']')

save_trajectory(trajectory, 'trajectory.json')


# policy_net.load_state_dict(torch.load('trained.model'))
# print(policy_net.state_dict())
# exit(0)
# env.reset_leo()
# state = env.get_state()
# num_iters = 10000
# for t in range(num_iters):
#     action = select_action(state)
#     next_state = env.step(action.item(), 60)
#     reward = np.abs(np.linalg.norm(state[:3]) - 2e7) - np.abs(np.linalg.norm(next_state[:3]) - 2e7)

#     # Record data.
#     position = [state[0,0].item(), state[0,1].item(), state[0,2].item()]
#     trajectory.append(position + [t * 60, action.item(), reward.item()])

#     if t % 100 == 0:
#         print('Reward: ' + str(reward))
#     state = next_state

# save_trajectory(trajectory, 'trajectory.json')
