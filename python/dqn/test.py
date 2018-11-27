

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from eoe import eoe_to_pv

import dqn

GAMMA = 0.999


policy_net = dqn.DQN(6, 3)
policy_net.eval()
trajectory = []

env = dqn.Environment()


def select_action(state):
    # pos = state[:, :3].norm(dim=1, keepdim=True)
    # if pos < 2e7:
    #     return torch.tensor(2)
    # return torch.tensor(1)
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


def save_trajectory(data, path):
    with open(path, 'w') as f:
        f.write('[')
        f.write(',\n'.join(map(lambda line: '[' + ','.join(map(str, line)) + ']', data)))
        f.write(']')


policy_net.load_state_dict(torch.load('trained.model'))
print(policy_net.state_dict())
env.reset_leo()
state = env.get_state()
num_iters = 10000
rewards = []
for t in range(num_iters):
    action = select_action(state)
    #next_state = env.step(action, 60)
    next_state = env.step(action.item(), 60)
    reward = -1.0 * np.abs(state[:,0].item() - 2e7)
    rewards.append(reward)
    #reward = np.abs(np.linalg.norm(state[:3]) - 2e7) - np.abs(np.linalg.norm(next_state[:3]) - 2e7)

    # Record data.
    print(state[:,0])
    pv = eoe_to_pv(state[0].numpy(), 3.9860044188e14)
    trajectory.append(list(pv[:3]) + [t * 60, action[0,0].item(), 0])
    # position = [state[0,0].item(), state[0,1].item(), state[0,2].item()]
    # trajectory.append(position + [t * 60, action.item(), reward.item()])

    # if t % 100 == 0:
    #     print('Reward: ' + str(reward))
    state = next_state

save_trajectory(trajectory, 'trajectory.json')
plt.plot(rewards)
plt.show()