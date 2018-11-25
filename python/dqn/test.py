

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dqn


GAMMA = 0.999


policy_net = dqn.DQN(6, 3)
policy_net.eval()
trajectory = []

env = dqn.Environment()


def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


def save_trajectory(data, path):
    with open(path, 'w') as f:
        f.write('[')
        f.write(',\n'.join(map(lambda line: '[' + ','.join(map(str, line)) + ']', data)))
        f.write(']')


policy_net.load_state_dict(torch.load('trained.model'))
print(policy_net.state_dict())
exit(0)
env.reset_leo()
state = env.get_state()
num_iters = 10000
for t in range(num_iters):
    action = select_action(state)
    next_state = env.step(action.item(), 60)
    reward = np.abs(np.linalg.norm(state[:3]) - 2e7) - np.abs(np.linalg.norm(next_state[:3]) - 2e7)

    # Record data.
    position = [state[0,0].item(), state[0,1].item(), state[0,2].item()]
    trajectory.append(position + [t * 60, action.item(), reward.item()])

    if t % 100 == 0:
        print('Reward: ' + str(reward))
    state = next_state

save_trajectory(trajectory, 'trajectory.json')
