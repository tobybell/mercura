import os.path as path
import random
import subprocess
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # self.fc1 = nn.Linear(state_size, 12)
        # self.fc2 = nn.Linear(12, 12)
        # self.head = nn.Linear(12, action_size)
        self.fc1 = nn.Linear(1, 6)
        self.head = nn.Linear(6, action_size)

    def forward(self, state):
        pos = state[:, :3].norm(dim=1, keepdim=True)
        vel = state[:, 3:].norm(dim=1, keepdim=True)
        x = torch.cat((pos, vel), 1)
        # x = state
        x = F.relu(self.fc1(pos))
        # x = F.relu(self.fc2(x))
        return self.head(x)


class Environment(object):
    def __init__(self):
        dirname = path.dirname(__file__)
        exec_path = path.join(dirname, '../../c/interactive')
        self.sp = subprocess.Popen([exec_path],
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

    def get_state(self):
        return self.step(0, 0)

    def step(self, action, duration=60.0):
        thrust = 0.1 * action - 0.1
        cmd = 'a {} {}'.format(thrust, duration)
        self.sp.stdin.write(cmd.encode('utf-8'))
        self.sp.stdin.flush()
        out = self.sp.stdout.readline().decode("utf-8").strip('{}\n')
        pv = torch.tensor([list(map(float, out.split(',')))])
        return pv

    def reset(self, *pv):
        cmd = 'r {}'.format(' '.join(map(str, pv)))
        self.sp.stdin.write(cmd.encode('utf-8'))
        self.sp.stdin.flush()
        self.sp.stdout.readline()

    def reset_rand(self):
        if random.random() < 0.5:
            self.reset_leo()
        else:
            self.reset_geo()

    def reset_geo(self):
        self.reset(42241095.67708342, 0, 0, 0.017776962751035255, 3071.8591633446, 0)

    def reset_leo(self):
        self.reset(7255000.0, 0, 0, 0, 7412.2520611297, 0)
