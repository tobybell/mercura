import math
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
        self.fc1 = nn.Linear(state_size, 12)
        self.fc2 = nn.Linear(12, 12)
        self.head = nn.Linear(12, action_size)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class Environment(object):
    def __init__(self):
        self.sp = subprocess.Popen(['../c/interactive'],
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
        raise NotImplementedError('Random reset not implemented')

    def reset_geo(self):
        self.reset(42241095.67708342, 0, 0, 0.017776962751035255, 3071.8591633446, 0)

    def reset_leo():
        self.reset(7255000.0, 0, 0, 0, 7412.2520611297, 0)


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 10


policy_net = DQN(6, 3)
target_net = DQN(6, 3)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
env = Environment()


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(3)]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                      batch.next_state)), device=device, dtype=torch.uint8)
    #non_final_next_states = torch.cat([s for s in batch.next_state
    #                                            if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_states = torch.cat(batch.next_state)
    next_state_values = target_net(next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    print(i_episode)

    # Initialize the environment and state
    env.reset_geo()
    state = env.get_state()
    for t in range(10000):
        # Select and perform an action
        action = select_action(state)
        next_state = env.step(action.item(), 60)
        reward = np.abs(np.linalg.norm(state[:3]) - 2e7) - np.abs(np.linalg.norm(next_state[:3]) - 2e7) 
        reward = torch.tensor([reward])

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())






