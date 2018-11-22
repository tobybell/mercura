import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess
import numpy as np
from torch.autograd import Variable

class Policy(nn.Module):
    def __init__(self, state_size, action_size,):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.softmax(self.fc5(x))

policy_history = Variable(torch.Tensor())
reward_episode = []
reward_history = []
loss_history = []
policy = Policy(6, 3)
optimizer = optim.Adam(policy.parameters(), lr=.001)
environment = subprocess.Popen(['../c/interactive'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def select_action(state):
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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policy_history = torch.Tensor([])
    reward_episode.clear()

def get_state(s):
    s = s.decode("utf-8").strip('{}\n')
    eoe = list(map(float, s.split()))
    return np.array(eoe)

def take_action(action, timestep=60.0):
    s = str(action) + ' ' + str(timestep) + '\n'
    environment.stdin.write(s.encode('utf-8'))
    environment.stdin.flush()
    out = environment.stdout.readline()
    return get_state(out), 1


def main(epochs):
    s = environment.stdout.readline()
    state = get_state(s)
    for epoch in range(epochs):

        #state = starting state
        for time in range(100):
            action = 1.0 * select_action(state).item() - 1
            state, reward = take_action(action)
            print(time, state)
            reward_episode.append(reward)
        print(epoch)
        update_policy()

if __name__ == '__main__':
    main(100)

