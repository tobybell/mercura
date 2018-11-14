import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return F.relu(self.fc5(x))

policy_history = Variable(torch.Tensor())
reward_episode = []
reward_history = []
loss_history = []
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=.001)

def select_action(state):
    probs = policy(Variable(state))
    c = Categorical(probs)
    action = c.sample()
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action

def update_policy():
    R = 0
    rewards = []
    int i 
    for reward in reward_episode[::-1]:
        R = reward + 0.99 * R
        rewards.insert(0,R)
    
    loss = (torch.sum(torch.mul(policy_history, Variable(rewards)).mul(-1), -1))
    optimizer.zero_grad()
    loss.backward()
    optimerzer.step()
    policy.policy_history = Variable(torch.Tensor())
    reward_episode = []

def main(epochs):
    for epoch in range(epochs):

        #state = starting state
        for time in range(10000):
            action = select_action(state)
            state, reward = #simulator
            reward_episode.append(reward)
        update_policy()

