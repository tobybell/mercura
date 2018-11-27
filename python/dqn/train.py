import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import dqn
from model import InverseTransitionModel


BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.4
EPS_DECAY = 3000
TARGET_UPDATE = 1


policy_net = dqn.DQN(6, 3)
target_net = dqn.DQN(6, 3)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = dqn.ReplayMemory(30000)
env = dqn.Environment()
losses = []


steps_done = 0

#####################
# MB Model to guide training of the dqn
#####################

itm = InverseTransitionModel(6, 3)
itm.load_state_dict(torch.load('itm.model'))
itm.eval()


TARGET = torch.tensor([2e7, 0.0, 0.0, 0.0, 0.0, 0.0])
#[2e7, 0.0, 0.0, 0.0, 0.0, true_anomally + pi/20]


def select_action_mb(state):
    with torch.no_grad():
        bait = torch.tensor([2e7, 0.0, 0.0, 0.0, 0.0, state[0,5] + 0.1])
        
        a = itm(state.unsqueeze(0), bait.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=a[0])
        return dist.sample().item()

#####################
# End
#####################

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
        return torch.tensor([[select_action_mb(state)]])


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = dqn.Transition(*zip(*transitions))

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
    losses.append(loss.item())
    #print(loss.item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 70
for i_episode in range(num_episodes):
    print(i_episode)

    # Initialize the environment and state

    env.reset_rand()
    state = env.step(1,0)
    action_dist = [0,0,0]
    for t in range(1000):
        # Select and perform an action
        action = select_action(state)
        #print(action)
        action_dist[action] += 1
        next_state = env.step(action.item(), 60)
        reward = -1.0 * np.abs(state[:,0].item() - 2e7)
        # reward = np.abs(np.linalg.norm(state[:3]) - 2e7) - np.abs(np.linalg.norm(next_state[:3]) - 2e7) 
        reward = torch.tensor([reward])
        if t % 1000 == 0:
            print(t) 
            print(reward) 
            print(state)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if state[:,0] > 4e7 or state[:,0] < 2e3:
            env.reset_leo()
            state = env.get_state()
            print('RESET')
            target_net.load_state_dict(policy_net.state_dict())
            #plt.plot(losses)
            #plt.show()
            #losses = []
    print(action_dist)
    target_net.load_state_dict(policy_net.state_dict())
    #plt.plot(losses)
    #plt.show()
    #losses = []
    # Update the target network
    # if i_episode % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())


torch.save(policy_net.state_dict(), 'trained.model')
