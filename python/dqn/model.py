import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseTransitionModel(nn.Module):
  def __init__(self, state_size, n_actions):
    super().__init__()
    self.enc1 = nn.Linear(state_size, 8)
    self.enc2 = nn.Linear(8, 8)
    self.enc3 = nn.Linear(8, 8)
    self.feat = nn.Linear(8, 16)
    self.head = nn.Linear(16, n_actions)

  def forward(self, state, next_state):
    scale = torch.tensor([1e7, 1.0, 1.0, 1.0, 1.0, 1.0])
    state = state / scale
    state = F.relu(self.enc1(state))
    state = F.relu(self.enc2(state))
    state = F.relu(self.enc3(state))
    state = self.feat(state)
    next_state = next_state / scale
    next_state = F.relu(self.enc1(next_state))
    next_state = F.relu(self.enc2(next_state))
    next_state = F.relu(self.enc3(next_state))
    next_state = self.feat(next_state)
    diff = next_state - state
    return self.head(diff)


class TransitionModel(nn.Module):
  def __init__(self, state_size, n_actions):
    super().__init__()
    self.action_emb = nn.Embedding(n_actions, 16)
    self.enc1 = nn.Linear(state_size, 8)
    self.enc2 = nn.Linear(8, 8)
    self.enc3 = nn.Linear(8, 8)
    self.enc4 = nn.Linear(8, 8)
    self.enc5 = nn.Linear(8, 8)
    self.feat = nn.Linear(8, 16)
    self.dec1 = nn.Linear(16, 8)
    self.dec2 = nn.Linear(8, 8)
    self.dec3 = nn.Linear(8, 8)
    self.dec4 = nn.Linear(8, 8)
    self.dec5 = nn.Linear(8, 8)
    self.head = nn.Linear(8, state_size)

  def forward(self, state, action):
    state = F.relu(self.enc1(state))
    state = F.relu(self.enc2(state))
    state = F.relu(self.enc3(state))
    state = F.relu(self.enc4(state))
    state = F.relu(self.enc5(state))
    state = self.feat(state)
    next_state = state + self.action_emb(action)
    next_state = F.relu(self.dec1(next_state))
    next_state = F.relu(self.dec2(next_state))
    next_state = F.relu(self.dec3(next_state))
    next_state = F.relu(self.dec4(next_state))
    next_state = F.relu(self.dec5(next_state))
    return self.head(next_state)
