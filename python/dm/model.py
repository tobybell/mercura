import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseTransitionModel(nn.Module):
  def __init__(self, state_size, n_actions):
    super().__init__()
    self.enc1 = nn.Linear(state_size, 16)
    self.enc2 = nn.Linear(16, 16)
    self.enc3 = nn.Linear(16, 16)
    self.enc4 = nn.Linear(16, 16)
    self.enc5 = nn.Linear(16, 16)
    self.feat = nn.Linear(16, 16)
    self.dec1 = nn.Linear(16, 16)
    self.dec2 = nn.Linear(16, 16)
    self.dec3 = nn.Linear(16, 16)
    self.head = nn.Linear(16, n_actions)

  def forward(self, state, next_state):
    state = F.relu(self.enc1(state))
    state = F.relu(self.enc2(state))
    state = F.relu(self.enc3(state))
    state = F.relu(self.enc4(state))
    state = F.relu(self.enc5(state))
    state = self.feat(state)
    next_state = F.relu(self.enc1(next_state))
    next_state = F.relu(self.enc2(next_state))
    next_state = F.relu(self.enc3(next_state))
    next_state = F.relu(self.enc4(next_state))
    next_state = F.relu(self.enc5(next_state))
    next_state = self.feat(next_state)
    diff = next_state - state
    diff = F.relu(self.dec1(diff))
    diff = F.relu(self.dec2(diff))
    diff = F.relu(self.dec3(diff))
    return self.head(diff)
