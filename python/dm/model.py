import torch
import torch.nn as nn
import torch.nn.functional as F


class CostModel(nn.Module):
  def __init__(self, state_size, n_actions):
    super().__init__()
    self.enc1 = nn.Linear(state_size, 8)
    self.enc2 = nn.Linear(8, 8)
    self.enc3 = nn.Linear(8, 8)
    self.feat = nn.Linear(8, 16)
    self.head = nn.Linear(16, n_actions)

  def forward(self, state, next_state):
    state = F.relu(self.enc1(state))
    state = F.relu(self.enc2(state))
    state = F.relu(self.enc3(state))
    state = self.feat(state)
    next_state = F.relu(self.enc1(next_state))
    next_state = F.relu(self.enc2(next_state))
    next_state = F.relu(self.enc3(next_state))
    next_state = self.feat(next_state)
    diff = next_state - state
    return self.head(diff)
