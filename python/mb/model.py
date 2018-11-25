import torch
import torch.nn as nn


class InverseTransitionModel(nn.Module):
  def __init__(self, state_size, n_actions):
    super().__init__()
    self.fc1 = nn.Linear(state_size, )

  
