#!/usr/bin/env python

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import TransitionModel, InverseTransitionModel
from environment import Environment


itm = InverseTransitionModel(6, 3)
env = Environment()
optimizer = optim.RMSprop(itm.parameters())


def get_batch():
  """Returns a batch of 1000 samples"""
  x0 = []
  x1 = []
  y = []
  for i in range(100):
    action = random.randint(0, 2)
    state = env.reset_rand()
    env.step(action, 60)
    next_state = env.step(1, int(random.random() * 86400))
    x0.append(state)
    x1.append(next_state)
    y.append(torch.tensor(action))
  return (torch.stack(x0),
          torch.stack(x1),
          torch.stack(y))


losses = []

for i in range(500):
  s, ns, a = get_batch()

  # Predict and loss with forward model.
  optimizer.zero_grad()
  a_hat = itm(s, ns)
  loss = F.cross_entropy(a_hat, a)
  loss.backward()

  # # Predict and loss with inverse model.
  # opt1.zero_grad()
  # a_hat = itm(s, ns)
  # l1 = F.cross_entropy(a_hat, a)
  # l1.backward()

  if i % 100 == 0:
    print(i, loss.item())
  losses.append(loss.item())

  optimizer.step()
  # opt1.step()

torch.save(itm.state_dict(), 'itm.model')

plt.plot(np.log(losses))
plt.show()
