#!/usr/bin/env python

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from model import InverseTransitionModel
from environment import Environment


itm = InverseTransitionModel(6, 3)
env = Environment()
optimizer = optim.RMSprop(itm.parameters())


def get_batch():
  """Returns a batch of 100 samples"""
  x0 = []
  x1 = []
  y = []
  c = []
  for i in range(10):
    state = env.reset_rand()
    for j in range(10):
      action = random.randint(0, 2)
      duration = int(random.random() * 3600)
      cost = 0. if action == 1 else float(duration)
      next_state = env.step(action, duration)
      x0.append(state)
      x1.append(next_state)
      y.append(torch.tensor(action))
      c.append(torch.tensor(cost))
  return (torch.stack(x0),
          torch.stack(x1),
          torch.stack(y),
          torch.stack(c))


losses = []

for i in range(500):
  s, ns, a, c = get_batch()

  # Predict and loss with forward model.
  optimizer.zero_grad()
  c_hat = itm(s, ns)
  loss = F.mse_loss(c_hat.gather(1, a.unsqueeze(1)).squeeze(), c)
  loss.backward()

  # # Predict and loss with inverse model.
  # opt1.zero_grad()
  # a_hat = itm(s, ns)
  # l1 = F.cross_entropy(a_hat, a)
  # l1.backward()

  if i % 10 == 0:
    print(i, loss.item())
  losses.append(loss.item())

  optimizer.step()
  # opt1.step()

torch.save(itm.state_dict(), 'itm.model')

plt.plot(np.log(losses))
plt.show()
