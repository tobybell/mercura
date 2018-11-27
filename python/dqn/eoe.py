import numpy as np 

def eoe_to_pv(eoe, mu):
  l, h, k, p, q, L = eoe

  A = 1 + p * p + q * q
  f = np.array([1 - p * p + q * q, 2 * p * q, -2 * p]) / A
  g = np.array([2 * p * q, 1 + p * p - q * q, 2 * q]) / A

  b = 1 - h * h - k * k

  r = l / (1 + h * np.sin(L) + k * np.cos(L))
  X = r * np.cos(L)
  Y = r * np.sin(L)
  
  c = np.sqrt(mu / l)
  dX = -c * (h + np.sin(L))
  dY = c * (k + np.cos(L))

  return np.concatenate((X * f + Y * g, dX * f + dY * g))
