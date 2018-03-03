import numpy as np
import scipy
from scipy import ndimage

test = np.zeros((5,5))
for i in range (5):
  for j in range (5):
    if j >= 4 - i:
      test[i][j] = 10

def harris_guass():
  dx = ndimage.sobel(test, 0, mode = 'reflect')  # horizontal derivative
  dy = ndimage.sobel(test, 1, mode = 'reflect')  # vertical derivative
  imagex = dx ** 2
  imagey = dy ** 2
  imagexy = dx * dy
harris_guass()