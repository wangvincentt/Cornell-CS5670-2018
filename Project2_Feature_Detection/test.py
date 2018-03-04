import numpy as np
import scipy
from scipy import ndimage

test = np.zeros((5,5))
for i in range (5):
  for j in range (5):
    if j >= 4 - i:
      test[i][j] = 10

def harris_gauss():
  dx = ndimage.sobel(test, 0, mode = 'reflect')  # horizontal derivative
  dy = ndimage.sobel(test, 1, mode = 'reflect')  # vertical derivative
  dx = dx / 8
  dy = dy / 8
  imagex = dx ** 2
  imagey = dy ** 2
  imagexy = dx * dy
  a11 = ndimage.gaussian_filter(imagex, sigma = 0.5, mode = 'reflect', truncate = 4 ) 
  a12 = ndimage.gaussian_filter(imagexy, sigma = 0.5, mode = 'reflect', truncate = 4)
  a21 = a12
  a22 = ndimage.gaussian_filter(imagey, sigma = 0.5, mode = 'reflect', truncate = 4 )

  # Computer score 
  det_a = a11 * a22 - a12 * a21
  trace_a = a11 + a22
  score = det_a - 0.1 * trace_a ** 2
  print (score)

harris_gauss()