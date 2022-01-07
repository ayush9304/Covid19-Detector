import numpy as np


def dilate(img, kernel):
  m,n = img.shape
  blank = np.zeros((m,n), dtype=np.uint8)
  k = kernel.shape[0]
  constant = (k-1)//2
  for i in range(constant,m-constant):
    for j in range(constant,n-constant):
      temp = img[i-constant:i+constant+1, j-constant:j+constant+1]
      product = temp*kernel
      blank[i,j] = np.max(product)
  return blank

def erode(img, kernel):
  m,n = img.shape
  blank = np.zeros((m,n), dtype=np.uint8)
  k = kernel.shape[0]
  constant = (k-1)//2
  for i in range(constant,m-constant):
    for j in range(constant,n-constant):
      temp = img[i-constant:i+constant+1, j-constant:j+constant+1]
      product = temp*kernel
      blank[i,j] = np.min(product)
  return blank