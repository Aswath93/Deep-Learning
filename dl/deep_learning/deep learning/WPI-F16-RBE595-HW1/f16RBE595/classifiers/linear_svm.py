import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_ex=X.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1 # margin of the SVM
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  regularization = np.sum(np.square(W))
  scores = W.dot(X)
  scores= scores-scores[y,range(num_ex)]
  scores = np.add(np.ones(scores.shape),scores)
  scores[y,range(num_ex)] = 0
  scores = np.maximum(np.zeros(scores.shape),scores)
  temp = scores
  loss = np.sum(scores)/num_ex + reg*regularization

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  temp[temp > 0] = 1
  temp2 = -np.sum(temp,axis = 0)
  temp[y,range(num_ex)] = temp2
  dW = X.dot(temp.T)/num_ex + reg*W.T


  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW.T

