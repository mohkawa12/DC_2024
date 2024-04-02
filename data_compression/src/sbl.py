'''
Class to execute the SBL algorithm for denoising images.
'''
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

class SBL:
    def __init__(self):
        return

    '''
    Return a ones-matrix nxK (used to initialise the dictionary)
    '''
    # K: dimension of sparse vector x
    # n: dimension of one example
    def init_dict(self, K):
        return np.ones((8*8, K))

    '''
    Return a ones-matrix KxN (used to initialise the hyperparameter gamma)
    '''
    # N: number of examples
    def init_gamma(self, K, N):
        return np.ones((K, N))

    '''
    Return sigma, mu and gamma in the first for-loop in the paper
    '''
    def floop1(self, sigma2, A, gamma, Y):
        # sigma2: variance of noise
        # A: dictionary at current step
        # gamma: hyperparameter gamma (K x N)
        # Y: list of N examples

        sigma = [] # list of N x N matrices
        mu = [] # list of N dimensional vectors
        num_examples, _ = A.shape
        # Iterate over examples
        for k, y_k in enumerate(Y):

            # E-step
            gamma_k = np.diag(gamma[:, k])
            phi = np.linalg.inv(sigma2*np.eye(num_examples) + A*gamma_k*(np.transpose(A)))
            sigma_k = gamma_k - gamma_k*np.transpose(A)*phi*A*gamma_k
            mu_k = pow(sigma2, -2)*sigma_k*np.transpose(A)*y_k

            sigma.append(sigma_k)
            mu.append(mu_k)

            # M-step pt1
            gamma[:,k] = np.diag(mu_k * np.transpose(mu_k) + sigma_k)
        return sigma, mu, gamma


    def floop2(self):


