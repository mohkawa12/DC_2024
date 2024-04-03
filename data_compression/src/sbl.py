'''
Function to execute the SBL algorithm for denoising images.
'''
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


def run_sbl_am(sigma2, Y, num_atoms, tile_size=64, epsilon1=1, epsilon2=1):
        # sigma2: variance of noise
        # A: dictionary at current step
        # gamma: hyperparameter gamma (K x N):  N=number of examples, K=number of atoms
        # Y: list of N examples
        # r: number of iteration for update

        Sigma = [] # list of N x N matrices
        mu = [] # list of N dimensional vectors
        num_examples = len(Y)
        # Initialize dictionary and gamma
        gamma_current = np.ones([num_atoms, num_examples])
        A_current = np.ones([tile_size, num_examples])

        gamma_new = np.zeros(gamma_current.shape)
        A_new = np.zeros(A_current.shape)

        while np.linalg.norm(A_new - A_current) + np.linalg.norm(gamma_new-gamma_current, ord=2, axis=0) < epsilon1:
            for k, y_k in enumerate(Y):
                # E-step
                gamma_k = np.diag(gamma_current[:, k])
                phi = np.linalg.inv(sigma2*np.eye(num_atoms) + A_current*gamma_k*(np.transpose(A_current)))
                sigma_k = gamma_k - gamma_k*np.transpose(A_current)*phi*A_current*gamma_k
                mu_k = pow(sigma2, -2)*sigma_k*np.transpose(A_current)*y_k

                Sigma.append(sigma_k)
                mu.append(mu_k)

                # M-step pt1
                gamma_new[:,k] = np.diag(mu_k * np.transpose(mu_k) + sigma_k)

            # Update of A
            M = np.array(mu)
            Sigma_tot = np.sum(Sigma, axis=0) + M*np.transpose(M)

            while np.linalg.norm(A_new - A_current) < epsilon2:
                v = Y * np.transpose(M)
                for i in range(num_atoms):
                    v1 = 0
                    v2 = 0
                    for j in range(0, i):
                        v1 += Sigma_tot[i,j]*A_current[:,j]
                    for j in range(i+1, num_atoms):
                        v2 += Sigma_tot[i,j]*A_current[:,j]
                    v[:,i] = v[:,i] - v1 - v2

                    if np.any(v[:,i] != 0):
                        A_new[:,i] = v[:,i]/np.linalg.norm(v[:,i])
                    else:
                        A_new[:, i] = A_current[:,i]

        return mu, A_new


