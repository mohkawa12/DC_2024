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

        num_examples = len(Y)
        # Initialize dictionary and gamma
        gamma_current = np.ones([num_atoms, num_examples])
        A_current = np.ones([tile_size, num_atoms])

        gamma_new = np.zeros(gamma_current.shape)
        A_new = np.zeros(A_current.shape)

        while np.linalg.norm(A_new - A_current) + np.sum(np.linalg.norm(gamma_new-gamma_current, ord=2, axis=0)) > epsilon1:
            Sigma = []  # list of N x N matrices
            mu = []  # list of N dimensional vectors

            # Update
            if np.all(A_new != 0):
                A_current = A_new

            for k, y_k in enumerate(Y):
                # E-step
                gamma_k = np.diag(gamma_current[:, k])
                phi = np.linalg.inv(sigma2*np.eye(tile_size) + A_current @ gamma_k @ A_current.T)
                sigma_k = gamma_k - gamma_k @ np.transpose(A_current) @ phi @ A_current*gamma_k
                mu_k = pow(sigma2, -2)*sigma_k @ A_current.T @ y_k

                Sigma.append(sigma_k)
                mu.append(mu_k)

                # M-step pt1
                gamma_new[:,k] = np.diag(mu_k * np.transpose(mu_k) + sigma_k)

            # Update of A
            M = np.array(mu).T
            Sigma_tot = np.sum(Sigma, axis=0) + M @ M.T
            A_u_current = A_current
            A_u_new = np.zeros(A_u_current.shape)

            while np.linalg.norm(A_u_new - A_u_current) > epsilon2:

                # Update
                if np.all(A_u_new != 0):
                    A_u_current = A_u_new

                v = np.array(Y).T @ M.T
                for i in range(num_atoms):
                    v1 = 0
                    v2 = 0
                    for j in range(0, i):
                        v1 += Sigma_tot[i,j]*A_current[:,j]
                    for j in range(i+1, num_atoms):
                        v2 += Sigma_tot[i,j]*A_current[:,j]
                    v[:,i] = v[:,i] - v1 - v2

                    if np.any(v[:,i] != 0):
                        A_u_new[:,i] = v[:,i]/np.linalg.norm(v[:,i])
                    else:
                        A_u_new[:, i] = A_u_current[:,i]
                diff = np.linalg.norm(A_u_new - A_u_current)
                print("change in A:", diff)
            # update A
            A_new = A_u_new


        return mu, A_new


