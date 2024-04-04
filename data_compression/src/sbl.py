'''
Function to execute the SBL algorithm for denoising images.
'''
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


def run_sbl_am(sigma2, Y, num_atoms, tile_size=64, epsilon1=0.0006, epsilon2=0.0006):

    num_examples = len(Y)
    # Initialize dictionary and gamma
    gamma_current = np.ones([num_atoms, num_examples])
    A_current = np.ones([tile_size, num_atoms])

    gamma_new = np.zeros(gamma_current.shape)
    r = 0
    condition1 = True

    while condition1:
        r += 1
        print(r)
        Sigma = []  # list of N x N matrices, N: number of atoms
        mu = []  # list of N dimensional vectors

        for k, y_k in enumerate(Y):
            # E-step
            gamma_k = np.diag(gamma_current[:,k])
            phi = np.linalg.inv(pow(sigma2, 2)*np.eye(tile_size) + A_current @ gamma_k @ A_current.T)
            sigma_k = gamma_k - gamma_k @ A_current.T @ phi @ A_current @ gamma_k
            mu_k = pow(sigma2, -2) * sigma_k @ A_current.T @ y_k

            Sigma.append(sigma_k)
            mu.append(mu_k)

            # M-step pt1
            gamma_new[:,k] = np.diag(mu_k @ mu_k.T + sigma_k)

        # Update of A
        M = np.array(mu).T
        Sigma_tot = np.sum(Sigma, axis=0) + M @ M.T
        A_u_current = A_current
        A_u_new = np.zeros(A_u_current.shape)
        condition2 = True
        while condition2:

            v = np.array(Y).T @ M.T
            for i in range(num_atoms):
                v1 = 0
                v2 = 0
                for j in range(0, i):
                    v1 += Sigma_tot[i,j]*A_u_current[:,j]
                for j in range(i+1, num_atoms):
                    v2 += Sigma_tot[i,j]*A_u_current[:,j]
                v[:,i] = v[:,i] - v1 - v2

                if np.any(v[:,i] != 0):
                    A_u_new[:,i] = v[:,i]/np.linalg.norm(v[:,i])
                else:
                    A_u_new[:, i] = A_u_current[:,i]

            if np.linalg.norm(A_u_new - A_u_current) < epsilon2:
                A_new = A_u_new
                condition2 = False
            else:
                A_u_current = A_u_new

        if np.linalg.norm(A_new - A_current) + np.sum(np.linalg.norm(gamma_new - gamma_current, ord=2, axis=0)) < epsilon1 or r>8:
            condition1 = False
            return mu, A_new
        else:
            A_current = A_new
            gamma_current = gamma_new







