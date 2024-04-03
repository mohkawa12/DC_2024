'''
Class to execute the K-SVD algorithm for denoising images.
'''
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

class KSVD:

    def __init__(self):
        return

    '''
    Return a random matrix nxK (used to initialise the dictionary)
    ''' 
    def init_dict(self, K):
        A = np.random.random((8*8, K))
        A_normed = A / np.linalg.norm(A, axis=0)
        return A_normed
    
    '''
    First step in K-SVD. Given dictionary A and measurements ys, 
    find the sparse vectors xs. Termination of the algorithm is by default
    based on the number of non-zero entries in the solution, but can also 
    be based on the residual error
    '''
    def sparse_coding(self, A, ys, method="omp", s=None, tol=None):
        ys = np.transpose(np.array(ys))
        if method=="omp":
            if (s is not None):
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
            elif (tol is not None):
                omp = OrthogonalMatchingPursuit(tol=tol)
            else:
                omp = OrthogonalMatchingPursuit()
            reg = omp.fit(A,ys)
            return np.transpose(np.array(omp.coef_)), reg.score(A,ys)
        else:
            print("No other method besides omp is supported")
            return None

    '''
    Second step in K-SVD. Given sparse vectors xK, dictionary A and measurements yN, 
    find the next estimate of the dictionary A.
    '''
    def codebook_update(self, xK, yN, A):
        _, no_atoms = A.shape
        yN = np.transpose(np.array(yN))
        for atom_no in range(0, no_atoms):
            # Find om_atom_no, the set of examples that use the kth atom of A
            xk_row = xK[atom_no,:]
            om_atom_no = np.nonzero(xk_row)
            om_atom_no = om_atom_no[0]
            
            # If no example uses this atom, skip it
            if len(om_atom_no)==0:
                continue
            
            # Calculate the error matrix
            sum_out = 0
            for j in range(no_atoms):
                if j==atom_no:
                    continue
                else:
                    sum_out = sum_out + np.outer(A[:,j],xK[j,:])
            E_atom_no = yN-sum_out

            # Restrict E to only the columns corresponding to omega
            E_restricted = E_atom_no[:,om_atom_no]

            # Do SVD
            U, S, Vh = np.linalg.svd(E_restricted)

            # Update dictionary and xk
            A[:,atom_no] = U[:,0]
            xKR = S[0]*Vh[0,:]
            xKR_idx = 0
            for om_idx in om_atom_no:
                xK[atom_no,om_idx] = xKR[xKR_idx] 
                xKR_idx = xKR_idx+1
        return A

    '''
    Calculate the MSE ||Y-AX||_F
    '''
    def convergence_crit(self, yN, A, xK):
        yN = np.transpose(np.array(yN))
        error = yN - A@xK
        return np.linalg.norm(error)
