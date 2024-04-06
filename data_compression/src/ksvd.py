'''
Class to execute the K-SVD algorithm for denoising images.
'''
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm

class KSVD:

    def __init__(self):
        return

    def omp(self, A, y, tol=None, s=None):
        if tol is not None:
            error = len(y)*tol
            r = y
            z = np.zeros(A.shape[1])
            lambdas = set()
            while error>=np.sqrt(len(y))*tol:
                h = A.T@r
                k = np.argmax(abs(h))
                lambdas.add(k)
                Alambda = A[:,list(lambdas)]
                zlambda = np.linalg.inv(Alambda.T@Alambda)@Alambda.T@y
                z[list(lambdas)] = zlambda
                r = y - A@z
                error = np.linalg.norm(r, ord=2)
        elif s is not None:
            r = y
            z = np.zeros(A.shape[1])
            lambdas = set()
            for i in range(s):
                h = A.T@r
                k = np.argmax(abs(h))
                lambdas.add(k)
                Alambda = A[:,list(lambdas)]
                zlambda = np.linalg.inv(Alambda.T@Alambda)@Alambda.T@y
                z[list(lambdas)] = zlambda
                r = y - A@z
                error = np.linalg.norm(r, ord=2)
        else:
            print("Please specify error tolerance or sparsity level.")
            z = -1
        return z


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
        xK = []
        avg_sparsity = 0
        if method=="omp":
            row, col = ys.shape
            for i in range(col):
                y = ys[:,i]
                xk = self.omp(A, y, s=s, tol=tol)
                avg_sparsity += len(np.nonzero(xk)[0])
                xK.append(xk)
            avg_sparsity /= col
            print("Average sparsity", avg_sparsity)
            return np.array(xK).T
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
        skipped_atoms = 0
        for atom_no in tqdm(range(0, no_atoms)):
            # Find om_atom_no, the set of examples that use the kth atom of A
            xk_row = xK[atom_no,:]
            om_atom_no = np.nonzero(xk_row)
            om_atom_no = om_atom_no[0]
            
            # If no example uses this atom, skip it
            if len(om_atom_no)==0:
                skipped_atoms += 1
                continue
            
            # Calculate the error matrix
            sum_out = np.hstack((A[:,:atom_no], A[:,atom_no+1:])) @ np.vstack((xK[:atom_no,:], xK[atom_no+1:,:]))
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
        print("Skipped", skipped_atoms, "atoms in dictionary update")
        return A

    '''
    Calculate the MSE ||Y-AX||_F
    '''
    def convergence_crit(self, yN, A, xK):
        yN = np.transpose(np.array(yN))
        error = yN - A@xK
        return np.linalg.norm(error)**2
