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
        return np.random.random((8*8, K))
    
    '''
    First step in K-SVD. Given dictionary A and measurements ys, 
    find the sparse vectors xs. Termination of the algorithm is by default
    based on the number of non-zero entries in the solution, but can also 
    be based on the residual error
    '''
    def sparse_coding(self, A, ys, method="omp", s=None, tol=None):
        xs = []
        sum_score = 0
        for yk in ys:
            coeffs, score = self.sparse_coding_low(A, yk, method=method, s=s, tol=tol)
            sum_score = sum_score+score
            xs.append(coeffs)
        print(sum_score/len(ys)) 
        return np.transpose(np.array(xs))

    '''
    Sparse coding step for a single measurement y
    '''
    def sparse_coding_low(self, A, y, method="omp", s=None, tol=None):
        if method=="omp":
            if (s is not None):
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
            elif (tol is not None):
                omp = OrthogonalMatchingPursuit(tol=tol)
            else:
                omp = OrthogonalMatchingPursuit()
            reg = omp.fit(A,y)
            return np.array(omp.coef_), reg.score(A,y)
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
            om_atom_no = []
            xk_row = xK[atom_no,:]
            for example_no in range(0,len(xk_row)):
                xk = xk_row[example_no]
                if abs(xk)>0.000000001:
                    om_atom_no.append(example_no) 
            
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
            xKR = S[0]*Vh[:,0]
            xKR_idx = 0
            for om_idx in om_atom_no:
                xK[atom_no,om_idx] = xKR[xKR_idx] 
                xKR_idx = xKR_idx+1
        return A
