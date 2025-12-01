import numpy as np
import math
from scipy.linalg import polar
import qutip as qt

# Find L_4 norm of a matrix

def l4norm(matrix):
    flat = np.ndarray.flatten(matrix)
    return sum([ abs(x)**4 for x in flat])

# Example

A = np.random.rand(4,5) + 1j* np.random.rand(4,5)
print("L_4 norm of A=",l4norm(A) )

# Random unitary matrix 

def randomunitary(d):
    #return unitary_group.rvs(d)
    return qt.rand_unitary(d,distribution='haar').full()


# Returns normalized array

def normalize(array1):
    return array1/math.sqrt(  sum(abs(np.ndarray.flatten(array1) )**2 ) )

# Example

A = np.random.rand(4,5) + 1j* np.random.rand(4,5)
print("\n normalized array A=",normalize(A) )

# Find nearest isometry

def nearestU(matrix):
        d1,d2 =matrix.shape
        u,d,v = np.linalg.svd(matrix)
        return u@np.eye(d1,d2)@v

# Example

A = np.random.rand(4,5) + 1j* np.random.rand(4,5)
print("\n Nearest isometry to A=",nearestU(A) )

# L_p- PCA algorithm

def nearest_U_l4(matrix):
    
    # find shape m1 x n1.
    X = matrix
    m1,n1 = X.shape
    
    
    #norm
    lp=4
    
    #tolerance
    tol = 10**(-15)
    
    #initiate m1 x m1 matrix (column list)
    wl = [normalize(np.random.randn(m1,1) ) for _ in range(m1) ]
    
    nearest_U_l4_list=[]
    
    for i in range(30):
        
        wp = [ sum([      ((elm.conj().T@X[:,i:i+1])[0,0])**(lp/2-1)*(((X[:,i:i+1]).conj().T@elm)**(lp/2))[0,0]* X[:,i:i+1]  for i in range(n1)]) for elm in wl]
        
        Wp = polar(np.concatenate(wp,axis=1))[0]
        
        wl = [Wp[:,i:i+1] for i in range(len(Wp))]
        
        nearest_U_l4_list.append( l4norm(  Wp.conj().T@matrix ) )
        if i>3:
            if abs(nearest_U_l4_list[-2] - nearest_U_l4_list[-1])< tol:
                break
    return Wp.conj().T


# Example

A = np.random.rand(4,5) + 1j* np.random.rand(4,5)
print("\n L_4 PCA",nearest_U_l4(A) )

# Tensor Product

def tensor_product(list_of_matrices):
    from functools import reduce
    return reduce(np.kron, list_of_matrices)

# Example

u1,u2,u3,u4 = [randomunitary(3) for _ in range(4)]

print("\n tensor product of u1,u2,u3,u4 = ", tensor_product( [ u1,u2,u3,u4 ] ) )

# Algorithm to maximize L_4 norm for ququad states ( N=4, d=4)



def max_entropy(state):
    
    N=4
    ld=4
    
    psi=normalize(state)
    
    u1,u2,u3,u4 = [randomunitary(ld) for _ in range(N)]
    
    psi0=psi
    
    ld3 = ld**(N-1)
    
    idld = np.eye(ld)
    
    tol = 10**(-15)
    
    distlist = []
    
    for i in range(300):


        psi1 = tensor_product([idld,u2,u3,u4])@psi0
        psi2 = psi1.reshape(ld,ld3)
        u1 = nearest_U_l4(psi2)

        psi1 = tensor_product([u1,idld,u3,u4])@psi0
        psi2 = (np.einsum('abcd->bacd',psi1.reshape(ld,ld,ld,ld))).reshape(ld,ld3)
        u2 = nearest_U_l4(psi2)

        psi1 = tensor_product([u1,u2,idld,u4])@psi0
        psi2 = (np.einsum('abcd->cabd',psi1.reshape(ld,ld,ld,ld))).reshape(ld,ld3)
        v1 = nearest_U_l4(psi2)

        psi1 = tensor_product([u1,u2,u3,idld])@psi0
        psi2 = (  np.einsum('abcd->dabc',psi1.reshape(ld,ld,ld,ld))  ).reshape(ld,ld3)
        v2 = nearest_U_l4(psi2)

        currentval = l4norm(tensor_product([u1,u2,u3,u4]) @psi)

        distlist.append( currentval )

        #convergence test
        if i>3:
            if abs(distlist[-2] - distlist[-1])< tol:
                break
    
        results_list_inner = [-math.log(currentval), psi ,[u1,u2,u3,u4] ]

    return results_list_inner

# Example

N=4 # Number of parties
ld=4 # Local dimension d

state_N_4_d_4 = np.random.rand(N**ld,1) + 1j* np.random.rand(N**ld,1) 
max_entropy(state_N_4_d_4)

