## CODE TO GENERATE AME(6,2) STATES ##

#MODIFY AS REQUIRED
# Import and definitions
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Find nearest isometry to a given matrix

def nearestU(matrix):
        d1,d2 =matrix.shape
        u,d,v = np.linalg.svd(matrix)
        return u@np.eye(d1,d2)@v

# AME code

def diff(lstx1,lstx2):
    lstx3=[]
    for x in lstx1:
        if x not in lstx2:
            lstx3.append(x)
    return "".join(lstx3)
# number of parties
N=6

# local dimension
ld=2

# define uniformity

k = int(N/2)

#Construction of list of permutations
char_list = ['i','j','k','l','m','n','o','p','q','r','s']
list_of_combinations = list(itertools.combinations(char_list[0:N],k) )
list_of_permutations = ["".join(x)+diff( char_list[0:N],x ) for x in list_of_combinations ]
num_list = ['1','2','3','4','5','6','7','8','9','10','11']
list_of_combinations_num = list(itertools.combinations(num_list[0:N],k) )
list_of_permutations_num = ["".join(x)+diff( num_list[0:N],x ) for x in list_of_combinations_num ]

# Definition of reshaping operations and its inverse

def reshape_n(nx,ldx,statex,indx):
    kx = int(nx/2)
    assert statex.shape==(ldx**kx,ldx**(nx-kx))
    tpl = tuple(  [ldx for i in range(nx)] )
    return np.einsum( "".join(char_list[0:nx])+'->'+list_of_permutations[indx],statex.reshape(tpl) ).reshape((ldx**kx,ldx**(nx-kx)) )
def reshape_n_inverse(nx,ldx,statex,indx):
    kx = int(nx/2)
    assert statex.shape==(ldx**kx,ldx**(nx-kx))
    tpl = tuple(  [ldx for i in range(nx)] )
    return np.einsum( list_of_permutations[indx]+'->'+"".join(char_list[0:nx]),statex.reshape(tpl) ).reshape((ldx**kx,ldx**(nx-kx)) )

# algorithm

a0 = nearestU(np.random.randn(ld**k,ld**(N-k)) + 1j*np.random.randn(ld**k,ld**(N-k)) )

tol=10**(-10)

distlist = []
maxval=0
perm_range = len(list_of_permutations )
for i in range(10**4):
    blist = []
    for i1 in range(perm_range):
        blist.append(    reshape_n_inverse(N,ld, nearestU(reshape_n(N,ld,a0,i1)),i1   )      )

    b0 =  sum(blist)/perm_range
    current_value = np.trace( a0.conj().T@ b0)
    a0=b0

    if current_value>maxval:
        maxval = current_value
        max_uni = a0
    distlist.append(current_value)
    if i>3:
            if abs(distlist[-1] -distlist[-2]) < tol:
                break

# Plot convergence

xval = [i for i in range(len(distlist))]
plt.scatter(xval, np.real(distlist),s=1.5)
plt.show()


# Compute reduced density matrices and trace of its square

import qutip
psist = (max_uni).reshape(ld**N,1) 
qpsi = qutip.Qobj(psist,dims=[[ld]*N,[1]*N])
for i1 in range(6):
    for j1 in range(6):
        for l1 in range(6):
            if i1<j1<l1: 
                rho = qutip.ptrace(qpsi,[i1,j1,l1]).full()
                print(i1,j1,l1, np.trace(rho@rho) ) 


