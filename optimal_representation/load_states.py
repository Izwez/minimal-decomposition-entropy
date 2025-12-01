from sympy.physics.quantum import TensorProduct
import numpy as np
import pickle
import math

def l4norm(matrix):
    flat = np.ndarray.flatten(matrix)
    return sum([ abs(x)**4 for x in flat])

print('AME(4,6)')

with open('ame_46.pickle', 'rb') as handle:
        data_array = pickle.load(handle)
psi_46 = data_array[0]
u1,u2,u3,u4 = data_array[1]
psi_46_f = TensorProduct(u1,u2,u3,u4)@psi_46


print( "S_2=",-math.log( l4norm(psi_46)) )
print( "min S_2=",-math.log( l4norm(psi_46_f)) )

print('AME(5,2)')

with open('ame_52.pickle', 'rb') as handle:
        data_array = pickle.load(handle)
psi_52 = data_array[0]
u1,u2,u3,u4,u5 = data_array[1]
psi_52_f = TensorProduct(u1,u2,u3,u4,u5)@psi_52
print( "S_2=",-math.log( l4norm(psi_52)) )
print("min S_2=", -math.log( l4norm(psi_52_f)) )

print('AME(6,2)')

with open('ame_62.pickle', 'rb') as handle:
        data_array = pickle.load(handle)
psi_62 = data_array[0]
u1,u2,u3,u4,u5,u6 = data_array[1]
psi_62_f = TensorProduct(u1,u2,u3,u4,u5,u6)@psi_62
print( "S_2=", -math.log( l4norm(psi_62)) )
print("min S_2=", -math.log( l4norm(psi_62_f)) )
