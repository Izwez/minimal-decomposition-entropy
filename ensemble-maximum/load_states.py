import numpy as np
import math


def l4norm(matrix):
    flat = np.ndarray.flatten(matrix)
    return sum([ abs(x)**4 for x in flat])

state_N_3_d_2 = np.load('state_N_3_d_2.npy') 
print( "Haar random N=3, d=2 : max S_2^{min} = ",-math.log( l4norm(state_N_3_d_2) ) )


state_N_4_d_2 = np.load('state_N_4_d_2.npy') 
print( "Haar random N=4, d=2 : max S_2^{min} = ",-math.log( l4norm(state_N_4_d_2) ) )

state_N_3_d_3 = np.load('state_N_3_d_3.npy') 
print( "Haar random N=3, d=3 : max S_2^{min} = ",-math.log( l4norm(state_N_3_d_3) ) )

state_N_4_d_3 = np.load('state_N_4_d_3.npy') 
print( "Haar random N=4, d=3 : max S_2^{min} = ",-math.log( l4norm(state_N_4_d_3) ) )

state_N_4_d_4 = np.load('state_N_4_d_4.npy') 
print( "Haar random N=4, d=4 : max S_2^{min} = ",-math.log( l4norm(state_N_4_d_4) ) )

ame_state_N_4_d_4 = np.load('ame_state_N_4_d_4.npy') 
print( "AME state N=4, d=4 : max S_2^{min} = ",-math.log( l4norm(state_N_4_d_4) ) )
