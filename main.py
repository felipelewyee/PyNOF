#######################################################
#                                                     #
#  PyDoNOF (Python Doing Natural Orbital Functionals) #
#                                                     #
#######################################################
#                      Authors                        #
#    M. en C. Juan Felipe Huan Lew Yee                #
#    Dr. Jorge Martín del Campo Ramírez               #
#    Dr. Mario Piris                                  #
#######################################################

# Importamos librerías
import psi4
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from time import time
import energy
import parameters

# Parametros de control
PNOFi = 7
gradient = "analytical" # analytical/numerical


# Seleccionamos una molécula, y otros datos como la memoria del sistema y la base
psi4.set_memory('4 GB')

mol = psi4.geometry("""
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
  symmetry c1
""")

psi4.set_options({'basis': '6-31G'}),

# Paramdetros del sistema
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = parameters.param(mol,wfn)
p.ipnof = PNOFi
p.gradient = gradient
p.RI = True 
p.gpu = False
p.jit = False #True

p.autozeros()

t1 = time()
energy.compute_energy(mol,wfn,PNOFi,p,gradient)
t2 = time()
print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))
