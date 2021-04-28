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
from time import time
import energy
import parameters
import optimization
import guess

# Seleccionamos una molécula, y otros datos como la memoria del sistema y la base
psi4.set_memory('4 GB')

mol = psi4.geometry("""
C   0.00000000 0.00000000 0.00000000
O   0.00000000 0.00000000 1.34860000
O   1.00767985 0.00000000 -0.65619172
H   0.92243039 0.00000000 1.63239603
C  -1.41203055 0.00000000 -0.54650576
N  -1.51819222 0.00000000 -1.98459258
H  -1.92097701 0.87155833 -0.13352898
H  -1.92097701 -0.87155833 -0.13352898
H  -1.05352765 0.80858071 -2.37472446
H  -1.05352765 -0.80858071 -2.37472446
  symmetry c1
""")

psi4.set_options({'basis': 'cc-pVDZ'}),

# Parametros del sistema
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = parameters.param(mol,wfn)
p.ipnof = 7
p.gradient = "analytical"
p.optimizer = "Newton-CG"
p.RI = True 
p.gpu = True
p.jit = False#True
p.threshl = 10**-5   # Convergencia de los multiplicadores de Lagrange
p.threshe = 10**-7   # Convergencia de la energía

p.autozeros()
E,C,gamma,fmiug0 = energy.compute_energy(mol,wfn,p,p.gradient)

#print(fmiug0)
#import numpy as np
#fmiug0 = np.random.random((25))
#p.autozeros(restart=True)
#E,C,gamma,fmiug0 = energy.compute_energy(mol,wfn,p,p.gradient,C,gamma,fmiug0)
#print(fmiug0)
