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
mol = psi4.geometry("""
    C     0.746724339984    -0.768576459902     0.000000000000
    C    -0.086016217343     0.501420111809     0.000000000000
    O    -1.298355217572     0.554744362828     0.000000000000
    O     0.701082325216     1.606059720765     0.000000000000
    N    -0.005464235552    -2.009481157081     0.000000000000
    H     1.406810354307    -0.736094687977    -0.876352457582
    H    1.406810354307    -0.736094687977     0.876352457582
    H    -0.616687006606    -2.052008115699    -0.815033132467
    H    -0.616687006606    -2.052008115699     0.815033132467
    H     0.107891112776     2.384096666234     0.000000000000
  symmetry c1
""")

psi4.set_options({'basis': '6-31+G(d)'}),

# Paramdetros del sistema
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = parameters.param(mol,wfn)
p.ipnof = PNOFi
p.gradient = gradient


t1 = time()
energy.compute_energy(mol,wfn,PNOFi,p,gradient)
t2 = time()
print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))
