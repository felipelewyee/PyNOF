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
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
  symmetry c1
""")

psi4.set_options({'basis': 'cc-pVDZ'}),

# Parametros del sistema
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = parameters.param(mol,wfn)
p.ipnof = 7
p.gradient = "analytical"
p.optimizer = "Newton-CG"
p.RI = False 
p.gpu = True
p.jit = False
p.threshl = 10**-5
p.threshe = 10**-7

p.autozeros()

t1 = time()
optimization.optgeo(mol,wfn,p,p.gradient)
#E,C,gamma,fmiug0 = optimization.optgeo(mol,wfn,p,p.gradient)
t2 = time()
print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))
