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

psi4.set_options({'basis': 'cc-pVDZ'}),

# Paramdetros del sistema
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = parameters.param(mol,wfn)
p.ipnof = PNOFi
p.gradient = gradient
p.RI = False#True 
p.gpu = True
p.jit = False

p.threshl = 10**-3   # Convergencia de los multiplicadores de Lagrange
p.threshe = 10**-6   # Convergencia de los multiplicadores de Lagrange
#p.perdiis = False      # Aplica DIIS cada NDIIS (True) o después de NDIIS (False)
p.optimizer = "CG"
#p.C_guess = "Read"
#p.gamma_guess = "Read"
#p.fmiug0_guess = "Read"
#p.hfidr = False
#p.optimizer = "Newton-CG"

C,gamma,fmiug0 = guess.read_all()

p.autozeros()

t1 = time()
energy.compute_energy(mol,wfn,p,gradient,C,gamma,fmiug0)
energy.compute_energy(mol,wfn,p,gradient,C,gamma,fmiug0)
#optimization.optgeo(mol,wfn,p,gradient)
t2 = time()
print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))
