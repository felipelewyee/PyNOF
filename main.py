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
import projection

# Seleccionamos una molécula, y otros datos como la memoria del sistema y la base
psi4.set_memory('4 GB')

mol = pynof.molecule("""
0 1
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
  symmetry c1
""")

psi4.set_options({'basis': 'def2-TZVPD'})

# Paramdetros del sistema
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = pynof.param(mol,wfn)
#p.ipnof = 7
#p.gradient = "analytical"
#p.gradient = "numerical"
#p.optimizer = "Newton-CG"
#p.RI = False#True 
#p.gpu = True
#p.jit = False

p.autozeros()

#t1 = time()
E,C,gamma,fmiug0 = pynof.compute_energy(mol,wfn,p,p.gradient)
t2 = time()
print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))

#############################################

old_basis = wfn.basisset()

psi4.set_options({'basis': 'def2-TZVPD'})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = parameters.param(mol,wfn)
p.ipnof = 7
p.gradient = "analytical"
p.optimizer = "Newton-CG"
p.RI = False#True 
p.gpu = True
p.jit = False

C = projection.project_MO(mol,C,old_basis)

p.autozeros()
C,fmiug0 = projection.complete_projection(wfn,mol,p,C,fmiug0,False)

C_proj = C

p.autozeros(restart=True)
E,C,gamma,fmiug0 = energy.compute_energy(mol,wfn,p,p.gradient,C,gamma,fmiug0,True)

#######################################################
psi4.set_options({'basis': '6-311++G(d,p)'})

# Paramdetros del sistema
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
p = parameters.param(mol,wfn)
p.ipnof = 7
p.gradient = "analytical"
p.optimizer = "Newton-CG"
p.RI = False#True
p.gpu = True
p.jit = False

p.autozeros()

t1 = time()
E,C,gamma,fmiug0 = energy.compute_energy(mol,wfn,p,p.gradient)
t2 = time()
print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))

C_real = C

import numpy as np

for i in range(C_proj.shape[1]):
    print("{:2d} {:10.6f} {:10.6f} {:10.6f}".format(i,np.linalg.norm(C_real[:,i]-C_proj[:,i]),max(abs(C_real[:,i]-C_proj[:,i])),np.mean(abs(C_real[:,i]-C_proj[:,i]))))

