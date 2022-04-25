import numpy as np
import pynof

mol = pynof.molecule("""
0 1
B  0.000000  0.000000 -0.748417
N  0.000000  0.000000  0.534583
B  0.000000  0.000000  6.251583
N  0.000000  0.000000  7.534583
""")
#mol = pynof.molecule("""
#0 1
#  He    0.0000000    0.0000000    0.8490000
#  He    0.0000000    0.0000000   -0.8490000
#""")

p = pynof.param(mol,"cc-pVDZ")
#p = pynof.param(mol,"6-31G")
p.autozeros()

p.threshl=10**-4
p.threshe=10**-5

p.set_ncwo(1)

p.RI = True
p.gpu = True

p.title = "tmp"
C = pynof.read_C("tmp")
gamma = pynof.read_gamma("tmp")
############################################################################
#p.orbital_optimizer = "L-BFGS-B"
p.combined_optimizer = "trust-krylov"
##################################
#print("====================   Rotations Trust  ===========================")
#p.method = "Rotations"
#p.autozeros()
#E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True,check_hessian=True)
#print("====================   Combined  Trust  ===========================")
#p.method = "Combined"
#E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True,check_hessian=True)
print("====================   ID  ===========================")
p.method = "ID"
E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True,check_hessian=True)
