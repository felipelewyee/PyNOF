import numpy as np
import pynof

mol = pynof.molecule("""
0 1
B  0.000000  0.000000 -0.748417
N  0.000000  0.000000  0.534583
B  0.000000  0.000000  6.251583
N  0.000000  0.000000  7.534583
""")

p = pynof.param(mol,"cc-pVDZ")
p.autozeros()

p.threshl=10**-4
p.threshe=10**-5

p.set_ncwo(1)

p.RI = True
p.gpu = True

print("====================   ID  ===========================")
p.method = "ID"
E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True,check_hessian=True)