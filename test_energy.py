import pynof

mol = pynof.molecule("""
0 1
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
""")

p = pynof.param(mol,"cc-pvdz")
p.autozeros()

p.ipnof = 8

p.RI = False#True
p.pnof = 7
p.ista = 1
#p.gpu = True

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True,mbpt=True)
