import pynof

mol = pynof.molecule("""
0 1
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
""")

p = pynof.param(mol,"6-31G")
p.autozeros()

p.threshl=10**-4
p.threshe=10**-5

p.set_ncwo(1)

p.RI = True

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=False)
p.RI = False

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=False,C=C,gamma=gamma,fmiug0=fmiug0,mbpt=True)
