import pynof

mol = pynof.molecule("""
H   0.0000   0.0000   0.000
H   0.0000   0.0000   2.645
""")

p = pynof.param(mol,"6-31+G(d)")
p.autozeros()

p.ipnof = 5

#p.set_ncwo(2)

p.RI = True
#p.gpu = True

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True,erpa=True)
