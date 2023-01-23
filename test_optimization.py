import pynof

mol = pynof.molecule("""
0 1
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
""")

p = pynof.param(mol,"cc-pVDZ")
p.autozeros()
p.RI = True

pynof.optgeo(mol,p)
