import pynof

mol = pynof.molecule("""
0 1
  He    0.0000000    0.0000000    0.3540000
  He    0.0000000    0.0000000   -0.3540000
""")
#mol = pynof.molecule("""
#0 1
#O  0.0000   0.000   0.116
#H  0.0000   0.749  -0.453
#H  0.0000  -0.749  -0.453
#""")

p = pynof.param(mol,"6-31G")
p.autozeros()

p.threshl=10**-4
p.threshe=10**-5

p.set_ncwo(1)

p.RI = False

p.title = "Reference"
E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=False)
#p.RI = False

#E,C,gamma,fmiug0 = pynof.brute_force_energy(mol,p,20,hfidr=False)
