import pynof

mol = pynof.molecule("""
0 1
  O    0.0000   -0.0076    0.0000
  H    0.0000    0.3468   -0.7830
  H    0.0000    0.3468    0.7830
""")

p = pynof.param(mol,"cc-pvdz")
p.autozeros()

p.ipnof = 8

p.RI = True
#p.gpu = True

p.method = "ID"
E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True)
