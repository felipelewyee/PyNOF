import pynof

mol = pynof.molecule("""
0 1
  O   -4.9355124    2.3281965    0.0249981
  H   -3.9461861    2.3375266   -0.0168319
  H   -5.2248774    2.5343643   -0.8993212
""")

p = pynof.param(mol,"cc-pvdz")
p.autozeros()

p.threshl=10**-4
p.threshe=10**-5

p.set_ncwo(1)

p.RI = False

p.title = "Reference"
p.ipnof=8
E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True)
#p.RI = False

#E,C,gamma,fmiug0 = pynof.brute_force_energy(mol,p,20,hfidr=False)
