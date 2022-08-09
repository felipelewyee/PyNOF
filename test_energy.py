import pynof

mol = pynof.molecule("""
0 1
  C   -3.8198002    1.2608710    0.0099259
  C   -2.7184580    2.3164905    0.0417083
  C   -1.3308345    1.6700722   -0.0340180
  C   -0.2294924    2.7256919   -0.0022383
  H   -4.8120091    1.7562051    0.0660120
  H   -3.7177350    0.5702225    0.8740464
  H   -3.7670434    0.6760772   -0.9331277
  H   -2.8079941    2.9001710    0.9839137
  H   -2.8572068    3.0058174   -0.8197019
  H   -1.2412991    1.0863910   -0.9762229
  H   -1.1920849    0.9807459    0.8273927
  H   -0.3315599    3.4163399   -0.8663590
  H    0.7627164    2.2303580   -0.0583268
  H   -0.2822472    3.3104863    0.9408151
""")

p = pynof.param(mol,"cc-pvdz")
p.autozeros()

p.threshl=10**-4
p.threshe=10**-5

p.set_ncwo(1)

p.RI = True
p.jit = True
#p.gpu = True

p.title = "Reference"
p.ipnof=8
E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True)
#p.RI = False

#E,C,gamma,fmiug0 = pynof.brute_force_energy(mol,p,20,hfidr=False)
