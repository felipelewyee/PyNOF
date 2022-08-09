import pynof

mol = pynof.molecule("""
0 1
  C   -1.5535949   -0.0794622   -0.0153861
  C   -0.7417785    1.2112916    0.0353439
  C    0.7565252    0.9248780    0.0085429
  H   -2.6372892    0.1615747    0.0055264
  H   -1.3146911   -0.7204689    0.8599395
  H   -1.3314437   -0.6388306   -0.9491501
  H   -0.9917337    1.7640358    0.9665488
  H   -1.0084410    1.8454487   -0.8375148
  H    1.0470825    0.3063263    0.8844023
  H    1.3188963    1.8815529    0.0465079
  H    1.0303285    0.3879676   -0.9246872
""")

p = pynof.param(mol,"cc-pvdz")
p.autozeros()

p.threshl=10**-4
p.threshe=10**-5

p.set_ncwo(-1)

p.RI = True
p.jit = True
#p.gpu = True

p.title = "Reference"
p.ipnof=8
C,gamma,fmiug0 = pynof.read_all(p.title)
E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,C,gamma,fmiug0,hfidr=False)
#p.RI = False

#E,C,gamma,fmiug0 = pynof.brute_force_energy(mol,p,20,hfidr=False)
