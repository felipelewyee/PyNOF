import psi4
import numpy as np
from scipy.linalg import eigh
from time import time
import pynof
from scipy.optimize import minimize

def optgeo(mol,p,C=None,n=None,fmiug0=None):
   
    wfn = p.wfn
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    if(C is None or n is None or fmiug0 is None):
        E_t = pynof.compute_energy(mol,p,C,n,fmiug0,hfidr=True,printmode=True)
    
    print("Initial Geometry (Bohrs)")
    print("========================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    coord = coord.flatten()
    res = minimize(energy_optgeo, coord, args=(symbols,p,True), jac=True, method='CG')

    coord = res.x

    if(res.success):
        print("\n\n================¡Converged! :) ================\n\n")
    else:
        print("\n\n================¡Not Converged! :( ================\n\n")

    E,grad = energy_optgeo(coord,symbols,p,printmode=True)

    coord = np.reshape(coord,(int(len(coord)/3),3))

    print("Final Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))
    print("Final Geometry (Angstroms)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0]*0.529177,xyz[1]*0.529177,xyz[2]*0.529177))

    return coord


def energy_optgeo(coord,symbols,p,printmode=False):

    coord = np.reshape(coord,(int(len(coord)/3),3))
    print("Iter Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    mol_string = "{} {} \n".format(p.charge,p.mul)
    for symbol,xyz in zip(symbols,coord):
        mol_string += "{:s} {} {} {}\n".format(symbol,xyz[0],xyz[1],xyz[2])
    mol_string += "units bohr"
    mol = psi4.geometry(mol_string)
    
    # Paramdetros del sistema
    p.wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
  
    try:
        C,n,fmiug0 = pynof.read_all(p.title)
    except:
        C = pynof.read_C(p.title)
        n = pynof.read_n(p.title)
        fmiug0 = None

    #p.autozeros()
    p.autozeros(restart=True)
    
    t1 = time()
    E_t,C,n,fmiug0,grad = pynof.compute_energy(mol,p,C,n,fmiug0,hfidr=False,gradients=True,printmode=printmode)
    t2 = time()
    print("                       Total Energy:", E_t)

    print("====Gradient====")
    for i in range(p.natoms):
        print("Atom {:2d} {:10.4f} {:10.4f} {:10.4f}".format(i,grad[i*3+0],grad[i*3+1],grad[i*3+2]))

    return E_t,grad


