import psi4
import numpy as np
from scipy.linalg import eigh
from time import time
import minimization
import integrals
import utils
import energy
from scipy.optimize import minimize

def optgeo(mol,wfn,p=None,gradient="analytical"):
    
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    
    print("Initial Geometry (Bohrs)")
    print("========================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    res = minimize(energy_optgeo, coord, args=(symbols,p,gradient), method='Nelder-Mead')

    coord = res.x

    energy_optgeo(coord,symbols,p,gradient)

    coord = np.reshape(coord,(int(len(coord)/3),3))

    print("Final Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))


def energy_optgeo(coord,symbols,p,gradient):

    coord = np.reshape(coord,(int(len(coord)/3),3))
    print("Iter Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    mol_string = ""
    for symbol,xyz in zip(symbols,coord):
        mol_string += "{:s} {} {} {}\n".format(symbol,xyz[0],xyz[1],xyz[2])
    mol_string += "units bohr"
    mol = psi4.geometry(mol_string)
    
    # Paramdetros del sistema
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
    
    #p.autozeros()
    
    t1 = time()
    E_t = energy.compute_energy(mol,wfn,p,gradient,printmode=False)
    t2 = time()
    print("                       Total Energy:", E_t)

    return E_t


