import psi4
import numpy as np
from scipy.linalg import eigh
from time import time
import minimization
import integrals
import utils
import energy
import guess
from scipy.optimize import minimize

def optgeo(mol,wfn,p=None,gradient="analytical"):
    
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    E_t = energy.compute_energy(mol,wfn,p,p.gradient,printmode=False)
    
    print("Initial Geometry (Bohrs)")
    print("========================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    res = minimize(energy_optgeo, coord, args=(symbols,p,gradient), jac=False, method='Nelder-Mead')

    print(res)
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
   
    C,gamma,fmiug0 = guess.read_all()

    p.autozeros()
    #p.autozeros(restart=True)
    
    t1 = time()
    E_t,C,gamma,fmiug0 = energy.compute_energy(mol,wfn,p,p.gradient,C,gamma,fmiug0,gradients=False,printmode=False)
    #E_t,C,gamma,fmiug0,grad = energy.compute_energy(mol,wfn,p,p.gradient,None,gamma,fmiug0,printmode=False)
    #E_t,C,gamma,fmiug0,grad = energy.compute_energy(mol,wfn,p,p.gradient,C=None,gamma=None,fmiug0=None,gradients=True,printmode=False)
    t2 = time()
    print("                       Total Energy:", E_t)

    #print("====Gradient====")
    #for i in range(p.natoms):
    #    print("Atom {:2d} {} {} {}".format(i,grad[i*3+0],grad[i*3+1],grad[i*3+2]))

    return E_t#,grad


