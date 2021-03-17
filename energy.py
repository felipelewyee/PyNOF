import psi4
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from time import time
import minimization
import cupy as cp
import integrals
import utils

def compute_energy(mol,wfn,PNOFi=7,p=None,gradient="analytical"):


    S,T,V,H,I,b_mnl = integrals.compute_integrals(wfn,mol,p)

    # Energía Nuclear
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C) mediante (HC = SCe)
    E_i,C = eigh(H, S)

    C = utils.check_ortho(C,S,p)

    E,C,fmiug0 = minimization.hfidr(C,H,I,b_mnl,E_nuc,p)

    gamma = np.zeros((p.nbf5))
    for i in range(p.ndoc):
        gamma[i] = np.arccos(np.sqrt(2.0*0.999-1.0))
        for j in range(p.ncwo-1):
            ig = p.ndoc+(i)*(p.ncwo-1)+j
            gamma[ig] = np.arcsin(np.sqrt(1.0/(p.ncwo-j)))

    elag = np.zeros((p.nbf,p.nbf)) #temporal
    gamma,elag,n,cj12,ck12 = minimization.occoptr(gamma,True,False,elag,C,H,I,b_mnl,p)

    iloop = 0
    itlim = 1
    E_old = E
    sumdiff_old = 0

    print('{:^7} {:^7} {:^14} {:^14} {:^14} {:^14}'.format("Nitext","Nitint","Eelec","Etot","Ediff","maxdiff"))
    for i_ext in range(1000):
        t1 = time()
        #orboptr
        convgdelag,E_old,sumdiff_old,itlim,fmiug0,C = minimization.orboptr(C,n,H,I,b_mnl,cj12,ck12,E_old,sumdiff_old,i_ext,itlim,fmiug0,E_nuc,p)

        #occopt
        gamma,elag,n,cj12,ck12 = minimization.occoptr(gamma,False,convgdelag,elag,C,H,I,b_mnl,p)

        if(convgdelag):
            break
        t2 = time()
        print(t2-t1)
    #######
    
