import psi4
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from time import time
import minimization
import cupy as cp

def compute_energy(mol,wfn,PNOFi=7,p=None,gradient="analytical"):
 
    # Integrador
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    # Overlap, Kinetics, Potential
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = T + V
    
    # Integrales de Repulsión Electrónica, ERIs (mu nu | sigma lambda)
    I = np.asarray(mints.ao_eri())

    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pVDZ")

    orb = wfn.basisset()
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

    Ppq = mints.ao_eri(orb, orb, aux, zero_bas)

    metric = mints.ao_eri(aux, zero_bas, aux, zero_bas)
    metric.power(-0.5, 1.e-14)

    Ppq = np.squeeze(Ppq)
    metric = np.squeeze(metric)

    b_mnl = np.einsum('pqP,PQ->pqQ', Ppq, metric, optimize=True)
    b_mnl = cp.array(b_mnl)

    # Energía Nuclear
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C) mediante (HC = SCe)
    E_i,C = eigh(H, S)

    # Revisa ortonormalidad
    orthonormality = True
    CTSC = np.matmul(np.matmul(np.transpose(C),S),C)
    ortho_deviation = np.abs(CTSC - np.identity(p.nbf))
    if (np.any(ortho_deviation > 10**-6)):
        orthonormality = False
    if not orthonormality:
        print("Orthonormality violations {:d}, Maximum Violation {:f}".format((ortho_deviation > 10**-6).sum(),ortho_deviation.max()))
    else:
        print("No violations of the orthonormality")
    for j in range(p.nbf):
        #Obtiene el índice del coeficiente con mayor valor absoluto del MO
        idxmaxabsval = 0
        for i in range(p.nbf):
            if(abs(C[i][j])>abs(C[idxmaxabsval][j])):
                 idxmaxabsval = i
    # Ajusta el signo del MO
    sign = np.sign(C[idxmaxabsval][j])
    C[0:p.nbf,j] = sign*C[0:p.nbf,j]

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
    
