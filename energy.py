import numpy as np
from scipy.linalg import eigh
from time import time
import minimization
import integrals
import utils
import postpnof

def compute_energy(mol,wfn,p=None,gradient="analytical",C=None,gamma=None,fmiug0=None,hfidr=True,nofmp2=False,gradients=False,printmode=True):

    S,T,V,H,I,b_mnl = integrals.compute_integrals(wfn,mol,p)

    print("Number of basis functions                   (NBF)    =",p.nbf)
    if(p.RI):
        print("Number of auxiliary basis functions         (NBFAUX) =",p.nbfaux)
    print("Inactive Doubly occupied orbitals up to     (NO1)    =",p.no1)
    print("No. considered Strongly Doubly occupied MOs (NDOC)   =",p.ndoc)
    print("No. considered Strongly Singly occupied MOs (NSOC)   =",p.nsoc)
    print("NO. of Weakly occ. per St. Doubly occ.  MOs (NCWO)   =",p.ncwo)
    print("Dimension of the Active Nat. Orb. subspace  (NBF5)   =",p.nbf5)
    print("No. of electrons                                     =",p.ne)
    print("Multiplicity                                         =",p.mul)
    print("")

    # Energía Nuclear
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C)
    Cguess = C
    if(C is None):
        E_i,Cguess = eigh(H, S)  # (HC = SCe)
    Cguess = utils.check_ortho(Cguess,S,p)

    if (hfidr):
        EHF,Cguess,fmiug0guess = minimization.hfidr(Cguess,H,I,b_mnl,E_nuc,p,printmode)

    if(C is None):
        C = Cguess
    C = utils.check_ortho(C,S,p)

    if(gamma is None):
        gamma = np.zeros((p.nbf5))
        for i in range(p.ndoc):
            gamma[i] = np.arccos(np.sqrt(2.0*0.999-1.0))
            for j in range(p.ncwo-1):
                ig = p.ndoc+i*(p.ncwo-1)+j
                gamma[ig] = np.arcsin(np.sqrt(1.0/(p.ncwo-j)))

    elag = np.zeros((p.nbf,p.nbf)) #temporal
    gamma,n,cj12,ck12 = minimization.occoptr(gamma,True,False,C,H,I,b_mnl,p)

    iloop = 0
    itlim = 0
    E_old = 9999#EHF
    E_diff = 9999
    sumdiff_old = 0

    if(printmode):
        print("")
        print("PNOF{} Calculation".format(p.ipnof))
        print("==================")
        print("")
        print('{:^7} {:^7} {:^14} {:^14} {:^14} {:^14}'.format("Nitext","Nitint","Eelec","Etot","Ediff","maxdiff"))
    for i_ext in range(p.maxit):
        #t1 = time()
        #orboptr
        convgdelag,E_old,E_diff,sumdiff_old,itlim,fmiug0,C,elag = minimization.orboptr(C,n,H,I,b_mnl,cj12,ck12,E_old,E_diff,sumdiff_old,i_ext,itlim,fmiug0,E_nuc,p,printmode)
        #t2 = time()

        #occopt
        gamma,n,cj12,ck12 = minimization.occoptr(gamma,False,convgdelag,C,H,I,b_mnl,p)
        #t3 = time()

        if(convgdelag):
            break
        #print(t2-t1,t3-t2)

    np.save(p.title+"_C.npy",C)
    np.save(p.title+"_gamma.npy",gamma)
    np.save(p.title+"_fmiug0.npy",fmiug0)
 
    if(printmode):
        print("")
        print("RESULTS OF THE OCCUPATION OPTIMIZATION")
        print("========================================")
        for i in range(p.nbeta):
            print(" {:3d}    {:9.7f}  {:10.8f}".format(i+1,2*n[i],elag[i][i]))
        for i in range(p.nbeta,p.nalpha):
            if(not p.HighSpin):
                print(" {:3d}    {:9.7f}  {:10.8f}".format(i+1,2*n[i],elag[i][i]))
            else:
                print(" {:3d}    {:9.7f}  {:10.8f}".format(i+1,n[i],elag[i][i]))
        for i in range(p.nalpha,p.nbf5):
            print(" {:3d}    {:9.7f}  {:10.8f}".format(i+1,2*n[i],elag[i][i]))

        print("")

        print("----------------")
        print(" Final Energies ")
        print("----------------")
        
        if(hfidr):
            print("       HF Total Energy = {:15.7f}".format(E_nuc + EHF))
        print("Final NOF Total Energy = {:15.7f}".format(E_nuc + E_old))
        if(hfidr):
            print("    Correlation Energy = {:15.7f}".format(E_old-EHF))
        print("")
        print("")

    E_t = E_nuc + E_old

    if(nofmp2):
        postpnof.nofmp2(n,C,H,I,b_mnl,E_nuc,p)

    if gradients:
        grad = integrals.compute_der_integrals(wfn,mol,n,C,cj12,ck12,elag,p)
        return E_t,C,gamma,fmiug0,grad.flatten()
    else:
        return E_t,C,gamma,fmiug0
