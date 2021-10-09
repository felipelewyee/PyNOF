import numpy as np
from scipy.linalg import eigh
from time import time
import pynof

def compute_energy(mol,p=None,gradient="analytical",C=None,gamma=None,fmiug0=None,hfidr=True,nofmp2=False,gradients=False,printmode=True):
 
    t1 = time()

    wfn = p.wfn

    S,T,V,H,I,b_mnl = pynof.compute_integrals(wfn,mol,p)

    if(printmode):
        print("Number of basis functions                   (NBF)    =",p.nbf)
        if(p.RI):
            print("Number of auxiliary basis functions         (NBFAUX) =",p.nbfaux)
        print("Inactive Doubly occupied orbitals up to     (NO1)    =",p.no1)
        print("No. considered Strongly Doubly occupied MOs (NDOC)   =",p.ndoc)
        print("No. considered Strongly Singly occupied MOs (NSOC)   =",p.nsoc)
        print("NO. of Weakly occ. per St. Doubly occ.  MOs (NCWO)   =",p.ncwo)
        print("Dimension of the Nat. Orb. subspace         (NBF5)   =",p.nbf5)
        print("No. of electrons                                     =",p.ne)
        print("Multiplicity                                         =",p.mul)
        print("")

    # Energía Nuclear
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C)
    Cguess = C
    if(C is None):
        E_i,Cguess = eigh(H, S)  # (HC = SCe)
    Cguess = pynof.check_ortho(Cguess,S,p)

    if (hfidr):
        EHF,Cguess,fmiug0guess = pynof.hfidr(Cguess,H,I,b_mnl,E_nuc,p,printmode)

    if(C is None):
        C = Cguess
    C = pynof.check_ortho(C,S,p)

    if(gamma is None):
        gamma = np.zeros((p.nbf5))
        for i in range(p.ndoc):
            gamma[i] = np.arccos(np.sqrt(2.0*0.999-1.0))
            for j in range(p.ncwo-1):
                ig = p.ndoc+i*(p.ncwo-1)+j
                gamma[ig] = np.arcsin(np.sqrt(1.0/(p.ncwo-j)))

    elag = np.zeros((p.nbf,p.nbf)) #temporal
    gamma,n,cj12,ck12 = pynof.occoptr(gamma,True,False,C,H,I,b_mnl,p)

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
        convgdelag,E_old,E_diff,sumdiff_old,itlim,fmiug0,C,elag = pynof.orboptr(C,n,H,I,b_mnl,cj12,ck12,E_old,E_diff,sumdiff_old,i_ext,itlim,fmiug0,E_nuc,p,printmode)
        #t2 = time()

        #occopt
        gamma,n,cj12,ck12 = pynof.occoptr(gamma,False,convgdelag,C,H,I,b_mnl,p)
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

        key = np.zeros((p.nbf5))
        val = 0
        for i in range(p.ndoc):       
            val += 1
            idx = p.no1 + i
            # inicio y fin de los orbitales acoplados a los fuertemente ocupados
            ll = p.no1 + p.ndns + p.ncwo*(p.ndoc-i-1)
            ul = p.no1 + p.ndns + p.ncwo*(p.ndoc-i)
            key[idx] = val
            key[ll:ul] = val
        for i in range(p.nsoc):
            idx = p.ndoc + i
            val += 1
            key[idx] = val

        e_val = elag[np.diag_indices(p.nbf5)]
        sort_indices = np.array(list(e_val[:p.nbeta].argsort()) + list(e_val[p.nbeta:p.nalpha].argsort()) + list(e_val[p.nalpha:p.nbf5].argsort()))
        sort_indices[p.nbeta:p.nalpha] += p.nbeta
        sort_indices[p.nalpha:p.nbf5] += p.nalpha
        n_sorted = n[sort_indices]
        e_sorted = e_val[sort_indices]
        key_sorted = key[sort_indices]
        print(" {:^3}    {:^9}   {:>12}  {:^3}".format("Idx","n","E (Hartree)", "Key"))
        for i in range(p.nbeta):
            print(" {:3d}    {:9.7f}  {:12.8f}  {:3d}".format(i+1,2*n_sorted[i],e_sorted[i],int(key_sorted[i])))
        for i in range(p.nbeta,p.nalpha):
            if(not p.HighSpin):
                print(" {:3d}    {:9.7f}  {:12.8f}  {:3d}".format(i+1,2*n_sorted[i],e_sorted[i],int(key_sorted[i])))
            else:
                print(" {:3d}    {:9.7f}  {:12.8f}  {:3d}".format(i+1,n_sorted[i],e_sorted[i],int(key_sorted[i])))
        for i in range(p.nalpha,p.nbf5):
            print(" {:3d}    {:9.7f}  {:12.8f}  {:3d}".format(i+1,2*n_sorted[i],e_sorted[i],int(key_sorted[i])))

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

    pynof.fchk(p.title,wfn,mol,"Energy",E_t,elag,n,C,p)

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))

    if(nofmp2):
        pynof.nofmp2(n,C,H,I,b_mnl,E_nuc,p)


    if gradients:
        grad = pynof.compute_der_integrals(wfn,mol,n,C,cj12,ck12,elag,p)
        return E_t,C,gamma,fmiug0,grad.flatten()
    else:
        return E_t,C,gamma,fmiug0



def brute_force_energy(mol,p,intents=5,C=None,gamma=None,fmiug0=None,hfidr=True,RI_last=False,gpu_last=False):
    t1 = time()
    
    if(C=None and gamma=None and fmiug0=None):
        E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,p.gradient,hfidr)
    else:
        E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,p.gradient,C,gamma,fmiug0,hfidr)
    E_min = E
    C_min = C
    gamma_min = gamma
    fmiug0_min = fmiug0
    
    for i in range(intents):
        p.autozeros()
        E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,p.gradient,C,gamma,None,hfidr=False,nofmp2=False)
        if(E<E_min):
            E_min = E
            C_min = C
            gamma_min = gamma
            fmiug0_min = fmiug0
    
    p.RI = RI_last
    p.gpu = gpu_last
    p.jit = True
    E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,p.gradient,C_min,gamma_min,fmiug0_min,hfidr=False,nofmp2=False)
    
    t2 = time()
    
    print("Best Total NOF Energy {}".format(E))
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))
   
    return E,C,gamma,fmiug0
    
