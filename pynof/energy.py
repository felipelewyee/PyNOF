import numpy as np
from scipy.linalg import eigh
from time import time
import pynof

def compute_energy(mol,p=None,C=None,n=None,fmiug0=None,hfidr=True,nofmp2=False,mbpt=False,gradients=False,printmode=True,ekt=False,mulliken_pop=False,lowdin_pop=False,m_diagnostic=False,perturb=False,erpa=False):
 
    t1 = time()

    wfn = p.wfn
    S,T,V,H,I,b_mnl,Dipole = pynof.compute_integrals(wfn,mol,p)

    if(printmode):
        print("Number of basis functions                   (NBF)    =",p.nbf)
        if(p.RI):
            print("Number of auxiliary basis functions         (NBFAUX) =",p.nbfaux)
        print("Inactive Doubly occupied orbitals up to     (NO1)    =",p.no1)
        print("No. considered Strongly Doubly occupied MOs (NDOC)   =",p.ndoc)
        print("No. considered Strongly Singly occupied MOs (NSOC)   =",p.nsoc)
        print("No. of Weakly occ. per St. Doubly occ.  MOs (NCWO)   =",p.ncwo)
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

    if(n is None):
        gamma = pynof.compute_gammas_trigonometric(p.nv,p.ndoc,p.ncwo)
        if p.occ_method == "Softmax":
            p.nv = p.nbf5 - p.no1 - p.nsoc 
            n,dn_dgamma = pynof.ocupacion_trigonometric(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
            gamma = pynof.n_to_gammas_softmax(n,p.nv,p.no1,p.ndoc,p.ndns,p.ncwo)
    else:
        if p.occ_method == "Trigonometric":
            gamma = pynof.n_to_gammas_trigonometric(n,p.nv,p.no1,p.ndoc,p.ndns,p.ncwo)
        if p.occ_method == "Softmax":
            p.nv = p.nbf5 - p.no1 - p.nsoc 
            gamma = pynof.n_to_gammas_softmax(n,p.nv,p.no1,p.ndoc,p.ndns,p.ncwo)

    elag = np.zeros((p.nbf,p.nbf))

    E_occ,nit_occ,success_occ,gamma,n,cj12,ck12 = pynof.occoptr(gamma,C,H,I,b_mnl,p)

    iloop = 0
    itlim = 0
    E,E_old,E_diff = 9999,9999,9999
    Estored,Cstored,gammastored = 0,0,0
    last_iter = 0

    if(printmode):
        print("")
        print("PNOF{} Calculation (ID Optimization)".format(p.ipnof))
        print("==================")
        print("")
        print('{:^7} {:^7}  {:^7}  {:^14} {:^14} {:^14}   {:^6}   {:^6} {:^6} {:^6}'.format("Nitext","Nit_orb","Nit_occ","Eelec","Etot","Ediff","Grad_orb","Grad_occ","Conv Orb","Conv Occ"))
    for i_ext in range(p.maxit):
        #orboptr
        #t1 = time()
        if(p.orb_method=="ID"):
            E_orb,C,nit_orb,success_orb,itlim,fmiug0 = pynof.orboptr(C,n,H,I,b_mnl,cj12,ck12,i_ext,itlim,fmiug0,p,printmode)
        if(p.orb_method=="Rotations"):
            E_orb,C,nit_orb,success_orb = pynof.orbopt_rotations(gamma,C,H,I,b_mnl,p)
        #t2 = time()

        #occopt
        E_occ,nit_occ,success_occ,gamma,n,cj12,ck12 = pynof.occoptr(gamma,C,H,I,b_mnl,p)
        #t3 = time()
        #print("t_orb: {:3.1e} t_occ: {:3.1e}".format(t2-t1,t3-t2))
        if(p.occ_method=="Softmax"):
            C,gamma = pynof.order_occupations_softmax(C,gamma,H,I,b_mnl,p)

        E = E_orb
        E_diff = E-E_old
        E_old = E

        y = np.zeros((p.nvar))
        grad_orb = pynof.calcorbg(y,n,cj12,ck12,C,H,I,b_mnl,p)
        J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)
        grad_occ = pynof.calcoccg(gamma,J_MO,K_MO,H_core,p)
        print("{:6d} {:6d} {:6d}   {:14.8f} {:14.8f} {:15.8f}      {:3.1e}    {:3.1e}   {}   {}".format(i_ext,nit_orb,nit_occ,E,E+E_nuc,E_diff,np.linalg.norm(grad_orb),np.linalg.norm(grad_occ),success_orb,success_occ))

        if(success_orb or (np.linalg.norm(grad_orb) < 1e-3 and np.linalg.norm(grad_occ)< 1e-3)):

            if perturb and E - Estored < -1e-4:
                y = np.zeros((p.nvar))
                grad_orb = pynof.calcorbg(y,n,cj12,ck12,C,H,I,b_mnl,p)
                J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)
                grad_occ = pynof.calcoccg(gamma,J_MO,K_MO,H_core,p)
                print("Increasing Gradient")
                last_iter = i_ext
                Estored,Cstored,gammastored = E,C.copy(),gamma.copy()
                C,gamma = pynof.perturb_solution(C,gamma,grad_orb,grad_occ,p)
            else:
                print("Solution does not improve anymore")
                if(Estored<E):
                    E,C,gamma = Estored,Cstored,gammastored
                break

    if(p.orb_method=="ID"):
        np.save(p.title+"_fmiug0.npy",fmiug0)
    
    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = pynof.PNOFi_selector(n,p)
    E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)
    print("\nLagrage sumdiff {:3.1e} maxfdiff {:3.1e}".format(sumdiff,maxdiff))

    C,n,elag = pynof.order_subspaces(C,n,elag,H,I,b_mnl,p)

    np.save(p.title+"_C.npy",C)
    np.save(p.title+"_n.npy",n)

    if(printmode):
        print("")
        print("RESULTS OF THE OCCUPATION OPTIMIZATION")
        print("========================================")

        e_val = elag[np.diag_indices(p.nbf5)]
        print(" {:^3}    {:^9}   {:>12}".format("Idx","n","E (Hartree)"))
        for i in range(p.nbeta):
            print(" {:3d}    {:9.7f}  {:12.8f}".format(i+1,2*n[i],e_val[i]))
        for i in range(p.nbeta,p.nalpha):
            if(not p.HighSpin):
                print(" {:3d}    {:9.7f}  {:12.8f}".format(i+1,2*n[i],e_val[i]))
            else:
                print(" {:3d}    {:9.7f}  {:12.8f}".format(i+1,n[i],e_val[i]))
        for i in range(p.nalpha,p.nbf5):
            print(" {:3d}    {:9.7f}  {:12.8f}".format(i+1,2*n[i],e_val[i]))

        print("")

        print("----------------")
        print(" Final Energies ")
        print("----------------")
        
        if(hfidr):
            print("       HF Total Energy = {:15.7f}".format(E_nuc + EHF))
        print("Final NOF Total Energy = {:15.7f}".format(E_nuc + E))
        if(hfidr):
            print("    Correlation Energy = {:15.7f}".format(E-EHF))
        print("")
        print("")

    E_t = E_nuc + E

    pynof.fchk(p.title,wfn,mol,"Energy",E_t,elag,n,C,p)

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))

    if(nofmp2):
        pynof.nofmp2(n,C,H,I,b_mnl,E_nuc,p)

    if(mbpt):
        pynof.mbpt(n,C,H,I,b_mnl,Dipole,E_nuc,E,p)

    if(ekt):
        pynof.ext_koopmans(p,elag,n)

    if(mulliken_pop):
        pynof.mulliken_pop(p,wfn,n,C,S)

    if(lowdin_pop):
        pynof.lowdin_pop(p,wfn,n,C,S)

    if(m_diagnostic):
        pynof.M_diagnostic(p,n)

    if(erpa):
        pynof.ERPA(wfn,mol,n,C,H,I,b_mnl,cj12,ck12,elag,p)


    if gradients:
        grad = pynof.compute_geom_gradients(wfn,mol,n,C,cj12,ck12,elag,p)
        return E_t,C,n,fmiug0,grad.flatten()
    else:
        return E_t,C,n,fmiug0


def brute_force_energy(mol,p,intents=5,C=None,gamma=None,fmiug0=None,hfidr=True,RI_last=False,gpu_last=False,ekt=False,mulliken_pop=False,lowdin_pop=False,m_diagnostic=False):
    t1 = time()
    
    E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,C,gamma,fmiug0,hfidr)
    E_min = E
    C_min = C
    gamma_min = gamma
    fmiug0_min = fmiug0
    
    for i in range(intents):
        p.autozeros()
        E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,C,gamma,None,hfidr=False,nofmp2=False)
        if(E<E_min):
            E_min = E
            C_min = C
            gamma_min = gamma
            fmiug0_min = fmiug0
    
    p.RI = RI_last
    p.gpu = gpu_last
    p.jit = True
    E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,C_min,gamma_min,fmiug0_min,hfidr=False,nofmp2=False,ekt=ekt,mulliken_pop=mulliken_pop,lowdin_pop=lowdin_pop,m_diagnostic=m_diagnostic)
    
    t2 = time()
    
    print("Best Total NOF Energy {}".format(E))
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))
   
    return E,C,gamma,fmiug0
    
