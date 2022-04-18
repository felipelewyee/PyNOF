import numpy as np
from scipy.optimize import minimize
from time import time
import pynof 

def hfidr(C,H,I,b_mnl,E_nuc,p,printmode):

    no1_ori = p.no1
    p.no1 = p.nbeta

    n = np.zeros((p.nbf5))
    n[0:p.nbeta] = 1.0
    n[p.nbeta:p.nalpha] = 0.5

    cj12 = 2*np.einsum('i,j->ij',n,n)
    ck12 = np.einsum('i,j->ij',n,n)
    if(p.MSpin==0 and p.nsoc>1):
        ck12[p.nbeta:p.nalpha,p.nbeta:p.nalpha] = 2*np.einsum('i,j->ij',n[p.nbeta:p.nalpha],n[p.nbeta:p.nalpha])

    if(printmode):
        print("Hartree-Fock")
        print("============")
        print("")

        print('{:^7} {:^7} {:^14} {:^14} {:^15} {:^14}'.format("Nitext","Nitint","Eelec","Etot","Ediff","maxdiff"))

    E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

    fmiug0 = None

    ext = True
    # iteraciones externas
    for i_ext in range(p.maxitid):
        if i_ext==0:
            maxlp = 1
        else:
            maxlp = p.maxloop

        # iteraciones internas
        for i_int in range(maxlp):
            E_old = E

            if(p.scaling):
                fmiug = pynof.fmiug_scaling(fmiug0,elag,i_ext,p.nzeros,p.nbf,p.noptorb)

            fmiug0, W = np.linalg.eigh(fmiug)
            C = np.matmul(C,W)
            E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

            E_diff = E-E_old
            if(abs(E_diff)<p.thresheid):
                if(printmode):
                    print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))
                for i in range(p.nbf):
                    fmiug0[i] = elag[i][i]
                ext = False
                break

        if(not ext):
            break
        if(printmode):
            print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))


    # Regresamos no1 a su estado original
    p.no1 = no1_ori

    return E,C,fmiug0

def occoptr(gamma,firstcall,convgdelag,C,H,I,b_mnl,p):

    J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)

    if (not convgdelag and p.ndoc>0):
        if(p.gradient=="analytical"):
            res = minimize(pynof.calce, gamma[:p.nv], args=(J_MO,K_MO,H_core,p), jac=pynof.calcg, method=p.optimizer)
        elif(p.gradient=="numerical"):
            res = minimize(pynof.calce, gamma[:p.nv], args=(J_MO,K_MO,H_core,p),  method=p.optimizer)
        gamma = res.x
    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    cj12,ck12 = pynof.PNOFi_selector(n,p)

    return gamma,n,cj12,ck12

def orboptr(C,n,H,I,b_mnl,cj12,ck12,E_old,E_diff,sumdiff_old,i_ext,itlim,fmiug0,E_nuc,p,printmode):

    convgdelag = False

    E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

    #E_diff = E-E_old
    #P_CONV = abs(E_diff)
    #E_old = E

    if(maxdiff<p.threshl and abs(E_diff)<p.threshe):
        convgdelag = True
        if(printmode):
            print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,0,E,E+E_nuc,E_diff,maxdiff),p.nzeros)
        return convgdelag,E_old,E_diff,sumdiff_old,itlim,fmiug0,C,elag

    if (p.scaling and i_ext>1 and i_ext >= itlim and sumdiff > sumdiff_old):
        p.nzeros = p.nzeros + 1
        itlim = i_ext + p.itziter
        #if (p.nzeros>p.nzerosm):
        #    p.nzeros = p.nzerosr
        if (p.nzeros>abs(int(np.log10(maxdiff)))+1):
            p.nzeros = p.nzerosr
            #p.nzeros = abs(int(np.log10(maxdiff)))
    sumdiff_old = sumdiff

    if i_ext==0:
        maxlp = 1
    else:
        maxlp = p.maxloop

    fmiug = np.zeros((p.noptorb,p.noptorb))
    fk = np.zeros((30,p.noptorb,p.noptorb))
    bdiis = np.zeros((31,31))
    cdiis = np.zeros((31))
    iloop = 0
    idiis = 0

    for i_int in range(maxlp):
        iloop = iloop + 1
        E_old2 = E
        
        #scaling
        if(p.scaling):
            fmiug = pynof.fmiug_scaling(fmiug0,elag,i_ext,p.nzeros,p.nbf,p.noptorb)
        if(p.diis and maxdiff < p.thdiis):
            fk,fmiug,idiis,bdiis = pynof.fmiug_diis(fk,fmiug,idiis,bdiis,cdiis,maxdiff,p.noptorb,p.ndiis,p.perdiis)

        eigval, eigvec = np.linalg.eigh(fmiug)
        fmiug0 = eigval

        C = np.matmul(C,eigvec)

        E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

        E_diff2 = E-E_old2

        if(abs(E_diff2)<p.threshec or i_int==maxlp-1):
            E_diff = E-E_old
            E_old = E
            if(printmode):
                print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext+1,i_int,E,E+E_nuc,E_diff,maxdiff),p.nzeros)
            break

    return convgdelag,E_old,E_diff,sumdiff_old,itlim,fmiug0,C,elag


def orbopt_rotations(gamma,C,H,I,b_mnl,p):

    y = np.zeros((int(p.nbf*(p.nbf-1)/2)))

#    res = minimize(pynof.calcorbe, y, args=(gamma,C,H,I,b_mnl,p))
#    res = minimize(pynof.calcorbe, y, args=(gamma,C,H,I,b_mnl,p),jac=pynof.calcorbg)
#    res = minimize(pynof.calcorbe, y, args=(gamma,C,H,I,b_mnl,p),jac=pynof.calcorbg,hess=pynof.calcorbh2,method="trust-exact")
    res = minimize(pynof.calcorbe, y, args=(gamma,C,H,I,b_mnl,p),jac=pynof.calcorbg_num,hess=pynof.calcorbh_num,method="trust-exact")
#    res = minimize(pynof.calcorbe, y, args=(gamma,C,H,I,b_mnl,p),jac=pynof.calcorbg,hess="2-points",method="trust-ncg")

#    y = np.zeros((int(p.nbf*(p.nbf-1)/2)))
#    r = 0.1
#    maxr = 2.0
#    y,r = pynof.optimize_trust(y,r,maxr,pynof.calcorbe,pynof.calcorbg_num,pynof.calcorbh_num,gamma,C,H,I,b_mnl,p)  
#    stop

    E = res.fun
    y = res.x
    C = pynof.rotate_orbital(y,C,p)

    return E,C,res.nit,res.success

def comb(gamma,C,H,I,b_mnl,p):

    nvar = int(p.nbf*(p.nbf-1)/2)
    x = np.zeros((nvar+p.nv))
    x[nvar:] = gamma
    E = pynof.calccombe(x,C,H,I,b_mnl,p)
    print("{:3d} {:14.8f}".format(0,E))

    #res = minimize(pynof.calccombe, x, args=(C,H,I,b_mnl,p))
    res = minimize(pynof.calccombe, x, args=(C,H,I,b_mnl,p),jac=pynof.calccombg_num,hess=pynof.calccombh_num,method="trust-exact")

    #r = 0.04
    #maxr = 2.0
    #y,r = optimize_trust(x,r,maxr,pynof.calccombe,pynof.calccombg,pynof.calccombh,C,H,I,b_mnl,p)

    E = res.fun
    x = res.x
    y = x[:int(p.nbf*(p.nbf-1)/2)]
    gamma = x[int(p.nbf*(p.nbf-1)/2):]
    C = pynof.rotate_orbital(y,C,p)

    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)

    return E,C,gamma,n,res.nit,res.success

def comb2(gamma,C,H,I,b_mnl,p):

    nvar = int(p.nbf5*(p.nbf5-1)/2)
    x = np.zeros((nvar+p.nv))
    x[nvar:] = gamma
    E = pynof.calccombe(x,C,H,I,b_mnl,p)
    print("E inicial:",E)

    for i in range(3):

        r_phi = 0.3
        r_gamma = 0.10
        maxr = 2.0

        p_phi = np.zeros((nvar))
        p_gamma = np.zeros((p.nv))
        for i in range(100):

            # ========================== Compute Hess and grad ==========================

            x = np.zeros((nvar+p.nv))
            x[nvar:] = gamma
            J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)

            grad = pynof.calccombg_num(x,C,H,I,b_mnl,p)
            Hess = pynof.calccombh_num(x,C,H,I,b_mnl,p)

            Hess_phiphi = Hess[:nvar,:nvar]
            Hess_gammagamma = Hess[nvar:,nvar:]
            Hess_phigamma = Hess[:nvar,nvar:]
            Hess_gammaphi = Hess[nvar:,:nvar]
            grad_phi = grad[:nvar]
            grad_gamma = grad[nvar:]
            Hess_gammagamma = Hess[nvar:,nvar:]
            grad_gamma = grad[nvar:]

            # ========================== Compute Hess and grad ==========================

            # ========================== Gamma Step ==========================
            print("  == Gamma Step ==")
            grad_modif = grad_gamma+np.einsum("ij,j->i",Hess_gammaphi,p_phi)
            J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)
            p_gamma,r_gamma = pynof.optimize_trust2(gamma,r_gamma,maxr,pynof.calce,grad_modif,Hess_gammagamma,J_MO,K_MO,H_core,p)
    
            gamma = gamma + p_gamma
            x = np.zeros((nvar+p.nv))
            x[nvar:] = gamma
            # ========================== Phi Step ==========================
            print("  == Phi Step ==")

            y = np.zeros((nvar))
            grad_modif = grad_phi + np.einsum("ij,j->i",Hess_phigamma,p_gamma)
            p_phi,r_phi = pynof.optimize_trust2(y,r_phi,maxr,pynof.calcorbe,grad_modif,Hess_phiphi,gamma,C,H,I,b_mnl,p)

            y = y + p_phi
            C = pynof.rotate_orbital(y,C,p)
            x = np.zeros((nvar+p.nv))
            x[nvar:] = gamma

            p_gamma = np.zeros((p.nv))

    return E,C

