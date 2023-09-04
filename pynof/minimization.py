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

def occoptr(gamma,convgdelag,C,H,I,b_mnl,p):

    J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)

    E = 0
    nit = 0
    success = True

    pynof.calcocce(gamma,J_MO,K_MO,H_core,p)

    if (p.ndoc>0):
        res = minimize(pynof.calcocce, gamma, args=(J_MO,K_MO,H_core,p), jac=pynof.calcoccg, method=p.occupation_optimizer)
        gamma = res.x
        E = res.fun
        nit = res.nit
        success = res.success

    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    cj12,ck12 = pynof.PNOFi_selector(n,p)

    return E,nit,success,gamma,n,cj12,ck12

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
        if (p.nzeros>p.nzerosm):
            p.nzeros = p.nzerosr
        #if (p.nzeros>abs(int(np.log10(maxdiff)))+1):
        #    p.nzeros = p.nzerosr
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
            if(printmode):
                print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext+1,i_int,E,E+E_nuc,E_diff,maxdiff),p.nzeros)
            break

    return convgdelag,E,E_diff,sumdiff_old,itlim,fmiug0,C,elag

def orbopt_rotations(gamma,C,H,I,b_mnl,p):

    y = np.zeros((p.nvar))

    n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    cj12,ck12 = pynof.PNOFi_selector(n,p)

    if("trust" in p.orbital_optimizer or "Newton-CG" in p.orbital_optimizer):
        res = minimize(pynof.calcorbeg, y, args=(n,cj12,ck12,C,H,I,b_mnl,p),jac=True,hess="2-point",method=p.orbital_optimizer,options={"maxiter":p.maxloop})
    else:
        res = minimize(pynof.calcorbeg, y, args=(n,cj12,ck12,C,H,I,b_mnl,p),jac=True,method=p.orbital_optimizer,options={"maxiter":p.maxloop})

    E = res.fun
    y = res.x
    grad = res.jac
    nit = res.nit
    success = res.success

    C = pynof.rotate_orbital(y,C,p)

    return E,C,nit,success

def comb(gamma,C,H,I,b_mnl,p):

    x = np.zeros((p.nvar+p.nv))
    x[p.nvar:] = gamma

    if("trust" in p.orbital_optimizer or "Newton-CG" in p.orbital_optimizer):
        res = minimize(pynof.calccombeg, x, args=(C,H,I,b_mnl,p),jac=True,hess="2-point",method=p.combined_optimizer,options={"maxiter":p.maxloop})
    else:
        res = minimize(pynof.calccombeg, x, args=(C,H,I,b_mnl,p),jac=True,method=p.combined_optimizer,options={"maxiter":p.maxloop})

    E = res.fun
    x = res.x
    grad = res.jac
    y = x[:p.nvar]
    gamma = x[p.nvar:]
    C = pynof.rotate_orbital(y,C,p)

    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)

    return E,C,gamma,n,grad,res.nit,res.success
