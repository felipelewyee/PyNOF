import numpy as np
import utils 
import integrals
from scipy.optimize import minimize
import pnof
from time import time

def hfidr(C,H,I,b_mnl,E_nuc,p):

    no1_ori = p.no1
    p.no1 = p.nbeta

    n = np.zeros((p.nbf5))
    n[0:p.nbeta] = 1.0
    n[p.nbeta:p.nalpha] = 0.5

    cj12 = 2*np.einsum('i,j->ij',n,n)
    ck12 = np.einsum('i,j->ij',n,n)

    print("Hartree-Fock")
    print("============")
    print("")

    print('{:^7} {:^7} {:^14} {:^14} {:^15} {:^14}'.format("Nitext","Nitint","Eelec","Etot","Ediff","maxdiff"))

    E,elag,sumdiff,maxdiff = utils.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

    fmiug0 = np.zeros((p.nbf))

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
                fmiug = utils.fmiug_scaling(fmiug0,elag,i_ext,p.nzeros,p)

            fmiug0, W = np.linalg.eigh(fmiug)
            C = np.matmul(C,W)
            E,elag,sumdiff,maxdiff = utils.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

            E_diff = E-E_old
            if(abs(E_diff)<p.thresheid):
                print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))
                for i in range(p.nbf):
                    fmiug0[i] = elag[i][i]
                ext = False
                break

        if(not ext):
            break
        print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))


    # Regresamos no1 a su estado original
    p.no1 = no1_ori

    return E,C,fmiug0

def occoptr(gamma,firstcall,convgdelag,elag,C,H,I,b_mnl,p):

    J_MO,K_MO,H_core = integrals.computeJKH_core_MO(C,H,I,b_mnl,p)

    if (not convgdelag):
        if(p.gradient=="analytical"):
            res = minimize(pnof.calce, gamma[:p.nv], args=(J_MO,K_MO,H_core,p), jac=pnof.calcg, method='CG')
        elif(p.gradient=="numerical"):
            res = minimize(pnof.calce, gamma[:p.nv], args=(J_MO,K_MO,H_core,p),  method='CG')
        gamma = res.x
    n,DR = pnof.ocupacion(gamma,p)
    cj12,ck12 = pnof.PNOFi_selector(n,p)

    if (firstcall):
        elag_diag = np.zeros((p.nbf))

        # RO (H_core + J)
        elag_diag[:p.nbeta] = np.einsum('i,i->i',n[:p.nbeta],H_core[:p.nbeta]+np.diagonal(J_MO)[:p.nbeta])
        elag_diag[p.nbeta:p.nalpha] = np.einsum('i,i->i',n[p.nbeta:p.nalpha],H_core[p.nbeta:p.nalpha])
        elag_diag[p.nalpha:p.nbf5] = np.einsum('i,i->i',n[p.nalpha:p.nbf5],H_core[p.nalpha:p.nbf5]+np.diagonal(J_MO)[p.nalpha:p.nbf5])

        # CJ12 J_MO
        elag_diag[:p.nbf5] += np.einsum('ij,ji->i',cj12,J_MO)
        elag_diag[:p.nbf5] -= np.einsum('ii,ii->i',cj12,J_MO)

        # CK12 K_MO
        elag_diag[:p.nbf5] -= np.einsum('ij,ji->i',ck12,K_MO)
        elag_diag[:p.nbf5] += np.einsum('ii,ii->i',ck12,K_MO)

        for i in range(p.nbf):
            elag[i][i] = elag_diag[i]

    return gamma,elag,n,cj12,ck12

def orboptr(C,n,H,I,b_mnl,cj12,ck12,E_old,E_diff,sumdiff_old,i_ext,itlim,fmiug0,E_nuc,p):

    convgdelag = False

    E,elag,sumdiff,maxdiff = utils.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

    #E_diff = E-E_old
    #P_CONV = abs(E_diff)
    #E_old = E

    if(maxdiff<p.threshl and abs(E_diff)<p.threshe):
        convgdelag = True
        print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,0,E,E+E_nuc,E_diff,maxdiff))
        return convgdelag,E_old,E_diff,sumdiff_old,itlim,fmiug0,C

    if (p.scaling and i_ext>1 and i_ext >= itlim and sumdiff > sumdiff_old):
        p.nzeros = p.nzeros + 1
        itlim = i_ext + p.itziter
        if (p.nzeros>p.nzerosm):
            p.nzeros = p.nzerosr
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
            fmiug = utils.fmiug_scaling(fmiug0,elag,i_ext,p.nzeros,p)
        if(p.diis and maxdiff < p.thdiis):
            fk,fmiug,idiis,bdiis = utils.fmiug_diis(fk,fmiug,idiis,bdiis,cdiis,maxdiff,p)

        eigval, eigvec = np.linalg.eigh(fmiug)
        fmiug0 = eigval

        C = np.matmul(C,eigvec)

        E,elag,sumdiff,maxdiff = utils.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

        E_diff2 = E-E_old2

        if(abs(E_diff2)<p.threshec or i_int==maxlp-1):
            E_diff = E-E_old
            E_old = E
            print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext+1,i_int,E,E+E_nuc,E_diff,maxdiff))
            break
    return convgdelag,E_old,E_diff,sumdiff_old,itlim,fmiug0,C

