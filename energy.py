import numpy as np
from scipy.linalg import eigh
from time import time
import minimization
import integrals
import utils

def compute_energy(mol,wfn,p=None,gradient="analytical",C=None,gamma=None,fmiug0=None,printmode=True):

    S,T,V,H,I,b_mnl = integrals.compute_integrals(wfn,mol,p)

    # Energía Nuclear
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C)
    Cguess = C
    if(C is None):
        E_i,Cguess = eigh(H, S)  # (HC = SCe)
    Cguess = utils.check_ortho(Cguess,S,p)

    if (p.hfidr):
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

    np.save("C.npy",C)
    np.save("gamma.npy",gamma)
    np.save("fmiug0.npy",fmiug0)

    if(printmode):
        print("")
        print("RESULTS OF THE OCCUPATION OPTIMIZATION")
        print("========================================")
        for i,ni in enumerate(n):
            print(" {:d}    {:9.7f}  {:10.8f}".format(i,2*ni,elag[i][i]))
        print("")

        print("----------------")
        print(" Final Energies ")
        print("----------------")
        
        if(p.hfidr):
            print("       HF Total Energy = {:15.7f}".format(E_nuc + EHF))
        print("Final NOF Total Energy = {:15.7f}".format(E_nuc + E_old))
        if(p.hfidr):
            print("    Correlation Energy = {:15.7f}".format(E_old-EHF))
        print("")
        print("")

    E_t = E_nuc + E_old

    ################

    occ = n[p.no1:p.nbf5]
    vec = C[:,p.no1:p.nbf]

    Icpu = I.get()
    DM = np.einsum("mj,nj->mn",C[:,:p.nbeta],C[:,:p.nbeta])
    J = np.einsum("ls,mnsl->mn",DM,Icpu)
    K = np.einsum("ls,mlsn->mn",DM,Icpu)

    F = H + 2*J - K

    EHFL = np.trace(np.matmul(DM,H)+np.matmul(DM,F))
    F_MO = np.matmul(np.matmul(np.transpose(vec),F),vec)

    eig = np.einsum("ii->i",F_MO[:p.nbf-p.no1,:p.nbf-p.no1])

    iajb = np.einsum('mi,na,mnsl,sj,lb->iajb',C[:,p.no1:p.nbeta],C[:,p.nbeta:p.nbf],I,C[:,p.no1:p.nbeta],C[:,p.nbeta:p.nbf])

    FI1 = np.ones(p.nbf-p.no1)
    FI2 = np.ones(p.nbf-p.no1)

    for i in range(p.nbf5-p.no1):   
       Ci = 1 - abs(1-2*occ[i])
       FI1[i] = 1 - Ci*Ci
    
    for i in range(p.nbeta-p.no1,p.nbf5-p.no1):
        Ci = abs(1-2*occ[i])
        FI2[i] = Ci*Ci

    npair = np.zeros((p.nvir))
    for i in range(p.ndoc):
        ll = p.ncwo*(p.ndoc - i - 1)
        ul = p.ncwo*(p.ndoc - i)
        npair[ll:ul] = i + 1

    A = np.zeros((2*p.ndoc**2*p.nvir**2*(p.nbf-p.no1)))
    IROW = np.zeros((2*p.ndoc**2*p.nvir**2*(p.nbf-p.no1)))
    ICOL = np.zeros((2*p.ndoc**2*p.nvir**2*(p.nbf-p.no1)))

    nnz = -1
    for ib in range(p.nvir):
        for ia in range(p.nvir):
            for j in range(p.ndoc):
                for i in range(p.ndoc):
                    jab =     (j)*p.ndoc + (ia)*p.ndoc*p.ndoc + (ib)*p.ndoc*p.ndoc*p.nvir
                    iab = i              + (ia)*p.ndoc*p.ndoc + (ib)*p.ndoc*p.ndoc*p.nvir
                    ijb = i + (j)*p.ndoc                      + (ib)*p.ndoc*p.ndoc*p.nvir
                    ija = i + (j)*p.ndoc + (ia)*p.ndoc*p.ndoc
                    ijab= i + (j)*p.ndoc + (ia)*p.ndoc*p.ndoc + (ib)*p.ndoc*p.ndoc*p.nvir

                    nnz = nnz + 1
                    A[nnz] = F_MO[ia+p.ndoc,ia+p.ndoc] + F_MO[ib+p.ndoc,ib+p.ndoc] - F_MO[i,i] - F_MO[j,j]
                    IROW[nnz] = ijab
                    ICOL[nnz] = i + jab
          
                    for k in range(i-1):
                        if(abs(F_MO[i,k])>1e-10):
                            nnz += 1
                            Cki = FI2[k]*FI2[i]
                            A[nnz] = - Cki*F_MO[i,k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k + jab
          
                    for k in range(j-1):
                        if(abs(F_MO[j,k])>1e-10):
                            nnz += 1
                            Ckj = FI2[k]*FI2[j]
                            A[nnz] = - Ckj*F_MO[j,k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k*p.ndoc + iab
 
                    for k in range(ia-1):
                        if(abs(F_MO[ia+p.ndoc,k+p.ndoc])>1e-10):
                            nnz += 1
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+p.ndoc]*FI1[ia+p.ndoc]
                            else:
                                Ckia = FI2[k+p.ndoc]*FI2[ia+p.ndoc]
                            A[nnz] = Ckia*F_MO[ia+p.ndoc,k+p.ndoc]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k*p.ndoc*p.ndoc + ijb

                    for k in range(ib-1):
                        if(abs(F_MO[ib+p.ndoc,k+p.ndoc])>1e-10):
                            nnz += 1
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+p.ndoc]*FI1[ib+p.ndoc]
                            else:
                                Ckib = FI2[k+p.ndoc]*FI2[ib+p.ndoc]
                            A[nnz] = Ckib*F_MO[ib+p.ndoc,k+p.ndoc]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k*p.ndoc*p.ndoc*p.nvir + ija

    from scipy.sparse import csr_matrix
    A_CSR = csr_matrix((A, (IROW.astype(int), ICOL.astype(int))))

    B = np.zeros((p.ndoc**2*p.nvir**2))
    for i in range(p.ndoc):
        lmin_i = p.ndoc+p.ncwo*(p.ndoc-i-1)
        lmax_i = p.ndoc+p.ncwo*(p.ndoc-i-1)+p.ncwo
        for j in range(p.ndoc):
            if(i==j):
                for k in range(p.nvir):
                    ik = i + k*p.ndoc
                    kn = k + p.ndoc
                    for l in range(p.nvir):
                        ln = l + p.ndoc
                        if(lmin_i <= kn and kn <= lmax_i and lmin_i <= ln and ln <= lmax_i):
                            Ciikl = FI1[kn]*FI1[ln]*FI1[i]*FI1[i]
                        else:
                            Ciikl = FI2[kn]*FI2[ln]*FI2[i]*FI2[i]
                        iikl =  i + i*p.ndoc + k*p.ndoc*p.ndoc + l*p.ndoc*p.ndoc*p.nvir
                        B[iikl] = - Ciikl*iajb[i,k,i,l]
            else:
                for k in range(p.nvir):
                    ik = i + k*p.ndoc
                    kn = k + p.ndoc
                    for l in range(p.nvir):
                        ln = l + p.ndoc
                        ijkl =  i + j*p.ndoc + k*p.ndoc*p.ndoc + l*p.ndoc*p.ndoc*p.nvir
                        Cijkl = FI2[kn]*FI2[ln]*FI2[i]*FI2[j]
                        B[ijkl] = - Cijkl*iajb[j,k,i,l]

    from scipy.sparse.linalg import spsolve_triangular

    Tijab = spsolve_triangular(A_CSR,B,lower=True)

    ECd = 0
    for k in range(p.nvir):
        for l in range(p.nvir):
            for i in range(p.ndoc):
                for j in range(p.ndoc):
                    Xijkl = iajb[j,k,i,l]
                    ijkl = i+j*p.ndoc+k*p.ndoc*p.ndoc+l*p.ndoc*p.ndoc*p.nvir
                    ijlk = i+j*p.ndoc+l*p.ndoc*p.ndoc+k*p.ndoc*p.ndoc*p.nvir
                    ECd = ECd + Xijkl*(2*Tijab[ijkl]-Tijab[ijlk])

    fi = 2*n*(1-n)

    CK12nd = np.outer(fi,fi)

    beta = np.sqrt((1-abs(1-2*n))*n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - l)
        CK12nd[p.no1+l,ll:ul] = beta[p.no1+l]*beta[ll:ul]
        CK12nd[ll:ul,p.no1+l] = beta[ll:ul]*beta[p.no1+l]
        CK12nd[ll:ul,ll:ul] = -np.outer(beta[ll:ul],beta[ll:ul])

    #C^K KMO
    J_MO,K_MO,H_core = integrals.computeJKH_MO(C,H,I,b_mnl,p)
    ECndl = - np.einsum('ij,ji',CK12nd,K_MO) # sum_ij
    ECndl += np.einsum('ii,ii',CK12nd,K_MO) # Quita i=j

    print("Ehfc",EHFL+E_nuc)
    print("ECd",ECd)
    print("ECnd",ECndl)
    print("Ecorre",ECd+ECndl)
    print("E(NOFMP2)",EHFL+ECd+ECndl+E_nuc)



    return E_t,C,gamma,fmiug0


