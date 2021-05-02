import numpy as np
from scipy.sparse import csr_matrix
from time import time
import minimization
import integrals
import utils
from numba import prange,njit,jit
from scipy.sparse.linalg import spsolve

def nofmp2(n,C,H,I,b_mnl,E_nuc,p):

    ti = time()
    occ = n[p.no1:p.nbf5]
    vec = C[:,p.no1:p.nbf]
    t1 = time()

    I_cpu = I.get()
    t2 = time()
    DM = np.einsum("mj,nj->mn",C[:,:p.nbeta],C[:,:p.nbeta])
    J = np.einsum("ls,mnsl->mn",DM,I_cpu)
    K = np.einsum("ls,mlsn->mn",DM,I_cpu)
    t3 = time()
    F = H + 2*J - K
    t4 = time()

    t5 = time()
    EHFL = np.trace(np.matmul(DM,H)+np.matmul(DM,F))
    F_MO = np.matmul(np.matmul(np.transpose(vec),F),vec)
    t6 = time()

    t7 = time()
    eig = np.einsum("ii->i",F_MO[:p.nbf-p.no1,:p.nbf-p.no1])
    t8 = time()

    t9 = time()
    iajb = np.einsum('mi,na,mnsl,sj,lb->iajb',C[:,p.no1:p.nbeta],C[:,p.nbeta:p.nbf],I_cpu,C[:,p.no1:p.nbeta],C[:,p.nbeta:p.nbf])

    t10 = time()
    FI1 = np.ones(p.nbf-p.no1)
    FI2 = np.ones(p.nbf-p.no1)
    t11 = time()

    t12 = time()
    for i in range(p.nbf5-p.no1):   
       Ci = 1 - abs(1-2*occ[i])
       FI1[i] = 1 - Ci*Ci
    t13 = time()
    
    for i in range(p.nbeta-p.no1,p.nbf5-p.no1):
        Ci = abs(1-2*occ[i])
        FI2[i] = Ci*Ci
    t14 = time()

    t15 = time()
    Tijab = CalTijab(iajb,F_MO,eig,FI1,FI2,p)
    t16 = time()
    
    t17 = time()
    ECd = 0
    for k in range(p.nvir):
        for l in range(p.nvir):
            for i in range(p.ndoc):
                for j in range(p.ndoc):
                    Xijkl = iajb[j,k,i,l]
                    ijkl = i+j*p.ndoc+k*p.ndoc*p.ndoc+l*p.ndoc*p.ndoc*p.nvir
                    ijlk = i+j*p.ndoc+l*p.ndoc*p.ndoc+k*p.ndoc*p.ndoc*p.nvir
                    ECd = ECd + Xijkl*(2*Tijab[ijkl]-Tijab[ijlk])
    t18 = time()

    fi = 2*n*(1-n)

    CK12nd = np.outer(fi,fi)

    beta = np.sqrt((1-abs(1-2*n))*n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - l)
        CK12nd[p.no1+l,ll:ul] = beta[p.no1+l]*beta[ll:ul]
        CK12nd[ll:ul,p.no1+l] = beta[ll:ul]*beta[p.no1+l]
        CK12nd[ll:ul,ll:ul] = -np.outer(beta[ll:ul],beta[ll:ul])
    t19 = time()

    #C^K KMO
    J_MO,K_MO,H_core = integrals.computeJKH_MO(C,H,I,b_mnl,p)
    ECndl = - np.einsum('ij,ji',CK12nd,K_MO) # sum_ij
    ECndl += np.einsum('ii,ii',CK12nd,K_MO) # Quita i=j
    t20 = time()

    print("Ehfc",EHFL+E_nuc)
    print("ECd",ECd)
    print("ECnd",ECndl)
    print("Ecorre",ECd+ECndl)
    print("E(NOFMP2)",EHFL+ECd+ECndl+E_nuc)

    tf = time()
    print(tf-ti)
    print(t1-ti)
    print(t2-t1)
    print(t3-t2)
    print(t4-t3)
    print(t5-t4)
    print(t6-t5)
    print(t7-t6)
    print(t8-t7)
    print(t9-t8)
    print(t10-t9)
    print(t11-t10)
    print(t12-t11)
    print(t13-t12)
    print(t14-t13)
    print(t15-t14)
    print(t16-t15)
    print(t17-t16)
    print(t18-t17)
    print(t19-t18)
    print(t20-t19)

def CalTijab(iajb,F_MO,eig,FI1,FI2,p):
    ti = time()
    npair = np.zeros((p.nvir))
    for i in range(p.ndoc):
        ll = p.ncwo*(p.ndoc - i - 1)
        ul = p.ncwo*(p.ndoc - i)
        npair[ll:ul] = i + 1
    t1 = time()

    A = np.zeros((2*p.ndoc**2*p.nvir**2*(p.nbf-p.no1)))
    IROW = np.zeros((2*p.ndoc**2*p.nvir**2*(p.nbf-p.no1)))
    ICOL = np.zeros((2*p.ndoc**2*p.nvir**2*(p.nbf-p.no1)))
    t2 = time()

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
          
                    for k in range(i):
                        if(abs(F_MO[i,k])>1e-10):
                            nnz += 1
                            Cki = FI2[k]*FI2[i]
                            A[nnz] = - Cki*F_MO[i,k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k + jab
          
                    for k in range(j):
                        if(abs(F_MO[j,k])>1e-10):
                            nnz += 1
                            Ckj = FI2[k]*FI2[j]
                            A[nnz] = - Ckj*F_MO[j,k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k*p.ndoc + iab
 
                    for k in range(ia):
                        if(abs(F_MO[ia+p.ndoc,k+p.ndoc])>1e-10):
                            nnz += 1
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+p.ndoc]*FI1[ia+p.ndoc]
                            else:
                                Ckia = FI2[k+p.ndoc]*FI2[ia+p.ndoc]
                            A[nnz] = Ckia*F_MO[ia+p.ndoc,k+p.ndoc]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k*p.ndoc*p.ndoc + ijb

                    for k in range(ib):
                        if(abs(F_MO[ib+p.ndoc,k+p.ndoc])>1e-10):
                            nnz += 1
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+p.ndoc]*FI1[ib+p.ndoc]
                            else:
                                Ckib = FI2[k+p.ndoc]*FI2[ib+p.ndoc]
                            A[nnz] = Ckib*F_MO[ib+p.ndoc,k+p.ndoc]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k*p.ndoc*p.ndoc*p.nvir + ija

    t3 = time()
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


    t4 = time()
    AA = np.zeros((2*(nnz+1)))
    IIROW = np.zeros((2*(nnz+1)))
    IICOL = np.zeros((2*(nnz+1)))
    IIROW[:nnz+1] = IROW[:nnz+1]
    IICOL[:nnz+1] = ICOL[:nnz+1]
    AA[:nnz+1] = A[:nnz+1]
    NZ = nnz+1
    for i in range(nnz+1):
        if(IIROW[i]>IICOL[i]):   
            IIROW[NZ] = IICOL[i]
            IICOL[NZ] = IIROW[i]
            AA[NZ]    = AA[i]
            NZ = NZ + 1
    t5 = time()

    print("Entrada")
    AA = AA[:NZ]
    IIROW = IIROW[:NZ]
    IICOL = IICOL[:NZ]
    A_CSR = csr_matrix((AA, (IIROW.astype(int), IICOL.astype(int))))
    print("Entrada")
    from scipy.sparse.linalg import cg,minres
    
    Tijab = np.zeros(p.nvir**2*p.ndoc**2)
    for ia in range(p.nvir):
        for i in range(p.ndoc):
            for ib in range(p.nvir):
                for j in range(p.ndoc):
                    ijab = i + (j)*p.ndoc + (ia)*p.ndoc*p.ndoc + (ib)*p.ndoc*p.ndoc*p.nvir
                    Eijab = eig[ib+p.ndoc] + eig[ia+p.ndoc] - eig[j] - eig[i]
                    Tijab[ijab] = - iajb[j,ia,i,ib]/Eijab

    t6 = time()
    Tijab,info = cg(A_CSR, B,x0=Tijab)
    t7 = time()
    print("cg",t7-t6)

    #t8 = time()
    #Tijab = spsolve(A_CSR,B)
    #t9 = time()
    #print("spsolve",t9-t8)
    #print(info,max(Tijab1-Tijab))
    tf = time()

    print("================entrada======================")
    print(tf-ti)
    print(t1-ti)
    print(t2-t1)
    print(t3-t2)
    print(t4-t3)
    print(t5-t4)
    print(tf-t5)
    print("================salida======================")

    return Tijab
