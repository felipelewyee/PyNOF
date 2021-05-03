import numpy as np
from scipy.sparse import csr_matrix
from time import time
import minimization
import integrals
import utils
from numba import prange,njit,jit
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg,minres

def nofmp2(n,C,H,I,b_mnl,E_nuc,p):

    print(" NOF-MP2")
    print("=========")

    occ = n[p.no1:p.nbf5]
    vec = C[:,p.no1:p.nbf]

    D,J,K = integrals.computeJK_HF(C,I,b_mnl,p)
    F = H + 2*J - K

    EHFL = np.trace(np.matmul(D,H)+np.matmul(D,F))
    F_MO = np.matmul(np.matmul(np.transpose(vec),F),vec)

    eig = np.einsum("ii->i",F_MO[:p.nbf-p.no1,:p.nbf-p.no1])

    iajb = integrals.compute_iajb(C,I,b_mnl,p)
    FI1 = np.ones(p.nbf-p.no1)
    FI2 = np.ones(p.nbf-p.no1)

    FI1[:p.nbf5-p.no1] = 1 - (1 - abs(1-2*occ[:p.nbf5-p.no1]))**2

    FI2[p.nbeta-p.no1:p.nbf5-p.no1] = abs(1-2*occ[p.nbeta-p.no1:p.nbf5-p.no1])**2

    Tijab = CalTijab(iajb,F_MO,eig,FI1,FI2,p)
    
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

    print("      Ehfc      = {:f}".format(EHFL+E_nuc))
    print("")
    print("      ECd       = {:f}".format(ECd))
    print("      ECnd      = {:f}".format(ECndl))
    print("      Ecorre    = {:f}".format(ECd+ECndl))
    print("      E(NOFMP2) = {:f}".format(EHFL+ECd+ECndl+E_nuc))
    print("")

def CalTijab(iajb,F_MO,eig,FI1,FI2,p):

    print("")

    A,IROW,ICOL = build_A(F_MO,FI1,FI2,p.no1,p.ndoc,p.nvir,p.ncwo,p.nbf)
    A_CSR = csr_matrix((A, (IROW.astype(int), ICOL.astype(int))))
    print("A matrix has {}/{} elements with Tol = {}".format(len(A),p.nvir**4*p.ndoc**4,1e-10))

    B = build_B(iajb,FI1,FI2,p.ndoc,p.nvir,p.ncwo)
    print("B vector Computed")
    
    Tijab = Tijab_guess(iajb,eig,p.ndoc,p.nvir)
    print("Tijab Guess Computed")
    Tijab = solve_Tijab(A_CSR,B,Tijab,p)
    print("Tijab Computed")

    print("")

    return Tijab

@njit
def build_A(F_MO,FI1,FI2,no1,ndoc,nvir,ncwo,nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = ncwo*(ndoc - i - 1)
        ul = ncwo*(ndoc - i)
        npair[ll:ul] = i + 1

    A = []
    IROW = []
    ICOL = []

    nnz = -1
    for ib in range(nvir):
        for ia in range(nvir):
            for j in range(ndoc):
                for i in range(ndoc):
                    jab =     (j)*ndoc + (ia)*ndoc*ndoc + (ib)*ndoc*ndoc*nvir
                    iab = i              + (ia)*ndoc*ndoc + (ib)*ndoc*ndoc*nvir
                    ijb = i + (j)*ndoc                      + (ib)*ndoc*ndoc*nvir
                    ija = i + (j)*ndoc + (ia)*ndoc*ndoc
                    ijab= i + (j)*ndoc + (ia)*ndoc*ndoc + (ib)*ndoc*ndoc*nvir

                    nnz = nnz + 1
                    A.append(F_MO[ia+ndoc,ia+ndoc] + F_MO[ib+ndoc,ib+ndoc] - F_MO[i,i] - F_MO[j,j])
                    IROW.append(ijab)
                    ICOL.append(i + jab)

                    for k in range(i):
                        if(abs(F_MO[i,k])>1e-10):
                            nnz += 1
                            Cki = FI2[k]*FI2[i]
                            A.append(- Cki*F_MO[i,k])
                            IROW.append(ijab)
                            ICOL.append(k + jab)
                            A.append(- Cki*F_MO[i,k])
                            ICOL.append(ijab)
                            IROW.append(k + jab)

                    for k in range(j):
                        if(abs(F_MO[j,k])>1e-10):
                            nnz += 1
                            Ckj = FI2[k]*FI2[j]
                            A.append(- Ckj*F_MO[j,k])
                            IROW.append(ijab)
                            ICOL.append(k*ndoc + iab)
                            A.append(- Ckj*F_MO[j,k])
                            ICOL.append(ijab)
                            IROW.append(k*ndoc + iab)

                    for k in range(ia):
                        if(abs(F_MO[ia+ndoc,k+ndoc])>1e-10):
                            nnz += 1
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+ndoc]*FI1[ia+ndoc]
                            else:
                                Ckia = FI2[k+ndoc]*FI2[ia+ndoc]
                            A.append(Ckia*F_MO[ia+ndoc,k+ndoc])
                            IROW.append(ijab)
                            ICOL.append(k*ndoc*ndoc + ijb)
                            A.append(Ckia*F_MO[ia+ndoc,k+ndoc])
                            ICOL.append(ijab)
                            IROW.append(k*ndoc*ndoc + ijb)

                    for k in range(ib):
                        if(abs(F_MO[ib+ndoc,k+ndoc])>1e-10):
                            nnz += 1
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+ndoc]*FI1[ib+ndoc]
                            else:
                                Ckib = FI2[k+ndoc]*FI2[ib+ndoc]
                            A.append(Ckib*F_MO[ib+ndoc,k+ndoc])
                            IROW.append(ijab)
                            ICOL.append(k*ndoc*ndoc*nvir + ija)
                            A.append(Ckib*F_MO[ib+ndoc,k+ndoc])
                            ICOL.append(ijab)
                            IROW.append(k*ndoc*ndoc*nvir + ija)
    A = np.array(A)
    IROW = np.array(IROW)
    ICOL = np.array(ICOL)

    return A,IROW,ICOL

@njit
def build_B(iajb,FI1,FI2,ndoc,nvir,ncwo):
    B = np.zeros((ndoc**2*nvir**2))
    for i in range(ndoc):
        lmin_i = ndoc+ncwo*(ndoc-i-1)
        lmax_i = ndoc+ncwo*(ndoc-i-1)+ncwo
        for j in range(ndoc):
            if(i==j):
                for k in range(nvir):
                    ik = i + k*ndoc
                    kn = k + ndoc
                    for l in range(nvir):
                        ln = l + ndoc
                        if(lmin_i <= kn and kn <= lmax_i and lmin_i <= ln and ln <= lmax_i):
                            Ciikl = FI1[kn]*FI1[ln]*FI1[i]*FI1[i]
                        else:
                            Ciikl = FI2[kn]*FI2[ln]*FI2[i]*FI2[i]
                        iikl =  i + i*ndoc + k*ndoc*ndoc + l*ndoc*ndoc*nvir
                        B[iikl] = - Ciikl*iajb[i,k,i,l]
            else:
                for k in range(nvir):
                    ik = i + k*ndoc
                    kn = k + ndoc
                    for l in range(nvir):
                        ln = l + ndoc
                        ijkl =  i + j*ndoc + k*ndoc*ndoc + l*ndoc*ndoc*nvir
                        Cijkl = FI2[kn]*FI2[ln]*FI2[i]*FI2[j]
                        B[ijkl] = - Cijkl*iajb[j,k,i,l]

    return B

@njit
def Tijab_guess(iajb,eig,ndoc,nvir):
    Tijab = np.zeros(nvir**2*ndoc**2)
    for ia in range(nvir):
        for i in range(ndoc):
            for ib in range(nvir):
                for j in range(ndoc):
                    ijab = i + (j)*ndoc + (ia)*ndoc*ndoc + (ib)*ndoc*ndoc*nvir
                    Eijab = eig[ib+ndoc] + eig[ia+ndoc] - eig[j] - eig[i]
                    Tijab[ijab] = - iajb[j,ia,i,ib]/Eijab
    return Tijab 

def solve_Tijab(A_CSR,B,Tijab,p):
    Tijab,info = cg(A_CSR, B,x0=Tijab)
    return Tijab
