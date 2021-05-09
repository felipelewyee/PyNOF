import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.optimize import root
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
    if(p.MSpin==0):
        if(p.nsoc>0):
            Dalpha,Jalpha,Kalpha = integrals.computeJKalpha_HF(C,I,b_mnl,p)
            D = D + Dalpha
            J = J + Jalpha
            K = K + Kalpha
        F = H + 2*J - K
        EHFL = np.trace(np.matmul(D,H)+np.matmul(D,F))
    elif(not p.MSpin==0):
        Dalpha,Jalpha,Kalpha = integrals.computeJKalpha_HF(C,I,b_mnl,p)
        F = 2*J - K
        EHFL = 2*np.trace(np.matmul(D+Dalpha,H)+np.matmul(D+2*Dalpha,F))
        F = H + F
        if(p.nsoc>1):
            Falpha = J - K
            EHFL = EHFL + 2*np.trace(np.matmul(Dalpha,Falpha))
            F = F + Falpha

    F_MO = np.matmul(np.matmul(np.transpose(vec),F),vec)

    eig = np.einsum("ii->i",F_MO[:p.nbf-p.no1,:p.nbf-p.no1])

    iajb = integrals.compute_iajb(C,I,b_mnl,p)
    FI1 = np.ones(p.nbf-p.no1)
    FI2 = np.ones(p.nbf-p.no1)

    FI1[:p.nbf5-p.no1] = 1 - (1 - abs(1-2*occ[:p.nbf5-p.no1]))**2

    FI2[p.nalpha-p.no1:p.nbf5-p.no1] = abs(1-2*occ[p.nalpha-p.no1:p.nbf5-p.no1])**2

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

    print("Starting CalTijab")

    B = build_B(iajb,FI1,FI2,p.ndoc,p.ndns,p.nvir,p.ncwo)
    print("....B vector Computed")
    
    Tijab = Tijab_guess(iajb,eig,p.ndoc,p.ndns,p.nvir)
    print("....Tijab Guess Computed")

    #A_CSR = csr_matrix(build_A(F_MO,FI1,FI2,p.no1,p.ndoc,p.ndns,p.nvir,p.ncwo,p.nbf))
    #print("A matrix has {}/{} elements with Tol = {}".format(len(A),p.nvir**4*p.ndoc**4,1e-10))
    #Tijab = solve_Tijab(A_CSR,B,Tijab,p)

    res = root(build_R, Tijab, args=(B,F_MO,FI1,FI2,p.no1,p.ndoc,p.ndns,p.nvir,p.ncwo,p.nbf),method="krylov")
    if(res.success):
        print("....Tijab found as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
    else:
        print("....WARNING! Tijab NOT FOUND as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
        print(res)
    Tijab = res.x
    print("")

    return Tijab

@njit
def build_A(F_MO,FI1,FI2,no1,ndoc,ndns,nvir,ncwo,nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = ncwo*(ndoc - i - 1)
        ul = ncwo*(ndoc - i)
        npair[ll:ul] = i + 1

    A = np.empty((2*ndns**2*nvir**2*(nbf-no1)))
    IROW = np.empty((2*ndns**2*nvir**2*(nbf-no1)),dtype=np.int32)
    ICOL = np.empty((2*ndns**2*nvir**2*(nbf-no1)),dtype=np.int32)

    nnz = -1
    for ib in range(nvir):
        for ia in range(nvir):
            for j in range(ndns):
                for i in range(ndns):
                    #print(nnz)
                    jab =     (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    iab = i            + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    ijb = i + (j)*ndns                  + (ib)*ndns*ndns*nvir
                    ija = i + (j)*ndns + (ia)*ndns*ndns
                    ijab= i + (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir

                    nnz = nnz + 1
                    A[nnz] = (F_MO[ia+ndns,ia+ndns] + F_MO[ib+ndns,ib+ndns] - F_MO[i,i] - F_MO[j,j])
                    IROW[nnz] = (ijab)
                    ICOL[nnz] = (i + jab)

                    for k in range(i):
                        if(abs(F_MO[i,k])>1e-10):
                            Cki = FI2[k]*FI2[i]
                            nnz += 1
                            A[nnz]=(- Cki*F_MO[i,k])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k + jab)
                            nnz += 1
                            A[nnz]=(- Cki*F_MO[i,k])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k + jab)

                    for k in range(j):
                        if(abs(F_MO[j,k])>1e-10):
                            Ckj = FI2[k]*FI2[j]
                            nnz += 1
                            A[nnz]=(- Ckj*F_MO[j,k])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k*ndns + iab)
                            nnz += 1
                            A[nnz]=(- Ckj*F_MO[j,k])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k*ndns + iab)

                    for k in range(ia):
                        if(abs(F_MO[ia+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+ndns]*FI1[ia+ndns]
                            else:
                                Ckia = FI2[k+ndns]*FI2[ia+ndns]
                            nnz += 1
                            A[nnz]=(Ckia*F_MO[ia+ndns,k+ndns])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k*ndns*ndns + ijb)
                            nnz += 1
                            A[nnz]=(Ckia*F_MO[ia+ndns,k+ndns])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k*ndns*ndns + ijb)

                    for k in range(ib):
                        if(abs(F_MO[ib+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+ndns]*FI1[ib+ndns]
                            else:
                                Ckib = FI2[k+ndns]*FI2[ib+ndns]
                            nnz += 1
                            A[nnz]=(Ckib*F_MO[ib+ndns,k+ndns])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k*ndns*ndns*nvir + ija)
                            nnz += 1
                            A[nnz]=(Ckib*F_MO[ib+ndns,k+ndns])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k*ndns*ndns*nvir + ija)

    A = A[:nnz+1]
    IROW = IROW[:nnz+1]
    ICOL = ICOL[:nnz+1]

    return A,(IROW,ICOL)

@njit(parallel=True)
def build_R(T,B,F_MO,FI1,FI2,no1,ndoc,ndns,nvir,ncwo,nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = ncwo*(ndoc - i - 1)
        ul = ncwo*(ndoc - i)
        npair[ll:ul] = i + 1

    Bp = np.zeros((ndns**2*nvir**2))
    
    for ib in prange(nvir):
        for ia in prange(nvir):
            for j in prange(ndns):
                for i in prange(ndns):
                    jab =     (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    iab = i            + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    ijb = i + (j)*ndns                  + (ib)*ndns*ndns*nvir
                    ija = i + (j)*ndns + (ia)*ndns*ndns
                    ijab= i + (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir

                    Bp[ijab] += (F_MO[ia+ndns,ia+ndns] + F_MO[ib+ndns,ib+ndns] - F_MO[i,i] - F_MO[j,j])*T[i+jab]

                    for k in range(i):
                        if(abs(F_MO[i,k])>1e-10):
                            Cki = FI2[k]*FI2[i]
                            Bp[ijab] += (- Cki*F_MO[i,k])*T[k+jab]
                    for k in range(i+1,ndns):
                        if(abs(F_MO[i,k])>1e-10):
                            Cki = FI2[k]*FI2[i]
                            Bp[ijab] += (- Cki*F_MO[i,k])*T[k+jab]

                    for k in range(j):
                        if(abs(F_MO[j,k])>1e-10):
                            Ckj = FI2[k]*FI2[j]
                            Bp[ijab] += (- Ckj*F_MO[j,k])*T[k*ndns+iab]
                    for k in range(j+1,ndns):
                        if(abs(F_MO[j,k])>1e-10):
                            Ckj = FI2[k]*FI2[j]
                            Bp[ijab] += (- Ckj*F_MO[j,k])*T[k*ndns+iab]

                    for k in range(ia):
                        if(abs(F_MO[ia+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+ndns]*FI1[ia+ndns]
                            else:
                                Ckia = FI2[k+ndns]*FI2[ia+ndns]
                            Bp[ijab] += (Ckia*F_MO[ia+ndns,k+ndns]) * T[k*ndns*ndns + ijb] 
                    for k in range(ia+1,nvir):
                        if(abs(F_MO[ia+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+ndns]*FI1[ia+ndns]
                            else:
                                Ckia = FI2[k+ndns]*FI2[ia+ndns]
                            Bp[ijab] += (Ckia*F_MO[ia+ndns,k+ndns]) * T[k*ndns*ndns + ijb] 

                    for k in range(ib):
                        if(abs(F_MO[ib+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+ndns]*FI1[ib+ndns]
                            else:
                                Ckib = FI2[k+ndns]*FI2[ib+ndns]
                            Bp[ijab] += (Ckib*F_MO[ib+ndns,k+ndns]) * T[k*ndns*ndns*nvir + ija] 
                    for k in range(ib+1,nvir):
                        if(abs(F_MO[ib+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+ndns]*FI1[ib+ndns]
                            else:
                                Ckib = FI2[k+ndns]*FI2[ib+ndns]
                            Bp[ijab] += (Ckib*F_MO[ib+ndns,k+ndns]) * T[k*ndns*ndns*nvir + ija] 


    R = B-Bp
    return R

@njit
def build_B(iajb,FI1,FI2,ndoc,ndns,nvir,ncwo):
    B = np.zeros((ndns**2*nvir**2))
    for i in range(ndns):
        lmin_i = ndoc+ncwo*(ndns-i-1)
        lmax_i = ndoc+ncwo*(ndns-i-1)+ncwo
        for j in range(ndns):
            if(i==j):
                for k in range(nvir):
                    ik = i + k*ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndns
                        if(lmin_i <= kn and kn <= lmax_i and lmin_i <= ln and ln <= lmax_i):
                            Ciikl = FI1[kn]*FI1[ln]*FI1[i]*FI1[i]
                        else:
                            Ciikl = FI2[kn]*FI2[ln]*FI2[i]*FI2[i]
                        iikl =  i + i*ndns + k*ndns*ndns + l*ndns*ndns*nvir
                        B[iikl] = - Ciikl*iajb[i,k,i,l]
            else:
                for k in range(nvir):
                    ik = i + k*ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndoc
                        ijkl =  i + j*ndns + k*ndns*ndns + l*ndns*ndns*nvir
                        Cijkl = FI2[kn]*FI2[ln]*FI2[i]*FI2[j]
                        B[ijkl] = - Cijkl*iajb[j,k,i,l]

    return B

@njit
def Tijab_guess(iajb,eig,ndoc,ndns,nvir):
    Tijab = np.zeros(nvir**2*ndns**2)
    for ia in range(nvir):
        for i in range(ndns):
            for ib in range(nvir):
                for j in range(ndns):
                    ijab = i + (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    Eijab = eig[ib+ndns] + eig[ia+ndns] - eig[j] - eig[i]
                    Tijab[ijab] = - iajb[j,ia,i,ib]/Eijab
    return Tijab 

def solve_Tijab(A_CSR,B,Tijab,p):
    Tijab,info = cg(A_CSR, B,x0=Tijab)
    return Tijab
