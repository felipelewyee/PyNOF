from time import time
import psi4
import numpy as np
import cupy as cp
from numba import prange,njit

def compute_integrals(wfn,mol,p):

    # Integrador
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    # Overlap, Kinetics, Potential
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = T + V
    
    I = []
    b_mnl = []
    if (not p.RI):
        # Integrales de Repulsión Electrónica, ERIs (mu nu | sigma lambda)
        I = np.asarray(mints.ao_eri())
    else:

        orb = wfn.basisset()
        aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", orb.blend())
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

        Ppq = mints.ao_eri(orb, orb, aux, zero_bas)
        
        metric = mints.ao_eri(aux, zero_bas, aux, zero_bas)
        metric.power(-0.5, 1.e-14)
        p.nbfaux = metric.shape[0]

        Ppq = np.squeeze(Ppq)
        metric = np.squeeze(metric)

        b_mnl = np.einsum('pqP,PQ->pqQ', Ppq, metric, optimize=True)

    if(p.gpu):
        I = cp.array(I)
        b_mnl = cp.array(b_mnl)

    return S,T,V,H,I,b_mnl

######################################### J_mn^(j) K_mn^(j) #########################################

def computeJKj(C,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            J,K = JKj_RI_GPU(C,b_mnl,p)
        else:
            J,K = JKj_Full_GPU(C,I,p)
    else:
        if(p.jit):
            if(p.RI):
                J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
            else:
                J,K = JKj_Full_jit(C,I,p.nbf,p.nbf5)
        else:
            if(p.RI):
                J,K = JKj_RI(C,b_mnl,p)
            else:
                J,K = JKj_Full(C,I,p)

    return J,K

#########################################


def JKj_Full(C,I,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)
    #hstarj
    J = np.tensordot(D, I, axes=([1,2],[2,3]))
    #hstark
    K = np.tensordot(D, I, axes=([1,2],[1,3]))
    
    return J,K

def JKj_RI(C,b_mnl,p):
    
    #b_transform
    b_qnl = np.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_qql = np.einsum('nq,qnl->ql',C[:,0:p.nbf5],b_qnl, optimize=True)
    #hstarj
    J = np.tensordot(b_qql, b_mnl, axes=([1],[2]))
    #hstark
    K = np.einsum('qml,qnl->qmn', b_qnl, b_qnl, optimize=True)
    
    return J,K

def JKj_Full_GPU(C,I,p):

    #denmatj
    D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)
    #hstarj
    J = cp.tensordot(D, I, axes=([1,2],[2,3]))
    #hstark
    K = cp.tensordot(D, I, axes=([1,2],[1,3]))
    
    return J.get(),K.get()

def JKj_RI_GPU(C,b_mnl,p):

    C = cp.array(C)
    #b_transform
    b_qnl = cp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_qql = cp.einsum('nq,qnl->ql',C[:,0:p.nbf5],b_qnl, optimize=True)
    #hstarj
    J = cp.tensordot(b_qql, b_mnl, axes=([1],[2]))
    #hstark
    K = cp.einsum('qml,qnl->qmn', b_qnl, b_qnl, optimize=True)

    return J.get(),K.get()

@njit(parallel=True)
def JKj_Full_jit(C,I,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf5,nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbf5):
                D[i][mu][nu] = C[mu][i]*C[nu][i]
                D[i][nu][mu] = D[i][mu][nu]

    #hstarj
    J = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for n in prange(m+1):
                for s in range(nbf):
                    for l in range(nbf):
                        J[i][m][n] += D[i][s][l]*I[m][n][s][l]
                J[i][n][m] = J[i][m][n]

    #hstark
    K = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for s in prange(m+1):
                for n in range(nbf):
                    for l in range(nbf):
                        K[i][m][s] += D[i][n][l]*I[m][n][s][l]
                K[i][s][m] = K[i][m][s]

    return J,K

@njit(parallel=True)
def JKj_RI_jit(C,b_mnl,nbf,nbf5,nbfaux):

    #denmatj
    b_qnl = np.zeros((nbf5,nbf,nbfaux))
    for q in prange(nbf5):
        for n in prange(nbf):
            for l in prange(nbfaux):
                for m in range(nbf):
                    b_qnl[q][n][l] += C[m][q]*b_mnl[m][n][l] 
    b_qql = np.zeros((nbf5,nbfaux))
    for q in prange(nbf5):
        for l in prange(nbfaux):
            for n in range(nbf):
                b_qql[q][l] += C[n][q]*b_qnl[q][n][l] 

    #hstarj
    J = np.zeros((nbf5,nbf,nbf))
    for q in prange(nbf5):
        for m in prange(nbf):
            for n in prange(nbf):
                for l in range(nbfaux):
                    J[q][m][n] += b_qql[q][l]*b_mnl[m][n][l] 

    #hstark
    K = np.zeros((nbf5,nbf,nbf))
    for q in prange(nbf5):
        for m in prange(nbf):
            for n in prange(nbf):
                for l in range(nbfaux):
                    K[q][m][n] += b_qnl[q][m][l]*b_qnl[q][n][l] 

    return J,K

######################################### J_pq K_pq #########################################

def computeJKH_MO(C,H,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            J_MO,K_MO,H_core = JKH_MO_RI_GPU(C,H,b_mnl,p)
        else:
            J_MO,K_MO,H_core = JKH_MO_Full_GPU(C,H,I,p)
    else:
        if(p.jit):
            if(p.RI):
                J_MO,K_MO,H_core = JKH_MO_RI_jit(C,H,b_mnl,p.nbf,p.nbf5,p.nbfaux)
            else:
                J_MO,K_MO,H_core = JKH_MO_Full_jit(C,H,I,p.nbf,p.nbf5)
        else:
            if(p.RI):
                J_MO,K_MO,H_core = JKH_MO_RI(C,H,b_mnl,p)
            else:
                J_MO,K_MO,H_core = JKH_MO_Full(C,H,I,p)

    return J_MO,K_MO,H_core

#########################################

def JKH_MO_Full(C,H,I,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #QJMATm
    J = np.tensordot(D, I, axes=([1,2],[2,3]))
    J_MO = np.tensordot(J, D,axes=((1,2),(1,2)))
    #QKMATm
    K = np.tensordot(D, I, axes=([1,2],[1,3]))
    K_MO = np.tensordot(K, D, axes=([1,2],[1,2]))
    #QHMATm
    H_core = np.tensordot(D, H, axes=([1,2],[0,1]))
    
    return J_MO,K_MO,H_core

def JKH_MO_RI(C,H,b_mnl,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #b transform
    b_pnl = np.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_pql = np.einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)
    #QJMATm
    J_MO = np.einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)
    #QKMATm
    K_MO = np.einsum('pql,pql->pq', b_pql, b_pql, optimize=True)
    #QHMATm
    H_core = np.tensordot(D,H, axes=([1,2],[0,1]))
    
    return J_MO,K_MO,H_core

def JKH_MO_Full_GPU(C,H,I,p):

    C = cp.array(C)
    H = cp.array(H)
    #denmatj
    D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #QJMATm
    J = cp.tensordot(D, I, axes=([1,2],[2,3]))
    J_MO = cp.tensordot(J, D,axes=((1,2),(1,2)))
    #QKMATm
    K = cp.tensordot(D, I, axes=([1,2],[1,3]))
    K_MO = cp.tensordot(K, D, axes=([1,2],[1,2]))
    #QHMATm
    H_core = cp.tensordot(D, H, axes=([1,2],[0,1]))

    return J_MO.get(),K_MO.get(),H_core.get()

def JKH_MO_RI_GPU(C,H,b_mnl,p):

    C = cp.array(C)
    H = cp.array(H)
    #denmatj
    D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #b transform 
    b_pnl = cp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_pql = cp.einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)
    #QJMATm
    J_MO = cp.einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)
    #QKMATm
    K_MO = cp.einsum('pql,pql->pq', b_pql, b_pql, optimize=True)
    #QHMATm
    H_core = cp.tensordot(D,H, axes=([1,2],[0,1]))

    return J_MO.get(),K_MO.get(),H_core.get()

@njit(parallel=True)
def JKH_MO_Full_jit(C,H,I,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf5,nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbf5):
                D[i][mu][nu] = C[mu][i]*C[nu][i]
                D[i][nu][mu] = D[i][mu][nu]
    #QJMATm
    J = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for n in prange(m+1):
                for s in range(nbf):
                    for l in range(nbf):
                        J[i][m][n] += D[i][s][l]*I[m][n][s][l]
                J[i][n][m] = J[i][m][n]
    J_MO = np.zeros((nbf5,nbf5))
    for i in prange(nbf5):
        for j in prange(i+1):
            for m in range(nbf):
                for n in range(nbf):
                    J_MO[i][j] += D[j][m][n]*J[i][m][n]
            J_MO[j][i] = J_MO[i][j]

    #QKMATm
    K = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for s in prange(m+1):
                for n in range(nbf):
                    for l in range(nbf):
                        K[i][m][s] += D[i][n][l]*I[m][n][s][l]
                K[i][s][m] = K[i][m][s]
    K_MO = np.zeros((nbf5,nbf5))
    for i in prange(nbf5):
        for j in prange(i+1):
            for m in range(nbf):
                for s in range(nbf):
                    K_MO[i][j] += D[j][m][s]*K[i][m][s]
            K_MO[j][i] = K_MO[i][j]

    #QHMATm
    H_core = np.zeros((nbf5))
    for i in prange(nbf5):
        for m in range(nbf):
            for n in range(m):
                H_core[i] += 2*D[i][m][n]*H[m][n]
            H_core[i] += D[i][m][m]*H[m][m]

    return J_MO,K_MO,H_core

@njit(parallel=True)
def JKH_MO_RI_jit(C,H,b_mnl,nbf,nbf5,nbfaux):

    #denmatj
    D = np.zeros((nbf5,nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbf5):
                D[i][mu][nu] = C[mu][i]*C[nu][i]
                D[i][nu][mu] = D[i][mu][nu]

    b_pnl = np.zeros((nbf5,nbf,nbfaux))
    for p in prange(nbf5):
        for n in prange(nbf):
            for l in prange(nbfaux):
                for m in range(nbf):
                    b_pnl[p][n][l] += C[m][p]*b_mnl[m][n][l]
    b_pql = np.zeros((nbf5,nbf5,nbfaux))
    for p in prange(nbf5):
        for q in prange(p+1):
            for l in prange(nbfaux):
                for n in range(nbf):
                    b_pql[p][q][l] += C[n][q]*b_pnl[p][n][l]
            b_pql[q][p][l] = b_pql[p][q][l]

    #hstarj
    J_MO = np.zeros((nbf5,nbf5))
    for p in prange(nbf5):
        for q in prange(p+1):
            for l in range(nbfaux):
                J_MO[p][q] += b_pql[p][p][l]*b_pql[q][q][l]
            J_MO[q][p] = J_MO[p][q]

    #hstark
    K_MO = np.zeros((nbf5,nbf5))
    for p in prange(nbf5):
        for q in prange(p+1):
            for l in range(nbfaux):
                K_MO[p][q] += b_pql[p][q][l]*b_pql[p][q][l]
            K_MO[q][p] = K_MO[p][q]

    H_core = np.zeros((nbf5))
    for i in prange(nbf5):
        for m in range(nbf):
            for n in range(m):
                H_core[i] += 2*D[i][m][n]*H[m][n]
            H_core[i] += D[i][m][m]*H[m][m]

    return J_MO,K_MO,H_core

######################################### J_mn^(j) K_mn^(j) #########################################

def computeJK_HF(C,I,b_mnl,p):

    if(p.gpu):
#        if(p.RI):
#            J,K = JKj_RI_GPU(C,b_mnl,p)
#        else:
        D,J,K = JK_HF_Full_GPU(C,I,p)
    else:
#        if(p.RI):
#            J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
#        else:
        D,J,K = JK_HF_Full_jit(C,I,p.nbeta,p.nbf,p.nbf5)

    return D,J,K

@njit(parallel=True)
def JK_HF_Full_jit(C,I,nbeta,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbeta):
                D[mu][nu] += C[mu][i]*C[nu][i]
            D[nu][mu] = D[mu][nu]

    #hstarj
    J = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for n in prange(m+1):
            for s in range(nbf):
                for l in range(nbf):
                    J[m][n] += D[s][l]*I[m][n][s][l]
            J[n][m] = J[m][n]

    #hstark
    K = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for s in prange(m+1):
            for n in range(nbf):
                for l in range(nbf):
                    K[m][s] += D[n][l]*I[m][n][s][l]
            K[s][m] = K[m][s]

    return D,J,K

def JK_HF_Full_GPU(C,I,p):

    #denmatj
    D = cp.einsum("mj,nj->mn",C[:,:p.nbeta],C[:,:p.nbeta],optimize=True)
    J = cp.einsum("ls,mnsl->mn",D,I,optimize=True)
    K = cp.einsum("nl,mnsl->ms",D,I,optimize=True)
    
    return D.get(),J.get(),K.get()

def computeJKalpha_HF(C,I,b_mnl,p):

    if(p.gpu):
#        if(p.RI):
#            J,K = JKj_RI_GPU(C,b_mnl,p)
#        else:
        D,J,K = JKalpha_HF_Full_GPU(C,I,p)
    else:
#        if(p.RI):
#            J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
#        else:
        D,J,K = JKalpha_HF_Full_jit(C,I,p.nbeta,p.nalpha,p.nbf,p.nbf5)

    return D,J,K

@njit(parallel=True)
def JKalpha_HF_Full_jit(C,I,nbeta,nalpha,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbeta,nalpha):
                D[mu][nu] += C[mu][i]*C[nu][i]
            D[nu][mu] = D[mu][nu]

    #hstarj
    J = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for n in prange(m+1):
            for s in range(nbf):
                for l in range(nbf):
                    J[m][n] += D[s][l]*I[m][n][s][l]
            J[n][m] = J[m][n]

    #hstark
    K = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for s in prange(m+1):
            for n in range(nbf):
                for l in range(nbf):
                    K[m][s] += D[n][l]*I[m][n][s][l]
            K[s][m] = K[m][s]

    return D,J,K

def JKalpha_HF_Full_GPU(C,I,p):

    #denmatj
    D = cp.einsum("mj,nj->mn",C[:,p.nbeta:p.nalpha],C[:,p.nbeta:p.nalpha],optimize=True)
    J = cp.einsum("ls,mnsl->mn",D,I,optimize=True)
    K = cp.einsum("nl,mnsl->ms",D,I,optimize=True)

    return D.get(),J.get(),K.get()

def compute_iajb(C,I,b_mnl,p):

    if(p.gpu):
        iajb = iajb_Full_GPU(C,I,p)
    else:
        iajb = iajb_Full_jit(C,I,p.no1,p.nalpha,p.nbf,p.nbf5)

    return iajb

def iajb_Full_jit(C,I,no1,nalpha,nbf,nbf5):

    iajb = np.einsum('mi,na,mnsl,sj,lb->iajb',C[:,no1:nalpha],C[:,nalpha:nbf],I,C[:,no1:nalpha],C[:,nalpha:nbf],optimize=True)

    return iajb

def iajb_Full_GPU(C,I,p):

    iajb = cp.einsum('mi,na,mnsl,sj,lb->iajb',C[:,p.no1:p.nalpha],C[:,p.nalpha:p.nbf],I,C[:,p.no1:p.nalpha],C[:,p.nalpha:p.nbf],optimize=True)

    return iajb.get()

