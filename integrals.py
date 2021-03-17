import numpy as np
from time import time
from einsumt import einsumt as einsum
from numba import jit,guvectorize,prange,njit,cuda
import cupy as cp

@njit(parallel=True,cache=True,fastmath=True)
def computeJK(C,I,nbf5,nbf):

    #denmatj
    #D = np.einsum('mi,ni->imn', C[:,0:nbf5], C[:,0:nbf5], optimize=True)
    D = np.zeros((nbf5,nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbf5):
                D[i][mu][nu] = C[mu][i]*C[nu][i]
                D[i][nu][mu] = D[i][mu][nu]


    #hstarj
    #J = np.einsum('isl,mnsl->imn', D, I, optimize=True)
    #J = np.tensordot(D, I, axes=([1,2],[2,3]))
    J = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for n in prange(m+1):
                for s in range(nbf):
                    for l in range(nbf):
                        J[i][m][n] += D[i][s][l]*I[m][n][s][l]
                J[i][n][m] = J[i][m][n]

    #hstark
    #K = np.einsum('inl,mnsl->ims', D, I, optimize=True)
    #K = np.tensordot(D, I, axes=([1,2],[1,3]))
    K = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for s in prange(m+1):
                for n in range(nbf):
                    for l in range(nbf):
                        K[i][m][s] += D[i][n][l]*I[m][n][s][l]
                K[i][s][m] = K[i][m][s]

    return J,K


@njit(parallel=True,cache=True,fastmath=True)
def computeJKH_core_MO(C,H,I,nbf5,nbf):

    #denmatj
    #D = np.einsum('mi,ni->imn', C[:,0:nbf5], C[:,0:nbf5],optimize=True)
    D = np.zeros((nbf5,nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbf5):
                D[i][mu][nu] = C[mu][i]*C[nu][i]
                D[i][nu][mu] = D[i][mu][nu]
    #QJMATm

    #J = einsum('isl,mnsl->imn', D, I, optimize=True)
    #J = np.tensordot(D, I, axes=([1,2],[2,3]))
    #J_MO = einsum('jmn,imn->ij', D, J, optimize=True)
    #J_MO = np.tensordot(J, D,axes=((1,2),(1,2)))
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
    #K = einsum('inl,mnsl->ims', D, I, optimize=True)
    #K = np.tensordot(D, I, axes=([1,2],[1,3]))
    #K_MO = einsum('jms,ims->ij', D, K, optimize=True)
    #K_MO = np.tensordot(K, D, axes=([1,2],[1,2]))
    K = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for s in prange(m+1):
                for n in range(nbf):
                    for l in range(nbf):
                        K[i][m][s] += D[i][n][l]*I[m][n][s][l]
    K_MO = np.zeros((nbf5,nbf5))
    for i in prange(nbf5):
        for j in prange(i+1):
            for m in range(nbf):
                for s in range(nbf):
                    K_MO[i][j] += D[j][m][s]*K[i][m][s]
            K_MO[j][i] = K_MO[i][j]

    #QHMATm
    #H_core = einsum('imn,mn->i', D, H, optimize=True)
    #H_core = np.tensordot(D, H, axes=([1,2],[0,1]))
    H_core = np.zeros((nbf5))
    for i in prange(nbf5):
        for m in range(nbf):
            for n in range(m+1):
                H_core[i] += D[i][m][n]*H[m][n]


    return J_MO,K_MO,H_core

def computeJK_RI(C,I,b_mnl,p):

    C = cp.array(C)
    #denmatj
    #b_qnl = cp.einsum('mp,mnl->pnl',C[:,0:p.nbf5],b_mnl, optimize=True)
    b_qnl = cp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_qql = cp.einsum('nq,qnl->ql',C[:,0:p.nbf5],b_qnl, optimize=True)

    #hstarj
    #J = cp.einsum('ql,mnl->qmn', b_qql, b_mnl, optimize=True)
    J = np.tensordot(b_qql, b_mnl, axes=([1],[2]))

    #hstark
    K = cp.einsum('qml,qnl->qmn', b_qnl, b_qnl, optimize=True)

    return J.get(),K.get()

def computeJKH_core_MO_RI(C,H,I,b_mnl,p):

    C = cp.array(C)
    H = cp.array(H)
    #denmatj
    D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)

    #b_pnl = cp.einsum('mp,mnl->pnl',C[:,0:p.nbf5],b_mnl, optimize=True)
    b_pnl = np.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_pql = cp.einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)

    #QJMATm
    J_MO = cp.einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)

    #QKMATm
    K_MO = cp.einsum('pql,pql->pq', b_pql, b_pql, optimize=True)

    #QHMATm
    #H_core = cp.einsum('imn,mn->i', D, H, optimize=True)
    H_core = cp.tensordot(D,H, axes=([1,2],[0,1]))

    return J_MO.get(),K_MO.get(),H_core.get()

