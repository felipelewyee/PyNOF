import pynof
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
    
    Dipole = np.asarray(mints.ao_dipole())

    I = []
    b_mnl = []
    if (not p.RI):
        # Integrales de Repulsión Electrónica, ERIs (mu nu | sigma lambda)
        I = np.asarray(mints.ao_eri())
    else:

        orb = wfn.basisset()
        aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", orb.blend())
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

        Ppq = np.transpose(mints.ao_eri(aux, zero_bas, orb, orb))
        
        metric = mints.ao_eri(aux, zero_bas, aux, zero_bas)
        metric.power(-0.5, 1.e-14)
        p.nbfaux = metric.shape[0]

        Ppq = np.squeeze(Ppq)
        metric = np.squeeze(metric)
        p.metric = metric

        p.mnl = Ppq

        b_mnl = np.einsum('pqP,PQ->pqQ', Ppq, metric, optimize=True)

    if(p.gpu):
        I = cp.array(I)
        b_mnl = cp.array(b_mnl)

    return S,T,V,H,I,b_mnl,Dipole

def compute_der_integrals(wfn,mol,n,C,cj12,ck12,elag,p):

    mints = psi4.core.MintsHelper(wfn.basisset())

    RDM1 = 2*np.einsum('p,mp,np->mn',n,C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)
    lag = 2*np.einsum('mq,qp,np->mn',C,elag,C,optimize=True)
    
    grad = np.zeros((p.natoms,3))

    grad += np.array(mol.nuclear_repulsion_energy_deriv1())

    for i in range(p.natoms):
        dSx,dSy,dSz = np.array(mints.ao_oei_deriv1("OVERLAP",i))
        grad[i,0] -= np.einsum('mn,mn->',lag,dSx,optimize=True)
        grad[i,1] -= np.einsum('mn,mn->',lag,dSy,optimize=True)
        grad[i,2] -= np.einsum('mn,mn->',lag,dSz,optimize=True)

        dTx,dTy,dTz = np.array(mints.ao_oei_deriv1("KINETIC",i))
        grad[i,0] += np.einsum('mn,mn->',RDM1,dTx,optimize=True)
        grad[i,1] += np.einsum('mn,mn->',RDM1,dTy,optimize=True)
        grad[i,2] += np.einsum('mn,mn->',RDM1,dTz,optimize=True)

        dVx,dVy,dVz = np.array(mints.ao_oei_deriv1("POTENTIAL",i))
        grad[i,0] += np.einsum('mn,mn->',RDM1,dVx,optimize=True)
        grad[i,1] += np.einsum('mn,mn->',RDM1,dVy,optimize=True)
        grad[i,2] += np.einsum('mn,mn->',RDM1,dVz,optimize=True)

    np.fill_diagonal(cj12,0) # Remove diag.
    np.fill_diagonal(ck12,0) # Remove diag.
    RDM2 = np.einsum('qp,mp,np,sq,lq->msnl',cj12,C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)
    RDM2 += np.einsum('p,mp,np,sp,lp->msnl',n[:p.nbeta],C[:,:p.nbeta],C[:,:p.nbeta],C[:,:p.nbeta],C[:,:p.nbeta],optimize=True)
    RDM2 += np.einsum('p,mp,np,sp,lp->msnl',n[p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],optimize=True)
    RDM2 -= np.einsum('qp,mp,lp,sq,nq->msnl',ck12,C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)

    for i in range(p.natoms):
        derix,deriy,deriz = np.array(mints.ao_tei_deriv1(i))
        grad[i,0] += np.einsum("msnl,mnsl->",RDM2,derix,optimize=True)
        grad[i,1] += np.einsum("msnl,mnsl->",RDM2,deriy,optimize=True)
        grad[i,2] += np.einsum("msnl,mnsl->",RDM2,deriz,optimize=True)

    return grad


######################################### J_mn^(j) K_mn^(j) #########################################

def computeJKj(S,C,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            J,K = JKj_RI_GPU(C,b_mnl,p)
        else:
            J,K = JKj_Full_GPU(C,I,p)
    else:
        if(p.jit):
            if(p.RI):
                J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
            elif(p.PARI):
                metric = p.metric.copy()
                #metric = np.real(inv(metric))
                #metric = np.real(sqrtm(metric))
                pop,pop_aux = pynof.local_domains(p,C,S)
                J,K = JKj_RI_jit_tmp(pop,C,p.mnl,metric,p.nbf,p.nbf5,p.nbfaux)
                #J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
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
    K = np.einsum("inl,mnsl->ims",D,I,optimize=True)
    #K = cp.tensordot(D, I, axes=([1,2],[1,3]))
    
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

@njit(parallel=True, cache=True)
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

@njit(parallel=True, cache=True)
def JKj_RI_jit(C,b_mnl,nbf,nbf5,nbfaux):

    #denmatj
    J = np.zeros((nbf5,nbf,nbf))
    K = np.zeros((nbf5,nbf,nbf))
    for q in prange(nbf5):
        b_qnl = np.zeros((nbf,nbfaux))
        b_qql = np.zeros((nbfaux))
        for l in range(nbfaux):
            for n in range(nbf):
                for m in range(nbf):
                    b_qnl[n][l] += C[m][q]*b_mnl[m][n][l]
                b_qql[l] += C[n][q]*b_qnl[n][l]
            for n in range(nbf):
                for m in range(nbf):
                    K[q][m][n] += b_qnl[m][l]*b_qnl[n][l]
                    J[q][m][n] += b_qql[l]*b_mnl[m][n][l]

    return J,K

@njit(parallel=True, cache=True)
def JKj_RI_jit_tmp(pop,C,b_mnk,G,nbf,nbf5,nbfaux):

    #denmatj
    b_qnk = np.zeros((nbf5,nbf,nbfaux))
    b_qnl = np.zeros((nbf5,nbf,nbfaux))
    K = np.zeros((nbf5,nbf,nbf))
    #hstark
    for q in prange(nbf5):
        for n in range(nbf):
            if(pop[q,n] == 1):
                for k in range(nbfaux):
                    for m in range(nbf):
                        b_qnk[q][n][k] += C[m][q]*b_mnk[m][n][k]

    np.save("b_qnk.npy",b_qnk)
    #pynof.check_sparsity(b_qnk)
    #pdata(b_qnk)

    for q in prange(nbf5):
        for n in range(nbf):
            for l in range(nbfaux):
                for k in range(nbfaux):
                    b_qnl[q][n][l] += b_qnk[q][n][k]*G[k][l]

        for n in range(nbf):
            if(pop[q,n] == 1):
                for m in range(nbf):
                    if(pop[q,m] == 1):
                        for l in range(nbfaux):
                            K[q][m][n] += b_qnl[q][m][l]*b_qnl[q][n][l]

    b_mnl = np.zeros((nbf,nbf,nbfaux))
    for m in prange(nbf):
        for n in range(nbf):
            for l in range(nbfaux):
                for k in range(nbfaux):
                    b_mnl[m][n][l] += b_mnk[m][n][k]*G[k][l]

    #hstarj
    b_qql = np.zeros((nbf5,nbfaux))
    J = np.zeros((nbf5,nbf,nbf))
    for q in prange(nbf5):
        for l in range(nbfaux):
            for n in range(nbf):
                if(pop[q,n] == 1):
                    b_qql[q][l] += C[n][q]*b_qnl[q][n][l]

        for l in range(nbfaux):
            for m in range(nbf):
                for n in range(nbf):
                    J[q][m][n] += b_qql[q][l]*b_mnl[m][n][l]


    return J,K


######################################### J_pq K_pq #########################################

def computeJKH_MO(S,C,H,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            J_MO,K_MO,H_core = JKH_MO_RI_GPU(C,H,b_mnl,p)
        else:
            J_MO,K_MO,H_core = JKH_MO_Full_GPU(C,H,I,p)
    else:
        if(p.jit):
            if(p.RI):
                J_MO,K_MO,H_core = JKH_MO_RI_jit(C,H,b_mnl,p.nbf,p.nbf5,p.nbfaux)
            elif(p.PARI):
                metric = p.metric.copy()
                #metric = np.real(inv(metric))
                #metric = np.real(sqrtm(metric))
                pop, pop_aux = pynof.local_domains(p,C,S)
                J_MO,K_MO,H_core = JKH_MO_RI_jit_tmp(pop,C,H,p.mnl.copy(),metric,p.nbf,p.nbf5,p.nbfaux)                
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

@njit(parallel=True, cache=True)
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

@njit(parallel=True, cache=True)
def JKH_MO_RI_jit(C,H,b_mnl,nbf,nbf5,nbfaux):

    b_pql = np.zeros((nbf5,nbf5,nbfaux))
    for p in prange(nbf5):
        for n in range(nbf):
            for l in range(nbfaux):
                b_pnl = 0
                for m in range(nbf):
                    b_pnl += C[m][p]*b_mnl[m][n][l]
                for q in range(p+1):
                    b_pql[p][q][l] += C[n][q]*b_pnl

    H_core = np.zeros((nbf5))
    J_MO = np.zeros((nbf5,nbf5))
    K_MO = np.zeros((nbf5,nbf5))
    for p in prange(nbf5):
        for m in range(nbf):
            for n in range(nbf):
                H_core[p] += C[m][p]*H[m][n]*C[n][p]
        for q in range(p+1):
            for l in range(nbfaux):
                J_MO[p][q] += b_pql[p][p][l]*b_pql[q][q][l]
                K_MO[p][q] += b_pql[p][q][l]*b_pql[p][q][l]
            J_MO[q][p] = J_MO[p][q]
            K_MO[q][p] = K_MO[p][q]

    return J_MO,K_MO,H_core

@njit(parallel=True, cache=True)
def JKH_MO_RI_jit_tmp(pop,C,H,b_mnk,G,nbf,nbf5,nbfaux):

    #denmatj
    D = np.zeros((nbf5,nbf,nbf))
    #b_pnl = np.zeros((nbf5,nbf,nbfaux))
    b_pnk = np.zeros((nbf5,nbf,nbfaux))
    b_pqk = np.zeros((nbf5,nbf5,nbfaux))
    b_pql = np.zeros((nbf5,nbf5,nbfaux))
    H_core = np.zeros((nbf5))

    for p in prange(nbf5):
        for m in range(nbf):
            for n in range(m+1):
                D[p][m][n] = C[m][p]*C[n][p]
                D[p][n][m] = D[p][m][n]
        for m in range(nbf):
            for n in range(m):
                H_core[p] += 2*D[p][m][n]*H[m][n]
            H_core[p] += D[p][m][m]*H[m][m]

    #hstark
    K_MO = np.zeros((nbf5,nbf5))
    for p in prange(nbf5):
        for n in range(nbf):
            if(pop[p,n] == 1):
                for l in range(nbfaux):
                    for m in range(nbf):
                        b_pnk[p][n][l] += C[m][p]*b_mnk[m][n][l]

        for q in range(p+1):
            for k in range(nbfaux):
                for n in range(nbf):
                    if(pop[p,n] == 1):
                        b_pqk[p][q][k] += C[n][q]*b_pnk[p][n][k]
       #         b_pqk[q][p][k] = b_pqk[p][q][k]

        #for q in range(p+1):
            for l in range(nbfaux):
                for k in range(nbfaux):
                    b_pql[p][q][l] += b_pqk[p][q][k]*G[k,l]
        #        b_pql[q][p][l] = b_pql[p][q][l]

        #for q in range(p+1):
        #    for l in range(nbfaux):
                K_MO[p][q] += b_pql[p][q][l]*b_pql[p][q][l]
            K_MO[q][p] = K_MO[p][q]

    #hstarj
    b_ppk = np.zeros((nbf5,nbfaux))
    for p in prange(nbf5):
        for k in prange(nbfaux):
            for n in range(nbf):
                if(pop[p,n] == 1):
                    b_ppk[p][k] += C[n][p]*b_pnk[p][n][k]

    b_ppl = np.zeros((nbf5,nbfaux))
    for p in prange(nbf5):
        for l in prange(nbfaux):
            for k in range(nbfaux):
                b_ppl[p][l] += b_ppk[p][k]*G[k,l]

    J_MO = np.zeros((nbf5,nbf5))
    for p in prange(nbf5):
        for q in prange(p+1):
            for l in range(nbfaux):
                #J_MO[p][q] += b_pql[p][p][l]*b_pql[q][q][l]
                J_MO[p][q] += b_ppl[p][l]*b_ppl[q][l]
            J_MO[q][p] = J_MO[p][q]

    return J_MO,K_MO,H_core


######################################### J_mn^(j) K_mn^(j) #########################################

def computeD_HF(C,p):

    if(p.gpu):
        D = cp.einsum('mj,nj->mn',C[:,:p.nbeta],C[:,:p.nbeta],optimize=True)
        return D.get()
    else:
        D = np.einsum('mj,nj->mn',C[:,:p.nbeta],C[:,:p.nbeta],optimize=True)
        return D

def computeDalpha_HF(C,p):

    if(p.gpu):
        D = cp.einsum('mj,nj->mn',C[:,p.nbeta:p.nalpha],C[:,p.nbeta:p.nalpha],optimize=True)
        return D.get()
    else:
        D = np.einsum('mj,nj->mn',C[:,p.nbeta:p.nalpha],C[:,p.nbeta:p.nalpha],optimize=True)
        return D

def computeJK_HF(D,I,b_mnl,p):

    if(p.gpu):
#        if(p.RI):
#            J,K = JKj_RI_GPU(C,b_mnl,p)
#        else:
        J,K = JK_HF_Full_GPU(D,I,p)
    else:
#        if(p.RI):
#            J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
#        else:
        J,K = JK_HF_Full_jit(D,I,p.nbeta,p.nbf,p.nbf5)

    return J,K

def JK_HF_Full_jit(D,I,nbeta,nbf,nbf5):

    J = np.einsum("ls,mnsl->mn",D,I,optimize=True)
    K = np.einsum("nl,mnsl->ms",D,I,optimize=True)

    return J,K

def JK_HF_Full_GPU(D,I,p):

    #denmatj
    J = cp.einsum("ls,mnsl->mn",D,I,optimize=True)
    K = cp.einsum("nl,mnsl->ms",D,I,optimize=True)

    return J.get(),K.get()

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

@njit(parallel=True, cache=True)
def JKalpha_HF_Full_jit(C,I,nbeta,nalpha,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu):
            for i in prange(nbeta,nalpha):
                D[mu][nu] += C[mu][i]*C[nu][i]
            D[nu][mu] = D[mu][nu]

    #hstarj
    J = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for n in prange(m):
            for s in range(nbf):
                for l in range(nbf):
                    J[m][n] += D[s][l]*I[m][n][s][l]
            J[n][m] = J[m][n]

    #hstark
    K = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for s in prange(m):
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

def compute_pqrt(C,I,b_mnl,p):

    if(p.gpu):
        pqrt = pqrt_Full_GPU(C,I,p)
    else:
        pqrt = pqrt_Full_jit(C,I,p.no1,p.nalpha,p.nbf,p.nbf5)

    return pqrt


def pqrt_Full_jit(C,I,no1,nalpha,nbf,nbf5):

    pqrt = np.einsum('mp,nq,mnsl,sr,lt->pqrt',C,C,I,C,C,optimize=True)

    return pqrt

def pqrt_Full_GPU(C,I,p):

    pqrt = cp.einsum('mp,nq,mnsl,sr,lt->pqrt',C,C,I,C,C,optimize=True)

    return pqrt.get()

def JKH_MO_tmp(C,H,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            H_MO, b_MO = Integrals_MO_RI_GPU(C,H,b_mnl,p)
            pass
        else:
            H_MO, I_MO = Integrals_MO_Full_GPU(C,H,I,p)
    else:
        if(p.RI):
            H_MO, b_MO = Integrals_MO_RI_CPU(C,H,b_mnl,p)
        else:
            H_MO, I_MO = Integrals_MO_Full_CPU(C,H,I,p)

    if(p.RI):
        return H_MO,b_MO
    else:
        return H_MO,I_MO

def Integrals_MO_Full_CPU(C,H,I,p):

    H_mat = np.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    I_MO = np.einsum("mp,nq,mnsl,sr,lt->pqrt",C,C[:,:p.nbf5],I,C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)

    return H_mat,I_MO

def Integrals_MO_Full_GPU(C,H,I,p):

    H_mat = cp.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    I_MO = cp.einsum("mp,nq,mnsl,sr,lt->pqrt",C,C[:,:p.nbf5],I,C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)

    return H_mat,I_MO

def Integrals_MO_RI_CPU(C,H,b_mnl,p):

    H_mat = np.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    b_MO = np.einsum("mp,nq,mnl->pql",C,C[:,:p.nbf5],b_mnl,optimize=True)

    return H_mat,b_MO

def Integrals_MO_RI_GPU(C,H,b_mnl,p):

    H_mat = cp.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    b_MO = cp.einsum("mp,nq,mnl->pql",C,C[:,:p.nbf5],b_mnl,optimize=True)

    return H_mat,b_MO

