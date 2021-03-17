import numpy as np
from time import time
from einsumt import einsumt as einsum
import cupy as cp
import psi4

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
        aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pVDZ")

        orb = wfn.basisset()
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

        Ppq = mints.ao_eri(orb, orb, aux, zero_bas)

        metric = mints.ao_eri(aux, zero_bas, aux, zero_bas)
        metric.power(-0.5, 1.e-14)

        Ppq = np.squeeze(Ppq)
        metric = np.squeeze(metric)

        b_mnl = np.einsum('pqP,PQ->pqQ', Ppq, metric, optimize=True)

    if(p.gpu):
        I = cp.array(I)
        b_mnl = cp.array(b_mnl)

    return S,T,V,H,I,b_mnl


def computeJK(C,I,b_mnl,p):

    if(not p.RI):
        if(not p.gpu):
            #denmatj
            D = einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)

            #hstarj
            #J = np.einsum('isl,mnsl->imn', D, I, optimize=True)
            J = np.tensordot(D, I, axes=([1,2],[2,3]))

            #hstark
            #K = np.einsum('inl,mnsl->ims', D, I, optimize=True)
            K = np.tensordot(D, I, axes=([1,2],[1,3]))
            return J,K
        else:
            #denmatj
            D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)

            #hstarj
            #J = np.einsum('isl,mnsl->imn', D, I, optimize=True)
            J = cp.tensordot(D, I, axes=([1,2],[2,3]))

            #hstark
            #K = np.einsum('inl,mnsl->ims', D, I, optimize=True)
            K = cp.tensordot(D, I, axes=([1,2],[1,3]))
            return J.get(),K.get()

    else:
        if(not p.gpu):
            #denmatj
            #b_qnl = np.einsum('mp,mnl->pnl',C[:,0:p.nbf5],b_mnl, optimize=True)
            b_qnl = np.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
            b_qql = einsum('nq,qnl->ql',C[:,0:p.nbf5],b_qnl, optimize=True)

            #hstarj
            #J = cp.einsum('ql,mnl->qmn', b_qql, b_mnl, optimize=True)
            J = np.tensordot(b_qql, b_mnl, axes=([1],[2]))

            #hstark
            K = einsum('qml,qnl->qmn', b_qnl, b_qnl, optimize=True)
            return J,K
        else:
            C = cp.array(C)
            #denmatj
            #b_qnl = np.einsum('mp,mnl->pnl',C[:,0:p.nbf5],b_mnl, optimize=True)
            b_qnl = cp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
            b_qql = cp.einsum('nq,qnl->ql',C[:,0:p.nbf5],b_qnl, optimize=True)

            #hstarj
            #J = cp.einsum('ql,mnl->qmn', b_qql, b_mnl, optimize=True)
            J = cp.tensordot(b_qql, b_mnl, axes=([1],[2]))

            #hstark
            K = cp.einsum('qml,qnl->qmn', b_qnl, b_qnl, optimize=True)

            return J.get(),K.get()


def computeJKH_core_MO(C,H,I,b_mnl,p):

    if (not p.RI):
        if (not p.gpu):
            #denmatj
            D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    
            #QJMATm
            #J = einsum('isl,mnsl->imn', D, I, optimize=True)
            J = np.tensordot(D, I, axes=([1,2],[2,3]))
            #J_MO = einsum('jmn,imn->ij', D, J, optimize=True)
            J_MO = np.tensordot(J, D,axes=((1,2),(1,2)))
    
            #QKMATm
            #K = einsum('inl,mnsl->ims', D, I, optimize=True)
            K = np.tensordot(D, I, axes=([1,2],[1,3]))
            #K_MO = einsum('jms,ims->ij', D, K, optimize=True)
            K_MO = np.tensordot(K, D, axes=([1,2],[1,2]))
    
            #QHMATm
            #H_core = einsum('imn,mn->i', D, H, optimize=True)
            H_core = np.tensordot(D, H, axes=([1,2],[0,1]))
            return J_MO,K_MO,H_core
        else:
            #denmatj
            D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    
            #QJMATm
            #J = einsum('isl,mnsl->imn', D, I, optimize=True)
            J = cp.tensordot(D, I, axes=([1,2],[2,3]))
            #J_MO = einsum('jmn,imn->ij', D, J, optimize=True)
            J_MO = cp.tensordot(J, D,axes=((1,2),(1,2)))
    
            #QKMATm
            #K = einsum('inl,mnsl->ims', D, I, optimize=True)
            K = cp.tensordot(D, I, axes=([1,2],[1,3]))
            #K_MO = einsum('jms,ims->ij', D, K, optimize=True)
            K_MO = cp.tensordot(K, D, axes=([1,2],[1,2]))
    
            #QHMATm
            #H_core = einsum('imn,mn->i', D, H, optimize=True)
            H_core = np.tensordot(D, H, axes=([1,2],[0,1]))
            return J_MO.get(),K_MO.get(),H_core.get()
   
   
    else:
        if(not p.gpu):
            #denmatj
            D = einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)

            #b_pnl = cp.einsum('mp,mnl->pnl',C[:,0:p.nbf5],b_mnl, optimize=True)
            b_pnl = np.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
            b_pql = einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)

            #QJMATm
            J_MO = einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)

            #QKMATm
            K_MO = einsum('pql,pql->pq', b_pql, b_pql, optimize=True)

            #QHMATm
            #H_core = cp.einsum('imn,mn->i', D, H, optimize=True)
            H_core = np.tensordot(D,H, axes=([1,2],[0,1]))
            return J_MO,K_MO,H_core
        else:
            C = cp.array(C)
            #denmatj
            D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)

            #b_pnl = cp.einsum('mp,mnl->pnl',C[:,0:p.nbf5],b_mnl, optimize=True)
            b_pnl = cp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
            b_pql = cp.einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)

            #QJMATm
            J_MO = cp.einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)

            #QKMATm
            K_MO = cp.einsum('pql,pql->pq', b_pql, b_pql, optimize=True)

            #QHMATm
            #H_core = cp.einsum('imn,mn->i', D, H, optimize=True)
            H_core = cp.tensordot(D,H, axes=([1,2],[0,1]))
            return J_MO.get(),K_MO.get(),H_core.get()


   

