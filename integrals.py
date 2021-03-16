import numpy as np
from time import time
from einsumt import einsumt as einsum

def computeJK(C,I,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)

    #hstarj
    J = einsum('isl,mnsl->imn', D, I, optimize=True)
    #J = np.tensordot(D, I, axes=([1,2],[2,3]))

    #hstark
    K = einsum('inl,mnsl->ims', D, I, optimize=True)
    #K = np.tensordot(D, I, axes=([1,2],[1,3]))

    return J,K

def computeJKH_core_MO(C,H,I,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)

    #QJMATm

    J = einsum('isl,mnsl->imn', D, I, optimize=True)
    #J = np.tensordot(D, I, axes=([1,2],[2,3]))
    J_MO = einsum('jmn,imn->ij', D, J, optimize=True)
    #J_MO = np.tensordot(J, D,axes=((1,2),(1,2)))

    #QKMATm
    K = einsum('inl,mnsl->ims', D, I, optimize=True)
    #K = np.tensordot(D, I, axes=([1,2],[1,3]))
    K_MO = einsum('jms,ims->ij', D, K, optimize=True)
    #K_MO = np.tensordot(K, D, axes=([1,2],[1,2]))

    #QHMATm
    H_core = einsum('imn,mn->i', D, H, optimize=True)
    H_core = np.tensordot(D, H, axes=([1,2],[0,1]))

    return J_MO,K_MO,H_core

