import numpy as np

def computeJK(C,I,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)

    #hstarj
    J = np.einsum('isl,mnsl->imn', D, I, optimize=True)

    #hstark
    K = np.einsum('inl,mnsl->ims', D, I, optimize=True)

    return J,K

def computeJKH_core_MO(C,H,I,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)

    #QJMATm
    J = np.einsum('isl,mnsl->imn', D, I, optimize=True)
    J_MO = np.einsum('jmn,imn->ij', D, J, optimize=True)

    #QKMATm
    K = np.einsum('inl,mnsl->ims', D, I, optimize=True)
    K_MO = np.einsum('jms,ims->ij', D, K, optimize=True)

    #QHMATm
    H_core = np.einsum('imn,mn->i', D, H, optimize=True)

    return J_MO,K_MO,H_core

