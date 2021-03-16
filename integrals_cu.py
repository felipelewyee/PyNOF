import numpy as np
import cupy

def computeJK(C,I,p):

    C = cupy.array(C)
    I = cupy.array(I)
    #denmatj
    D = cupy.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)

    #hstarj
    #J = np.einsum('isl,mnsl->imn', D, I, optimize=True)
    J = cupy.tensordot(D, I, axes=([1,2],[2,3]))

    #hstark
    #K = np.einsum('inl,mnsl->ims', D, I, optimize=True)
    K = cupy.tensordot(D, I, axes=([1,2],[1,3]))

    return J.get(),K.get()

def computeJKH_core_MO(C,H,I,p):

    C = cupy.array(C)
    H = cupy.array(H)
    I = cupy.array(I)
    #denmatj
    D = cupy.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)

    #QJMATm
    #J = np.einsum('isl,mnsl->imn', D, I, optimize=True)
    J = cupy.tensordot(D, I, axes=([1,2],[2,3]))
    #J_MO = np.einsum('jmn,imn->ij', D, J, optimize=True)
    J_MO = cupy.tensordot(J, D,axes=((1,2),(1,2)))

    #QKMATm
    #K = np.einsum('inl,mnsl->ims', D, I, optimize=True)
    K = cupy.tensordot(D, I, axes=([1,2],[1,3]))
    #K_MO = np.einsum('jms,ims->ij', D, K, optimize=True)
    K_MO = cupy.tensordot(K, D, axes=([1,2],[1,2]))

    #QHMATm
    #H_core = np.einsum('imn,mn->i', D, H, optimize=True)
    H_core = cupy.tensordot(D, H, axes=([1,2],[0,1]))

    return J_MO.get(),K_MO.get(),H_core.get()

