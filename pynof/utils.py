import numpy as np
import pynof
from numba import prange,njit,jit
from time import time
import cupy as cp
from scipy.linalg import eigh

def computeF_RC_driver(J,K,n,H,cj12,ck12,p):

    if(p.gpu):
        F = computeF_RC_GPU(J,K,n,H,cj12,ck12,p)
    else:
        F = computeF_RC_CPU(J,K,n,H,cj12,ck12,p)

    return F

def computeF_RC_CPU(J,K,n,H,cj12,ck12,p):

    # Matriz de Fock Generalizada
    F = np.zeros((p.nbf5,p.nbf,p.nbf))

    ini = 0
    if(p.no1>1):
        ini = p.no1

    # nH
    F += np.einsum('i,mn->imn',n,H,optimize=True)        # i = [1,nbf5]

    # nJ
    F[ini:p.nbeta,:,:] += np.einsum('i,imn->imn',n[ini:p.nbeta],J[ini:p.nbeta,:,:],optimize=True)        # i = [ini,nbeta]
    F[p.nalpha:p.nbf5,:,:] += np.einsum('i,imn->imn',n[p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)  # i = [nalpha,nbf5]

    # C^J J
    np.fill_diagonal(cj12[ini:,ini:],0) # Remove diag.
    F += np.einsum('ij,jmn->imn',cj12,J,optimize=True)                                                # i = [1,nbf5]
    #F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    np.fill_diagonal(ck12[ini:,ini:],0) # Remove diag.
    F -= np.einsum('ij,jmn->imn',ck12,K,optimize=True)                                                # i = [1,nbf5]
    #F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    return F

def computeF_RC_GPU(J,K,n,H,cj12,ck12,p):

    # Matriz de Fock Generalizada
    F = cp.zeros((p.nbf5,p.nbf,p.nbf))

    ini = 0
    if(p.no1>1):
        ini = p.no1

    # nH
    F += cp.einsum('i,mn->imn',n,H,optimize=True)        # i = [1,nbf5]

    # nJ
    F[ini:p.nbeta,:,:] += cp.einsum('i,imn->imn',n[ini:p.nbeta],J[ini:p.nbeta,:,:],optimize=True)        # i = [ini,nbeta]
    F[p.nalpha:p.nbf5,:,:] += cp.einsum('i,imn->imn',n[p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)  # i = [nalpha,nbf5]

    # C^J J
    np.fill_diagonal(cj12[ini:,ini:],0) # Remove diag.
    F += cp.einsum('ij,jmn->imn',cj12,J,optimize=True)                                                # i = [1,nbf5]
    #F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    np.fill_diagonal(ck12[ini:,ini:],0) # Remove diag.
    F -= cp.einsum('ij,jmn->imn',ck12,K,optimize=True)                                                # i = [1,nbf5]
    #F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    return F.get()

def computeF_RO_driver(J,K,n,H,cj12,ck12,p):

    if(p.gpu):
        F = computeF_RO_GPU(J,K,n,H,cj12,ck12,p)
    else:
        F = computeF_RO_CPU(J,K,n,H,cj12,ck12,p)

    return F


def computeF_RO_CPU(J,K,n,H,cj12,ck12,p):

    # Matriz de Fock Generalizada
    F = np.zeros((p.nbf5,p.nbf,p.nbf))

    ini = 0
    if(p.no1>1):
        ini = p.no1

    # nH
    F[:p.nbeta,:,:] += np.einsum('i,mn->imn',n[:p.nbeta],H,optimize=True)                      # i = [1,nbf5]
    F[p.nbeta:p.nalpha,:,:] += 0.5*H                                                           # i = [nbeta,nalpha]
    F[p.nalpha:p.nbf5,:,:] += np.einsum('i,mn->imn',n[p.nalpha:p.nbf5],H,optimize=True)        # i = [nalpha,nbf5]

    # nJ
    F[ini:p.nbeta,:,:] += np.einsum('i,imn->imn',n[ini:p.nbeta],J[ini:p.nbeta,:,:],optimize=True)        # i = [ini,nbeta]
    F[p.nalpha:p.nbf5,:,:] += np.einsum('i,imn->imn',n[p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)  # i = [nalpha,nbf5]

    # C^J J
    np.fill_diagonal(cj12[ini:,ini:],0) # Remove diag.
    F[:p.nbeta,:,:] += np.einsum('ij,jmn->imn',cj12[:p.nbeta,:p.nbeta],J[:p.nbeta,:,:],optimize=True)                               # i = [1,nbeta]
    F[:p.nbeta,:,:] += np.einsum('ij,jmn->imn',cj12[:p.nbeta,p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)                               # i = [1,nbeta]
    F[p.nalpha:p.nbf5,:,:] += np.einsum('ij,jmn->imn',cj12[p.nalpha:p.nbf5,:p.nbeta],J[:p.nbeta,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    F[p.nalpha:p.nbf5,:,:] += np.einsum('ij,jmn->imn',cj12[p.nalpha:p.nbf5,p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    #F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    np.fill_diagonal(ck12[ini:,ini:],0) # Remove diag.
    F[:p.nbeta,:,:] -= np.einsum('ij,jmn->imn',ck12[:p.nbeta,:p.nbeta],K[:p.nbeta,:,:],optimize=True)                                                # i = [1,nbeta]
    F[:p.nbeta,:,:] -= np.einsum('ij,jmn->imn',ck12[:p.nbeta,p.nalpha:p.nbf5],K[p.nalpha:p.nbf5,:,:],optimize=True)                                                # i = [1,nbeta]
    F[p.nalpha:p.nbf5,:,:] -= np.einsum('ij,jmn->imn',ck12[p.nalpha:p.nbf5,:p.nbeta],K[:p.nbeta,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    F[p.nalpha:p.nbf5,:,:] -= np.einsum('ij,jmn->imn',ck12[p.nalpha:p.nbf5,p.nalpha:p.nbf5],K[p.nalpha:p.nbf5,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    #F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # SUMij
    F[:p.nbeta,:,:] += np.einsum('i,jmn->imn',n[:p.nbeta],J[p.nbeta:p.nalpha,:,:]-0.5*K[p.nbeta:p.nalpha,:,:])
    F[p.nbeta:p.nalpha,:,:] += 0.5*np.einsum('jmn->mn',J[p.nbeta:p.nalpha,:,:]-K[p.nbeta:p.nalpha,:,:])
    F[p.nbeta:p.nalpha,:,:] -= 0.5*(J[p.nbeta:p.nalpha,:,:]-K[p.nbeta:p.nalpha,:,:]) #Remove diag.
    F[p.nalpha:p.nbf5,:,:] += np.einsum('i,jmn->imn',n[p.nalpha:p.nbf5],J[p.nbeta:p.nalpha,:,:]-0.5*K[p.nbeta:p.nalpha,:,:])
    
    # PRODWROij
    F[p.nbeta:p.nalpha,:,:] += np.einsum('j,jmn->mn',n[:p.nbeta],J[:p.nbeta,:,:]) - 0.5*np.einsum('j,jmn->mn',n[:p.nbeta],K[:p.nbeta,:,:])
    F[p.nbeta:p.nalpha,:,:] += np.einsum('j,jmn->mn',n[p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:]) - 0.5*np.einsum('j,jmn->mn',n[p.nalpha:p.nbf5],K[p.nalpha:p.nbf5,:,:])

    return F

def computeF_RO_GPU(J,K,n,H,cj12,ck12,p):

    # Matriz de Fock Generalizada
    F = cp.zeros((p.nbf5,p.nbf,p.nbf))

    ini = 0
    if(p.no1>1):
        ini = p.no1

    # nH
    F[:p.nbeta,:,:] += cp.einsum('i,mn->imn',n[:p.nbeta],H,optimize=True)                      # i = [1,nbf5]
    F[p.nbeta:p.nalpha,:,:] += 0.5*cp.array(H)                                                           # i = [nbeta,nalpha]
    F[p.nalpha:p.nbf5,:,:] += cp.einsum('i,mn->imn',n[p.nalpha:p.nbf5],H,optimize=True)        # i = [nalpha,nbf5]

    # nJ
    F[ini:p.nbeta,:,:] += cp.einsum('i,imn->imn',n[ini:p.nbeta],J[ini:p.nbeta,:,:],optimize=True)        # i = [ini,nbeta]
    F[p.nalpha:p.nbf5,:,:] += cp.einsum('i,imn->imn',n[p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)  # i = [nalpha,nbf5]

    # C^J J
    np.fill_diagonal(cj12[ini:,ini:],0) # Remove diag.
    F[:p.nbeta,:,:] += cp.einsum('ij,jmn->imn',cj12[:p.nbeta,:p.nbeta],J[:p.nbeta,:,:],optimize=True)                               # i = [1,nbeta]
    F[:p.nbeta,:,:] += cp.einsum('ij,jmn->imn',cj12[:p.nbeta,p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)                               # i = [1,nbeta]
    F[p.nalpha:p.nbf5,:,:] += cp.einsum('ij,jmn->imn',cj12[p.nalpha:p.nbf5,:p.nbeta],J[:p.nbeta,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    F[p.nalpha:p.nbf5,:,:] += cp.einsum('ij,jmn->imn',cj12[p.nalpha:p.nbf5,p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    #F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    np.fill_diagonal(ck12[ini:,ini:],0) # Remove diag.
    F[:p.nbeta,:,:] -= cp.einsum('ij,jmn->imn',ck12[:p.nbeta,:p.nbeta],K[:p.nbeta,:,:],optimize=True)                                                # i = [1,nbeta]
    F[:p.nbeta,:,:] -= cp.einsum('ij,jmn->imn',ck12[:p.nbeta,p.nalpha:p.nbf5],K[p.nalpha:p.nbf5,:,:],optimize=True)                                                # i = [1,nbeta]
    F[p.nalpha:p.nbf5,:,:] -= cp.einsum('ij,jmn->imn',ck12[p.nalpha:p.nbf5,:p.nbeta],K[:p.nbeta,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    F[p.nalpha:p.nbf5,:,:] -= cp.einsum('ij,jmn->imn',ck12[p.nalpha:p.nbf5,p.nalpha:p.nbf5],K[p.nalpha:p.nbf5,:,:],optimize=True)                                      # i = [nalpha,nbf5]
    #F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # SUMij
    F[:p.nbeta,:,:] += cp.einsum('i,jmn->imn',n[:p.nbeta],J[p.nbeta:p.nalpha,:,:]-0.5*K[p.nbeta:p.nalpha,:,:])
    F[p.nbeta:p.nalpha,:,:] += 0.5*cp.einsum('jmn->mn',J[p.nbeta:p.nalpha,:,:]-K[p.nbeta:p.nalpha,:,:])
    F[p.nbeta:p.nalpha,:,:] -= 0.5*cp.array((J[p.nbeta:p.nalpha,:,:]-K[p.nbeta:p.nalpha,:,:])) #Remove diag.
    F[p.nalpha:p.nbf5,:,:] += cp.einsum('i,jmn->imn',n[p.nalpha:p.nbf5],J[p.nbeta:p.nalpha,:,:]-0.5*K[p.nbeta:p.nalpha,:,:])

    # PRODWROij
    F[p.nbeta:p.nalpha,:,:] += cp.einsum('j,jmn->mn',n[:p.nbeta],J[:p.nbeta,:,:]) - 0.5*cp.einsum('j,jmn->mn',n[:p.nbeta],K[:p.nbeta,:,:])
    F[p.nbeta:p.nalpha,:,:] += cp.einsum('j,jmn->mn',n[p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:]) - 0.5*cp.einsum('j,jmn->mn',n[p.nalpha:p.nbf5],K[p.nalpha:p.nbf5,:,:])

    return F.get()

def computeLagrange(F,C,p):

    G = np.einsum('imn,ni->mi',F,C[:,0:p.nbf5],optimize=True)

    #Compute Lagrange multipliers
    elag = np.zeros((p.nbf,p.nbf))
    elag[0:p.noptorb,0:p.nbf5] = np.einsum('mi,mj->ij',C[:,0:p.noptorb],G,optimize=True)[0:p.noptorb,0:p.nbf5]

    return elag


def computeE_elec(H,C,n,elag,p):
    #EELECTRr
    E = 0

    E = E + np.einsum('ii',elag[:p.nbf5,:p.nbf5],optimize=True)
    E = E + np.einsum('i,mi,mn,ni',n[:p.nbeta],C[:,:p.nbeta],H,C[:,:p.nbeta],optimize=True)
    if(not p.HighSpin):
        E = E + np.einsum('i,mi,mn,ni',n[p.nbeta:p.nalpha],C[:,p.nbeta:p.nalpha],H,C[:,p.nbeta:p.nalpha],optimize=True)
    elif(p.HighSpin):
        E = E + 0.5*np.einsum('mi,mn,ni',C[:,p.nbeta:p.nalpha],H,C[:,p.nbeta:p.nalpha],optimize=True)

    E = E + np.einsum('i,mi,mn,ni',n[p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],H,C[:,p.nalpha:p.nbf5],optimize=True)

    return E


def computeLagrangeConvergency(elag):
    # Convergency

    sumdiff = np.sum(np.abs(elag-elag.T))
    maxdiff = np.max(np.abs(elag-elag.T))

    return sumdiff,maxdiff


def ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p):

    J,K = pynof.computeJKj(C,I,b_mnl,p)

    if(p.MSpin==0):
        F = computeF_RC_driver(J,K,n,H,cj12,ck12,p)
    elif(not p.MSpin==0):
        F = computeF_RO_driver(J,K,n,H,cj12,ck12,p)

    elag = computeLagrange(F,C,p)

    E = computeE_elec(H,C,n,elag,p)

    sumdiff,maxdiff = computeLagrangeConvergency(elag)

    return E,elag,sumdiff,maxdiff

def fmiug_scaling(fmiug0,elag,i_ext,nzeros,nbf,noptorb):

    #scaling
    fmiug = np.zeros((nbf,nbf))
    if(i_ext == 0 and fmiug0 is None):
        fmiug[:noptorb,:noptorb] = ((elag[:noptorb,:noptorb] + elag[:noptorb,:noptorb].T) / 2)

    else:
        fmiug[:noptorb,:noptorb] = (elag[:noptorb,:noptorb] - elag[:noptorb,:noptorb].T)
        fmiug = np.tril(fmiug,-1) + np.tril(fmiug,-1).T
        for k in range(nzeros+9+1):
            fmiug[(abs(fmiug) > 10**(9-k)) & (abs(fmiug) < 10**(10-k))] *= 0.1
        np.fill_diagonal(fmiug[:noptorb,:noptorb],fmiug0[:noptorb])

    return fmiug

def fmiug_diis(fk,fmiug,idiis,bdiis,cdiis,maxdiff,p):

    fk[idiis,0:p.noptorb,0:p.noptorb] = fmiug[0:p.noptorb,0:p.noptorb]
    for m in range(idiis+1):
        bdiis[m][idiis] = 0
        for i in range(p.noptorb):
            for j in range(i):
                bdiis[m][idiis] = bdiis[m][idiis] + fk[m][i][j]*fk[idiis][j][i]
        bdiis[idiis][m] = bdiis[m][idiis]
        bdiis[m][idiis+1] = -1
        bdiis[idiis+1][m] = -1
    bdiis[idiis+1][idiis+1] = 0

    if(idiis>=p.ndiis):
        cdiis = np.zeros((idiis+2))
        cdiis[0:idiis+1] = 0
        cdiis[idiis+1] = -1
        x = np.linalg.solve(bdiis[0:idiis+2,0:idiis+2],cdiis[0:idiis+2])

        for i in range(p.noptorb):
            for j in range(i):
                fmiug[i][j] = 0
                for k in range(idiis+1):
                    fmiug[i][j] = fmiug[i][j] + x[k]*fk[k][i][j]
                fmiug[j][i] = fmiug[i][j]

    if(idiis>=p.ndiis):
        if(p.perdiis):
            idiis = 0
    else:
        idiis = idiis + 1

    return fk,fmiug,idiis,bdiis

def check_ortho(C,S,p):

    # Revisa ortonormalidad
    orthonormality = True
    CTSC = np.matmul(np.matmul(np.transpose(C),S),C)
    ortho_deviation = np.abs(CTSC - np.identity(p.nbf))
    if (np.any(ortho_deviation > 10**-6)):
        orthonormality = False
    if not orthonormality:
        print("Orthonormality violations {:d}, Maximum Violation {:f}".format((ortho_deviation > 10**-6).sum(),ortho_deviation.max()))
        print("Trying to orthonormalize")
        C = orthonormalize(C,S)
        C = check_ortho(C,S,p)
    else:
        print("No violations of the orthonormality")
    for j in range(p.nbf):
        #Obtiene el índice del coeficiente con mayor valor absoluto del MO
        idxmaxabsval = 0
        for i in range(p.nbf):
            if(abs(C[i][j])>abs(C[idxmaxabsval][j])):
                 idxmaxabsval = i
    # Ajusta el signo del MO
    sign = np.sign(C[idxmaxabsval][j])
    C[0:p.nbf,j] = sign*C[0:p.nbf,j]

    return C

def orthonormalize(C,S):
    eigval,eigvec = eigh(S) 
    S_12 = np.einsum('ij,j->ij',eigvec,eigval**(-1/2),optimize=True)

    Cnew = np.einsum('ik,kj->ij',S,C,optimize=True)

    Cnew2 = np.einsum('ki,kj->ij',S_12,Cnew)

    for i in range(Cnew2.shape[1]):
        norm = np.einsum('k,k->',Cnew2[:,i],Cnew2[:,i],optimize=True)
        Cnew2[:,i] = Cnew2[:,i]/np.sqrt(norm)
        for j in range(i+1,Cnew2.shape[1]):
            val = -np.einsum('k,k->',Cnew2[:,i],Cnew2[:,j],optimize=True)
            Cnew2[:,j] = Cnew2[:,j] + val*Cnew2[:,i]

    C = np.einsum("ik,kj->ij",S_12,Cnew2,optimize=True)

    return C




