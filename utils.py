import numpy as np
import integrals

def computeF(J,K,n,H,cj12,ck12,p):

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
    F += np.einsum('ij,jmn->imn',cj12,J,optimize=True)                                                # i = [1,nbf5]
    F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    F -= np.einsum('ij,jmn->imn',ck12,K,optimize=True)                                                # i = [1,nbf5]
    F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    return F

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
    E = E + np.einsum('i,mi,mn,ni',n[:p.nbf5],C[:,:p.nbf5],H,C[:,:p.nbf5],optimize=True)

    return E


    # Definimos una función que calcule la convergencia de los multiplicadores de lagrange

    # In[14]:


def computeLagrangeConvergency(elag):
    # Convergency

    sumdiff = np.sum(np.abs(elag-elag.T))
    maxdiff = np.max(np.abs(elag-elag.T))

    return sumdiff,maxdiff


def ENERGY1r(C,n,H,I,cj12,ck12,p):

    J,K = integrals.computeJK(C,I,p)

    F = computeF(J,K,n,H,cj12,ck12,p)

    elag = computeLagrange(F,C,p)

    E = computeE_elec(H,C,n,elag,p)

    sumdiff,maxdiff = computeLagrangeConvergency(elag)

    return E,elag,sumdiff,maxdiff

def fmiug_scaling(fmiug0,elag,i_ext,nzeros,p):

    #scaling
    fmiug = np.zeros((p.nbf,p.nbf))
    if(i_ext == 0):
        fmiug[:p.noptorb,:p.noptorb] = ((elag[:p.noptorb,:p.noptorb] + elag[:p.noptorb,:p.noptorb].T) / 2)

    else:
        fmiug[:p.noptorb,:p.noptorb] = (elag[:p.noptorb,:p.noptorb] - elag[:p.noptorb,:p.noptorb].T)
        fmiug = np.tril(fmiug,-1) + np.tril(fmiug,-1).T
        for k in range(nzeros+9+1):
            fmiug[(abs(fmiug) > 10**(9-k)) & (abs(fmiug) < 10**(10-k))] *= 0.1
        np.fill_diagonal(fmiug[:p.noptorb,:p.noptorb],fmiug0[:p.noptorb])

    return fmiug

def fmiug_diis(fk,fmiug,idiis,bdiis,cdiis,maxdiff,p):
    if(maxdiff<p.thdiis):

        #restart_diis = False
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

        #    restart_diis=True
        #idiis = idiis + 1
        #if(restart_diis):
        if(p.perdiis):
            idiis = 0

    return fk,fmiug,idiis,bdiis




