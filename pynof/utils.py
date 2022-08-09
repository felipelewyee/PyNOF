import numpy as np
import pynof
from numba import prange,njit,jit
from time import time
import cupy as cp
from scipy.linalg import eigh,expm,solve
from scipy.optimize import root
import trustregion

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
    F[ini:p.nbeta,:,:] += cp.einsum('i,imn->imn',n[ini:p.nbeta],J[ini:p.nbeta,:,:],optimize=True)
    F[p.nalpha:p.nbf5,:,:] += cp.einsum('i,imn->imn',n[p.nalpha:p.nbf5],J[p.nalpha:p.nbf5,:,:],optimize=True)

    # C^J J
    np.fill_diagonal(cj12[ini:,ini:],0) # Remove diag.
    F += cp.einsum('ij,jmn->imn',cj12,J,optimize=True)

    # -C^K K
    np.fill_diagonal(ck12[ini:,ini:],0) # Remove diag.
    F -= cp.einsum('ij,jmn->imn',ck12,K,optimize=True)

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

    #p.gpu = False
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

@njit
def fmiug_diis(fk,fmiug,idiis,bdiis,cdiis,maxdiff,noptorb,ndiis,perdiis):

    fk[idiis,0:noptorb,0:noptorb] = fmiug[0:noptorb,0:noptorb]
    for m in range(idiis+1):
        bdiis[m][idiis] = 0
        for i in range(noptorb):
            for j in range(i):
                bdiis[m][idiis] = bdiis[m][idiis] + fk[m][i][j]*fk[idiis][j][i]
        bdiis[idiis][m] = bdiis[m][idiis]
        bdiis[m][idiis+1] = -1.
        bdiis[idiis+1][m] = -1.
    bdiis[idiis+1][idiis+1] = 0.

    if(idiis>=ndiis):
        cdiis = np.zeros((idiis+2))
        cdiis[0:idiis+1] = 0.
        cdiis[idiis+1] = -1.
        x = np.linalg.solve(bdiis[0:idiis+2,0:idiis+2],cdiis[0:idiis+2])

        for i in range(noptorb):
            for j in range(i):
                fmiug[i][j] = 0.
                for k in range(idiis+1):
                    fmiug[i][j] = fmiug[i][j] + x[k]*fk[k][i][j]
                fmiug[j][i] = fmiug[i][j]
#        print("idiis", idiis)
    if(idiis>=ndiis and perdiis):
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

def rotate_orbital(y,C,p):

    ynew = np.zeros((p.nbf,p.nbf))

    n = 0
    for i in range(p.nbf5):
        for j in range(i+1,p.nbf):
            ynew[i,j] =  y[n]
            ynew[j,i] = -y[n]
            n += 1

    U = expm(ynew)
    Cnew = np.einsum("mr,rp->mp",C,U,optimize=True)

    return Cnew


def optimize_trust(y0,r,maxr,func,grad,hess,*args):

    ####### Initialization of variables #######
    neg_eig_orig = []
    ynorm_orig = 0

    y = y0
    f = func(y,*args)
    f_old = f
    print("f inicial:",f)
    print(" Step    radius    Neg Eig    Norm    New Norm    Old F(x)    New F(x)    Diff       Pred     Ratio     New r  Status  Grad Norm")
    ####### Optimization #######
    for i in range(10000):

        r_orig = r

        g = grad(y,*args)
        H = hess(y,*args)
        eigval, eigvec = eigh(H)
        neg_eig_orig = eigval[eigval<0]
        minval = min(eigval)

        p = solve(H,-g)
        pnorm_orig = np.linalg.norm(p)
        if(minval<0 or np.linalg.norm(p) > r ):
            p = trustregion.solve(g, H, r, sl=None, su=None, verbose_output=False)

            #if(minval<0):
            #    lambd = -minval + 1e-3
            #elif(np.linalg.norm(p) > r):
            #    lambd = 1e-3
            #def phi(lambd):
            #    B = H + lambd * np.identity(len(H))
            #    p = solve(B,-g)
            #    return 1/r - 1/np.linalg.norm(p)
            #res = root(phi,lambd)
            #lambd = res.x
            #B = H + lambd * np.identity(len(H))
            #p = solve(B,-g)
            #eigval, eigvec = eigh(B)

        f_new = func(y+p,*args)
        diff = f_new - f_old
        pred = np.dot(p,g) + 1/2*np.einsum("m,mn,n->",p,H,p)
        ratio = diff/pred

        if ratio > 0 and diff < 0:
            status = "Accepted"
            y = y + p
            f_old = f
            f = f_new
            if(ratio > 0.75 and np.linalg.norm(p) >= r):
                r = min(1.2*r,maxr)
            elif(ratio < 0.25):
                r = 0.7*r

            g = grad(y+p,*args)
            H = hess(y+p,*args)
            eigval, eigvec = eigh(H)
            print(" {: 3d}     {:3.1e}    {: 3d}      {:3.1e}   {:3.1e}   {:8.3f}    {:8.3f}   {: 3.1e}   {: 3.1e}   {: 6.2f}   {:3.1e}  {} {: 4.1e}".format(i,r_orig,len(neg_eig_orig),pnorm_orig,np.linalg.norm(p),f_old,f_new,diff,pred,ratio,r, status,np.linalg.norm(g)))
            if(np.linalg.norm(g)<1e-4 and len(neg_eig_orig) == 0):
                break
        else:
            status= "Rejected"
            r = 0.7*r
            #r = max(minr,0.7*r)

            print(" {: 3d}     {:3.1e}    {: 3d}      {:3.1e}   {:3.1e}   {:8.3f}    {:8.3f}   {: 3.1e}   {: 3.1e}   {: 6.2f}   {:3.1e}  {} {: 4.1e}".format(i,r_orig,len(neg_eig_orig),pnorm_orig,np.linalg.norm(p),f_old,f_new,diff,pred,ratio,r, status,np.linalg.norm(g)))

    return y,r,f_new,i,True

def optimize_trust2(y0,r,maxr,func,g,H,*args):


    ####### Initialization of variables #######
    neg_eig_orig = []
    ynorm_orig = 0

    y = y0
    f = func(y,*args)
    f_old = f
    print("f inicial:",f)
    print(" Step    radius    Neg Eig    Norm    New Norm    Old F(x)    New F(x)    Diff       Pred     Ratio     New r  Status  Grad Norm")
    ####### Optimization #######
    for i in range(10):

        r_orig = r

        eigval, eigvec = eigh(H)
        neg_eig_orig = eigval[eigval<0]
        minval = min(eigval)

        p = solve(H,-g)
        pnorm_orig = np.linalg.norm(p)
        if(minval<0 or np.linalg.norm(p) > r ):
            p = trustregion.solve(g, H, r, sl=None, su=None, verbose_output=False)
            #if(minval<0):
            #    lambd = -minval + 1e-5
            #elif(np.linalg.norm(p) > r):
            #    lambd = 1e-5
            #def phi(lambd):
            #    B = H + lambd * np.identity(len(H))
            #    p = solve(B,-g)
            #    return 1/r - 1/np.linalg.norm(p)
            #res = root(phi,lambd)
            #lambd = res.x
            #B = H + lambd * np.identity(len(H))
            #p = solve(B,-g)
            #eigval, eigvec = eigh(B)

        f_new = func(y+p,*args)
        diff = f_new - f_old
        pred = np.dot(p,g) + 1/2*np.einsum("m,mn,n->",p,H,p)
        ratio = diff/pred

        if ratio > 0 and diff < 0:
            status = "Accepted"
            y = y + p
            f_old = f
            f = f_new
            if(ratio > 0.75 and np.linalg.norm(p) > 0.8*r ):
                r = min(2*r,maxr)
            elif(ratio < 0.25):
                r = 0.5*r
                if(r< 1e-4):
                    r = 1e-4
                    print("Canceled Step by Small Radius")

            print(" {: 3d}     {:3.1e}    {: 3d}      {:3.1e}   {:3.1e}   {:8.3f}    {:8.3f}   {: 3.1e}   {: 3.1e}   {: 6.2f}   {:3.1e}  {} {: 4.1e}".format(i,r_orig,len(neg_eig_orig),pnorm_orig,np.linalg.norm(p),f_old,f_new,diff,pred,ratio,r, status,np.linalg.norm(g)))
            break
        else:
            status= "Rejected"
            r = 0.5*r
            #r = max(minr,0.7*r)
            if(r< 1e-4):
                r = 1e-4
                print("Canceled Step by Small Radius")
                break

        print(" {: 3d}     {:3.1e}    {: 3d}      {:3.1e}   {:3.1e}   {:8.3f}    {:8.3f}   {: 3.1e}   {: 3.1e}   {: 6.2f}   {:3.1e}  {} {: 4.1e}".format(i,r_orig,len(neg_eig_orig),pnorm_orig,np.linalg.norm(p),f_old,f_new,diff,pred,ratio,r, status,np.linalg.norm(g)))

    return p,r

@njit
def extract_tiu_tensor(t,k):
    dim = len(t)
    var = int(dim*(dim-1)/2)
    t_extracted = np.zeros((var,var))
    i = 0
    for p in range(dim):
        for q in range(p+k,dim):
            j = 0
            for r in range(dim):
                for s in range(r+k,dim):
                    t_extracted[i,j] = t[p,q,r,s]
                    j += 1
            i += 1

    return t_extracted

def check_grads(gamma,C,H,I,b_mnl,p):
    y = np.zeros((int(p.nbf*(p.nbf-1)/2)))

    print("======Gradient Check======")
    print("Grad Analytical")
    grad_a = pynof.calcorbg(y,gamma,C,H,I,b_mnl,p)
    print("Grad Numerical")
    grad_n = pynof.calcorbg_num(y,gamma,C,H,I,b_mnl,p)

    print("Max Diff {:3.1e}".format(np.max(np.abs(grad_a-grad_n))))

def check_hessian(gamma,C,H,I,b_mnl,p):
    y = np.zeros((int(p.nbf*(p.nbf-1)/2)))

    print("======Hessian Check======")
    print("Hess Analytical")
    hess_a = pynof.calcorbh(y,gamma,C,H,I,b_mnl,p)
    print("Hess Numerical")
    hess_n = pynof.calcorbh_num(y,gamma,C,H,I,b_mnl,p)

    print("Max Diff {:3.1e}".format(np.max(np.abs(hess_a-hess_n))))

def check_hessian_eigvals(tol,gamma,C,H,I,b_mnl,p,printeig=False):
        y = np.zeros((int(p.nbf*(p.nbf-1)/2)))
        hess = pynof.calcorbh(y,gamma,C,H,I,b_mnl,p)
        eigval, eigvec = eigh(hess)
        while(tol >= -0.1):
            neg_eig_orig = eigval[eigval<tol]
            if(len(neg_eig_orig)>0):
                print("{} Eigenvalues < {:3.1e} in the Orbital Hessian".format(len(neg_eig_orig),tol))
                if(printeig):
                    print(neg_eig_orig)
            else:
                print("No Eigenvalues < {:3.1e} in the Orbital Hessian".format(tol))
            tol = tol*10

def noise(radius,dim):

    Y = np.random.normal(size=dim)
    Y /= np.linalg.norm(Y)

    U = np.random.uniform()

    epsilon_vec = radius * Y * U**(1/dim)

    return epsilon_vec

@njit
def perturb_gradient(grad,tol):
    dim = grad.shape[0]
    for i in range(dim):
        if(np.abs(grad[i])<tol):
            grad[i] = np.sign(grad[i])*tol

    return grad

def perturb_solution(C,gamma,grad_orb,grad_occ,p):
    grad_orb = pynof.perturb_gradient(grad_orb,p.tol_gorb)
    y = -grad_orb
    C = pynof.rotate_orbital(y,C,p)
    grad_occ = pynof.perturb_gradient(grad_occ,p.tol_gocc)
    gamma += -grad_occ

    return C,gamma
