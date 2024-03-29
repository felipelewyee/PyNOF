import pynof
import numpy as np
from numba import prange,njit
try:
    import cupy as cp
except:
    pass

#CJCKD5
@njit(parallel=True, cache=True)
def CJCKD5(n,no1,ndoc,nsoc,nbeta,nalpha,ndns,ncwo,MSpin):

    # Interpair Electron correlation #
    cj12 = 2*np.outer(n,n)
    ck12 = np.outer(n,n)

    # Intrapair Electron Correlation

    if(MSpin==0 and nsoc>1):
        ck12[nbeta:nalpha,nbeta:nalpha] = 2*np.outer(n[nbeta:nalpha],n[nbeta:nalpha])

    for l in prange(ndoc):
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ncwo*ndoc

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        cj12[ldx,ll:ul:ndoc] = 0
        cj12[ll:ul:ndoc,ldx] = 0

        cj12[ll:ul:ndoc,ll:ul:ndoc] = 0

        ck12[ldx,ll:ul:ndoc] = np.sqrt(n_strong*n_weak)
        ck12[ll:ul:ndoc,ldx] = np.sqrt(n_strong*n_weak)

        ck12[ll:ul:ndoc,ll:ul:ndoc] = -np.sqrt(np.outer(n_weak,n_weak))

    return cj12,ck12

@njit(parallel=True, cache=True)
def der_CJCKD5(n,ista,dn_dgamma,no1,ndoc,nalpha,nbeta,nv,nbf5,ndns,ncwo):

    # Interpair Electron correlation #

    Dcj12r = np.zeros((nbf5,nbf5,nv))
    Dck12r = np.zeros((nbf5,nbf5,nv))
    for k in prange(nv):
        Dcj12r[:,:,k] = 2*np.outer(dn_dgamma[:,k],n)
        Dck12r[:,:,k] = np.outer(dn_dgamma[:,k],n)

    # Intrapair Electron Correlation

    for l in prange(ndoc):
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ncwo*ndoc

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        Dcj12r[ldx,ll:ul:ndoc,:nv] = 0
        Dcj12r[ll:ul:ndoc,ldx,:nv] = 0

        Dcj12r[ll:ul:ndoc,ll:ul:ndoc,:nv] = 0

        a = max(n_strong,10**-15)
        b = n_weak.copy()
        b[b<10**-15] = 10**-15

        for k in range(nv):
            dn_strong = dn_dgamma[ldx,k]
            dn_weak = dn_dgamma[ll:ul:ndoc,k]
            Dck12r[ldx,ll:ul:ndoc,k] = 1/2 * 1/np.sqrt(a) * dn_strong * np.sqrt(n_weak)
            Dck12r[ll:ul:ndoc,ldx,k] = 1/2 * 1/np.sqrt(b) * dn_weak * np.sqrt(n_strong)
            Dck12r[ll:ul:ndoc,ll:ul:ndoc,k] = - 1/2 * np.outer(1/np.sqrt(b)*dn_weak,np.sqrt(n_weak))

    return Dcj12r,Dck12r

#CJCKD7
@njit(parallel=True, cache=True)
def CJCKD7(n,ista,no1,ndoc,nsoc,nbeta,nalpha,ndns,ncwo,MSpin):

    if(ista==0):
        fi = n*(1-n)
        fi[fi<=0] = 0
        fi = np.sqrt(fi)      
    else:
        fi = 2*n*(1-n)

    # Interpair Electron correlation #

    cj12 = 2*np.outer(n,n)
    ck12 = np.outer(n,n) + np.outer(fi,fi)
    
    # Intrapair Electron Correlation

    if(MSpin==0 and nsoc>1):
        ck12[nbeta:nalpha,nbeta:nalpha] = 2*np.outer(n[nbeta:nalpha],n[nbeta:nalpha])

    for l in prange(ndoc):            
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndns + ncwo*ndoc

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        cj12[ldx,ll:ul:ndoc] = 0    
        cj12[ll:ul:ndoc,ldx] = 0    
    
        cj12[ll:ul:ndoc,ll:ul:ndoc] = 0    
        
        ck12[ldx,ll:ul:ndoc] = np.sqrt(n_strong*n_weak)
        ck12[ll:ul:ndoc,ldx] = np.sqrt(n_strong*n_weak)

        ck12[ll:ul:ndoc,ll:ul:ndoc] = -np.sqrt(np.outer(n_weak,n_weak))

    return cj12,ck12        
       
@njit(parallel=True, cache=True)
def der_CJCKD7(n,ista,dn_dgamma,no1,ndoc,nalpha,nbeta,nv,nbf5,ndns,ncwo):

    if(ista==0):
        fi = n*(1-n)
        fi[fi<=0] = 0
        fi = np.sqrt(fi)      
    else:
        fi = 2*n*(1-n)
            
    dfi_dgamma = np.zeros((nbf5,nv))
    for i in prange(no1,nbf5):
        a = max(fi[i],10**-15)
        for k in range(nv):
            if(ista==0):
                dfi_dgamma[i,k] = 1/(2*a)*(1-2*n[i])*dn_dgamma[i][k]
            else:
                dfi_dgamma[i,k] = 2*(1-2*n[i])*dn_dgamma[i][k]
                
   
    # Interpair Electron correlation #

    Dcj12r = np.zeros((nbf5,nbf5,nv))
    Dck12r = np.zeros((nbf5,nbf5,nv))
    for k in prange(nv):
        Dcj12r[:,:,k] = 2*np.outer(dn_dgamma[:,k],n)    
        Dck12r[:,:,k] = np.outer(dn_dgamma[:,k],n) + np.outer(dfi_dgamma[:,k],fi)

    # Intrapair Electron Correlation

    for l in prange(ndoc):            
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndns + ncwo*ndoc
 
        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]
 
        Dcj12r[ldx,ll:ul:ndoc,:nv] = 0
        Dcj12r[ll:ul:ndoc,ldx,:nv] = 0

        Dcj12r[ll:ul:ndoc,ll:ul:ndoc,:nv] = 0   

        a = max(n_strong,10**-15)
        b = n_weak.copy()
        b[b<10**-15] = 10**-15        
        
        for k in range(nv):
            dn_strong = dn_dgamma[ldx,k]
            dn_weak = dn_dgamma[ll:ul:ndoc,k] 
            Dck12r[ldx,ll:ul:ndoc,k] = 1/2 * 1/np.sqrt(a) * dn_strong * np.sqrt(n_weak)
            Dck12r[ll:ul:ndoc,ldx,k] = 1/2 * 1/np.sqrt(b) * dn_weak * np.sqrt(n_strong)
            Dck12r[ll:ul:ndoc,ll:ul:ndoc,k] = - 1/2 * np.outer(1/np.sqrt(b)*dn_weak,np.sqrt(n_weak))
                        
    return Dcj12r,Dck12r

#CJCKD8
@njit(parallel=True, cache=True)
def CJCKD8(n,no1,ndoc,nsoc,nbeta,nalpha,ndns,ncwo,MSpin):

    h_cut = 0.02*np.sqrt(2.0)
    n_d = np.zeros((len(n)))

    for i in prange(ndoc):
        idx = no1 + i
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ndns + ncwo*ndoc

        n_strong = n[idx]
        n_weak = n[ll:ul:ndoc]

        h = 1.0 - n_strong
        coc = h / h_cut
        arg = -coc ** 2
        F = np.exp(arg)  # ! Hd/Hole
        n_d[idx] = n_strong * F
        n_d[ll:ul:ndoc] = n_weak * F  # ROd = RO*Hd/Hole

    n_d12 = np.sqrt(n_d)
    fi = n*(1-n)
    fi[fi<=0] = 0
    fi = np.sqrt(fi)

    # Interpair Electron correlation #

    cj12 = 2*np.outer(n,n)
    ck12 = np.outer(n,n)

    ck12[no1:nbeta,nalpha:] += np.outer(fi[no1:nbeta],fi[nalpha:])
    ck12[nalpha:,no1:nbeta] += np.outer(fi[nalpha:],fi[no1:nbeta])
    ck12[nalpha:,nalpha:] += np.outer(fi[nalpha:],fi[nalpha:])

    # Intrapair Electron Correlation

    if(MSpin==0 and nsoc>0):
        half = np.zeros((nsoc)) + 0.5
        ck12[no1:nbeta,nbeta:nalpha] += 0.5*np.outer(fi[no1:nbeta],half)
        ck12[nbeta:nalpha,no1:nbeta] += 0.5*np.outer(half,fi[no1:nbeta])
        ck12[nbeta:nalpha,nalpha:] += np.outer(half,fi[nalpha:])
        ck12[nalpha:,nbeta:nalpha] += np.outer(fi[nalpha:],half)

    if(MSpin==0 and nsoc>1): #then
        ck12[nbeta:nalpha,nbeta:nalpha] = 0.5

    ck12[no1:nbeta,nalpha:] += np.outer(n_d12[no1:nbeta],n_d12[nalpha:]) - np.outer(n_d[no1:nbeta],n_d[nalpha:])
    ck12[nalpha:,no1:nbeta] += np.outer(n_d12[nalpha:],n_d12[no1:nbeta]) - np.outer(n_d[nalpha:],n_d[no1:nbeta])
    ck12[nalpha:,nalpha:] += -np.outer(n_d12[nalpha:],n_d12[nalpha:]) - np.outer(n_d[nalpha:],n_d[nalpha:])

    for l in prange(ndoc):
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ncwo*ndoc

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        cj12[ldx,ll:ul:ndoc] = 0
        cj12[ll:ul:ndoc,ldx] = 0

        cj12[ll:ul:ndoc,ll:ul:ndoc] = 0

        ck12[ldx,ll:ul:ndoc] = np.sqrt(n_strong*n_weak)
        ck12[ll:ul:ndoc,ldx] = np.sqrt(n_strong*n_weak)

        ck12[ll:ul:ndoc,ll:ul:ndoc] = -np.sqrt(np.outer(n_weak,n_weak))

    return cj12,ck12

@njit(parallel=True, cache=True)
def der_CJCKD8(n,dn_dgamma,no1,ndoc,nalpha,nbeta,nv,nbf5,ndns,ncwo,MSpin,nsoc):

    h_cut = 0.02*np.sqrt(2.0)
    n_d = np.zeros((len(n)))
    dn_d_dgamma = np.zeros((len(n),nv))
    dn_d12_dgamma = np.zeros((len(n),nv))

    for i in prange(ndoc):
        idx = no1 + i
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ncwo*ndoc

        n_strong = n[idx]
        n_weak = n[ll:ul:ndoc]
        dn_strong = dn_dgamma[idx,:]
        dn_weak = dn_dgamma[ll:ul:ndoc,:]

        h_idx = 1.0-n[idx]
        coc = h_idx/h_cut
        arg = -coc**2
        F_idx = np.exp(arg)                # Hd/Hole
        n_d[idx] = n_strong * F_idx
        n_d[ll:ul:ndoc] = n_weak * F_idx      # n_d = RO*Hd/Hole
        dn_d_dgamma[idx,:] = F_idx*dn_strong * (1-n_strong*(-2*coc/h_cut))
        dn_d_dgamma[ll:ul:ndoc,:] = F_idx*(dn_weak - (- 2*coc/h_cut)*np.outer(n_weak,dn_strong))

    n_d12 = np.sqrt(n_d)
    for i in prange(nbf5):
        dn_d12_dgamma[i, :] = 0.5 * dn_d_dgamma[i, :] / max(n_d12[i], 1e-15)

    fi = n*(1-n)
    fi[fi<=0] = 0
    fi = np.sqrt(fi)


    dfi_dgamma = np.zeros((nbf5,nv))
    for i in prange(no1,nbf5):
        a = max(fi[i],10**-15)
        for k in range(nv):
            dfi_dgamma[i,k] = 1/(2*a)*(1-2*n[i])*dn_dgamma[i][k]

    # Interpair Electron correlation #

    Dcj12r = np.zeros((nbf5,nbf5,nv))
    Dck12r = np.zeros((nbf5,nbf5,nv))
    for k in prange(nv):
        Dcj12r[:,:,k] = 2*np.outer(dn_dgamma[:,k],n)
        Dck12r[:,:,k] = np.outer(dn_dgamma[:,k],n)

    for i in prange(nalpha,nbf5):
        Dck12r[no1:nbeta,i,:] += dfi_dgamma[no1:nbeta,:]*fi[i]
        Dck12r[i,no1:nbeta,:] += np.outer(fi[no1:nbeta],dfi_dgamma[i,:])
        Dck12r[i,nalpha:nbf5,:] += np.outer(fi[nalpha:nbf5],dfi_dgamma[i,:])

    if(MSpin==0 and nsoc>0):
        for i in prange(nbeta,nalpha):
            Dck12r[no1:nbeta,i,:] += 0.5*dfi_dgamma[no1:nbeta,:]*0.5
            Dck12r[nalpha:nbf5,i,:] += dfi_dgamma[nalpha:nbf5,:]*0.5

    if(MSpin==0 and nsoc>1):
        Dck12r[nbeta:nalpha, nbeta:nalpha, :] = 0.0

    for i in prange(nalpha,nbf5):
        Dck12r[no1:nbeta,i,:] += dn_d12_dgamma[no1:nbeta,:]*n_d12[i] - dn_d_dgamma[no1:nbeta,:]*n_d[i]
        Dck12r[i,no1:nbeta,:] += np.outer(n_d12[no1:nbeta],dn_d12_dgamma[i,:]) - np.outer(n_d[no1:nbeta],dn_d_dgamma[i,:])
        Dck12r[i,nalpha:nbf5,:] += -np.outer(n_d12[nalpha:nbf5],dn_d12_dgamma[i,:]) - np.outer(n_d[nalpha:nbf5],dn_d_dgamma[i,:])

    # Intrapair Electron Correlation

    for l in prange(ndoc):
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndns + ncwo*ndoc

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        Dcj12r[ldx,ll:ul:ndoc,:nv] = 0
        Dcj12r[ll:ul:ndoc,ldx,:nv] = 0

        Dcj12r[ll:ul:ndoc,ll:ul:ndoc,:nv] = 0

        a = max(n_strong,10**-15)
        b = n_weak.copy()
        b[b<10**-15] = 10**-15

        for k in range(nv):
            dn_strong = dn_dgamma[ldx,k]
            dn_weak = dn_dgamma[ll:ul:ndoc,k]
            Dck12r[ldx,ll:ul:ndoc,k] = 1/2 * 1/np.sqrt(a) * dn_strong * np.sqrt(n_weak)
            Dck12r[ll:ul:ndoc,ldx,k] = 1/2 * 1/np.sqrt(b) * dn_weak * np.sqrt(n_strong)
            Dck12r[ll:ul:ndoc,ll:ul:ndoc,k] = - 1/2 * np.outer(1/np.sqrt(b)*dn_weak,np.sqrt(n_weak))

    return Dcj12r,Dck12r

def compute_2RDM(pp,n):

    norb = len(n)
    # PNOF5
    # Interpair Electron correlation #
    Id = np.identity(norb)

    inter = np.outer(n,n)
    intra = 0*np.outer(n,n)

    # Intrapair Electron Correlation
    for l in prange(pp.ndoc):
        ldx = pp.no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = pp.no1 + pp.ndns + (pp.ndoc - l - 1)
        ul = ll + pp.ncwo*pp.ndoc

        n_strong = n[ldx]
        n_weak = n[ll:ul:pp.ndoc]

        inter[ldx,ldx] = 0
        inter[ldx,ll:ul:pp.ndoc] = 0
        inter[ll:ul:pp.ndoc,ldx] = 0
        inter[ll:ul:pp.ndoc,ll:ul:pp.ndoc] = 0

        intra[ldx,ldx] = np.sqrt(n_strong*n_strong)
        intra[ldx,ll:ul:pp.ndoc] = -np.sqrt(n_strong*n_weak)
        intra[ll:ul:pp.ndoc,ldx] = -np.sqrt(n_strong*n_weak)
        intra[ll:ul:pp.ndoc,ll:ul:pp.ndoc] = np.sqrt(np.outer(n_weak,n_weak))

    for i in range(pp.nbeta,pp.nalpha):
        inter[i,i] = 0

    Daa = np.einsum('pq,pr,qt->pqrt',inter,Id,Id,optimize=True) - np.einsum('pq,pt,qr->pqrt',inter,Id,Id,optimize=True)
    Dab = np.einsum('pq,pr,qt->pqrt',inter,Id,Id,optimize=True) + np.einsum('pr,pq,rt->pqrt',intra,Id,Id,optimize=True)

    # PNOF7
    if(pp.ipnof == 7 or pp.ipnof==8):
        fi = n*(1-n)
        fi[fi<=0] = 0
        fi = np.sqrt(fi)
        Pi_s = np.outer(fi,fi)
        # Intrapair Electron Correlation
        for l in prange(pp.ndoc):
            ldx = pp.no1 + l
            # inicio y fin de los orbitales acoplados a los fuertemente ocupados
            ll = pp.no1 + pp.ndns + (pp.ndoc - l - 1)
            ul = ll + pp.ncwo*pp.ndoc

            Pi_s[ldx,ldx] = 0
            Pi_s[ldx,ll:ul:pp.ndoc] = 0
            Pi_s[ll:ul:pp.ndoc,ldx] = 0
            Pi_s[ll:ul:pp.ndoc,ll:ul:pp.ndoc] = 0
        for i in range(pp.nbeta,pp.nalpha):
            Pi_s[i,i] = 0

        inter2 = 0*Pi_s.copy()
        inter2[pp.nbeta:pp.nalpha,pp.nbeta:pp.nalpha] = Pi_s[pp.nbeta:pp.nalpha,pp.nbeta:pp.nalpha]
        Pi_s[pp.nbeta:pp.nalpha,pp.nbeta:pp.nalpha] = 0

        Dab -= np.einsum('pr,pt,qr->pqrt',inter2,Id,Id,optimize=True)

        if(pp.ipnof==8):
            Pi_s[:pp.nbeta,:pp.nbeta] = 0
            Pi_s[:pp.nbeta,pp.nbeta:pp.nalpha] *= 0.5
            Pi_s[pp.nbeta:pp.nalpha,:pp.nbeta] *= 0.5

        Dab -= np.einsum('pr,pq,rt->pqrt',Pi_s,Id,Id,optimize=True)

        if(pp.ipnof==8):

            h_cut = 0.02*np.sqrt(2.0)
            n_d = np.zeros((len(n)))

            for i in prange(pp.ndoc):
                idx = pp.no1 + i
                # inicio y fin de los orbitales acoplados a los fuertemente ocupados
                ll = pp.no1 + pp.ndns + (pp.ndoc - i - 1)
                ul = ll + pp.ncwo*pp.ndoc

                n_strong = n[idx]
                n_weak = n[ll:ul:pp.ndoc]

                h = 1.0 - n_strong
                coc = h / h_cut
                arg = -coc ** 2
                F = np.exp(arg)  # ! Hd/Hole
                n_d[idx] = n_strong * F
                n_d[ll:ul:pp.ndoc] = n_weak * F  # ROd = RO*Hd/Hole

            n_d12 = np.sqrt(n_d)

            inter = np.outer(n_d12,n_d12) - np.outer(n_d,n_d)
            inter2 = np.outer(n_d12,n_d12) + np.outer(n_d,n_d)
            # Intrapair Electron Correlation
            for l in prange(pp.ndoc):
                ldx = pp.no1 + l
                # inicio y fin de los orbitales acoplados a los fuertemente ocupados
                ll = pp.no1 + pp.ndns + (pp.ndoc - l - 1)
                ul = ll + pp.ncwo*pp.ndoc

                inter[ldx,ldx] = 0
                inter[ldx,ll:ul:pp.ndoc] = 0
                inter[ll:ul:pp.ndoc,ldx] = 0
                inter[ll:ul:pp.ndoc,ll:ul:pp.ndoc] = 0
                inter2[ldx,ldx] = 0
                inter2[ldx,ll:ul:pp.ndoc] = 0
                inter2[ll:ul:pp.ndoc,ldx] = 0
                inter2[ll:ul:pp.ndoc,ll:ul:pp.ndoc] = 0

            inter[pp.nbeta:,pp.nbeta:] = 0
            inter[:pp.nalpha,:pp.nalpha] = 0
            inter2[pp.nbeta:,pp.nbeta:] = 0
            inter2[:pp.nalpha,:] = 0
            inter2[:,:pp.nalpha] = 0

            Pi_d = inter - inter2

            Dab -= np.einsum('pr,pq,rt->pqrt',Pi_d,Id,Id,optimize=True)

    return Daa, Dab

# Creamos un seleccionador de PNOF

def PNOFi_selector(n,p):
    if(p.ipnof==5):
        cj12,ck12 = CJCKD5(n,p.no1,p.ndoc,p.nsoc,p.nbeta,p.nalpha,p.ndns,p.ncwo,p.MSpin)
    if(p.ipnof==7):
        cj12,ck12 = CJCKD7(n,p.ista,p.no1,p.ndoc,p.nsoc,p.nbeta,p.nalpha,p.ndns,p.ncwo,p.MSpin)
    if(p.ipnof==8):
        cj12,ck12 = CJCKD8(n,p.no1,p.ndoc,p.nsoc,p.nbeta,p.nalpha,p.ndns,p.ncwo,p.MSpin)

    return cj12,ck12

def der_PNOFi_selector(n,dn_dgamma,p):
    if(p.ipnof==5):
        Dcj12r,Dck12r = der_CJCKD5(n,p.ista,dn_dgamma,p.no1,p.ndoc,p.nalpha,p.nbeta,p.nv,p.nbf5,p.ndns,p.ncwo)
    if(p.ipnof==7):
        Dcj12r,Dck12r = der_CJCKD7(n,p.ista,dn_dgamma,p.no1,p.ndoc,p.nalpha,p.nbeta,p.nv,p.nbf5,p.ndns,p.ncwo)
    if(p.ipnof==8):
        Dcj12r,Dck12r = der_CJCKD8(n,dn_dgamma,p.no1,p.ndoc,p.nalpha,p.nbeta,p.nv,p.nbf5,p.ndns,p.ncwo,p.MSpin,p.nsoc)
        
    return Dcj12r,Dck12r

@njit(cache=True)
def ocupacion(gamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo,HighSpin,occ_method):
    if occ_method == "Trigonometric":
        n,dn_dgamma = ocupacion_trigonometric(gamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo,HighSpin)
    elif occ_method == "Softmax":
        n,dn_dgamma = ocupacion_softmax(gamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo,HighSpin)
    return n,dn_dgamma

@njit(cache=True)
def ocupacion_trigonometric(gamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo,HighSpin):

    n = np.zeros((nbf5))
    dn_dgamma = np.zeros((nbf5,nv))
    dni_dgammai = np.zeros((nbf5))

    n[0:no1] = 1                                              # [1,no1]

    n[no1:no1+ndoc] = 1/2 * (1 + np.cos(gamma[:ndoc])**2)     # (no1,no1+ndoc]
    dni_dgammai[no1:no1+ndoc] = - 1/2 * np.sin(2*gamma[:ndoc])
    for i in range(ndoc):
        # dn_g/dgamma_g
        dn_dgamma[no1+i,i] = dni_dgammai[no1+i]

    if(not HighSpin):
        n[no1+ndoc:no1+ndns] = 0.5   # (no1+ndoc,no1+ndns]
    elif(HighSpin):
        n[no1+ndoc:no1+ndns] = 1.0   # (no1+ndoc,no1+ndns]

    h = 1 - n
    for i in range(ndoc):
        ll_n = no1 + ndns + (ndoc - i - 1)
        ul_n = ll_n + ncwo*ndoc
        n_pi = n[ll_n:ul_n:ndoc]
        ll_gamma = ndoc + (ndoc - i - 1)
        ul_gamma = ll_gamma + (ncwo-1)*ndoc
        gamma_pi = gamma[ll_gamma:ul_gamma:ndoc]

        # n_pi
        n_pi[:] = h[no1+i]
        for kw in range(ncwo-1):
            n_pi[kw] *= np.sin(gamma_pi[kw])**2
            n_pi[kw+1:] *= np.cos(gamma_pi[kw])**2

        # dn_pi/dgamma_g
        dn_pi_dgamma_g = dn_dgamma[ll_n:ul_n:ndoc,i]
        dn_pi_dgamma_g[:] = -dni_dgammai[no1+i]
        for kw in range(ncwo-1):
            dn_pi_dgamma_g[kw] *= np.sin(gamma_pi[kw])**2
            dn_pi_dgamma_g[kw+1:] *= np.cos(gamma_pi[kw])**2

        # dn_pi/dgamma_pj (j<i)
        dn_pi_dgamma_pj = dn_dgamma[ll_n:ul_n:ndoc,ll_gamma:ul_gamma:ndoc]
        for jw in range(ncwo-1):
            dn_pi_dgamma_pj[jw+1:,jw] = n[no1+i] - 1
            for kw in range(jw):
                dn_pi_dgamma_pj[jw+1:,jw] *= np.cos(gamma_pi[kw])**2
            dn_pi_dgamma_pj[jw+1:,jw] *= np.sin(2*gamma_pi[jw])
            for kw in range(jw+1,ncwo-1):
                dn_pi_dgamma_pj[kw,jw] *= np.sin(gamma_pi[kw])**2
                dn_pi_dgamma_pj[kw+1:,jw] *= np.cos(gamma_pi[kw])**2

        # dn_pi/dgamma_i
        for jw in range(ncwo-1):
            dn_pi_dgamma_pj[jw,jw] = 1 - n[no1+i]
        for kw in range(ncwo-1):
            dn_pi_dgamma_pj[kw,kw] *= np.sin(2*gamma_pi[kw])
            for lw in range(kw+1,ncwo-1):
                dn_pi_dgamma_pj[lw,lw] *= np.cos(gamma_pi[kw])**2

    return n,dn_dgamma

@njit(cache=True)
def ocupacion_softmax(x,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo,HighSpin):

    n = np.zeros((nbf5))
    dn_dx = np.zeros((nbf5,nv))

    if(not HighSpin):
         n[no1+ndoc:no1+ndns] = 0.5   # (no1+ndoc,no1+ndns]
    elif(HighSpin):
         n[no1+ndoc:no1+ndns] = 1.0   # (no1+ndoc,no1+ndns]

    exp_x = np.exp(x)

    for i in range(ndoc):
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ncwo*ndoc
        n_pi = n[ll:ul:ndoc]

        ll_x = ll - ndns + ndoc - no1
        ul_x = ll_x + ncwo*ndoc
        dn_pi_dx_pi = dn_dx[ll:ul:ndoc,ll_x:ul_x:ndoc]
        dn_g_dx_pi = dn_dx[i,ll_x:ul_x:ndoc]
        dn_pi_dx_g = dn_dx[ll:ul:ndoc,i]

        exp_x_pi = exp_x[ll_x:ul_x:ndoc]

        sum_exp = exp_x[i] + np.sum(exp_x_pi)

        n[i] = exp_x[i]/sum_exp
        n_pi[:] = exp_x_pi/sum_exp

        dn_pi_dx_pi[:,:] = -np.outer(exp_x_pi,exp_x_pi)/sum_exp**2

        dn_g_dx_pi[:] = -exp_x_pi*exp_x[i]/sum_exp**2
        dn_pi_dx_g[:] = -exp_x_pi*exp_x[i]/sum_exp**2

        dn_dx[i,i] = exp_x[i]*(sum_exp-exp_x[i])/sum_exp**2

        for j in range(ncwo):
            dn_pi_dx_pi[j,j] = exp_x_pi[j]*(sum_exp-exp_x_pi[j])/sum_exp**2

    return n,dn_dx

def calce(n,cj12,ck12,J_MO,K_MO,H_core,p):

    E = 0

    if(p.MSpin==0):

        # 2H + J
        E = E + 2*np.einsum('i,i',n,H_core,optimize=True)
        E = E + np.einsum('i,i',n[:p.nbeta],np.diagonal(J_MO)[:p.nbeta],optimize=True)
        E = E + np.einsum('i,i',n[p.nalpha:p.nbf5],np.diagonal(J_MO)[p.nalpha:p.nbf5],optimize=True)
    
        #C^J JMO
        np.fill_diagonal(cj12,0) # Remove diag.
        E = E + np.einsum('ij,ji->',cj12,J_MO,optimize=True) # sum_ij
    
        #C^K KMO
        np.fill_diagonal(ck12,0) # Remove diag.
        E = E - np.einsum('ij,ji->',ck12,K_MO,optimize=True) # sum_ij
    
    elif(not p.MSpin==0):
        E = 0

        # 2H + J
        E = E + np.einsum('i,i',n[:p.nbeta],2*H_core[:p.nbeta]+np.diagonal(J_MO)[:p.nbeta],optimize=True) # [0,Nbeta]
        E = E + np.einsum('i,i',n[p.nbeta:p.nalpha],2*H_core[p.nbeta:p.nalpha],optimize=True)               # (Nbeta,Nalpha]
        E = E + np.einsum('i,i',n[p.nalpha:p.nbf5],2*H_core[p.nalpha:p.nbf5]+np.diagonal(J_MO)[p.nalpha:p.nbf5],optimize=True) # (Nalpha,Nbf5)

        #C^J JMO
        np.fill_diagonal(cj12,0) # Remove diag.
        E = E + np.einsum('ij,ji->',cj12[:p.nbeta,:p.nbeta],J_MO[:p.nbeta,:p.nbeta],optimize=True) # sum_ij
        E = E + np.einsum('ij,ji->',cj12[:p.nbeta,p.nalpha:p.nbf5],J_MO[p.nalpha:p.nbf5,:p.nbeta],optimize=True) # sum_ij
        E = E + np.einsum('ij,ji->',cj12[p.nalpha:p.nbf5,:p.nbeta],J_MO[:p.nbeta,p.nalpha:p.nbf5],optimize=True) # sum_ij
        E = E + np.einsum('ij,ji->',cj12[p.nalpha:p.nbf5,p.nalpha:p.nbf5],J_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True) # sum_ij

        #C^K KMO
        np.fill_diagonal(ck12,0) # Remove diag.
        E = E - np.einsum('ij,ji->',ck12[:p.nbeta,:p.nbeta],K_MO[:p.nbeta,:p.nbeta],optimize=True) # sum_ij
        E = E - np.einsum('ij,ji->',ck12[:p.nbeta,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,:p.nbeta],optimize=True) # sum_ij
        E = E - np.einsum('ij,ji->',ck12[p.nalpha:p.nbf5,:p.nbeta],K_MO[:p.nbeta,p.nalpha:p.nbf5],optimize=True) # sum_ij
        E = E - np.einsum('ij,ji->',ck12[p.nalpha:p.nbf5,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True) # sum_ij

        #n JMO
        E = E + 2*np.einsum('i,ji->',n[:p.nbeta],J_MO[p.nbeta:p.nalpha,:p.nbeta],optimize=True) # sum_ij
        E = E + 2*np.einsum('i,ji->',n[p.nalpha:p.nbf5],J_MO[p.nbeta:p.nalpha,p.nalpha:p.nbf5],optimize=True) # sum_ij
        E = E + 0.5*(np.einsum('i,ji->',n[p.nbeta:p.nalpha],J_MO[p.nbeta:p.nalpha,p.nbeta:p.nalpha],optimize=True) - np.einsum('i,ii->',n[p.nbeta:p.nalpha],J_MO[p.nbeta:p.nalpha,p.nbeta:p.nalpha],optimize=True))

        #n KMO
        E = E - np.einsum('i,ji->',n[:p.nbeta],K_MO[p.nbeta:p.nalpha,:p.nbeta],optimize=True) # sum_ij
        E = E - np.einsum('i,ji->',n[p.nalpha:p.nbf5],K_MO[p.nbeta:p.nalpha,p.nalpha:p.nbf5],optimize=True) # sum_ij
        E = E - 0.5*(np.einsum('i,ji->',n[p.nbeta:p.nalpha],K_MO[p.nbeta:p.nalpha,p.nbeta:p.nalpha],optimize=True) + np.einsum('i,ii->',n[p.nbeta:p.nalpha],K_MO[p.nbeta:p.nalpha,p.nbeta:p.nalpha],optimize=True))

    return E

def calcocce(gamma,J_MO,K_MO,H_core,p):

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = PNOFi_selector(n,p)

    E = calce(n,cj12,ck12,J_MO,K_MO,H_core,p)

    return E

def calcoccg(gamma,J_MO,K_MO,H_core,p):

    grad = np.zeros((p.nv))

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    Dcj12r,Dck12r = der_PNOFi_selector(n,dn_dgamma,p)

    if(p.MSpin==0):

        # dn_dgamma (2H+J)
        grad += np.einsum('ik,i->k',dn_dgamma,2*H_core+np.diagonal(J_MO),optimize=True)

        # 2 dCJ_dgamma J_MO
        diag = np.diag_indices(p.nbf5)
        Dcj12r[diag] = 0
        grad += 2*np.einsum('ijk,ji->k',Dcj12r,J_MO,optimize=True)
    
        # -2 dCK_dgamma K_MO
        diag = np.diag_indices(p.nbf5)
        Dck12r[diag] = 0
        grad -= 2*np.einsum('ijk,ji->k',Dck12r,K_MO,optimize=True)

    elif(not p.MSpin==0):
    
        # dn_dgamma (2H+J)
        grad += np.einsum('ik,i->k',dn_dgamma[p.no1:p.nbeta,:p.nv],2*H_core[p.no1:p.nbeta]+np.diagonal(J_MO)[p.no1:p.nbeta],optimize=True) # [0,Nbeta]
        grad += np.einsum('ik,i->k',dn_dgamma[p.nalpha:p.nbf5,:p.nv],2*H_core[p.nalpha:p.nbf5]+np.diagonal(J_MO)[p.nalpha:p.nbf5],optimize=True) # [Nalpha,Nbf5]

        # 2 dCJ_dgamma J_MO
        Dcj12r[np.diag_indices(p.nbf5)] = 0
        grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.no1:p.nbeta,:p.nbeta,:p.nv],J_MO[:p.nbeta,p.no1:p.nbeta],optimize=True)
        grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.no1:p.nbeta,p.nalpha:p.nbf5,:p.nv],J_MO[p.nalpha:p.nbf5,p.no1:p.nbeta],optimize=True)
        grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.nalpha:p.nbf5,:p.nbeta,:p.nv],J_MO[:p.nbeta,p.nalpha:p.nbf5],optimize=True)
        grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:p.nv],J_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)

        # -2 dCK_dgamma K_MO
        Dck12r[np.diag_indices(p.nbf5)] = 0
        grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.no1:p.nbeta,:p.nbeta,:p.nv],K_MO[:p.nbeta,p.no1:p.nbeta],optimize=True)
        grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.no1:p.nbeta,p.nalpha:p.nbf5,:p.nv],K_MO[p.nalpha:p.nbf5,p.no1:p.nbeta],optimize=True)
        grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.nalpha:p.nbf5,:p.nbeta,:p.nv],K_MO[:p.nbeta,p.nalpha:p.nbf5],optimize=True)
        grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:p.nv],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)

        # 2 dn_dgamma J_MO
        grad += 2*np.einsum('jk,ij->k',dn_dgamma[p.no1:p.nbeta,:p.nv],J_MO[p.nbeta:p.nalpha,p.no1:p.nbeta],optimize=True)
        grad += 2*np.einsum('jk,ij->k',dn_dgamma[p.nalpha:p.nbf5,:p.nv],J_MO[p.nbeta:p.nalpha,p.nalpha:p.nbf5],optimize=True)

        # - dn_dgamma K_MO
        grad -= np.einsum('jk,ij->k',dn_dgamma[p.no1:p.nbeta,:p.nv],K_MO[p.nbeta:p.nalpha,p.no1:p.nbeta],optimize=True)
        grad -= np.einsum('jk,ij->k',dn_dgamma[p.nalpha:p.nbf5,:p.nv],K_MO[p.nbeta:p.nalpha,p.nalpha:p.nbf5],optimize=True)

    return grad

def calcorbe(y,n,cj12,ck12,C,H,I,b_mnl,p):

    Cnew = pynof.rotate_orbital(y,C,p)

    J_MO,K_MO,H_core = pynof.computeJKH_MO(Cnew,H,I,b_mnl,p)

    E = calce(n,cj12,ck12,J_MO,K_MO,H_core,p)

    return E

def calcorbg(y,n,cj12,ck12,C,H,I,b_mnl,p):

    Cnew = pynof.rotate_orbital(y,C,p)

    elag,Hmat = pynof.computeLagrange2(n,cj12,ck12,Cnew,H,I,b_mnl,p)

    grad = 4*(elag - elag.T)

    grads = np.zeros((int(p.nbf*(p.nbf-1)/2) - int(p.no0*(p.no0-1)/2)))

    k = 0
    for i in range(p.nbf5):
        for j in range(i+1,p.nbf):
            grads[k] = grad[i,j]
            k += 1
    grad = grads

    return grad

def calcorbeg(y,n,cj12,ck12,C,H,I,b_mnl,p):

    Cnew = pynof.rotate_orbital(y,C,p)

    elag,Hmat = pynof.computeLagrange2(n,cj12,ck12,Cnew,H,I,b_mnl,p)

    grad = 4*(elag - elag.T)

    grads = np.zeros((int(p.nbf*(p.nbf-1)/2) - int(p.no0*(p.no0-1)/2)))

    k = 0
    for i in range(p.nbf5):
        for j in range(i+1,p.nbf):
            grads[k] = grad[i,j]
            k += 1
    grad = grads

    E = np.einsum('ii',elag[:p.nbf5,:p.nbf5],optimize=True)
    E = E + np.einsum('i,ii',n[:p.nbf5],Hmat[:p.nbf5,:p.nbf5],optimize=True)

    return E,grad
