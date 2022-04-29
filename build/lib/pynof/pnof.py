import pynof
import numpy as np
import numdifftools as nd
from numba import prange,njit,jit
import numdifftools.nd_statsmodels as nds
from time import time as time
import cupy as cp

#CJCKD5
@njit(parallel=True)
def CJCKD5(n,ista,no1,ndoc,nsoc,nbeta,nalpha,ndns,ncwo,MSpin):

    # Interpair Electron correlation #
    cj12 = 2*np.outer(n,n)
    ck12 = np.outer(n,n)

    # Intrapair Electron Correlation

    if(MSpin==0 and nsoc>1):
        ck12[nbeta:nalpha,nbeta:nalpha] = 2*np.outer(n[nbeta:nalpha],n[nbeta:nalpha])

    for l in prange(ndoc):
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + ncwo*(ndoc - l - 1)
        ul = no1 + ndns + ncwo*(ndoc - l)

        cj12[ldx,ll:ul] = 0
        cj12[ll:ul,ldx] = 0

        cj12[ll:ul,ll:ul] = 0

        ck12[ldx,ll:ul] = np.sqrt(n[ldx]*n[ll:ul])
        ck12[ll:ul,ldx] = np.sqrt(n[ldx]*n[ll:ul])

        ck12[ll:ul,ll:ul] = -np.sqrt(np.outer(n[ll:ul],n[ll:ul]))

    return cj12,ck12

def der_CJCKD5(n,dn_dgamma,p):

    # Interpair Electron correlation #

    Dcj12r = 2*np.einsum('ik,j->ijk',dn_dgamma,n)    
    Dck12r = np.einsum('ik,j->ijk',dn_dgamma,n)    

    # Interpair Electron Correlation
    
    for l in range(p.ndoc):            
        ldx = p.no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc-l-1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc-l)

        Dcj12r[ldx,ll:ul,:p.nv] = 0
        Dcj12r[ll:ul,ldx,:p.nv] = 0

        Dcj12r[ll:ul,ll:ul,:p.nv] = 0   
        
        a = n[ldx] 
        a = max(a,10**-15)
        b = n[ll:ul]
        b[b<10**-15] = 10**-15        
        
        Dck12r[ldx,ll:ul,:p.nv] = 1/2 * 1/np.sqrt(a) * np.einsum('j,i->ij',dn_dgamma[ldx,:p.nv],np.sqrt(n[ll:ul]))
        Dck12r[ll:ul,ldx,:p.nv] = 1/2 * np.einsum('i,ij->ij', 1/np.sqrt(b),dn_dgamma[ll:ul,:p.nv])*np.sqrt(n[ldx])
        
        for k in range(p.nv):
            Dck12r[ll:ul,ll:ul,k] = - 1/2 * np.einsum('i,i,j->ij',1/np.sqrt(b),dn_dgamma[ll:ul,k],np.sqrt(n[ll:ul]))
                        
    return Dcj12r,Dck12r


#CJCKD7
@njit(parallel=True)
def CJCKD7(n,ista,no1,ndoc,nsoc,nbeta,nalpha,ndns,ncwo,MSpin):

    if(ista==0):
        fi = n*(1-n)
        fi[fi<=0] = 0
        fi = np.sqrt(fi)      
    else:
        fi = 2*n*(1-n)

    # Interpair Electron correlation #

    #cj12 = 2*np.einsum('i,j->ij',n,n)
    #ck12 = np.einsum('i,j->ij',n,n) + np.einsum('i,j->ij',fi,fi)
    cj12 = 2*np.outer(n,n)
    ck12 = np.outer(n,n) + np.outer(fi,fi)
    
    # Intrapair Electron Correlation

    if(MSpin==0 and nsoc>1):
        ck12[nbeta:nalpha,nbeta:nalpha] = 2*np.outer(n[nbeta:nalpha],n[nbeta:nalpha])

    for l in prange(ndoc):            
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + ncwo*(ndoc - l - 1)
        ul = no1 + ndns + ncwo*(ndoc - l)

        cj12[ldx,ll:ul] = 0    
        cj12[ll:ul,ldx] = 0    
    
        cj12[ll:ul,ll:ul] = 0    
        
        ck12[ldx,ll:ul] = np.sqrt(n[ldx]*n[ll:ul])
        ck12[ll:ul,ldx] = np.sqrt(n[ldx]*n[ll:ul])

        ck12[ll:ul,ll:ul] = -np.sqrt(np.outer(n[ll:ul],n[ll:ul]))

    return cj12,ck12        
       
@njit(parallel=True)
def der_CJCKD7(n,ista,dn_dgamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo):

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
    #Dcj12r = 2*np.einsum('ik,j->ijk',dn_dgamma,n)    
    #Dck12r = np.einsum('ik,j->ijk',dn_dgamma,n) + np.einsum('ik,j->ijk',dfi_dgamma,fi)    

    # Intrapair Electron Correlation

    for l in prange(ndoc):            
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + ncwo*(ndoc - l - 1)
        ul = no1 + ndns + ncwo*(ndoc - l)

        Dcj12r[ldx,ll:ul,:nv] = 0
        Dcj12r[ll:ul,ldx,:nv] = 0

        Dcj12r[ll:ul,ll:ul,:nv] = 0   

        a = max(n[ldx],10**-15)
        b = n[ll:ul]
        b[b<10**-15] = 10**-15        
        
        for k in range(nv):
            Dck12r[ldx,ll:ul,k] = 1/2 * 1/np.sqrt(a) * dn_dgamma[ldx,k] * np.sqrt(n[ll:ul])
            Dck12r[ll:ul,ldx,k] = 1/2 * 1/np.sqrt(b) * dn_dgamma[ll:ul,k] * np.sqrt(n[ldx])
            Dck12r[ll:ul,ll:ul,k] = - 1/2 * np.outer(1/np.sqrt(b)*dn_dgamma[ll:ul,k],np.sqrt(n[ll:ul]))
        #Dck12r[ldx,ll:ul,:nv] = 1/2 * 1/np.sqrt(a) * np.einsum('j,i->ij',dn_dgamma[ldx,:nv],np.sqrt(n[ll:ul]))
        #Dck12r[ll:ul,ldx,:nv] = 1/2 * np.einsum('i,ij->ij', 1/np.sqrt(b),dn_dgamma[ll:ul,:nv])*np.sqrt(n[ldx])
        
        #for k in range(nv):
        #    Dck12r[ll:ul,ll:ul,k] = - 1/2 * np.einsum('i,i,j->ij',1/np.sqrt(b),dn_dgamma[ll:ul,k],np.sqrt(n[ll:ul]))
                        
    return Dcj12r,Dck12r

#CJCKD8
@njit(parallel=True)
def CJCKD8(n,ista,no1,ndoc,nsoc,nbeta,nalpha,ndns,ncwo,MSpin,lamb):

    nbf5 = len(n)
    delta = np.zeros((nbf5,nbf5))
    for i in range(nbf5):
        for j in range(nbf5):
            delta[i,j] = lamb*min(n[i]*n[j],(1-n[i])*(1-n[j]))
    pi = np.zeros((nbf5,nbf5))
    for i in range(nbf5):
        for j in range(nbf5):
            pi[i,j] = np.sqrt((n[i]*(1-n[j])+delta[i,j]) * (n[j]*(1-n[i])+delta[j,i]))

    # Interpair Electron correlation #

    #cj12 = 2*np.einsum('i,j->ij',n,n)
    #ck12 = np.einsum('i,j->ij',n,n) + np.einsum('i,j->ij',fi,fi)
    cj12 = 2*np.outer(n,n) - delta
    ck12 = np.outer(n,n) - delta + pi

    # Intrapair Electron Correlation

    if(MSpin==0 and nsoc>1):
        ck12[nbeta:nalpha,nbeta:nalpha] = 2*np.outer(n[nbeta:nalpha],n[nbeta:nalpha])

    for l in range(ndoc):
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + ncwo*(ndoc - l - 1)
        ul = no1 + ndns + ncwo*(ndoc - l)

        cj12[ldx,ll:ul] = 0
        cj12[ll:ul,ldx] = 0

        cj12[ll:ul,ll:ul] = 0

        ck12[ldx,ll:ul] = np.sqrt(n[ldx]*n[ll:ul])
        ck12[ll:ul,ldx] = np.sqrt(n[ldx]*n[ll:ul])

        ck12[ll:ul,ll:ul] = -np.sqrt(np.outer(n[ll:ul],n[ll:ul]))

    return cj12,ck12


# Creamos un seleccionador de PNOF

def PNOFi_selector(n,p):
    if(p.ipnof==5):
        cj12,ck12 = CJCKD5(n,p.ista,p.no1,p.ndoc,p.nsoc,p.nbeta,p.nalpha,p.ndns,p.ncwo,p.MSpin)
    if(p.ipnof==7):
        cj12,ck12 = CJCKD7(n,p.ista,p.no1,p.ndoc,p.nsoc,p.nbeta,p.nalpha,p.ndns,p.ncwo,p.MSpin)
    if(p.ipnof==8):
        cj12,ck12 = CJCKD8(n,p.ista,p.no1,p.ndoc,p.nsoc,p.nbeta,p.nalpha,p.ndns,p.ncwo,p.MSpin,p.lamb)
        
    return cj12,ck12

def der_PNOFi_selector(n,dn_dgamma,p):
    if(p.ipnof==5):
        Dcj12r,Dck12r = der_CJCKD5(n,dn_dgamma,p)
    if(p.ipnof==7):
        Dcj12r,Dck12r = der_CJCKD7(n,p.ista,dn_dgamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo)
        
    return Dcj12r,Dck12r

@njit
def ocupacion(gamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo,HighSpin):

    n = np.zeros((nbf5))
    dni_dgammai = np.zeros((nbf5))

    n[0:no1] = 1                                              # [1,no1]

    n[no1:no1+ndoc] = 1/2 * (1 + np.cos(gamma[:ndoc])**2)     # (no1,no1+ndoc]
    dni_dgammai[no1:no1+ndoc] = - 1/2 * np.sin(2*gamma[:ndoc])
    
    if(not HighSpin):
        n[no1+ndoc:no1+ndns] = 0.5   # (no1+ndoc,no1+ndns]
    elif(HighSpin):
        n[no1+ndoc:no1+ndns] = 1.0   # (no1+ndoc,no1+ndns]


    if(ncwo==1):
        dn_dgamma = np.zeros((nbf5,nv))

        for i in range(ndoc):
            dn_dgamma[no1+i][i] = dni_dgammai[no1+i]
            #cwo
            icf = nalpha + ndoc - i - 1
            n[icf] = 1/2*np.sin(gamma[i])**2
            dni_dgammai[icf]  = 1/2*np.sin(2*gamma[i])
            dn_dgamma[icf][i] = dni_dgammai[icf]
    else:
        dn_dgamma = np.zeros((nbf5,nv))
        h = 1 - n
        for i in range(ndoc):
            ll = no1 + ndns + ncwo*(ndoc - i - 1)
            ul = no1 + ndns + ncwo*(ndoc - i)
            n[ll:ul] = h[no1+i]
            for iw in range(ncwo-1):
                n[ll+iw] *= np.sin(gamma[ndoc+(ncwo-1)*i+iw])**2
                n[ll+iw+1:ul] *= np.cos(gamma[ndoc+(ncwo-1)*i+iw])**2

        for i in range(ndoc):
            # dn_g/dgamma_g
            dn_dgamma[no1+i][i] = dni_dgammai[no1+i]

            # dn_pi/dgamma_g
            ll = no1 + ndns + ncwo*(ndoc - i - 1)
            ul = no1 + ndns + ncwo*(ndoc - i)
            dn_dgamma[ll:ul,i] = -dni_dgammai[no1+i]
            for iw in range(ncwo-1):
                dn_dgamma[ll+iw][i] *= np.sin(gamma[ndoc+(ncwo-1)*i+iw])**2
                dn_dgamma[ll+iw+1:ul,i] *= np.cos(gamma[ndoc+(ncwo-1)*i+iw])**2

            # dn_pi/dgamma_pj (j<i)
            for iw in range(ncwo-1):
                dn_dgamma[ll+iw+1:ul,ndoc+(ncwo-1)*i+iw] = n[no1+i] - 1
                for ip in range(ll+iw+1,ul):
                    for jw in range(ip-ll):
                        if(jw==iw):
                            dn_dgamma[ip][ndoc+(ncwo-1)*i+iw] *= np.sin(2*gamma[ndoc+(ncwo-1)*i+jw])  
                        else:
                            dn_dgamma[ip][ndoc+(ncwo-1)*i+iw] *= np.cos(gamma[ndoc+(ncwo-1)*i+jw])**2  
                    if(ip-ll<ncwo-1):
                        dn_dgamma[ip][ndoc+(ncwo-1)*i+iw] *= np.sin(gamma[ndoc+(ncwo-1)*i+(ip-ll)])**2  

            # dn_pi/dgamma_i
            for iw in range(ncwo-1):
                dn_dgamma[ll+iw][ndoc+(ncwo-1)*i+iw] = 1 - n[no1+i]
                for jw in range(iw+1):
                    if(jw==iw):
                        dn_dgamma[ll+iw][ndoc+(ncwo-1)*i+iw] *= np.sin(2*gamma[ndoc+(ncwo-1)*i+jw])  
                    else:
                        dn_dgamma[ll+iw][ndoc+(ncwo-1)*i+iw] *= np.cos(gamma[ndoc+(ncwo-1)*i+jw])**2  

    return n,dn_dgamma

def calce(gamma,J_MO,K_MO,H_core,p):

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    cj12,ck12 = PNOFi_selector(n,p)

    E = 0

    if(p.MSpin==0):

        # 2H + J
        E = E + np.einsum('i,i',n[:p.nbeta],2*H_core[:p.nbeta]+np.diagonal(J_MO)[:p.nbeta],optimize=True) # [0,Nbeta]
        E = E + np.einsum('i,i',n[p.nbeta:p.nalpha],2*H_core[p.nbeta:p.nalpha],optimize=True)               # (Nbeta,Nalpha]
        E = E + np.einsum('i,i',n[p.nalpha:p.nbf5],2*H_core[p.nalpha:p.nbf5]+np.diagonal(J_MO)[p.nalpha:p.nbf5],optimize=True) # (Nalpha,Nbf5)
    
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

def calce2(gamma,J_MO,K_MO,H_core,p):

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    cj12,ck12 = PNOFi_selector(n,p)

    E = 0

    if(p.MSpin==0):

        # 2H + J
#        E = E + np.einsum('i,i',n[:p.nbeta],2*H_core[:p.nbeta],optimize=True) # [0,Nbeta]
#        E = E + np.einsum('i,i',n[p.nbeta:p.nalpha],2*H_core[p.nbeta:p.nalpha],optimize=True)               # (Nbeta,Nalpha]
#        E = E + np.einsum('i,i',n[p.nalpha:p.nbf5],2*H_core[p.nalpha:p.nbf5],optimize=True) # (Nalpha,Nbf5)
        E = E + np.einsum('i,i',n[:p.nbeta],np.diagonal(J_MO)[:p.nbeta],optimize=True) # [0,Nbeta]
        E = E + np.einsum('i,i',n[p.nalpha:p.nbf5],np.diagonal(J_MO)[p.nalpha:p.nbf5],optimize=True) # (Nalpha,Nbf5)

        #C^J JMO
#        np.fill_diagonal(cj12,0) # Remove diag.
#        E = E + np.einsum('ij,ji->',cj12,J_MO,optimize=True) # sum_ij

        #C^K KMO
#        np.fill_diagonal(ck12,0) # Remove diag.
#        E = E - np.einsum('ij,ji->',ck12,K_MO,optimize=True) # sum_ij

    return E

def calcg(gamma,J_MO,K_MO,H_core,p):

    grad = np.zeros((p.nv))

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    Dcj12r,Dck12r = der_PNOFi_selector(n,dn_dgamma,p)

    if(p.MSpin==0):

        # dn_dgamma (2H+J)
        grad += np.einsum('ik,i->k',dn_dgamma[p.no1:p.nbeta,:p.nv],2*H_core[p.no1:p.nbeta]+np.diagonal(J_MO)[p.no1:p.nbeta],optimize=True) # [0,Nbeta]
        grad += np.einsum('ik,i->k',dn_dgamma[p.nalpha:p.nbf5,:p.nv],2*H_core[p.nalpha:p.nbf5]+np.diagonal(J_MO)[p.nalpha:p.nbf5],optimize=True) # [Nalpha,Nbf5]
    
        # 2 dCJ_dgamma J_MO
        diag = np.diag_indices(p.nbf5)
        Dcj12r[diag] = 0
        grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.no1:p.nbeta,:p.nbf5,:p.nv],J_MO[:p.nbf5,p.no1:p.nbeta],optimize=True)
        #grad -= 2*np.einsum('iik,ii->k',Dcj12r[p.no1:p.nbeta,p.no1:p.nbeta,:p.nv],J_MO[p.no1:p.nbeta,p.no1:p.nbeta],optimize=True)
    
        grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.nalpha:p.nbf5,:p.nbf5,:p.nv],J_MO[:p.nbf5,p.nalpha:p.nbf5],optimize=True)
        #grad -= 2*np.einsum('iik,ii->k',Dcj12r[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:p.nv],J_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
    
        # -2 dCK_dgamma K_MO
        diag = np.diag_indices(p.nbf5)
        Dck12r[diag] = 0
        grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.no1:p.nbeta,:p.nbf5,:p.nv],K_MO[:p.nbf5,p.no1:p.nbeta],optimize=True)
        #grad += 2*np.einsum('iik,ii->k',Dck12r[p.no1:p.nbeta,p.no1:p.nbeta,:p.nv],K_MO[p.no1:p.nbeta,p.no1:p.nbeta],optimize=True)
    
        grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.nalpha:p.nbf5,:p.nbf5,:p.nv],K_MO[:p.nbf5,p.nalpha:p.nbf5],optimize=True)
        #grad += 2*np.einsum('iik,ii->k',Dck12r[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:p.nv],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)

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

def calcorbe(y,gamma,C,H,I,b_mnl,p):

    Cnew = pynof.rotate_orbital(y,C,p)

    J_MO,K_MO,H_core = pynof.computeJKH_MO(Cnew,H,I,b_mnl,p)
    E = calce(gamma,J_MO,K_MO,H_core,p)

    return E

def calcorbe2(y,gamma,C,H,I,b_mnl,p):

    Cnew = pynof.rotate_orbital(y,C,p)

    J_MO,K_MO,H_core = pynof.computeJKH_MO(Cnew,H,I,b_mnl,p)
    E = calce2(gamma,J_MO,K_MO,H_core,p)

    return E

def calcorbg(y,gamma,C,H,I,b_mnl,p):

    Cnew = pynof.rotate_orbital(y,C,p)

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    cj12,ck12 = PNOFi_selector(n,p)

    if p.RI:
        Hmat,b_MO = pynof.JKH_MO_tmp(Cnew,H,I,b_mnl,p)
    else:
        Hmat,I_MO = pynof.JKH_MO_tmp(Cnew,H,I,b_mnl,p)

    if p.gpu:
        grad = cp.zeros((p.nbf,p.nbf))
        np.fill_diagonal(cj12,0) # Remove diag.
        np.fill_diagonal(ck12,0) # Remove diag.
        n = cp.array(n)
        cj12 = cp.array(cj12)
        ck12 = cp.array(ck12)
        if p.RI:
            if(p.MSpin==0):
                # 2ndH/dy_ab
                grad[:,:p.nbf5] +=  2*cp.einsum('b,ab->ab',2*n,Hmat[:,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] += -2*cp.einsum('a,ab->ab',2*n,Hmat[:p.nbf5,:],optimize=True)

                # dJ_pp/dy_ab
                grad[:,:p.nbeta] +=  4*cp.einsum('b,abk,bbk->ab',n[:p.nbeta],b_MO[:,:p.nbeta,:],b_MO[:p.nbeta,:p.nbeta,:],optimize=True)
                grad[:,p.nalpha:p.nbf5] +=  4*cp.einsum('b,abk,bbk->ab',n[p.nalpha:p.nbf5],b_MO[:,p.nalpha:p.nbf5,:],b_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:],optimize=True)
                grad[:p.nbeta,:] += -4*cp.einsum('a,bak,aak->ab',n[:p.nbeta],b_MO[:,:p.nbeta,:],b_MO[:p.nbeta,:p.nbeta,:],optimize=True)
                grad[p.nalpha:p.nbf5,:] += -4*cp.einsum('a,bak,aak->ab',n[p.nalpha:p.nbf5],b_MO[:,p.nalpha:p.nbf5,:],b_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:],optimize=True)

                # C^J_pq dJ_pq/dy_ab 
                grad[:,:p.nbf5] +=  4*cp.einsum('bq,abk,qqk->ab',cj12,b_MO[:,:p.nbf5,:],b_MO[:p.nbf5,:p.nbf5,:],optimize=True)
                grad[:p.nbf5,:] += -4*cp.einsum('aq,abk,qqk->ab',cj12,b_MO[:p.nbf5,:,:],b_MO[:p.nbf5,:p.nbf5,:],optimize=True)

                # -C^K_pq dK_pq/dy_ab 
                grad[:,:p.nbf5] += -4*cp.einsum('bq,aqk,bqk->ab',ck12,b_MO[:,:p.nbf5,:],b_MO[:p.nbf5,:p.nbf5,:],optimize=True)
                grad[:p.nbf5,:] +=  4*cp.einsum('aq,aqk,bqk->ab',ck12,b_MO[:p.nbf5,:p.nbf5,:],b_MO[:,:p.nbf5,:],optimize=True)
        else:        
            if(p.MSpin==0):
                # 2ndH/dy_ab
                grad[:,:p.nbf5] +=  2*cp.einsum('b,ab->ab',2*n,Hmat[:,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] += -2*cp.einsum('a,ab->ab',2*n,Hmat[:p.nbf5,:],optimize=True)
        
                # dJ_pp/dy_ab
                grad[:,:p.nbeta] +=  4*cp.einsum('b,abbb->ab',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
                grad[:,p.nalpha:p.nbf5] +=  4*cp.einsum('b,abbb->ab',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
                grad[:p.nbeta,:] += -4*cp.einsum('a,baaa->ab',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
                grad[p.nalpha:p.nbf5,:] += -4*cp.einsum('a,baaa->ab',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
        
                # C^J_pq dJ_pq/dy_ab 
                np.fill_diagonal(cj12,0) # Remove diag.
                grad[:,:p.nbf5] +=  4*cp.einsum('bq,abqq->ab',cj12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] += -4*cp.einsum('aq,abqq->ab',cj12,I_MO[:p.nbf5,:,:p.nbf5,:p.nbf5],optimize=True)
        
                # -C^K_pq dK_pq/dy_ab 
                np.fill_diagonal(ck12,0) # Remove diag.
                grad[:,:p.nbf5] += -4*cp.einsum('bq,aqbq->ab',ck12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] +=  4*cp.einsum('aq,aqbq->ab',ck12,I_MO[:p.nbf5,:p.nbf5,:,:p.nbf5],optimize=True)
        
#        triu_idx = np.triu_indices(p.nbf,k=1)
#        grad = grad[triu_idx]
        grad = grad.get()    
    else:
        grad = np.zeros((p.nbf,p.nbf))
        if p.RI:
            if(p.MSpin==0):
                # 2ndH/dy_ab
                grad[:,:p.nbf5] +=  2*np.einsum('b,ab->ab',2*n,Hmat[:,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] += -2*np.einsum('a,ab->ab',2*n,Hmat[:p.nbf5,:],optimize=True)

                # dJ_pp/dy_ab
                grad[:,:p.nbeta] +=  4*np.einsum('b,abk,bbk->ab',n[:p.nbeta],b_MO[:,:p.nbeta,:],b_MO[:p.nbeta,:p.nbeta,:],optimize=True)
                grad[:,p.nalpha:p.nbf5] +=  4*np.einsum('b,abk,bbk->ab',n[p.nalpha:p.nbf5],b_MO[:,p.nalpha:p.nbf5,:],b_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:],optimize=True)
                grad[:p.nbeta,:] += -4*np.einsum('a,bak,aak->ab',n[:p.nbeta],b_MO[:,:p.nbeta,:],b_MO[:p.nbeta,:p.nbeta,:],optimize=True)
                grad[p.nalpha:p.nbf5,:] += -4*np.einsum('a,bak,aak->ab',n[p.nalpha:p.nbf5],b_MO[:,p.nalpha:p.nbf5,:],b_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:],optimize=True)

                # C^J_pq dJ_pq/dy_ab 
                np.fill_diagonal(cj12,0) # Remove diag.
                grad[:,:p.nbf5] +=  4*np.einsum('bq,abk,qqk->ab',cj12,b_MO[:,:p.nbf5,:],b_MO[:p.nbf5,:p.nbf5,:],optimize=True)
                grad[:p.nbf5,:] += -4*np.einsum('aq,abk,qqk->ab',cj12,b_MO[:p.nbf5,:,:],b_MO[:p.nbf5,:p.nbf5,:],optimize=True)

                # -C^K_pq dK_pq/dy_ab 
                np.fill_diagonal(ck12,0) # Remove diag.
                grad[:,:p.nbf5] += -4*np.einsum('bq,aqk,bqk->ab',ck12,b_MO[:,:p.nbf5,:],b_MO[:p.nbf5,:p.nbf5,:],optimize=True)
                grad[:p.nbf5,:] +=  4*np.einsum('aq,aqk,bqk->ab',ck12,b_MO[:p.nbf5,:p.nbf5,:],b_MO[:,:p.nbf5,:],optimize=True)
        else:
            if(p.MSpin==0):
                # 2ndH/dy_ab
                grad[:,:p.nbf5] +=  2*np.einsum('b,ab->ab',2*n,Hmat[:,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] += -2*np.einsum('a,ab->ab',2*n,Hmat[:p.nbf5,:],optimize=True)
    
                # dJ_pp/dy_ab
                grad[:,:p.nbeta] +=  4*np.einsum('b,abbb->ab',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
                grad[:,p.nalpha:p.nbf5] +=  4*np.einsum('b,abbb->ab',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
                grad[:p.nbeta,:] += -4*np.einsum('a,baaa->ab',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
                grad[p.nalpha:p.nbf5,:] += -4*np.einsum('a,baaa->ab',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
    
                # C^J_pq dJ_pq/dy_ab 
                np.fill_diagonal(cj12,0) # Remove diag.
                grad[:,:p.nbf5] +=  4*np.einsum('bq,abqq->ab',cj12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] += -4*np.einsum('aq,abqq->ab',cj12,I_MO[:p.nbf5,:,:p.nbf5,:p.nbf5],optimize=True)
    
                # -C^K_pq dK_pq/dy_ab 
                np.fill_diagonal(ck12,0) # Remove diag.
                grad[:,:p.nbf5] += -4*np.einsum('bq,aqbq->ab',ck12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
                grad[:p.nbf5,:] +=  4*np.einsum('aq,aqbq->ab',ck12,I_MO[:p.nbf5,:p.nbf5,:,:p.nbf5],optimize=True)

#        triu_idx = np.triu_indices(p.nbf,k=1)
#        grad = grad[triu_idx]

        grads = np.zeros((int(p.nbf*(p.nbf-1)/2) - int(p.no0*(p.no0-1)/2)))
        n = 0
        for i in range(p.nbf5):
            for j in range(i+1,p.nbf):
                grads[n] = grad[i,j]
                n += 1
        grad = grads

    return grad

def calcorbh(y,gamma,C,H,I,b_mnl,p):

    Cnew = pynof.rotate_orbital(y,C,p)

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin)
    cj12,ck12 = PNOFi_selector(n,p)
    np.fill_diagonal(cj12,0) # Remove diag.
    np.fill_diagonal(ck12,0) # Remove diag.

    Hmat,I_MO = pynof.JKH_MO_tmp(Cnew,H,I,b_mnl,p)

    if p.gpu:
        Hmat = cp.array(Hmat)
        n = cp.array(n)
        cj12 = cp.array(cj12)
        ck12 = cp.array(ck12)
        d2E_dycddyab = cp.zeros((p.nbf,p.nbf,p.nbf,p.nbf))
        if(p.MSpin==0):
    
            d2E_dycddyab[:,:p.nbf5,:,:p.nbf5] += 8*cp.einsum("bd,cdab->cdab",cj12,I_MO[:,:p.nbf5,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:p.nbf5,:] -= 8*cp.einsum("ad,cdab->cdab",cj12,I_MO[:,:p.nbf5,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:,:p.nbf5] += 4*cp.einsum("bc,adbc->cdab",ck12,I_MO[:,:,:p.nbf5,:p.nbf5],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:p.nbf5,:] -= 4*cp.einsum("ac,adbc->cdab",ck12,I_MO[:p.nbf5,:,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:,:p.nbf5] += 4*cp.einsum("bc,acbd->cdab",ck12,I_MO[:,:p.nbf5,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:p.nbf5,:] -= 4*cp.einsum("ac,acbd->cdab",ck12,I_MO[:p.nbf5,:p.nbf5,:,:],optimize=True)
            ###########
            d2E_dycddyab[:p.nbf5,:,:,:p.nbf5] -= 8*cp.einsum("bc,cdab->cdab",cj12,I_MO[:p.nbf5,:,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:p.nbf5,:] += 8*cp.einsum("ac,cdab->cdab",cj12,I_MO[:p.nbf5,:,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:,:p.nbf5] -= 4*cp.einsum("bd,acbd->cdab",ck12,I_MO[:,:,:p.nbf5,:p.nbf5],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:p.nbf5,:] += 4*cp.einsum("ad,acbd->cdab",ck12,I_MO[:p.nbf5,:,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:,:p.nbf5] -= 4*cp.einsum("bd,adbc->cdab",ck12,I_MO[:,:p.nbf5,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:p.nbf5,:] += 4*cp.einsum("ad,adbc->cdab",ck12,I_MO[:p.nbf5,:p.nbf5,:,:],optimize=True)
    
            tmp = cp.zeros((p.nbf,p.nbf))
            tmp[:,:p.nbf5] += cp.einsum('b,cb->cb',2*n,Hmat[:,:p.nbf5],optimize=True)
            tmp[:p.nbf5,:] +=  cp.einsum('c,cb->cb',2*n,Hmat[:p.nbf5,:],optimize=True)
            tmp[:,:p.nbeta] +=  2*cp.einsum('b,cbbb->cb',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
            tmp[:,p.nalpha:p.nbf5] +=  2*cp.einsum('b,cbbb->cb',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
            tmp[:p.nbeta,:] +=  2*cp.einsum('c,bccc->cb',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
            tmp[p.nalpha:p.nbf5,:] +=  2*cp.einsum('c,bccc->cb',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
            tmp[:,:p.nbf5] +=  2*cp.einsum('bq,cbqq->cb',cj12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
            tmp[:p.nbf5,:] +=  2*cp.einsum('cq,cbqq->cb',cj12,I_MO[:p.nbf5,:,:p.nbf5,:p.nbf5],optimize=True)
            tmp[:,:p.nbf5] += -2*cp.einsum('bq,cqbq->cb',ck12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
            tmp[:p.nbf5,:] += -2*cp.einsum('cq,cqbq->cb',ck12,I_MO[:p.nbf5,:p.nbf5,:,:p.nbf5],optimize=True)

            for i in range(p.nbf):
                tmp2 = tmp.copy()
                if(i < p.nbf5):
                    # d2(2sum_p n_p H_pp)/dycddyab
                    tmp2 += -2*2*n[i]*Hmat
    
                    # d2(sum_p n_p J_pp)/dycddyab
                    if(i < p.nbeta or i >= p.nalpha):
                        tmp2 += -4*n[i]*I_MO[:,:,i,i]
                        tmp2 += -8*n[i]*I_MO[:,i,:,i]
    
                    # d2(sum_pq C^J_pq J_pq)/dycddyab
                    tmp2 += -4*cp.einsum('q,cbqq->cb',cj12[i,:],I_MO[:,:,:p.nbf5,:p.nbf5],optimize=True)
    
                    # -d2(sum_pq C^K_pq K_pq)/dycddyab
                    tmp2 +=  4*cp.einsum('q,cqbq->cb',ck12[i,:],I_MO[:,:p.nbf5,:,:p.nbf5],optimize=True)
    
                d2E_dycddyab[:i,i,i,i+1:] +=  tmp2[:i,i+1:]
                d2E_dycddyab[i,i+1:,:i,i] +=  tmp2[i+1:,:i]
                d2E_dycddyab[:i,i,:i,i] -=  tmp2[:i,:i]
                d2E_dycddyab[i,i+1:,i,i+1:] -=  tmp2[i+1:,i+1:]
        d2E_dycddyab = d2E_dycddyab.get()
    else:
        d2E_dycddyab = np.zeros((p.nbf,p.nbf,p.nbf,p.nbf))
        if(p.MSpin==0):

            d2E_dycddyab[:,:p.nbf5,:,:p.nbf5] += 8*np.einsum("bd,cdab->cdab",cj12,I_MO[:,:p.nbf5,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:p.nbf5,:] -= 8*np.einsum("ad,cdab->cdab",cj12,I_MO[:,:p.nbf5,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:,:p.nbf5] += 4*np.einsum("bc,adbc->cdab",ck12,I_MO[:,:,:p.nbf5,:p.nbf5],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:p.nbf5,:] -= 4*np.einsum("ac,adbc->cdab",ck12,I_MO[:p.nbf5,:,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:,:p.nbf5] += 4*np.einsum("bc,acbd->cdab",ck12,I_MO[:,:p.nbf5,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:p.nbf5,:] -= 4*np.einsum("ac,acbd->cdab",ck12,I_MO[:p.nbf5,:p.nbf5,:,:],optimize=True)
            ###########
            d2E_dycddyab[:p.nbf5,:,:,:p.nbf5] -= 8*np.einsum("bc,cdab->cdab",cj12,I_MO[:p.nbf5,:,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:p.nbf5,:,:p.nbf5,:] += 8*np.einsum("ac,cdab->cdab",cj12,I_MO[:p.nbf5,:,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:,:p.nbf5] -= 4*np.einsum("bd,acbd->cdab",ck12,I_MO[:,:,:p.nbf5,:p.nbf5],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:p.nbf5,:] += 4*np.einsum("ad,acbd->cdab",ck12,I_MO[:p.nbf5,:,:,:p.nbf5],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:,:p.nbf5] -= 4*np.einsum("bd,adbc->cdab",ck12,I_MO[:,:p.nbf5,:p.nbf5,:],optimize=True)
            d2E_dycddyab[:,:p.nbf5,:p.nbf5,:] += 4*np.einsum("ad,adbc->cdab",ck12,I_MO[:p.nbf5,:p.nbf5,:,:],optimize=True)

            tmp = np.zeros((p.nbf,p.nbf))
            tmp[:,:p.nbf5] += np.einsum('b,cb->cb',2*n,Hmat[:,:p.nbf5],optimize=True)
            tmp[:p.nbf5,:] +=  np.einsum('c,cb->cb',2*n,Hmat[:p.nbf5,:],optimize=True)
            tmp[:,:p.nbeta] +=  2*np.einsum('b,cbbb->cb',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
            tmp[:,p.nalpha:p.nbf5] +=  2*np.einsum('b,cbbb->cb',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
            tmp[:p.nbeta,:] +=  2*np.einsum('c,bccc->cb',n[:p.nbeta],I_MO[:,:p.nbeta,:p.nbeta,:p.nbeta],optimize=True)
            tmp[p.nalpha:p.nbf5,:] +=  2*np.einsum('c,bccc->cb',n[p.nalpha:p.nbf5],I_MO[:,p.nalpha:p.nbf5,p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
            tmp[:,:p.nbf5] +=  2*np.einsum('bq,cbqq->cb',cj12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
            tmp[:p.nbf5,:] +=  2*np.einsum('cq,cbqq->cb',cj12,I_MO[:p.nbf5,:,:p.nbf5,:p.nbf5],optimize=True)
            tmp[:,:p.nbf5] += -2*np.einsum('bq,cqbq->cb',ck12,I_MO[:,:p.nbf5,:p.nbf5,:p.nbf5],optimize=True)
            tmp[:p.nbf5,:] += -2*np.einsum('cq,cqbq->cb',ck12,I_MO[:p.nbf5,:p.nbf5,:,:p.nbf5],optimize=True)

            for i in range(p.nbf):
                tmp2 = tmp.copy()
                if(i < p.nbf5):
                    # d2(2sum_p n_p H_pp)/dycddyab
                    tmp2 += -2*2*n[i]*Hmat
                    # d2(sum_p n_p J_pp)/dycddyab
                    if(i < p.nbeta or i >= p.nalpha):
                        tmp2 += -4*n[i]*I_MO[:,:,i,i]
                        tmp2 += -8*n[i]*I_MO[:,i,:,i]
                    # d2(sum_pq C^J_pq J_pq)/dycddyab
                    tmp2 += -4*np.einsum('q,cbqq->cb',cj12[i,:],I_MO[:,:,:p.nbf5,:p.nbf5],optimize=True)

                    # -d2(sum_pq C^K_pq K_pq)/dycddyab
                    tmp2 +=  4*np.einsum('q,cqbq->cb',ck12[i,:],I_MO[:,:p.nbf5,:,:p.nbf5],optimize=True)

                d2E_dycddyab[:i,i,i,i+1:] +=  tmp2[:i,i+1:]
                d2E_dycddyab[i,i+1:,:i,i] +=  tmp2[i+1:,:i]
                d2E_dycddyab[:i,i,:i,i] -=  tmp2[:i,:i]
                d2E_dycddyab[i,i+1:,i,i+1:] -=  tmp2[i+1:,i+1:]

    hess = pynof.extract_tiu_tensor(d2E_dycddyab,1)
    return hess

def calcorbg_num(y,gamma,C,H,I,b_mnl,p):

    grad = nds.Gradient(calcorbe)(y,gamma,C,H,I,b_mnl,p)

    return grad

def calcorbh_num(y,gamma,C,H,I,b_mnl,p):

    hess = nds.Hessian(calcorbe)(y,gamma,C,H,I,b_mnl,p)

    return hess

def calcorbh_num2(y,gamma,C,H,I,b_mnl,p):

    hess = nds.Hessian(calcorbe2)(y,gamma,C,H,I,b_mnl,p)

    return hess


def calccombe(x,C,H,I,b_mnl,p):

    nvar = int(p.nbf*(p.nbf-1)/2) - int(p.no0*(p.no0-1)/2)
    y = x[:nvar]
    gamma = x[nvar:]

    Cnew = pynof.rotate_orbital(y,C,p)

    J_MO,K_MO,H_core = pynof.computeJKH_MO(Cnew,H,I,b_mnl,p)
    E = calce(gamma,J_MO,K_MO,H_core,p)

    return E

def calccombg(x,C,H,I,b_mnl,p):

    nvar = int(p.nbf*(p.nbf-1)/2) - int(p.no0*(p.no0-1)/2)
    y = x[:nvar]
    gamma = x[nvar:]

    Cnew = pynof.rotate_orbital(y,C,p)

    J_MO,K_MO,H_core = pynof.computeJKH_MO(Cnew,H,I,b_mnl,p)

    grad = np.zeros(nvar + p.nv)

    grad[:nvar] = calcorbg(y,gamma,Cnew,H,I,b_mnl,p)
    grad[nvar:] = calcg(gamma,J_MO,K_MO,H_core,p)

    return grad


def calccombg_num(x,C,H,I,b_mnl,p):

    grad = nds.Gradient(calccombe)(x,C,H,I,b_mnl,p)

    return grad

def calccombh_num(x,C,H,I,b_mnl,p):

    hess = nds.Hessian(calccombe)(x,C,H,I,b_mnl,p)

    return hess

