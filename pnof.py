import numpy as np
from numba import prange,njit,jit

#CJCKD5
def CJCKD5(n,p):    
    
    # Interpair Electron correlation #

    cj12 = 2*np.einsum('i,j->ij',n,n)
    ck12 = np.einsum('i,j->ij',n,n)    
    
    # Interpair Electron Correlation
    
    for l in range(p.ndoc):            
        ldx = p.no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc-l-1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc-l)

        cj12[ldx,ll:ul] = 0    
        cj12[ll:ul,ldx] = 0    
    
        cj12[ll:ul,ll:ul] = 0    
        
        ck12[ldx,ll:ul] = np.sqrt(n[ldx]*n[ll:ul])
        ck12[ll:ul,ldx] = np.sqrt(n[ldx]*n[ll:ul])

        ck12[ll:ul,ll:ul] = -np.outer(np.sqrt(n[ll:ul]),np.sqrt(n[ll:ul]))

    return cj12,ck12        
        
def der_CJCKD5(n,gamma,dn_dgamma,p):

    # Interpair Electron correlation #

    Dcj12r = 2*np.einsum('ik,j->ijk',dn_dgamma,n)    
    Dck12r = np.einsum('ik,j->ijk',dn_dgamma,n)    

    # Interpair Electron Correlation
    
    for l in range(p.ndoc):            
        ldx = p.no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = p.no1 + p.ndns + ncwo*(p.ndoc-l-1)
        ul = p.no1 + p.ndns + ncwo*(p.ndoc-l)

        Dcj12r[ldx,ll:ul,:nv] = 0
        Dcj12r[ll:ul,ldx,:nv] = 0

        Dcj12r[ll:ul,ll:ul,:nv] = 0   
        
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
@njit
def CJCKD7(n,ista,no1,ndoc,nalpha,ndns,ncwo):

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
       
@njit
def der_CJCKD7(n,ista,dn_dgamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo):

    if(ista==0):
        fi = n*(1-n)
        fi[fi<=0] = 0
        fi = np.sqrt(fi)      
    else:
        fi = 2*n*(1-n)
            
    dfi_dgamma = np.zeros((nbf5,nv))
    for i in range(no1,nbf5):
        a = max(fi[i],10**-15)
        for k in range(nv):
            if(ista==0):
                dfi_dgamma[i,k] = 1/(2*a)*(1-2*n[i])*dn_dgamma[i][k]
            else:
                dfi_dgamma[i,k] = 2*(1-2*n[i])*dn_dgamma[i][k]
                
   
    # Interpair Electron correlation #

    Dcj12r = np.zeros((nbf5,nbf5,nv))
    Dck12r = np.zeros((nbf5,nbf5,nv))
    for k in range(nv):
        Dcj12r[:,:,k] = 2*np.outer(dn_dgamma[:,k],n)    
        Dck12r[:,:,k] = np.outer(dn_dgamma[:,k],n) + np.outer(dfi_dgamma[:,k],fi)
    #Dcj12r = 2*np.einsum('ik,j->ijk',dn_dgamma,n)    
    #Dck12r = np.einsum('ik,j->ijk',dn_dgamma,n) + np.einsum('ik,j->ijk',dfi_dgamma,fi)    

    # Intrapair Electron Correlation

    for l in range(ndoc):            
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


# Creamos un seleccionador de PNOF

def PNOFi_selector(n,p):
    if(p.ipnof==5):
        cj12,ck12 = CJCKD5(n,p)
    if(p.ipnof==7):
        cj12,ck12 = CJCKD7(n,p.ista,p.no1,p.ndoc,p.nalpha,p.ndns,p.ncwo)
        
    return cj12,ck12

def der_PNOFi_selector(n,dn_dgamma,p):
    if(p.ipnof==5):
        Dcj12r,Dck12r = der_CJCKD5(n,dn_dgamma,p)
    if(p.ipnof==7):
        Dcj12r,Dck12r = der_CJCKD7(n,p.ista,dn_dgamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo)
        
    return Dcj12r,Dck12r

@njit
def ocupacion(gamma,no1,ndoc,nalpha,nv,nbf5,ndns,ncwo):

    n = np.zeros((nbf5))
    dni_dgammai = np.zeros((nbf5))

    n[0:no1] = 1                                              # [1,no1]

    n[no1:no1+ndoc] = 1/2 * (1 + np.cos(gamma[:ndoc])**2)     # (no1,no1+ndoc]
    dni_dgammai[no1:no1+ndoc] = - 1/2 * np.sin(2*gamma[:ndoc])
    
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

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo)
    cj12,ck12 = PNOFi_selector(n,p)

    E = 0

    # 2H + J
    E = E + np.einsum('i,i',n[:p.nbeta],2*H_core[:p.nbeta]+np.diagonal(J_MO)[:p.nbeta]) # [0,Nbeta]
    E = E + np.einsum('i,i',n[p.nbeta:p.nalpha],2*H_core[p.nbeta:p.nalpha])               # (Nbeta,Nalpha]
    E = E + np.einsum('i,i',n[p.nalpha:p.nbf5],2*H_core[p.nalpha:p.nbf5]+np.diagonal(J_MO)[p.nalpha:p.nbf5]) # (Nalpha,Nbf5)

    #C^J JMO
    E = E + np.einsum('ij,ji',cj12,J_MO) # sum_ij
    E = E - np.einsum('ii,ii',cj12,J_MO) # Quita i=j

    #C^K KMO
    E = E - np.einsum('ij,ji',ck12,K_MO) # sum_ij
    E = E + np.einsum('ii,ii',ck12,K_MO) # Quita i=j

    return E

def calcg(gamma,J_MO,K_MO,H_core,p):

    grad = np.zeros((p.nv))

    n,dn_dgamma = ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo)
    Dcj12r,Dck12r = der_PNOFi_selector(n,dn_dgamma,p)

    # dn_dgamma (2H+J)
    grad += np.einsum('ik,i->k',dn_dgamma[p.no1:p.nbeta,:p.nv],2*H_core[p.no1:p.nbeta]+np.diagonal(J_MO)[p.no1:p.nbeta],optimize=True) # [0,Nbeta]
    grad += np.einsum('ik,i->k',dn_dgamma[p.nalpha:p.nbf5,:p.nv],2*H_core[p.nalpha:p.nbf5]+np.diagonal(J_MO)[p.nalpha:p.nbf5],optimize=True) # [Nalpha,Nbf5]

    # 2 dCJ_dgamma J_MO
    grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.no1:p.nbeta,:p.nbf5,:p.nv],J_MO[:p.nbf5,p.no1:p.nbeta],optimize=True)
    grad -= 2*np.einsum('iik,ii->k',Dcj12r[p.no1:p.nbeta,p.no1:p.nbeta,:p.nv],J_MO[p.no1:p.nbeta,p.no1:p.nbeta],optimize=True)

    grad += 2*np.einsum('ijk,ji->k',Dcj12r[p.nalpha:p.nbf5,:p.nbf5,:p.nv],J_MO[:p.nbf5,p.nalpha:p.nbf5],optimize=True)
    grad -= 2*np.einsum('iik,ii->k',Dcj12r[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:p.nv],J_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)

    # -2 dCK_dgamma K_MO
    grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.no1:p.nbeta,:p.nbf5,:p.nv],K_MO[:p.nbf5,p.no1:p.nbeta],optimize=True)
    grad += 2*np.einsum('iik,ii->k',Dck12r[p.no1:p.nbeta,p.no1:p.nbeta,:p.nv],K_MO[p.no1:p.nbeta,p.no1:p.nbeta],optimize=True)

    grad -= 2*np.einsum('ijk,ji->k',Dck12r[p.nalpha:p.nbf5,:p.nbf5,:p.nv],K_MO[:p.nbf5,p.nalpha:p.nbf5],optimize=True)
    grad += 2*np.einsum('iik,ii->k',Dck12r[p.nalpha:p.nbf5,p.nalpha:p.nbf5,:p.nv],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)

    return grad

