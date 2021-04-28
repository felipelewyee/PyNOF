import numpy as np

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
def CJCKD7(n,p):    

    fi = n*(1-n)
    fi[fi<=0] = 0
    fi = np.sqrt(fi)      
   
    # Interpair Electron correlation #

    cj12 = 2*np.einsum('i,j->ij',n,n)
    ck12 = np.einsum('i,j->ij',n,n) + np.einsum('i,j->ij',fi,fi)
    
    # Intrapair Electron Correlation

    for l in range(p.ndoc):            
        ldx = p.no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - l)

        cj12[ldx,ll:ul] = 0    
        cj12[ll:ul,ldx] = 0    
    
        cj12[ll:ul,ll:ul] = 0    
        
        ck12[ldx,ll:ul] = np.sqrt(n[ldx]*n[ll:ul])
        ck12[ll:ul,ldx] = np.sqrt(n[ldx]*n[ll:ul])

        ck12[ll:ul,ll:ul] = -np.sqrt(np.outer(n[ll:ul],n[ll:ul]))

    return cj12,ck12        

#CJCKD7
#def CJCKD7(n,p):
#
#    fi = n*(1-n)
#    fi[fi<=0] = 0
#    fi = np.sqrt(fi)
#
#    cj12 = np.zeros((len(n),len(n)))
#    ck12 = np.zeros((len(n),len(n)))
#
#    # Intrapair Electron correlation #
#
#    for i in range(p.ndoc):
#        idx = p.no1 + i
#        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
#        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - i - 1)
#        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - i)
#        for idx_w1 in range(ll,ul):
#            ck12[idx,idx_w1] = np.sqrt(n[idx]*n[idx_w1])
#            ck12[idx_w1,idx] = np.sqrt(n[idx_w1]*n[idx])
#            for idx_w2 in range(ll,ul):
#                if(idx_w1!=idx_w2):
#                    ck12[idx_w1][idx_w2] = -np.sqrt(n[idx_w1]*n[idx_w2])
#
#    # Interpair Electron Correlation
#
#    cj12pp = np.zeros((len(n),len(n)))
#    for i in range(p.ndoc):
#        idx = p.no1 + i
#        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
#        lli = p.no1 + p.ndns + p.ncwo*(p.ndoc - i - 1)
#        uli = p.no1 + p.ndns + p.ncwo*(p.ndoc - i)
#        for j in range(p.ndoc):
#            if(i!=j):
#                jdx = p.no1 + j
#                # inicio y fin de los orbitales acoplados a los fuertemente ocupados
#                llj = p.no1 + p.ndns + p.ncwo*(p.ndoc - j - 1)
#                ulj = p.no1 + p.ndns + p.ncwo*(p.ndoc - j)
#                for idx_w in range(lli,uli):
#                    for jdx_w in range(llj,ulj):
#                        cj12[idx_w][jdx_w] = 2*n[idx_w]*n[jdx_w]
#                        ck12[idx_w][jdx_w] = n[idx_w]*n[jdx_w] + fi[idx_w]*fi[jdx_w]
#                        cj12[idx][jdx_w] = 2*n[idx]*n[jdx_w]
#                        ck12[idx][jdx_w] = n[idx]*n[jdx_w] + fi[idx]*fi[jdx_w]
#                    cj12[idx_w][jdx] = 2*n[idx_w]*n[jdx]
#                    ck12[idx_w][jdx] = n[idx_w]*n[jdx]
#                    ck12[idx_w][jdx] += fi[idx_w]*fi[jdx]
#                cj12[idx][jdx] = 2*n[idx]*n[jdx]
#                ck12[idx][jdx] = n[idx]*n[jdx] + fi[idx]*fi[jdx]
#  
#    # Interpair Electron correlation #
#
#    cj12[:p.no1,:p.no1] += 2*np.einsum('i,j->ij',n[:p.no1],n[:p.no1])
#    cj12[:p.no1,p.no1:len(n)] += 2*np.einsum('i,j->ij',n[:p.no1],n[p.no1:len(n)])
#    cj12[p.no1:len(n),:p.no1] += 2*np.einsum('i,j->ij',n[p.no1:len(n)],n[:p.no1])
#    cj12p = 2*np.einsum('i,j->ij',n,n)
#    ck12p = np.einsum('i,j->ij',n,n) + np.einsum('i,j->ij',fi,fi)
#    ck12[:p.no1,:p.no1] += np.einsum('i,j->ij',n[:p.no1],n[:p.no1]) + np.einsum('i,j->ij',fi[:p.no1],fi[:p.no1])
#    ck12[:p.no1,p.no1:len(n)] += np.einsum('i,j->ij',n[:p.no1],n[p.no1:len(n)]) + np.einsum('i,j->ij',fi[:p.no1],fi[p.no1:len(n)])
#    ck12[p.no1:len(n),:p.no1] += np.einsum('i,j->ij',n[p.no1:len(n)],n[:p.no1]) + np.einsum('i,j->ij',fi[p.no1:len(n)],fi[:p.no1])
#    
#
#    return cj12,ck12


def der_CJCKD7(n,dn_dgamma,p):
    
    fi = n*(1-n)
    fi[fi<=0] = 0
    fi = np.sqrt(fi)      
            
    dfi_dgamma = np.zeros((p.nbf5,p.nv))
    for i in range(p.no1,p.nbf5):
        a = max(fi[i],10**-15)
        for k in range(p.nv):
            dfi_dgamma[i,k] = 1/(2*a)*(1-2*n[i])*dn_dgamma[i][k]
   
    # Interpair Electron correlation #

    Dcj12r = 2*np.einsum('ik,j->ijk',dn_dgamma,n)    
    Dck12r = np.einsum('ik,j->ijk',dn_dgamma,n) + np.einsum('ik,j->ijk',dfi_dgamma,fi)    

    # Interpair Electron Correlation

    for l in range(p.ndoc):            
        ldx = p.no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - l)

        Dcj12r[ldx,ll:ul,:p.nv] = 0
        Dcj12r[ll:ul,ldx,:p.nv] = 0

        Dcj12r[ll:ul,ll:ul,:p.nv] = 0   

        a = max(n[ldx],10**-15)
        b = n[ll:ul]
        b[b<10**-15] = 10**-15        
        
        Dck12r[ldx,ll:ul,:p.nv] = 1/2 * 1/np.sqrt(a) * np.einsum('j,i->ij',dn_dgamma[ldx,:p.nv],np.sqrt(n[ll:ul]))
        Dck12r[ll:ul,ldx,:p.nv] = 1/2 * np.einsum('i,ij->ij', 1/np.sqrt(b),dn_dgamma[ll:ul,:p.nv])*np.sqrt(n[ldx])
        
        for k in range(p.nv):
            Dck12r[ll:ul,ll:ul,k] = - 1/2 * np.einsum('i,i,j->ij',1/np.sqrt(b),dn_dgamma[ll:ul,k],np.sqrt(n[ll:ul]))
                        
    return Dcj12r,Dck12r


# Creamos un seleccionador de PNOF

def PNOFi_selector(n,p):
    if(p.ipnof==5):
        cj12,ck12 = CJCKD5(n,p)
    if(p.ipnof==7):
        cj12,ck12 = CJCKD7(n,p)
        
    return cj12,ck12

def der_PNOFi_selector(n,dn_dgamma,p):
    if(p.ipnof==5):
        Dcj12r,Dck12r = der_CJCKD5(n,dn_dgamma,p)
    if(p.ipnof==7):
        Dcj12r,Dck12r = der_CJCKD7(n,dn_dgamma,p)
        
    return Dcj12r,Dck12r

def ocupacion(gamma,p):

    n = np.zeros((p.nbf5))
    dni_dgammai = np.zeros((p.nbf5))

    n[0:p.no1] = 1                                              # [1,no1]

    n[p.no1:p.no1+p.ndoc] = 1/2 * (1 + np.cos(gamma[:p.ndoc])**2)     # (no1,no1+ndoc]
    dni_dgammai[p.no1:p.no1+p.ndoc] = - 1/2 * np.sin(2*gamma[:p.ndoc])

    if(p.ncwo==1):
        dn_dgamma = np.zeros((p.nbf5,p.nv))

        for i in range(p.ndoc):
            dn_dgamma[p.no1+i][i] = dni_dgammai[p.no1+i]
            #cwo
            icf = p.nalpha + p.ndoc - i - 1
            n[icf] = 1/2*np.sin(gamma[i])**2
            dni_dgammai[icf]  = 1/2*np.sin(2*gamma[i])
            dn_dgamma[icf][i] = dni_dgammai[icf]

    return n,dn_dgamma

def calce(gamma,J_MO,K_MO,H_core,p):

    n,dn_dgamma = ocupacion(gamma,p)
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

    n,dn_dgamma = ocupacion(gamma,p)
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
