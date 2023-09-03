from time import time
import psi4
import numpy as np
import cupy as cp
from numba import prange,njit
import pynof

def ERPA(wfn,mol,n,C,cj12,ck12,elag,pp):

    mints = psi4.core.MintsHelper(wfn.basisset())

    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = T + V

    h = np.einsum("mi,mn,nj->ij",C[:,0:pp.nbf5],H,C[:,0:pp.nbf5],optimize=True)

    I = np.asarray(mints.ao_eri())

    I = np.einsum("mi,nj,mnsr,sk,rl->ijkl",C[:,0:pp.nbf5],C[:,0:pp.nbf5],I,C[:,0:pp.nbf5],C[:,0:pp.nbf5],optimize=True)

    c = np.sqrt(n)
    c[pp.ndoc:] *= -1

    I = np.einsum('rpsq->rspq',I,optimize=True)

    A = np.zeros((pp.nbf5,pp.nbf5,pp.nbf5,pp.nbf5))
    Id = np.identity(pp.nbf5)

    A += 2*np.einsum('sq,pr,p->rspq',h,Id,n,optimize=True)
    A -= 2*np.einsum('sq,pr,s->rspq',h,Id,n,optimize=True)
    A += 2*np.einsum('pr,sq,q->rspq',h,Id,n,optimize=True)
    A -= 2*np.einsum('pr,sq,r->rspq',h,Id,n,optimize=True)


    DaaJ, DaaK, Dab,Dabintra = pynof.compute_2RDM(pp,n)

######################################
    A += 2*np.einsum('stqu,purt->rspq',I,DaaJ+DaaK,optimize=True)
    A -= 2*np.einsum('stuq,purt->rspq',I,DaaJ+DaaK,optimize=True)
    A += 2*np.einsum('stqu,purt->rspq',I,Dab,optimize=True)
    A += 2*np.einsum('stuq,uprt->rspq',I,Dab,optimize=True)
    A += 2*np.einsum('uptr,stqu->rspq',I,DaaJ+DaaK,optimize=True)
    A -= 2*np.einsum('uprt,stqu->rspq',I,DaaJ+DaaK,optimize=True)
    A += 2*np.einsum('uptr,stqu->rspq',I,Dab,optimize=True)
    A += 2*np.einsum('uprt,stuq->rspq',I,Dab,optimize=True)
    ####
    A += np.einsum('pstu,turq->rspq',I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('psut,turq->rspq',I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('pstu,utrq->rspq',I,Dab,optimize=True)
    A -= np.einsum('psut,turq->rspq',I,Dab,optimize=True)
    A += np.einsum('tuqr,sptu->rspq',I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('turq,sptu->rspq',I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('tuqr,pstu->rspq',I,Dab,optimize=True)
    A -= np.einsum('turq,sptu->rspq',I,Dab,optimize=True)
    ####
    A += np.einsum('sq,tpwu,wurt->rspq',Id,I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('sq,tpuw,wurt->rspq',Id,I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('sq,tpwu,uwrt->rspq',Id,I,Dab,optimize=True)
    A -= np.einsum('sq,tpuw,wurt->rspq',Id,I,Dab,optimize=True)
    A += np.einsum('pr,tuwq,swtu->rspq',Id,I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('pr,tuqw,swtu->rspq',Id,I,DaaJ+DaaK,optimize=True)
    A -= np.einsum('pr,tuwq,wstu->rspq',Id,I,Dab,optimize=True)
    A -= np.einsum('pr,tuqw,swtu->rspq',Id,I,Dab,optimize=True)

    A /= 2





#    A += 2*np.einsum("p,ps,qr->pqsr",n,Id,h,optimize=True)
#    A -= 2*np.einsum("q,ps,qr->pqsr",n,Id,h,optimize=True)
#    A -= 2*np.einsum("p,qr,ps->pqsr",n,Id,h,optimize=True)
#    A += 2*np.einsum("q,qr,ps->pqsr",n,Id,h,optimize=True)
#
#
#    inter = np.outer(n,n)
#    intra = 0*np.outer(n,n)
#
#    # Intrapair Electron Correlation
#    for l in prange(pp.ndoc):
#        ldx = pp.no1 + l
#        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
#        ll = pp.no1 + pp.ndns + pp.ncwo*(pp.ndoc - l - 1)
#        ul = pp.no1 + pp.ndns + pp.ncwo*(pp.ndoc - l)
#
#        inter[ldx,ldx] = 0
#        inter[ldx,ll:ul] = 0
#        inter[ll:ul,ldx] = 0
#        inter[ll:ul,ll:ul] = 0
#
#        intra[ldx,ldx] = c[ldx]*c[ldx]#np.sqrt(n[ldx]*n[ldx])
#        intra[ldx,ll:ul] = c[ldx]*c[ll:ul]#-np.sqrt(n[ldx]*n[ll:ul])
#        intra[ll:ul,ldx] = c[ldx]*c[ll:ul]#-np.sqrt(n[ldx]*n[ll:ul])
#        intra[ll:ul,ll:ul] = np.outer(c[ll:ul],c[ll:ul])#np.sqrt(np.outer(n[ll:ul],n[ll:ul]))
#
#    for i in range(pp.nbeta,pp.nalpha):
#        inter[i,i] = 0
#
#    
#    A += 2*np.einsum("ps,pqrs->pqsr",intra,I,optimize=True)
#    A += 2*np.einsum("ps,pqsr->pqsr",intra,I,optimize=True)
#    A += 2*np.einsum("qr,pqrs->pqsr",intra,I,optimize=True)
#    A += 2*np.einsum("qr,pqsr->pqsr",intra,I,optimize=True)
#
#    A += 2*(2*np.einsum("ps,prqs->pqsr",inter,I,optimize=True) - np.einsum("ps,prsq->pqsr",inter,I,optimize=True))
#    A += 2*(2*np.einsum("qr,prqs->pqsr",inter,I,optimize=True) - np.einsum("qr,prsq->pqsr",inter,I,optimize=True))
#    A -= 2*(2*np.einsum("pr,prqs->pqsr",inter,I,optimize=True) - np.einsum("pr,prsq->pqsr",inter,I,optimize=True))
#    A -= 2*(2*np.einsum("qs,prqs->pqsr",inter,I,optimize=True) - np.einsum("qs,prsq->pqsr",inter,I,optimize=True))
#
#    A -= 2*np.einsum("qr,pt,pstt->pqsr",Id,intra,I,optimize=True)
#    A -= 2*np.einsum("ps,qt,qrtt->pqsr",Id,intra,I,optimize=True)
#    A -= 2*np.einsum("pr,rt,qstt->pqsr",Id,intra,I,optimize=True)
#    A -= 2*np.einsum("qs,st,prtt->pqsr",Id,intra,I,optimize=True)
#
#    A += 2*(2*np.einsum("ps,tp,qtrt->pqsr",Id,inter,I,optimize=True) - np.einsum("ps,tp,qttr->pqsr",Id,inter,I,optimize=True))
#    A -= 2*(2*np.einsum("ps,tq,qtrt->pqsr",Id,inter,I,optimize=True) - np.einsum("ps,tq,qttr->pqsr",Id,inter,I,optimize=True))
#    A += 2*(2*np.einsum("qr,tp,ptst->pqsr",Id,inter,I,optimize=True) - np.einsum("qr,tp,ptts->pqsr",Id,inter,I,optimize=True))
#    A -= 2*(2*np.einsum("qr,tq,ptst->pqsr",Id,inter,I,optimize=True) - np.einsum("qr,tq,ptts->pqsr",Id,inter,I,optimize=True))

    B = np.einsum("pqsr->pqrs",A,optimize=True)

    cp = np.add.outer(c,c)
    cm = np.subtract.outer(c,c)

    Ap = np.einsum("pq,pqrs,rs->pqrs",1/cp,A+B,1/cp,optimize=True)
    Am = np.einsum("pq,pqrs,rs->rspq",1/cm,A-B,1/cm,optimize=True)

    Dp = 1/2*np.einsum("pqrr,pq,r->pqr",B,1/cp,1/c,optimize=True)
    Dpt = 1/2*np.einsum("pqrr,pq,r->rpq",B,1/cp,1/c,optimize=True)

    Ep = 1/4*np.einsum("ppqq,p,q->pq",B,1/c,1/c,optimize=True)

    M = np.zeros((pp.nbf5**2,pp.nbf5**2))

    i = -1
    for q in range(pp.nbf5):
        for p in range(q+1,pp.nbf5):
            i += 1
            j = -1
            for s in range(pp.nbf5):
                for r in range(s+1,pp.nbf5):
                    j += 1
            for s in range(pp.nbf5):
                for r in range(s+1,pp.nbf5):
                    j += 1
                    M[i,j] = Am[p,q,r,s]
                    #M[i,j] = Am[r,s,p,q]
            for r in range(1,pp.nbf5):
                j += 1
                M[i,j] = Dp[p,q,r]
    for q in range(pp.nbf5):
        for p in range(q+1,pp.nbf5):
            i += 1
            j = -1
            for s in range(pp.nbf5):
                for r in range(s+1,pp.nbf5):
                    j += 1
                    M[i,j] = Ap[p,q,r,s]
            for s in range(pp.nbf5):
                for r in range(s+1,pp.nbf5):
                    j += 1
            for r in range(1,pp.nbf5):
                j += 1
                M[i,j] = Dp[p,q,r]
    for p in range(pp.nbf5):
        i += 1
        j = -1
        for s in range(pp.nbf5):
            for r in range(s+1,pp.nbf5):
                j += 1
                M[i,j] = 2*Dpt[p,s,r]
        for s in range(pp.nbf5):
            for r in range(s+1,pp.nbf5):
                j += 1
        for r in range(1,pp.nbf5):
            M[i,j] = Ep[p,r]

    V = np.identity(pp.nbf5**2)

    #for i in range(pp.nbf5*(pp.nbf5-1),pp.nbf5**2):
    #    V[i,i] = 0


#    M = M[:pp.nbf5*(pp.nbf5-1),:pp.nbf5*(pp.nbf5-1)]

    from scipy.linalg import eig

    vals,vecs = eig(M,V)
    #vals,vecs = eig(M)
    for i,val in enumerate(np.sort(vals)):
        print(i,val*27.2114)


#    t1 = 0
#    for q in range(pp.nbf5):
#        for p in range(q+1,pp.nbf5):
#            for s in range(pp.nbf5):
#                for r in range(s+1,pp.nbf5):
#                    t1 += np.abs(B[p,q,r,s] - B[r,s,p,q])
#                    print(p,q,r,s,B[p,q,r,s],B[r,s,p,q])
#    print("val t1: ",t1)
#    
#
#
#    t2 = 0
#    for q in range(pp.nbf5):
#        for p in range(q+1,pp.nbf5):
#            for s in range(pp.nbf5):
#                for r in range(s+1,pp.nbf5):
#                    t2 += np.abs(A[p,q,r,s] - A[r,s,p,q])
#                    print(p,q,r,s,A[p,q,r,s],A[r,s,p,q])
#    print("val t2: ",t2)


    print("test")

#    t1 = 0
#    for q in range(pp.nbf5):
#        for p in range(q+1,pp.nbf5):
#            for s in range(pp.nbf5):
#                for r in range(s+1,pp.nbf5):
#                    t1 += np.abs(Ap[p,q,r,s] - Ap[r,s,p,q])
#                    print(p,q,r,s,Ap[p,q,r,s],Ap[r,s,p,q])
#    print("val t1: ",t1)
#
#    t2 = 0
#    for q in range(pp.nbf5):
#        for p in range(q+1,pp.nbf5):
#            for s in range(pp.nbf5):
#                for r in range(s+1,pp.nbf5):
#                    t2 += np.abs(Am[p,q,r,s] - Am[r,s,p,q])
#                    print(p,q,r,s,Am[p,q,r,s],Am[r,s,p,q])
#    print("val t2: ",t2)



    return






















    cc = np.sqrt(nn.copy())
    cc[2*pp.ndoc:] *= -1 

    RDM2 = np.zeros((2*pp.nbf5,2*pp.nbf5,2*pp.nbf5,2*pp.nbf5))

    for i in range(pp.nbf5):
        for j in range(pp.nbf5):
            if((i>=pp.ndoc and j>=pp.ndoc) or (i==j)):
                RDM2[2*i,   2*i+1, 2*j,   2*j+1] =  np.sqrt(n[i]*n[j]) #abab
                RDM2[2*i+1, 2*i,   2*j+1, 2*j]   =  np.sqrt(n[i]*n[j]) #baba
                RDM2[2*i+1, 2*i,   2*j,   2*j+1] = -np.sqrt(n[i]*n[j]) #baab
                RDM2[2*i,   2*i+1, 2*j+1, 2*j]   = -np.sqrt(n[i]*n[j]) #abba
            else:
                RDM2[2*i,   2*i+1, 2*j,   2*j+1] = -np.sqrt(n[i]*n[j]) #abab
                RDM2[2*i+1, 2*i,   2*j+1, 2*j]   = -np.sqrt(n[i]*n[j]) #baba
                RDM2[2*i+1, 2*i,   2*j,   2*j+1] =  np.sqrt(n[i]*n[j]) #baab
                RDM2[2*i,   2*i+1, 2*j+1, 2*j]   =  np.sqrt(n[i]*n[j]) #abba

#            RDM2[2*i,2*i+1,2*j,2*j+1] =  cc[2*i]*cc[2*j] #abab
#            RDM2[2*i+1,2*i,2*j+1,2*j] =  cc[2*i]*cc[2*j] #baba
#            RDM2[2*i,2*i+1,2*j+1,2*j] = -cc[2*i]*cc[2*j] #abba
#            RDM2[2*i+1,2*i,2*j,2*j+1] = -cc[2*i]*cc[2*j] #baab

    h_s = np.zeros((2*pp.nbf5,2*pp.nbf5))
    for i in range(pp.nbf5):
        for j in range(pp.nbf5):
            h_s[2*i,2*j] = h[i,j]
            h_s[2*i+1,2*j+1] = h[i,j]

    I = np.einsum('rpsq->rspq',I,optimize=True)
    I_s = np.zeros((2*pp.nbf5,2*pp.nbf5,2*pp.nbf5,2*pp.nbf5))
    for i in range(pp.nbf5):
        for j in range(pp.nbf5):
            for k in range(pp.nbf5):
                for l in range(pp.nbf5):
                    I_s[2*i,  2*j,  2*k,  2*l] = I[i,j,k,l]
                    I_s[2*i+1,2*j,  2*k+1,2*l] = I[i,j,k,l]
                    I_s[2*i,  2*j+1,2*k,  2*l+1] = I[i,j,k,l]
                    I_s[2*i+1,2*j+1,2*k+1,2*l+1] = I[i,j,k,l]

    A = np.zeros((2*pp.nbf5,2*pp.nbf5,2*pp.nbf5,2*pp.nbf5))
    Id = np.identity(2*pp.nbf5)

    A += np.einsum('sq,pr,p->rspq',h_s,Id,nn,optimize=True)
    A -= np.einsum('sq,pr,s->rspq',h_s,Id,nn,optimize=True)
    A += np.einsum('pr,sq,q->rspq',h_s,Id,nn,optimize=True)
    A -= np.einsum('pr,sq,r->rspq',h_s,Id,nn,optimize=True)

    A += np.einsum('stqu,purt->rspq',I_s,RDM2,optimize=True) #alpha beta
    A -= np.einsum('stuq,purt->rspq',I_s,RDM2,optimize=True) #alpha beta
    A += np.einsum('uptr,stqu->rspq',I_s,RDM2,optimize=True) #alpha beta
    A -= np.einsum('uprt,stqu->rspq',I_s,RDM2,optimize=True) #alpha beta

    A += 1/2*np.einsum('pstu,turq->rspq',I_s,RDM2,optimize=True) #alpha beta
    A -= 1/2*np.einsum('psut,turq->rspq',I_s,RDM2,optimize=True) #alpha beta
    A += 1/2*np.einsum('tuqr,sptu->rspq',I_s,RDM2,optimize=True) #alpha beta
    A -= 1/2*np.einsum('turq,sptu->rspq',I_s,RDM2,optimize=True) #alpha beta

    A += 1/2*np.einsum('sq,tpwu,wurt->rspq',Id,I_s,RDM2,optimize=True) #alpha beta
    A -= 1/2*np.einsum('sq,tpuw,wurt->rspq',Id,I_s,RDM2,optimize=True) #alpha beta
    A += 1/2*np.einsum('pr,tuwq,swtu->rspq',Id,I_s,RDM2,optimize=True) #alpha beta
    A -= 1/2*np.einsum('pr,tuqw,swtu->rspq',Id,I_s,RDM2,optimize=True) #alpha beta

#    print("A 1")
    #print(A)
#    print(np.sum(A**2))

    ####
    
#    D = A.copy()
#
#    A = np.zeros((2*pp.nbf5,2*pp.nbf5,2*pp.nbf5,2*pp.nbf5))
#
#    A += np.einsum('sp,qr,r->rsqp',h_s,Id,nn,optimize=True)
#    A -= np.einsum('sp,qr,s->rsqp',h_s,Id,nn,optimize=True)
#    A += np.einsum('qr,sp,s->rsqp',h_s,Id,nn,optimize=True)
#    A -= np.einsum('qr,sp,r->rsqp',h_s,Id,nn,optimize=True)
# 
#    A += np.einsum('q,r,pqsr->rsqp',cc,cc,I_s,optimize=True)
#    A += np.einsum('q,r,pqrs->rsqp',cc,cc,I_s,optimize=True)
#    A += np.einsum('s,p,pqsr->rsqp',cc,cc,I_s,optimize=True)
#    A += np.einsum('s,p,pqrs->rsqp',cc,cc,I_s,optimize=True)
#
#    A -= np.einsum('sp,r,t,rqtt->rsqp',Id,cc,cc,I_s,optimize=True)
#    A -= np.einsum('qr,s,t,sptt->rsqp',Id,cc,cc,I_s,optimize=True)
#    A -= np.einsum('rp,p,t,qstt->rsqp',Id,cc,cc,I_s,optimize=True)
#    A -= np.einsum('sq,q,t,prtt->rsqp',Id,cc,cc,I_s,optimize=True)
#
#
#    print("A 2")
#    #print(A)
#    print(np.sum(A**2))
#
#    print("Summed diff",np.sum(A-D))
#    print("Summed squared diff",np.sum((A-D)**2))
#
#    exit()

####

####

    B = np.einsum("rspq->rsqp",A)

    dim = 2*pp.nbf5*(pp.nbf5-1) + 2*pp.nbf5

    print(dim)
    M = np.zeros((dim,dim))

    #First equation (15)
    i = -1
    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            j = -1
            # A_rspq X_pq aa
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r,2*s,2*p,2*q]
            # A_rspq X_pq bb
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r,2*s,2*p+1,2*q+1]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r,2*s,2*p,2*q]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r,2*s,2*p+1,2*q+1]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r,2*s,2*p,2*p]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r,2*s,2*p+1,2*p+1]
    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            j = -1
            # A_rspq X_pq aa
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r+1,2*s+1,2*p,2*q]
            # A_rspq X_pq bb
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r+1,2*s+1,2*p+1,2*q+1]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r+1,2*s+1,2*p,2*q]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r+1,2*s+1,2*p+1,2*q+1]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r+1,2*s+1,2*p,2*p]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r+1,2*s+1,2*p+1,2*p+1]

    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            j = -1
            # A_rspq X_pq aa
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r,2*s,2*p,2*q]
            # A_rspq X_pq bb
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r,2*s,2*p+1,2*q+1]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r,2*s,2*p,2*q]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r,2*s,2*p+1,2*q+1]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r,2*s,2*p,2*p]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r,2*s,2*p,2*p+1]
    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            j = -1
            # A_rspq X_pq aa
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r+1,2*s+1,2*p,2*q]
            # A_rspq X_pq bb
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = B[2*r+1,2*s+1,2*p+1,2*q+1]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r+1,2*s+1,2*p,2*q]
            # B_rspq Y_pq
            for q in range(pp.nbf5):
                for p in range(q+1,pp.nbf5):
                    j += 1
                    M[i,j] = A[2*r+1,2*s+1,2*p+1,2*q+1]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r+1,2*s+1,2*p,2*p]
            # A_rspp Z_p
            for p in range(pp.nbf5):
                j += 1
                M[i,j] = A[2*r+1,2*s+1,2*p+1,2*p+1]


    for r in range(pp.nbf5):
        i += 1
        j = -1
        # A_rspq X_pq aa
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = A[2*r,2*r,2*p,2*q]
        # A_rspq X_pq bb
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = A[2*r,2*r,2*p+1,2*q+1]
        # B_rspq Y_pq
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = B[2*r,2*r,2*p,2*q]
        # B_rspq Y_pq
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = B[2*r,2*r,2*p+1,2*q+1]
        # A_rspp Z_p
        for p in range(pp.nbf5):
            j += 1
            M[i,j] = A[2*r,2*r,2*p,2*p]
        # A_rspp Z_p
        for p in range(pp.nbf5):
            j += 1
            M[i,j] = A[2*r,2*r,2*p+1,2*p+1]
    for r in range(pp.nbf5):
        i += 1
        j = -1
        # A_rspq X_pq aa
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = A[2*r+1,2*r+1,2*p,2*q]
        # A_rspq X_pq bb
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = A[2*r+1,2*r+1,2*p+1,2*q+1]
        # B_rspq Y_pq
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = B[2*r+1,2*r+1,2*p,2*q]
        # B_rspq Y_pq
        for q in range(pp.nbf5):
            for p in range(q+1,pp.nbf5):
                j += 1
                M[i,j] = B[2*r+1,2*r+1,2*p+1,2*q+1]
        # A_rspp Z_p
        for p in range(pp.nbf5):
            j += 1
            M[i,j] = A[2*r+1,2*r+1,2*p,2*p]
        # A_rspp Z_p
        for p in range(pp.nbf5):
            j += 1
            M[i,j] = A[2*r+1,2*r+1,2*p+1,2*p+1]

    print(M)

#    for r in range(2*pp.nbf5*(pp.nbf5-1),2*pp.nbf5**2):
#        M[r,:] *= 1/M[r,r]
#        for s in range(2*pp.nbf5*(pp.nbf5-1),r):
#            M[s,:] -= M[r,:]*M[s,r]
#        for s in range(r+1,2*pp.nbf5**2):
#           M[s,:] -= M[r,:]*M[s,r]
#    for r in range(2*pp.nbf5*(pp.nbf5-1)):
#        for s in range(2*pp.nbf5*(pp.nbf5-1),2*pp.nbf5**2):
#            M[r,:] -= M[s,:]*M[r,s]


    from matplotlib import pyplot as plt
    plt.imshow(M)
    plt.colorbar()
    plt.show()
    M = M[:2*pp.nbf5*(pp.nbf5-1),:2*pp.nbf5*(pp.nbf5-1)]

    i = -1
    v = np.zeros((2*pp.nbf5*(pp.nbf5-1)))
    #v = np.zeros((dim))
    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            v[i] = -(nn[2*r] - nn[2*s])
    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            v[i] = -(nn[2*r+1] - nn[2*s+1])
    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            v[i] = (nn[2*r] - nn[2*s])
    for s in range(pp.nbf5):
        for r in range(s+1,pp.nbf5):
            i += 1
            v[i] = (nn[2*r+1] - nn[2*s+1])


    print(v)

    V = np.zeros((2*pp.nbf5*(pp.nbf5-1),2*pp.nbf5*(pp.nbf5-1)))
    #V = np.zeros((dim,dim))
    np.fill_diagonal(V, v)

    from matplotlib import pyplot as plt
    plt.imshow(M)
    plt.colorbar()
    plt.show()
    from scipy.linalg import eig
    vals,vecs = eig(M,V)
    print("Eigvals:")
    print(vals)

    #vals,vecs = np.linalg.eigh(M)

    vals_complex = np.array([val*27.2114 for val in vals if (np.abs(np.imag(val)) > 0.00000001)])
    print("Vals Complex:", np.size(vals_complex))

    print("Excitation energies (eV):")
    vals_real = np.array([val*27.2114 for val in vals if (np.abs(np.imag(val)) < 0.00000001 and np.real(val) > 0)])
    print(np.sort(vals_real))


    plt.imshow(vecs)
    plt.colorbar()
    plt.show()


    exit()






    M = np.zeros(((2*pp.nbf5)**2,(2*pp.nbf5)**2))

    # i is rows
    # j is columns

    #First equation (15)
    i = -1
    for s in range(2*pp.nbf5):
        for r in range(s+1,2*pp.nbf5):
            i += 1
            j = -1
            # A_rspq X_pq
            for q in range(2*pp.nbf5):
                for p in range(q+1,2*pp.nbf5):
                    j += 1
                    M[i,j] = A[r,s,p,q]
            # B_rspq Y_pq
            for q in range(2*pp.nbf5):
                for p in range(q+1,2*pp.nbf5):
                    j += 1
                    M[i,j] = B[r,s,p,q]
            # A_rspp Z_p
            for p in range(2*pp.nbf5):
                j += 1
                M[i,j] = A[r,s,p,p]

    #Second equation (16)
    for s in range(2*pp.nbf5):
        for r in range(s+1,2*pp.nbf5):
            i += 1
            j = -1
            # B_rspq X_pq
            for q in range(2*pp.nbf5):
                for p in range(q+1,2*pp.nbf5):
                    j += 1
                    M[i,j] = B[r,s,p,q]
            # A_rspq Y_pq
            for q in range(2*pp.nbf5):
                for p in range(q+1,2*pp.nbf5):
                    j += 1
                    M[i,j] = A[r,s,p,q]
            # B_rspp Z_p
            for p in range(2*pp.nbf5):
                j += 1
                M[i,j] = B[r,s,p,p]

    #Third equation (17)
    for r in range(2*pp.nbf5):
        i += 1
        j = -1
        # A_rrpq X_pq
        for q in range(2*pp.nbf5):
            for p in range(q+1,2*pp.nbf5):
                j += 1
                M[i,j] = A[r,r,p,q]
        # B_rrpq Y_pq
        for q in range(2*pp.nbf5):
            for p in range(q+1,2*pp.nbf5):
                j += 1
                M[i,j] = B[r,r,p,q]
        # A_rrpp Z_p
        for p in range(2*pp.nbf5):
            j += 1
            M[i,j] = A[r,r,p,p]

    print(M)
    print("============================")
    # Remove extra variables from (17)

    tmp1 = M[1,:].copy()
    tmp4 = M[4,:].copy()
    tmp7 = M[7,:].copy()
    tmp10 = M[10,:].copy()

    M[0,:] = M[0,:]
    M[1,:] = M[2,:]
    M[2,:] = M[3,:]
    M[3,:] = M[5,:]
    M[4,:] = M[6,:]
    M[5,:] = M[8,:]
    M[6,:] = M[9,:]
    M[7,:] = M[11,:]
    M[8,:] = M[12,:]
    M[9,:] = M[13,:]
    M[10,:] = M[14,:]
    M[11,:] = M[15,:]
    M[12,:] = tmp1
    M[13,:] = tmp4
    M[14,:] = tmp7
    M[15,:] = tmp10


    tmp1 = M[:,1].copy()
    tmp4 = M[:,4].copy()
    tmp7 = M[:,7].copy()
    tmp10 = M[:,10].copy()
    
    M[:,0] = M[:,0]
    M[:,1] = M[:,2]
    M[:,2] = M[:,3]
    M[:,3] = M[:,5]
    M[:,4] = M[:,6]
    M[:,5] = M[:,8]
    M[:,6] = M[:,9]
    M[:,7] = M[:,11]
    M[:,8] = M[:,12]
    M[:,9] = M[:,13]
    M[:,10] = M[:,14]
    M[:,11] = M[:,15]
    M[:,12] = tmp1
    M[:,13] = tmp4
    M[:,14] = tmp7
    M[:,15] = tmp10

    print(M)
#    for r in range(8,12):
#        print("r: {}*************************************************".format(r))
#        M[r,:] *= 1/M[r,r]
#        for s in range(8,r):
#            M[s,:] -= M[r,:]*M[s,r]
#        for s in range(r+1,12):
#            M[s,:] -= M[r,:]*M[s,r] 
#        print(M)
#    for r in range(8):
#        for s in range(8,12):
#            M[r,:] -= M[s,:]*M[r,s]
    M = M[:8,:8]

    i = -1
    v = np.zeros((12))
    for s in range(2*pp.nbf5):
        for r in range(s+1,2*pp.nbf5):
            i += 1
            v[i] = -(nn[r] - nn[s])
            print(i,r,s,nn[r],nn[s],v[i])
    for s in range(2*pp.nbf5):
        for r in range(s+1,2*pp.nbf5):
            i += 1
            v[i] = (nn[r] - nn[s])
            print(i,r,s,nn[r],nn[s],v[i])

    v[0] = v[0]
    v[1] = v[2]
    v[2] = v[3]
    v[3] = v[5]
    v[4] = v[6]
    v[5] = v[8]
    v[6] = v[9]
    v[7] = v[11]

    v = v[0:8]

    print(M)
    print(v)

    V = np.zeros((8,8))
    np.fill_diagonal(V, v)

    from scipy.linalg import eig
    vals,vecs = eig(M,V)

#    from matplotlib import pyplot as plt
#    plt.imshow(M)
    #plt.imshow(M[:pp.nbf5**2-pp.nbf5,:pp.nbf5**2-pp.nbf5])
#    plt.colorbar()
#    plt.show()

#    vals,vecs = np.linalg.eig(M)

    print(vecs)
    print(vals)

    return RDM2


def compute_integrals(wfn,mol,p):

    # Integrador
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    # Overlap, Kinetics, Potential
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = T + V
    
    Dipole = np.asarray(mints.ao_dipole())

    I = []
    b_mnl = []
    if (not p.RI):
        # Integrales de Repulsión Electrónica, ERIs (mu nu | sigma lambda)
        I = np.asarray(mints.ao_eri())
    else:

        orb = wfn.basisset()
        aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", orb.blend())
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

        mnk = np.transpose(mints.ao_eri(aux, zero_bas, orb, orb))
        #Ppq = mints.ao_eri(orb, orb, aux, zero_bas)
        
        metric = mints.ao_eri(aux, zero_bas, aux, zero_bas)
        metric.power(-0.5, 1.e-14)
        p.nbfaux = metric.shape[0]

        mnk = np.squeeze(mnk)
        metric = np.squeeze(metric)

        b_mnl = np.einsum('mnk,kl->mnl', mnk, metric, optimize=True)

    if(p.gpu):
        I = cp.array(I)
        b_mnl = cp.array(b_mnl)

    return S,T,V,H,I,b_mnl,Dipole

def compute_der_integrals(wfn,mol,n,C,cj12,ck12,elag,p):

    mints = psi4.core.MintsHelper(wfn.basisset())

    RDM1 = 2*np.einsum('p,mp,np->mn',n,C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)
    lag = 2*np.einsum('mq,qp,np->mn',C,elag,C,optimize=True)
    
    grad = np.zeros((p.natoms,3))

    grad += np.array(mol.nuclear_repulsion_energy_deriv1())

    print(grad)

    for i in range(p.natoms):
        dSx,dSy,dSz = np.array(mints.ao_oei_deriv1("OVERLAP",i))
        grad[i,0] -= np.einsum('mn,mn->',lag,dSx,optimize=True)
        grad[i,1] -= np.einsum('mn,mn->',lag,dSy,optimize=True)
        grad[i,2] -= np.einsum('mn,mn->',lag,dSz,optimize=True)

        dTx,dTy,dTz = np.array(mints.ao_oei_deriv1("KINETIC",i))
        grad[i,0] += np.einsum('mn,mn->',RDM1,dTx,optimize=True)
        grad[i,1] += np.einsum('mn,mn->',RDM1,dTy,optimize=True)
        grad[i,2] += np.einsum('mn,mn->',RDM1,dTz,optimize=True)

        dVx,dVy,dVz = np.array(mints.ao_oei_deriv1("POTENTIAL",i))
        grad[i,0] += np.einsum('mn,mn->',RDM1,dVx,optimize=True)
        grad[i,1] += np.einsum('mn,mn->',RDM1,dVy,optimize=True)
        grad[i,2] += np.einsum('mn,mn->',RDM1,dVz,optimize=True)

    np.fill_diagonal(cj12,0) # Remove diag.
    np.fill_diagonal(ck12,0) # Remove diag.

    if not p.RI:

        RDM2 = np.einsum('pq,mp,np,sq,lq->mnsl',cj12,C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)
        RDM2 += np.einsum('p,mp,np,sp,lp->mnsl',n[:p.nbeta],C[:,:p.nbeta],C[:,:p.nbeta],C[:,:p.nbeta],C[:,:p.nbeta],optimize=True)
        RDM2 += np.einsum('p,mp,np,sp,lp->mnsl',n[p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],optimize=True)
        RDM2 -= np.einsum('pq,mp,lp,sq,nq->mnsl',ck12,C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)

        for i in range(p.natoms):
            derix,deriy,deriz = np.array(mints.ao_tei_deriv1(i))
            grad[i,0] += np.einsum("mnsl,mnsl->",RDM2,derix,optimize=True)
            grad[i,1] += np.einsum("mnsl,mnsl->",RDM2,deriy,optimize=True)
            grad[i,2] += np.einsum("mnsl,mnsl->",RDM2,deriz,optimize=True)

    else:

        orb = wfn.basisset()
        aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", orb.blend())
        mints.set_basisset("aux",aux)
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

        mnP = np.transpose(mints.ao_eri(aux, zero_bas, orb, orb))
        mnP = np.squeeze(mnP)

        metric = mints.ao_eri(aux, zero_bas, aux, zero_bas)
        metric.power(-1.0, 1.e-14)
        metric = np.squeeze(metric)

        tmp1 = np.einsum('mp,mnP->pnP',C[:,:p.nbf5],mnP,optimize=True)
        tmp1 = np.einsum('nq,pnP->pqP',C[:,:p.nbf5],tmp1,optimize=True)
        tmp1 = np.einsum('pqP,PQ->pqQ',tmp1,metric,optimize=True)
        tmp2 = np.einsum('pq,pqQ->pqQ',-ck12,tmp1,optimize=True)
        tmp3 = np.einsum('sq,pqQ->psQ',C[:,:p.nbf5],tmp2,optimize=True)
        val1 = np.einsum('lp,psQ->lsQ',C[:,:p.nbf5],tmp3,optimize=True)
        tmp3 = None
        val2 = np.einsum('pqQ,pqR->QR',tmp1,tmp2,optimize=True)

        tmp2 = np.einsum('pq,ppQ->qQ',cj12,tmp1,optimize=True)
        val1 += np.einsum('sq,lq,qQ->slQ',C[:,:p.nbf5],C[:,:p.nbf5],tmp2,optimize=True)
        val2 += np.einsum('qqQ,qR->QR',tmp1,tmp2,optimize=True)

        tmp2 = np.einsum('p,ppQ->pQ',n[:p.nbeta],tmp1[:p.nbeta,:p.nbeta],optimize=True)
        val1 += np.einsum('lp,sp,pQ->lsQ',C[:,:p.nbeta],C[:,:p.nbeta],tmp2,optimize=True)
        val2 += np.einsum('ppQ,pR->QR',tmp1[:p.nbeta,:p.nbeta],tmp2,optimize=True)

        tmp2 = np.einsum('p,ppQ->pQ',n[p.nalpha:p.nbf5],tmp1[p.nalpha:p.nbf5,p.nalpha:p.nbf5],optimize=True)
        val1 += np.einsum('lp,sp,pQ->lsQ',C[:,p.nalpha:p.nbf5],C[:,p.nalpha:p.nbf5],tmp2,optimize=True)
        val2 += np.einsum('ppQ,pR->QR',tmp1[p.nalpha:p.nbf5,p.nalpha:p.nbf5],tmp2,optimize=True)

        for i in range(p.natoms):
            d3x,d3y,d3z = np.array(mints.ao_3center_deriv1(i,"aux"))
            d2x,d2y,d2z = np.array(mints.ao_metric_deriv1(i,"aux"))
            grad[i,0] += 2*np.einsum('lsQ,Qsl->',val1,d3x,optimize=True)
            grad[i,0] -= np.einsum('QR,QR->',val2,d2x,optimize=True)
            grad[i,1] += 2*np.einsum('lsQ,Qsl->',val1,d3y,optimize=True)
            grad[i,1] -= np.einsum('QR,QR->',val2,d2y,optimize=True)
            grad[i,2] += 2*np.einsum('lsQ,Qsl->',val1,d3z,optimize=True)
            grad[i,2] -= np.einsum('QR,QR->',val2,d2z,optimize=True)

    return grad


######################################### J_mn^(j) K_mn^(j) #########################################

def computeJKj(C,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            J,K = JKj_RI_GPU(C,b_mnl,p)
        else:
            J,K = JKj_Full_GPU(C,I,p)
    else:
        if(p.jit):
            if(p.RI):
                J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
            else:
                J,K = JKj_Full_jit(C,I,p.nbf,p.nbf5)
        else:
            if(p.RI):
                J,K = JKj_RI(C,b_mnl,p)
            else:
                J,K = JKj_Full(C,I,p)

    return J,K

#########################################


def JKj_Full(C,I,p):
    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)
    #hstarj
    J = np.tensordot(D, I, axes=([1,2],[2,3]))
    #hstark
    K = np.tensordot(D, I, axes=([1,2],[1,3]))
    
    return J,K

def JKj_RI(C,b_mnl,p):
    
    #b_transform
    b_qnl = np.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_qql = np.einsum('nq,qnl->ql',C[:,0:p.nbf5],b_qnl, optimize=True)
    #hstarj
    J = np.tensordot(b_qql, b_mnl, axes=([1],[2]))
    #hstark
    K = np.einsum('qml,qnl->qmn', b_qnl, b_qnl, optimize=True)
    
    return J,K

def JKj_Full_GPU(C,I,p):

    #denmatj
    D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5], optimize=True)
    #hstarj
    J = cp.tensordot(D, I, axes=([1,2],[2,3]))
    #hstark
    K = np.einsum("inl,mnsl->ims",D,I,optimize=True)
    #K = cp.tensordot(D, I, axes=([1,2],[1,3]))
    
    return J.get(),K.get()

def JKj_RI_GPU(C,b_mnl,p):

    C = cp.array(C)
    #b_transform
    b_qnl = cp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_qql = cp.einsum('nq,qnl->ql',C[:,0:p.nbf5],b_qnl, optimize=True)
    #hstarj
    J = cp.tensordot(b_qql, b_mnl, axes=([1],[2]))
    #hstark
    K = cp.einsum('qml,qnl->qmn', b_qnl, b_qnl, optimize=True)

    return J.get(),K.get()

@njit(parallel=True, cache=True)
def JKj_Full_jit(C,I,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf5,nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbf5):
                D[i][mu][nu] = C[mu][i]*C[nu][i]
                D[i][nu][mu] = D[i][mu][nu]

    #hstarj
    J = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for n in prange(m+1):
                for s in range(nbf):
                    for l in range(nbf):
                        J[i][m][n] += D[i][s][l]*I[m][n][s][l]
                J[i][n][m] = J[i][m][n]

    #hstark
    K = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for s in prange(m+1):
                for n in range(nbf):
                    for l in range(nbf):
                        K[i][m][s] += D[i][n][l]*I[m][n][s][l]
                K[i][s][m] = K[i][m][s]

    return J,K

@njit(parallel=True, cache=True)
def JKj_RI_jit(C,b_mnl,nbf,nbf5,nbfaux):

    #denmatj
    J = np.zeros((nbf5,nbf,nbf))
    K = np.zeros((nbf5,nbf,nbf))
    for q in prange(nbf5):
        b_qnl = np.zeros((nbf,nbfaux))
        b_qql = np.zeros((nbfaux))
        for l in range(nbfaux):
            for n in range(nbf):
                for m in range(nbf):
                    b_qnl[n][l] += C[m][q]*b_mnl[m][n][l]
                b_qql[l] += C[n][q]*b_qnl[n][l]
            for n in range(nbf):
                for m in range(nbf):
                    K[q][m][n] += b_qnl[m][l]*b_qnl[n][l]
                    J[q][m][n] += b_qql[l]*b_mnl[m][n][l]

    return J,K

######################################### J_pq K_pq #########################################

def computeJKH_MO(C,H,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            J_MO,K_MO,H_core = JKH_MO_RI_GPU(C,H,b_mnl,p)
        else:
            J_MO,K_MO,H_core = JKH_MO_Full_GPU(C,H,I,p)
    else:
        if(p.jit):
            if(p.RI):
                J_MO,K_MO,H_core = JKH_MO_RI_jit(C,H,b_mnl,p.nbf,p.nbf5,p.nbfaux)
            else:
                J_MO,K_MO,H_core = JKH_MO_Full_jit(C,H,I,p.nbf,p.nbf5)
        else:
            if(p.RI):
                J_MO,K_MO,H_core = JKH_MO_RI(C,H,b_mnl,p)
            else:
                J_MO,K_MO,H_core = JKH_MO_Full(C,H,I,p)

    return J_MO,K_MO,H_core

#########################################

def JKH_MO_Full(C,H,I,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #QJMATm
    J = np.tensordot(D, I, axes=([1,2],[2,3]))
    J_MO = np.tensordot(J, D,axes=((1,2),(1,2)))
    #QKMATm
    K = np.tensordot(D, I, axes=([1,2],[1,3]))
    K_MO = np.tensordot(K, D, axes=([1,2],[1,2]))
    #QHMATm
    H_core = np.tensordot(D, H, axes=([1,2],[0,1]))

    return J_MO,K_MO,H_core

def JKH_MO_RI(C,H,b_mnl,p):

    #denmatj
    D = np.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #b transform
    b_pnl = np.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_pql = np.einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)
    #QJMATm
    J_MO = np.einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)
    #QKMATm
    K_MO = np.einsum('pql,pql->pq', b_pql, b_pql, optimize=True)
    #QHMATm
    H_core = np.tensordot(D,H, axes=([1,2],[0,1]))

    return J_MO,K_MO,H_core

def JKH_MO_Full_GPU(C,H,I,p):

    C = cp.array(C)
    H = cp.array(H)
    #denmatj
    D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #QJMATm
    J = cp.tensordot(D, I, axes=([1,2],[2,3]))
    J_MO = cp.tensordot(J, D,axes=((1,2),(1,2)))
    #QKMATm
    K = cp.tensordot(D, I, axes=([1,2],[1,3]))
    K_MO = cp.tensordot(K, D, axes=([1,2],[1,2]))
    #QHMATm
    H_core = cp.tensordot(D, H, axes=([1,2],[0,1]))

    return J_MO.get(),K_MO.get(),H_core.get()

def JKH_MO_RI_GPU(C,H,b_mnl,p):

    C = cp.array(C)
    H = cp.array(H)
    #denmatj
    D = cp.einsum('mi,ni->imn', C[:,0:p.nbf5], C[:,0:p.nbf5],optimize=True)
    #b transform
    b_pnl = cp.tensordot(C[:,0:p.nbf5],b_mnl, axes=([0],[0]))
    b_pql = cp.einsum('nq,pnl->pql',C[:,0:p.nbf5],b_pnl, optimize=True)
    #QJMATm
    J_MO = cp.einsum('ppl,qql->pq', b_pql, b_pql, optimize=True)
    #QKMATm
    K_MO = cp.einsum('pql,pql->pq', b_pql, b_pql, optimize=True)
    #QHMATm
    H_core = cp.tensordot(D,H, axes=([1,2],[0,1]))

    return J_MO.get(),K_MO.get(),H_core.get()

@njit(parallel=True, cache=True)
def JKH_MO_Full_jit(C,H,I,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf5,nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu+1):
            for i in prange(nbf5):
                D[i][mu][nu] = C[mu][i]*C[nu][i]
                D[i][nu][mu] = D[i][mu][nu]
    #QJMATm
    J = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for n in prange(m+1):
                for s in range(nbf):
                    for l in range(nbf):
                        J[i][m][n] += D[i][s][l]*I[m][n][s][l]
                J[i][n][m] = J[i][m][n]
    J_MO = np.zeros((nbf5,nbf5))
    for i in prange(nbf5):
        for j in prange(i+1):
            for m in range(nbf):
                for n in range(nbf):
                    J_MO[i][j] += D[j][m][n]*J[i][m][n]
            J_MO[j][i] = J_MO[i][j]

    #QKMATm
    K = np.zeros((nbf5,nbf,nbf))
    for i in prange(nbf5):
        for m in prange(nbf):
            for s in prange(m+1):
                for n in range(nbf):
                    for l in range(nbf):
                        K[i][m][s] += D[i][n][l]*I[m][n][s][l]
                K[i][s][m] = K[i][m][s]
    K_MO = np.zeros((nbf5,nbf5))
    for i in prange(nbf5):
        for j in prange(i+1):
            for m in range(nbf):
                for s in range(nbf):
                    K_MO[i][j] += D[j][m][s]*K[i][m][s]
            K_MO[j][i] = K_MO[i][j]

    #QHMATm
    H_core = np.zeros((nbf5))
    for i in prange(nbf5):
        for m in range(nbf):
            for n in range(m):
                H_core[i] += 2*D[i][m][n]*H[m][n]
            H_core[i] += D[i][m][m]*H[m][m]

    return J_MO,K_MO,H_core

@njit(parallel=True, cache=True)
def JKH_MO_RI_jit(C,H,b_mnl,nbf,nbf5,nbfaux):

    b_pql = np.zeros((nbf5,nbf5,nbfaux))
    for p in prange(nbf5):
        for n in range(nbf):
            for l in range(nbfaux):
                b_pnl = 0
                for m in range(nbf):
                    b_pnl += C[m][p]*b_mnl[m][n][l]
                for q in range(p+1):
                    b_pql[p][q][l] += C[n][q]*b_pnl

    H_core = np.zeros((nbf5))
    J_MO = np.zeros((nbf5,nbf5))
    K_MO = np.zeros((nbf5,nbf5))
    for p in prange(nbf5):
        for m in range(nbf):
            for n in range(nbf):
                H_core[p] += C[m][p]*H[m][n]*C[n][p]
        for q in range(p+1):
            for l in range(nbfaux):
                J_MO[p][q] += b_pql[p][p][l]*b_pql[q][q][l]
                K_MO[p][q] += b_pql[p][q][l]*b_pql[p][q][l]
            J_MO[q][p] = J_MO[p][q]
            K_MO[q][p] = K_MO[p][q]

    return J_MO,K_MO,H_core

######################################### J_mn^(j) K_mn^(j) #########################################

def computeD_HF(C,p):

    if(p.gpu):
        D = cp.einsum('mj,nj->mn',C[:,:p.nbeta],C[:,:p.nbeta],optimize=True)
        return D.get()
    else:
        D = np.einsum('mj,nj->mn',C[:,:p.nbeta],C[:,:p.nbeta],optimize=True)
        return D

def computeDalpha_HF(C,p):

    if(p.gpu):
        D = cp.einsum('mj,nj->mn',C[:,p.nbeta:p.nalpha],C[:,p.nbeta:p.nalpha],optimize=True)
        return D.get()
    else:
        D = np.einsum('mj,nj->mn',C[:,p.nbeta:p.nalpha],C[:,p.nbeta:p.nalpha],optimize=True)
        return D

def computeJK_HF(D,I,b_mnl,p):

    if(p.gpu):
#        if(p.RI):
#            J,K = JKj_RI_GPU(C,b_mnl,p)
#        else:
        J,K = JK_HF_Full_GPU(D,I,p)
    else:
#        if(p.RI):
#            J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
#        else:
        J,K = JK_HF_Full_jit(D,I,p.nbeta,p.nbf,p.nbf5)

    return J,K

def JK_HF_Full_jit(D,I,nbeta,nbf,nbf5):

    J = np.einsum("ls,mnsl->mn",D,I,optimize=True)
    K = np.einsum("nl,mnsl->ms",D,I,optimize=True)

    return J,K

def JK_HF_Full_GPU(D,I,p):

    #denmatj
    J = cp.einsum("ls,mnsl->mn",D,I,optimize=True)
    K = cp.einsum("nl,mnsl->ms",D,I,optimize=True)

    return J.get(),K.get()

def computeJKalpha_HF(C,I,b_mnl,p):

    if(p.gpu):
#        if(p.RI):
#            J,K = JKj_RI_GPU(C,b_mnl,p)
#        else:
        D,J,K = JKalpha_HF_Full_GPU(C,I,p)
    else:
#        if(p.RI):
#            J,K = JKj_RI_jit(C,b_mnl,p.nbf,p.nbf5,p.nbfaux)
#        else:
        D,J,K = JKalpha_HF_Full_jit(C,I,p.nbeta,p.nalpha,p.nbf,p.nbf5)

    return D,J,K

@njit(parallel=True, cache=True)
def JKalpha_HF_Full_jit(C,I,nbeta,nalpha,nbf,nbf5):

    #denmatj
    D = np.zeros((nbf,nbf))
    for mu in prange(nbf):
        for nu in prange(mu):
            for i in prange(nbeta,nalpha):
                D[mu][nu] += C[mu][i]*C[nu][i]
            D[nu][mu] = D[mu][nu]

    #hstarj
    J = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for n in prange(m):
            for s in range(nbf):
                for l in range(nbf):
                    J[m][n] += D[s][l]*I[m][n][s][l]
            J[n][m] = J[m][n]

    #hstark
    K = np.zeros((nbf,nbf))
    for m in prange(nbf):
        for s in prange(m):
            for n in range(nbf):
                for l in range(nbf):
                    K[m][s] += D[n][l]*I[m][n][s][l]
            K[s][m] = K[m][s]

    return D,J,K

def JKalpha_HF_Full_GPU(C,I,p):

    #denmatj
    D = cp.einsum("mj,nj->mn",C[:,p.nbeta:p.nalpha],C[:,p.nbeta:p.nalpha],optimize=True)
    J = cp.einsum("ls,mnsl->mn",D,I,optimize=True)
    K = cp.einsum("nl,mnsl->ms",D,I,optimize=True)

    return D.get(),J.get(),K.get()

def compute_iajb(C,I,b_mnl,p):

    if(p.gpu):
        iajb = iajb_Full_GPU(C,I,p)
    else:
        iajb = iajb_Full_jit(C,I,p.no1,p.nalpha,p.nbf,p.nbf5)

    return iajb

def iajb_Full_jit(C,I,no1,nalpha,nbf,nbf5):

    iajb = np.einsum('mi,na,mnsl,sj,lb->iajb',C[:,no1:nalpha],C[:,nalpha:nbf],I,C[:,no1:nalpha],C[:,nalpha:nbf],optimize=True)

    return iajb

def iajb_Full_GPU(C,I,p):

    iajb = cp.einsum('mi,na,mnsl,sj,lb->iajb',C[:,p.no1:p.nalpha],C[:,p.nalpha:p.nbf],I,C[:,p.no1:p.nalpha],C[:,p.nalpha:p.nbf],optimize=True)

    return iajb.get()

def compute_pqrt(C,I,b_mnl,p):

    if(p.gpu):
        pqrt = pqrt_Full_GPU(C,I,p)
    else:
        pqrt = pqrt_Full_jit(C,I,p.no1,p.nalpha,p.nbf,p.nbf5)

    return pqrt


def pqrt_Full_jit(C,I,no1,nalpha,nbf,nbf5):

    pqrt = np.einsum('mp,nq,mnsl,sr,lt->pqrt',C,C,I,C,C,optimize=True)

    return pqrt

def pqrt_Full_GPU(C,I,p):

    pqrt = cp.einsum('mp,nq,mnsl,sr,lt->pqrt',C,C,I,C,C,optimize=True)

    return pqrt.get()

def JKH_MO_tmp(C,H,I,b_mnl,p):

    if(p.gpu):
        if(p.RI):
            H_MO, b_MO = Integrals_MO_RI_GPU(C,H,b_mnl,p)
            pass
        else:
            H_MO, I_MO = Integrals_MO_Full_GPU(C,H,I,p)
    else:
        if(p.RI):
            H_MO, b_MO = Integrals_MO_RI_CPU(C,H,b_mnl,p)
        else:
            H_MO, I_MO = Integrals_MO_Full_CPU(C,H,I,p)

    if(p.RI):
        return H_MO,b_MO
    else:
        return H_MO,I_MO

def Integrals_MO_Full_CPU(C,H,I,p):

    H_mat = np.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    I_MO = np.einsum("mp,nq,mnsl,sr,lt->pqrt",C,C[:,:p.nbf5],I,C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)

    return H_mat,I_MO

def Integrals_MO_Full_GPU(C,H,I,p):

    H_mat = cp.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    I_MO = cp.einsum("mp,nq,mnsl,sr,lt->pqrt",C,C[:,:p.nbf5],I,C[:,:p.nbf5],C[:,:p.nbf5],optimize=True)

    return H_mat,I_MO

def Integrals_MO_RI_CPU(C,H,b_mnl,p):

    H_mat = np.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    b_MO = np.einsum("mp,nq,mnl->pql",C,C[:,:p.nbf5],b_mnl,optimize=True)

    return H_mat,b_MO

def Integrals_MO_RI_GPU(C,H,b_mnl,p):

    H_mat = cp.einsum("mi,mn,nj->ij",C,H,C[:,:p.nbf5],optimize=True)
    b_MO = cp.einsum("mp,nq,mnl->pql",C,C[:,:p.nbf5],b_mnl,optimize=True)

    return H_mat,b_MO

