import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.optimize import root
from time import time
import pynof
from numba import prange,njit,jit
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg,minres
from scipy.linalg import fractional_matrix_power
from scipy.linalg import eigh
from scipy.linalg import eig
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.special import roots_legendre

@njit
def W(i,a,j,b,eri_at,wmn_at,eig,nab,bigomega,cfreq):

    w_iajb = eri_at
    for s in range(nab):
        w_iajb += wmn_at[i,a,s]*wmn_at[j,b,s]*(1/(cfreq-bigomega[s]) - 1/(cfreq+bigomega[s]))

    return w_iajb

@njit(parallel=True)
def integrated_omega(i,a,j,b,eri_at,wmn_at,eig,nab,bigomega,weights,freqs,cfreqs,order):

    integral = 0
    for ii in prange(order):
        integral += W(i,a,j,b,eri_at,wmn_at,eig,nab,bigomega,cfreqs[ii])*weights[ii] * 2*(eig[i] - eig[a])/((eig[i] - eig[a])**2 + freqs[ii]**2) *2*(eig[j] - eig[b])/((eig[j] - eig[b])**2 + freqs[ii]**2)

    if(np.abs(integral.imag)>1e-4):
        print("Warning, large imaginary component",integral)

    integral = integral.real

    return integral

@njit
def rpa_sosex(freqs,weights,sumw,order,wmn_at,eig,pqrt,pqrt_at,bigomega,nab,nbeta,nalpha,nbf):

    iEcRPA=0
    iEcSOSEX=0

    weights = weights/sumw
    freqs = (freqs + 1)/2

    weights = weights/(1-freqs)**2
    freqs = freqs/(1-freqs)
    cfreqs = freqs*1j

    for a in range(nalpha,nbf):
        for b in range(nalpha,nbf):
            for i in range(nbeta):
                for j in range(nbeta):
                    integral = integrated_omega(i,a,j,b,pqrt_at[i,b,j,a],wmn_at,eig,nab,bigomega,weights,freqs,cfreqs,order)
                    iEcRPA += -pqrt[i,b,j,a]*integral
                    iEcSOSEX += pqrt[i,a,j,b]*integral
                for j in range(nbeta,nalpha):
                    integral = integrated_omega(i,a,j,b,pqrt_at[i,b,j,a],wmn_at,eig,nab,bigomega,weights,freqs,cfreqs,order)
                    iEcRPA += -0.5*pqrt[i,b,j,a]*integral
                    iEcSOSEX += 0.5*pqrt[i,a,j,b]*integral
            for i in range(nbeta,nalpha):
                for j in range(nbeta):
                    integral = integrated_omega(i,a,j,b,pqrt_at[i,b,j,a],wmn_at,eig,nab,bigomega,weights,freqs,cfreqs,order)
                    iEcRPA += -0.5*pqrt[i,b,j,a]*integral
                    iEcSOSEX += 0.5*pqrt[i,a,j,b]*integral
                for j in range(nbeta,nalpha):
                    integral = integrated_omega(i,a,j,b,pqrt_at[i,b,j,a],wmn_at,eig,nab,bigomega,weights,freqs,cfreqs,order)
                    iEcRPA += -0.25*pqrt[i,b,j,a]*integral
                    iEcSOSEX += 0.25*pqrt[i,a,j,b]*integral

    iEcRPA = iEcRPA/np.pi
    iEcSOSEX = 0.5*iEcSOSEX/np.pi

    return iEcRPA, iEcSOSEX

def gw_gm_eq(wmn_at,pqrt,eig,bigomega,XpY,nab,p):
    EcGoWo = 0
    EcGMSOS = 0
    for a in range(p.nalpha,p.nbf):
        for b in range(p.nalpha,p.nbf):
            for i in range(p.nbeta):
                for j in range(p.nbeta):
                    l = j*(p.nbf-p.nalpha) + (b-p.nalpha)
                    for s in range(nab):
                        EcGoWo += wmn_at[i,a,s]*pqrt[j,a,i,b]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)
                        EcGMSOS -= wmn_at[i,a,s]*pqrt[j,b,i,a]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)
    if(p.nalpha != p.nbeta):
        for a in range(p.nalpha,p.nbf):
            for b in range(p.nalpha,p.nbf):
                for i in range(p.nbeta):
                    for j in range(p.nbeta,p.nalpha):
                        l = j*(p.nbf-p.nalpha) + (b-p.nalpha)
                        for s in range(nab):
                            EcGoWo += 0.5*wmn[i,a,s]*pqrt[j,a,i,b]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)
                            EcGMSOS -= 0.5*wmn[i,a,s]*pqrt[j,b,i,a]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)
                for i in range(p.nbeta,p.nalpha):
                    for j in range(p.nbeta,p.nalpha):
                        l = j*(p.nbf-p.nalpha) + (b-p.nalpha)
                        for s in range(nab):
                            EcGoWo += 0.5*wmn[i,a,s]*pqrt[j,a,i,b]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)
                            EcGMSOS -= 0.5*wmn[i,a,s]*pqrt[j,b,i,a]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)
                    for j in range(p.nbeta,p.nalpha):
                        if(i!=j):
                            l = j*(p.nbf-p.nalpha) + (b-p.nalpha)
                            for s in range(nab):
                                EcGoWo += 0.25*wmn[i,a,s]*pqrt[j,a,i,b]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)
                                EcGMSOS -= 0.25*wmn[i,a,s]*pqrt[j,b,i,a]*XpY[l,s]*np.sqrt(2)/(eig[i]-eig[a]-bigomega[s]+1e-10)

    return EcGoWo, EcGMSOS 

def build_wmn(pqrt,pqrt_at,XpY,nab,p):
    wmn = np.zeros((p.nbf,p.nbf,nab))
    wmn_at = np.zeros((p.nbf,p.nbf,nab))
    for k in range(nab):
        for i in range(p.nbf):
            for j in range(p.nbf):
                a = 0
                b = p.nalpha
                for l in range(nab):
                    wmn[i,j,k] += pqrt[a,j,i,b]*XpY[l,k]
                    wmn_at[i,j,k] += pqrt_at[a,j,i,b]*XpY[l,k]
                    b += 1
                    if(b==p.nbf):
                        b = p.nalpha
                        a += 1
                wmn[i,j,k] *= np.sqrt(2)
                wmn_at[i,j,k] *= np.sqrt(2)

    return wmn,wmn_at


def td_polarizability(EcRPA,C,Dipole,bigomega,XpY,nab,p):

    Dipole_MO = Dipole
    for i in range(3):
        Dipole_MO[i] = np.einsum("mi,mn,nj->ij",C,Dipole[i],C,optimize=True)

    oscstr = np.zeros((nab))
    tempm = np.zeros((3,nab))
    for k in range(nab):
        EcRPA = EcRPA + 0.5*np.abs(bigomega[k])
        dipsum = np.zeros((3))
        l=0
        for i in range(p.nalpha):
            for a in range(p.nalpha,p.nbf):
                for xyz in range(3):
                    dipsum[xyz] += np.sqrt(2)*Dipole_MO[xyz,i,a]*XpY[l,k]
                l += 1
        tempm[:,k] = dipsum[:]
        oscstr[k] = 2.0*(dipsum[0]**2 + dipsum[1]**2 + dipsum[2]**2)*bigomega[k]/3

    print("TD-H (RPA) CASIDA eq. solved")
    print("N. excitation   a.u.         eV            nm      osc. strenght")
    for i in range(nab):
        if(oscstr[i] > 10**-6):
            print("{:^ 10d} {: 11.5f} {: 11.5f} {: 11.5f} {: 11.5f} ".format(i,bigomega[i],bigomega[i]*27.211399,1239.84193/(bigomega[i]*27.211399),oscstr[i]))
    print("")
    
    print("Static Polarizability:")
    print("")
    staticpol = 2*np.einsum("ji,ki,i->jk",tempm,tempm,1/(bigomega + 1e-10),optimize=True)
    for xyz in range(3):
        print("{:15.5f} {:15.5f} {:15.5f}".format(staticpol[xyz,0],staticpol[xyz,1],staticpol[xyz,2]))
    print("")
    print("Trace of the static polarizability {:9.5f}".format((staticpol[0,0]+staticpol[1,1]+staticpol[2,2])/3))
    print("")

    return EcRPA

def build_F_MO(C,H,I,b_mnl,p):
    D = pynof.computeD_HF(C,p)
    if(p.MSpin==0):
        if(p.nsoc>0):
            Dalpha = pynof.computeDalpha_HF(C,p)
            D = D + 0.5*Dalpha
        J,K = pynof.computeJK_HF(D,I,b_mnl,p)
        F = H + 2*J - K
        EHFL = np.trace(np.matmul(D,H)+np.matmul(D,F))
    elif(not p.MSpin==0):
        Dalpha = pynof.computeDalpha_HF(C,p)
        J,K = pynof.computeJK_HF(D,I,b_mnl,p)
        F = 2*J - K
        EHFL = 2*np.trace(np.matmul(D+0.5*Dalpha,H))+np.trace(np.matmul(D+Dalpha,F))
        F = H + F
        if(p.nsoc>1):
            J,K = pynof.computeJK_HF(0.5*Dalpha,I,b_mnl,p)
            Falpha = J - K
            EHFL = EHFL + 2*np.trace(np.matmul(0.5*Dalpha,Falpha))
            F = F + Falpha

    F_MO = np.matmul(np.matmul(np.transpose(C),F),C)

    eig = np.einsum("ii->i",F_MO) #JFHLY: Print info

    return EHFL,F_MO

@njit
def ERIS_attenuated(pqrt,Cintra,Cinter,no1,ndoc,nsoc,ndns,ncwo,nbf5,nbf):
    subspaces = np.zeros((nbf))
    for i in range(no1):
        subspaces[i] = i
    for i in range(ndoc):
        subspaces[no1+i] = no1+i
        ll = no1 + ndns + ncwo*(ndoc-i-1)
        ul = no1 + ndns + ncwo*(ndoc-i)
        subspaces[ll:ul] = no1+i
    for i in range(nsoc):
        subspaces[no1+ndoc+i] = no1+ndoc+i
    subspaces[nbf5:] = -1

    pqrt_at = np.zeros((nbf,nbf,nbf,nbf))
    for pp in range(nbf):
        for q in range(nbf):
            for r in range(nbf):
                for t in range(nbf):
                    if(subspaces[pp]==subspaces[q] and subspaces[pp]==subspaces[r] and subspaces[pp]==subspaces[t] and subspaces[pp]!=-1):
                        pqrt_at[pp,q,r,t] = pqrt[pp,q,r,t] * Cintra[pp]*Cintra[q]*Cintra[r]*Cintra[t]
                    else:
                        pqrt_at[pp,q,r,t] = pqrt[pp,q,r,t] * Cinter[pp]*Cinter[q]*Cinter[r]*Cinter[t]

    return pqrt_at

@njit
def F_MO_attenuated(F_MO,Cintra,Cinter,no1,nalpha,ndoc,nsoc,ndns,ncwo,nbf5,nbf):

    F_MO_at = np.zeros((nbf,nbf))

    subspaces = np.zeros((nbf))
    for i in range(no1):
        subspaces[i] = i
    for i in range(ndoc):
        subspaces[no1+i] = no1+i
        ll = no1 + ndns + ncwo*(ndoc-i-1)
        ul = no1 + ndns + ncwo*(ndoc-i)
        subspaces[ll:ul] = no1+i
    for i in range(nsoc):
        subspaces[no1+ndoc+i] = no1+ndoc+i
    subspaces[nbf5:] = -1

    for pp in range(nbf):
        for q in range(nbf):
            if(pp != q):
                if(subspaces[pp]==subspaces[q]):
                    F_MO_at[pp,q] = F_MO[pp,q]*Cintra[pp]*Cintra[q]
                else:
                    F_MO_at[pp,q] = F_MO[pp,q]*Cinter[pp]*Cinter[q]
            else:
                F_MO_at[pp,q] = F_MO[pp,q]
    F_MO_at[:nalpha,nalpha:nbf] = 0.0
    F_MO_at[nalpha:nbf,:nalpha] = 0.0
 
    return F_MO_at

def mp2_eq(eig,pqrt,pqrt_at,p):

    EcMP2 = 0
    for a in range(p.nalpha,p.nbf):
        for b in range(p.nalpha,p.nbf):
            for i in range(p.nbeta):
                for j in range(p.nbeta):
                    EcMP2 += pqrt[i,a,j,b]*(2*pqrt_at[i,a,j,b]-pqrt_at[i,b,j,a])/(eig[i]+eig[j]-eig[a]-eig[b]+1e-10)
                for j in range(p.nbeta,p.nalpha):
                    EcMP2 += pqrt[i,a,j,b]*(pqrt_at[i,a,j,b]-0.5*pqrt_at[i,b,j,a])/(eig[i]+eig[j]-eig[a]-eig[b]+1e-10)
            for i in range(p.nbeta,p.nalpha):
                for j in range(p.nbeta):
                    EcMP2 += pqrt[i,a,j,b]*(pqrt_at[i,a,j,b]-0.5*pqrt_at[i,b,j,a])/(eig[i]+eig[j]-eig[a]-eig[b]+1e-10)
                for j in range(p.nbeta,p.nalpha):
                    EcMP2 += 0.5*pqrt[i,a,j,b]*(pqrt_at[i,a,j,b]-0.5*pqrt_at[i,b,j,a])/(eig[i]+eig[j]-eig[a]-eig[b]+1e-10)

    return EcMP2

def ECorrNonDyn(n,C,H,I,b_mnl,p):
    fi = 2*n*(1-n)

    CK12nd = np.outer(fi,fi)

    beta = np.sqrt((1-abs(1-2*n))*n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - l)
        CK12nd[p.no1+l,ll:ul] = beta[p.no1+l]*beta[ll:ul]
        CK12nd[ll:ul,p.no1+l] = beta[ll:ul]*beta[p.no1+l]
        CK12nd[ll:ul,ll:ul] = -np.outer(beta[ll:ul],beta[ll:ul])

    #C^K KMO
    J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)

    ECndHF = 0
    ECndl = 0
    if (p.MSpin==0):
       ECndHF = - np.einsum('ii,ii->',CK12nd[p.nbeta:p.nalpha,p.nbeta:p.nalpha],K_MO[p.nbeta:p.nalpha,p.nbeta:p.nalpha]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd,K_MO) # sum_ij
       ECndl += np.einsum('ii,ii->',CK12nd,K_MO) # Quita i=j
    elif (not p.MSpin==0):
       ECndl -= np.einsum('ij,ji->',CK12nd[p.no1:p.nbeta,p.no1:p.nbeta],K_MO[p.no1:p.nbeta,p.no1:p.nbeta]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd[p.no1:p.nbeta,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,p.no1:p.nbeta]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd[p.nalpha:p.nbf5,p.no1:p.nbeta],K_MO[p.no1:p.nbeta,p.nalpha:p.nbf5]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd[p.nalpha:p.nbf5,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5]) # sum_ij
       ECndl += np.einsum('ii,ii->',CK12nd[p.no1:p.nbeta,p.no1:p.nbeta],K_MO[p.no1:p.nbeta,p.no1:p.nbeta]) # Quita i=j
       ECndl += np.einsum('ii,ii->',CK12nd[p.nalpha:p.nbf5,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5]) # Quita i=j

    return ECndHF,ECndl


def mbpt(n,C,H,I,b_mnl,Dipole,E_nuc,E_elec,p):

    t1 = time()

    print(" MBPT")
    print("=========")

    nab = p.nvir*p.nalpha
    last_coup = p.nalpha + p.ncwo*(p.ndoc)

    print("Number of orbitals        (NBASIS) = {}".format(p.nbf))
    print("Number of frozen pairs       (NFR) = {}".format(p.no1))
    print("Number of occupied orbs.     (NOC) = {}".format(p.nalpha))
    print("Number of virtual orbs.     (NVIR) = {}".format(p.nvir))
    print("Size of A+B and A-B (NAB=NOCxNVIR) = {}".format(nab))
    print("")

    occ = np.zeros((p.nbf))
    occ[:p.nbf5] = n

    print(" ....Building F_MO")
    EHFL,F_MO = build_F_MO(C,H,I,b_mnl,p)

    print(" ....Transforming ERIs mnsl->pqrt")
    pqrt = pynof.compute_pqrt(C,I,b_mnl,p)

    Cintra = 1 - (1 - abs(1-2*occ))**2
    Cinter = abs(1-2*occ)**2
    Cinter[:p.nalpha] = 1.0

    print(" ....Attenuating F_MO")
    F_MO_at = F_MO_attenuated(F_MO,Cintra,Cinter,p.no1,p.nalpha,p.ndoc,p.nsoc,p.ndns,p.ncwo,p.nbf5,p.nbf)

    print(" ....Attenuating pqrt")
    pqrt_at = ERIS_attenuated(pqrt,Cintra,Cinter,p.no1,p.ndoc,p.nsoc,p.ndns,p.ncwo,p.nbf5,p.nbf)

    eig,C_can = eigh(F_MO_at)

    print(" ....Canonicalizing pqrt")
    pqrt = np.einsum("pqrt,pm,qn,rs,tl->mnsl",pqrt,C_can,C_can,C_can,C_can,optimize=True)
    print(" ....Canonicalizing pqrt_at")
    pqrt_at = np.einsum("pqrt,pm,qn,rs,tl->mnsl",pqrt_at,C_can,C_can,C_can,C_can,optimize=True)
    print("")

    print("List of qp-orbital energies (a.u.) and occ numbers used")
    print()
    mu = (eig[p.nalpha] + eig[p.nalpha-1])/2
    eig = eig - mu
    for i in range(p.nbf):
        print(" {: 8.6f} {:5.3f}".format(eig[i],occ[i]))
    print("Chemical potential used for qp-orbital energies: {}".format(mu))
    print("")

    AmB = np.zeros((nab,nab))
    ApB = np.zeros((nab,nab))

    EcRPA = 0
    
    k = 0
    for i in range(p.nalpha):
        for a in range(p.nalpha,p.nbf):
            l = 0
            for j in range(p.nalpha):
                for b in range(p.nalpha,p.nbf):
                    ApB[k,l] = 4*pqrt_at[i,a,j,b]
                    AmB[k,l] = 0.0
                    if(i==j and a==b):
                        AmB[k,l] = eig[a] - eig[i]
                        ApB[k,l] = ApB[k,l] + AmB[k,l]
                    if(k==l):
                        EcRPA = EcRPA - 0.25*(ApB[k,k] + AmB[k,k])
                    l += 1
            k += 1

    L = sp.linalg.cholesky(ApB, lower=True)

    bigomega2,tempm = np.linalg.eigh(np.matmul(np.matmul(np.transpose(L),AmB),L))
#    bigomega2,tempm = np.linalg.eigh(np.matmul(ApB,AmB))

    bigomega = np.sqrt(bigomega2)

    XmY = np.matmul(L,tempm)

    XpY = sp.linalg.solve(np.transpose(L),tempm)

    for i in range(nab):
        XmY[:,i] /= np.sqrt(bigomega[i]) 
        XpY[:,i] *= np.sqrt(bigomega[i]) 

    print(" ....Computing Polarizabilities")
    EcRPA = td_polarizability(EcRPA,C,Dipole,bigomega,XpY,nab,p)

    print(" ....Computing wmn")
    wmn,wmn_at = build_wmn(pqrt,pqrt_at,XpY,nab,p)

    print(" ....Computing gw_gm")
    EcGoWo, EcGMSOS = gw_gm_eq(wmn_at,pqrt,eig,bigomega,XpY,nab,p)

    EcGoWo *= 2
    EcGoWoSOS = EcGoWo + EcGMSOS

    print(" ....Computing mp2")
    EcMP2 = mp2_eq(eig,pqrt,pqrt_at,p)

    order = 40
    freqs, weights, sumw = roots_legendre(order, mu=True)
    print(" ....Computing rpa_sosex")
    iEcRPA, iEcSOSEX = rpa_sosex(freqs,weights,sumw,order,wmn_at,eig,pqrt,pqrt_at,bigomega,nab,p.nbeta,p.nalpha,p.nbf)

    iEcRPASOS = iEcRPA+iEcSOSEX

    ECndHF,ECndl = ECorrNonDyn(n,C,H,I,b_mnl,p)

    ESD = EHFL+E_nuc+ECndHF
    ESDc = ESD + ECndl
    EPNOF = E_elec + E_nuc

    print(" E(SD)                = {: f}".format(ESD))
    print(" E(SD+ND)             = {: f}".format(ESDc))
    print(" E(PNOFi)             = {: f}".format(EPNOF))
    print("")
    print(" Ec(ND)               = {: f}".format(ECndl))
    print(" Ec(RPA-FURCHE)       = {: f}".format(EcRPA))
    print(" Ec(RPA)              = {: f}".format(iEcRPA))
    print(" Ec(AC-SOSEX)         = {: f}".format(iEcSOSEX))
    print(" Ec(RPA+AC-SOSEX)     = {: f}".format(iEcRPASOS))
    print(" Ec(GW@GM)            = {: f}".format(EcGoWo))
    print(" Ec(SOSEX@GM)         = {: f}".format(EcGMSOS))
    print(" Ec(GW@GM+SOSEX@GM)   = {: f}".format(EcGoWoSOS))
    print(" Ec(MP2)              = {: f}".format(EcMP2))
    print("")
    print(" E(RPA-FURCHE)       = {: f}".format(ESD + EcRPA))
    print(" E(RPA)              = {: f}".format(ESD + iEcRPA))
    print(" E(RPA+AC-SOSEX)     = {: f}".format(ESD + iEcRPASOS))
    print(" E(GW@GM)            = {: f}".format(ESD + EcGoWo))
    print(" E(SOSEX@GM)         = {: f}".format(ESD + EcGMSOS))
    print(" E(GW@GM+SOSEX@GM)   = {: f}".format(ESD + EcGoWoSOS))
    print(" E(MP2)              = {: f}".format(ESD + EcMP2))
    print("")
    print(" E(NOF-c-RPA-FURCHE)       = {: f}".format(EPNOF + EcRPA))
    print(" E(NOF-c-RPA)              = {: f}".format(EPNOF + iEcRPA))
    print(" E(NOF-c-RPA+AC+SOSEX)     = {: f}".format(EPNOF + iEcRPASOS))
    print(" E(NOF-c-GW@GM)            = {: f}".format(EPNOF + EcGoWo))
    print(" E(NOF-c-SOSEX@GM)         = {: f}".format(EPNOF + EcGMSOS))
    print(" E(NOF-c-GW@GM+SOSEX@GM)   = {: f}".format(EPNOF + EcGoWoSOS))
    print(" E(NOF-c-MP2)              = {: f}".format(EPNOF + EcMP2))

    print("")

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))



def nofmp2(n,C,H,I,b_mnl,E_nuc,p):

    t1 = time()

    print(" NOF-MP2")
    print("=========")

    occ = n[p.no1:p.nbf5]
    vec = C[:,p.no1:p.nbf]

    D = pynof.computeD_HF(C,p)
    if(p.MSpin==0):
        if(p.nsoc>0):
            Dalpha = pynof.computeDalpha_HF(C,p)
            D = D + 0.5*Dalpha
        J,K = pynof.computeJK_HF(D,I,b_mnl,p)
        F = H + 2*J - K
        EHFL = np.trace(np.matmul(D,H)+np.matmul(D,F))
    elif(not p.MSpin==0):
        Dalpha = pynof.computeDalpha_HF(C,p)
        J,K = pynof.computeJK_HF(D,I,b_mnl,p)
        F = 2*J - K
        EHFL = 2*np.trace(np.matmul(D+0.5*Dalpha,H))+np.trace(np.matmul(D+Dalpha,F))
        F = H + F
        if(p.nsoc>1):
            J,K = pynof.computeJK_HF(0.5*Dalpha,I,b_mnl,p)
            Falpha = J - K
            EHFL = EHFL + 2*np.trace(np.matmul(0.5*Dalpha,Falpha))
            F = F + Falpha

    F_MO = np.matmul(np.matmul(np.transpose(vec),F),vec)

    eig = np.einsum("ii->i",F_MO[:p.nbf-p.no1,:p.nbf-p.no1])

    iajb = pynof.compute_iajb(C,I,b_mnl,p)
    FI1 = np.ones(p.nbf-p.no1)
    FI2 = np.ones(p.nbf-p.no1)

    FI1[:p.nbf5-p.no1] = 1 - (1 - abs(1-2*occ[:p.nbf5-p.no1]))**2

    FI2[p.nalpha-p.no1:p.nbf5-p.no1] = abs(1-2*occ[p.nalpha-p.no1:p.nbf5-p.no1])**2

    Tijab = CalTijab(iajb,F_MO,eig,FI1,FI2,p)
    ECd = 0
    for k in range(p.nvir):
        for l in range(p.nvir):
            for i in range(p.ndoc):
                for j in range(p.ndoc):
                    Xijkl = iajb[j,k,i,l]
                    ijkl = i+j*p.ndns+k*p.ndns*p.ndns+l*p.ndns*p.ndns*p.nvir
                    ijlk = i+j*p.ndns+l*p.ndns*p.ndns+k*p.ndns*p.ndns*p.nvir
                    ECd = ECd + Xijkl*(2*Tijab[ijkl]-Tijab[ijlk])
                for j in range(p.ndoc,p.ndns):
                    Xijkl = iajb[j,k,i,l]
                    ijkl = i+j*p.ndns+k*p.ndns*p.ndns+l*p.ndns*p.ndns*p.nvir
                    ijlk = i+j*p.ndns+l*p.ndns*p.ndns+k*p.ndns*p.ndns*p.nvir
                    ECd = ECd + Xijkl*(Tijab[ijkl]-0.5*Tijab[ijlk])
            for i in range(p.ndoc,p.ndns):
                for j in range(p.ndoc):
                    Xijkl = iajb[j,k,i,l]
                    ijkl = i+j*p.ndns+k*p.ndns*p.ndns+l*p.ndns*p.ndns*p.nvir
                    ijlk = i+j*p.ndns+l*p.ndns*p.ndns+k*p.ndns*p.ndns*p.nvir
                    ECd = ECd + Xijkl*(Tijab[ijkl]-0.5*Tijab[ijlk])
                for j in range(p.ndoc,p.ndns):
                    Xijkl = iajb[j,k,i,l]
                    ijkl = i+j*p.ndns+k*p.ndns*p.ndns+l*p.ndns*p.ndns*p.nvir
                    ijlk = i+j*p.ndns+l*p.ndns*p.ndns+k*p.ndns*p.ndns*p.nvir
                    if(j!=i):
                        ECd = ECd + Xijkl*(Tijab[ijkl]-0.5*Tijab[ijlk])/2

    fi = 2*n*(1-n)

    CK12nd = np.outer(fi,fi)

    beta = np.sqrt((1-abs(1-2*n))*n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - l)
        CK12nd[p.no1+l,ll:ul] = beta[p.no1+l]*beta[ll:ul]
        CK12nd[ll:ul,p.no1+l] = beta[ll:ul]*beta[p.no1+l]
        CK12nd[ll:ul,ll:ul] = -np.outer(beta[ll:ul],beta[ll:ul])

    #C^K KMO
    J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)

    ECndHF = 0
    ECndl = 0
    if (p.MSpin==0):
       ECndHF = - np.einsum('ii,ii->',CK12nd[p.nbeta:p.nalpha,p.nbeta:p.nalpha],K_MO[p.nbeta:p.nalpha,p.nbeta:p.nalpha]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd,K_MO) # sum_ij
       ECndl += np.einsum('ii,ii->',CK12nd,K_MO) # Quita i=j
    elif (not p.MSpin==0):
       ECndl -= np.einsum('ij,ji->',CK12nd[p.no1:p.nbeta,p.no1:p.nbeta],K_MO[p.no1:p.nbeta,p.no1:p.nbeta]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd[p.no1:p.nbeta,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,p.no1:p.nbeta]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd[p.nalpha:p.nbf5,p.no1:p.nbeta],K_MO[p.no1:p.nbeta,p.nalpha:p.nbf5]) # sum_ij
       ECndl -= np.einsum('ij,ji->',CK12nd[p.nalpha:p.nbf5,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5]) # sum_ij
       ECndl += np.einsum('ii,ii->',CK12nd[p.no1:p.nbeta,p.no1:p.nbeta],K_MO[p.no1:p.nbeta,p.no1:p.nbeta]) # Quita i=j
       ECndl += np.einsum('ii,ii->',CK12nd[p.nalpha:p.nbf5,p.nalpha:p.nbf5],K_MO[p.nalpha:p.nbf5,p.nalpha:p.nbf5]) # Quita i=j

    print("      Ehfc      = {:f}".format(EHFL+E_nuc+ECndHF))
    print("")
    print("      ECd       = {:f}".format(ECd))
    print("      ECnd      = {:f}".format(ECndl))
    print("      Ecorre    = {:f}".format(ECd+ECndl))
    print("      E(NOFMP2) = {:f}".format(EHFL+ECd+ECndl+E_nuc+ECndHF))
    print("")

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2-t1))

def CalTijab(iajb,F_MO,eig,FI1,FI2,p):

    print("Starting CalTijab")

    B = build_B(iajb,FI1,FI2,p.ndoc,p.ndns,p.nvir,p.ncwo)
    print("....B vector Computed")
    
    Tijab = Tijab_guess(iajb,eig,p.ndoc,p.ndns,p.nvir)
    print("....Tijab Guess Computed")

    #A_CSR = csr_matrix(build_A(F_MO,FI1,FI2,p.no1,p.ndoc,p.ndns,p.nvir,p.ncwo,p.nbf))
    #print("A matrix has {}/{} elements with Tol = {}".format(len(A),p.nvir**4*p.ndoc**4,1e-10))
    #Tijab = solve_Tijab(A_CSR,B,Tijab,p)

    res = root(build_R, Tijab, args=(B,F_MO,FI1,FI2,p.no1,p.ndoc,p.ndns,p.nvir,p.ncwo,p.nbf),method="krylov")
    if(res.success):
        print("....Tijab found as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
    else:
        print("....WARNING! Tijab NOT FOUND as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
        print(res)
    Tijab = res.x
    print("")

    return Tijab

@njit
def build_A(F_MO,FI1,FI2,no1,ndoc,ndns,nvir,ncwo,nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = ncwo*(ndoc - i - 1)
        ul = ncwo*(ndoc - i)
        npair[ll:ul] = i + 1

    A = np.empty((2*ndns**2*nvir**2*(nbf-no1)))
    IROW = np.empty((2*ndns**2*nvir**2*(nbf-no1)),dtype=np.int32)
    ICOL = np.empty((2*ndns**2*nvir**2*(nbf-no1)),dtype=np.int32)

    nnz = -1
    for ib in range(nvir):
        for ia in range(nvir):
            for j in range(ndns):
                for i in range(ndns):
                    #print(nnz)
                    jab =     (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    iab = i            + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    ijb = i + (j)*ndns                  + (ib)*ndns*ndns*nvir
                    ija = i + (j)*ndns + (ia)*ndns*ndns
                    ijab= i + (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir

                    nnz = nnz + 1
                    A[nnz] = (F_MO[ia+ndns,ia+ndns] + F_MO[ib+ndns,ib+ndns] - F_MO[i,i] - F_MO[j,j])
                    IROW[nnz] = (ijab)
                    ICOL[nnz] = (i + jab)

                    for k in range(i):
                        if(abs(F_MO[i,k])>1e-10):
                            Cki = FI2[k]*FI2[i]
                            nnz += 1
                            A[nnz]=(- Cki*F_MO[i,k])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k + jab)
                            nnz += 1
                            A[nnz]=(- Cki*F_MO[i,k])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k + jab)

                    for k in range(j):
                        if(abs(F_MO[j,k])>1e-10):
                            Ckj = FI2[k]*FI2[j]
                            nnz += 1
                            A[nnz]=(- Ckj*F_MO[j,k])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k*ndns + iab)
                            nnz += 1
                            A[nnz]=(- Ckj*F_MO[j,k])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k*ndns + iab)

                    for k in range(ia):
                        if(abs(F_MO[ia+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+ndns]*FI1[ia+ndns]
                            else:
                                Ckia = FI2[k+ndns]*FI2[ia+ndns]
                            nnz += 1
                            A[nnz]=(Ckia*F_MO[ia+ndns,k+ndns])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k*ndns*ndns + ijb)
                            nnz += 1
                            A[nnz]=(Ckia*F_MO[ia+ndns,k+ndns])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k*ndns*ndns + ijb)

                    for k in range(ib):
                        if(abs(F_MO[ib+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+ndns]*FI1[ib+ndns]
                            else:
                                Ckib = FI2[k+ndns]*FI2[ib+ndns]
                            nnz += 1
                            A[nnz]=(Ckib*F_MO[ib+ndns,k+ndns])
                            IROW[nnz]=(ijab)
                            ICOL[nnz]=(k*ndns*ndns*nvir + ija)
                            nnz += 1
                            A[nnz]=(Ckib*F_MO[ib+ndns,k+ndns])
                            ICOL[nnz]=(ijab)
                            IROW[nnz]=(k*ndns*ndns*nvir + ija)

    A = A[:nnz+1]
    IROW = IROW[:nnz+1]
    ICOL = ICOL[:nnz+1]

    return A,(IROW,ICOL)

@njit(parallel=True)
def build_R(T,B,F_MO,FI1,FI2,no1,ndoc,ndns,nvir,ncwo,nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = ncwo*(ndoc - i - 1)
        ul = ncwo*(ndoc - i)
        npair[ll:ul] = i + 1

    Bp = np.zeros((ndns**2*nvir**2))
    
    for ib in prange(nvir):
        for ia in prange(nvir):
            for j in prange(ndns):
                for i in prange(ndns):
                    jab =     (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    iab = i            + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    ijb = i + (j)*ndns                  + (ib)*ndns*ndns*nvir
                    ija = i + (j)*ndns + (ia)*ndns*ndns
                    ijab= i + (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir

                    Bp[ijab] += (F_MO[ia+ndns,ia+ndns] + F_MO[ib+ndns,ib+ndns] - F_MO[i,i] - F_MO[j,j])*T[i+jab]

                    for k in range(i):
                        if(abs(F_MO[i,k])>1e-10):
                            Cki = FI2[k]*FI2[i]
                            Bp[ijab] += (- Cki*F_MO[i,k])*T[k+jab]
                    for k in range(i+1,ndns):
                        if(abs(F_MO[i,k])>1e-10):
                            Cki = FI2[k]*FI2[i]
                            Bp[ijab] += (- Cki*F_MO[i,k])*T[k+jab]

                    for k in range(j):
                        if(abs(F_MO[j,k])>1e-10):
                            Ckj = FI2[k]*FI2[j]
                            Bp[ijab] += (- Ckj*F_MO[j,k])*T[k*ndns+iab]
                    for k in range(j+1,ndns):
                        if(abs(F_MO[j,k])>1e-10):
                            Ckj = FI2[k]*FI2[j]
                            Bp[ijab] += (- Ckj*F_MO[j,k])*T[k*ndns+iab]

                    for k in range(ia):
                        if(abs(F_MO[ia+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+ndns]*FI1[ia+ndns]
                            else:
                                Ckia = FI2[k+ndns]*FI2[ia+ndns]
                            Bp[ijab] += (Ckia*F_MO[ia+ndns,k+ndns]) * T[k*ndns*ndns + ijb] 
                    for k in range(ia+1,nvir):
                        if(abs(F_MO[ia+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ia]):
                                Ckia = FI1[k+ndns]*FI1[ia+ndns]
                            else:
                                Ckia = FI2[k+ndns]*FI2[ia+ndns]
                            Bp[ijab] += (Ckia*F_MO[ia+ndns,k+ndns]) * T[k*ndns*ndns + ijb] 

                    for k in range(ib):
                        if(abs(F_MO[ib+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+ndns]*FI1[ib+ndns]
                            else:
                                Ckib = FI2[k+ndns]*FI2[ib+ndns]
                            Bp[ijab] += (Ckib*F_MO[ib+ndns,k+ndns]) * T[k*ndns*ndns*nvir + ija] 
                    for k in range(ib+1,nvir):
                        if(abs(F_MO[ib+ndns,k+ndns])>1e-10):
                            if(npair[k]==npair[ib]):
                                Ckib = FI1[k+ndns]*FI1[ib+ndns]
                            else:
                                Ckib = FI2[k+ndns]*FI2[ib+ndns]
                            Bp[ijab] += (Ckib*F_MO[ib+ndns,k+ndns]) * T[k*ndns*ndns*nvir + ija] 


    R = B-Bp
    return R

@njit(parallel=True)
def build_B(iajb,FI1,FI2,ndoc,ndns,nvir,ncwo):
    B = np.zeros((ndns**2*nvir**2))
    for i in prange(ndns):
        lmin_i = ndns+ncwo*(ndns-i-1)
        lmax_i = ndns+ncwo*(ndns-i-1)+ncwo
        for j in range(ndns):
            if(i==j):
                for k in range(nvir):
                    ik = i + k*ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndns
                        if(lmin_i <= kn and kn < lmax_i and lmin_i <= ln and ln < lmax_i):
                            Ciikl = FI1[kn]*FI1[ln]*FI1[i]*FI1[i]
                        else:
                            Ciikl = FI2[kn]*FI2[ln]*FI2[i]*FI2[i]
                        iikl =  i + i*ndns + k*ndns*ndns + l*ndns*ndns*nvir
                        B[iikl] = - Ciikl*iajb[i,k,i,l]
            else:
                for k in range(nvir):
                    ik = i + k*ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndns
                        ijkl =  i + j*ndns + k*ndns*ndns + l*ndns*ndns*nvir
                        Cijkl = FI2[kn]*FI2[ln]*FI2[i]*FI2[j]
                        B[ijkl] = - Cijkl*iajb[j,k,i,l]
    return B

@njit(parallel=True)
def Tijab_guess(iajb,eig,ndoc,ndns,nvir):
    Tijab = np.zeros(nvir**2*ndns**2)
    for ia in prange(nvir):
        for i in range(ndns):
            for ib in range(nvir):
                for j in range(ndns):
                    ijab = i + (j)*ndns + (ia)*ndns*ndns + (ib)*ndns*ndns*nvir
                    Eijab = eig[ib+ndns] + eig[ia+ndns] - eig[j] - eig[i]
                    Tijab[ijab] = - iajb[j,ia,i,ib]/Eijab
    return Tijab 

def solve_Tijab(A_CSR,B,Tijab,p):
    Tijab,info = cg(A_CSR, B,x0=Tijab)
    return Tijab

def ext_koopmans(p,elag,n):
    elag_small = elag[:p.nbf5,:p.nbf5]
    nu = -np.einsum("qp,q,p->qp",elag_small,1/np.sqrt(n),1/np.sqrt(n))

    print("")
    print("---------------------------")
    print(" Extended Koopmans Theorem ")
    print("   Ionization Potentials   ")
    print("---------------------------")

    eigval, eigvec = np.linalg.eigh(nu)
    print(" OM        (eV)")
    print("---------------------------")
    for i,val in enumerate(eigval[::-1]):
        print("{: 3d}      {: 7.3f}".format(i,val*27.2114))

    print("")
    print("EKT IP: {: 7.3f} eV".format(eigval[0]*27.2114))
    print("")

def mulliken_pop(p,wfn,n,C,S):

    nPS = 2*np.einsum("i,mi,ni,nm->m",n,C[:,:p.nbf5],C[:,:p.nbf5],S,optimize=True)

    pop = np.zeros((p.natoms))

    for mu in range(C.shape[0]):
        iatom = wfn.basisset().function_to_center(mu)
        pop[iatom] += nPS[mu]

    print("")
    print("---------------------------------")
    print("  Mulliken Population Analysis   ")
    print("---------------------------------")
    print(" Idx  Atom   Population   Charge ")
    print("---------------------------------")
    for iatom in range(p.natoms):
        symbol = wfn.molecule().flabel(iatom)
        print("{: 3d}    {:2s}    {: 5.2f}      {: 5.2f}".format(iatom, symbol, pop[iatom], wfn.molecule().Z(iatom)-pop[iatom]))
    print("")

def lowdin_pop(p,wfn,n,C,S):

    S_12 = fractional_matrix_power(S, 0.5)

    S_12nPS_12 = 2*np.einsum("sm,i,mi,ni,ns->s",S_12,n,C[:,:p.nbf5],C[:,:p.nbf5],S_12,optimize=True)

    pop = np.zeros((p.natoms))

    for mu in range(C.shape[0]):
        iatom = wfn.basisset().function_to_center(mu)
        pop[iatom] += S_12nPS_12[mu]

    print("")
    print("---------------------------------")
    print("   Lowdin Population Analysis    ")
    print("---------------------------------")
    print(" Idx  Atom   Population   Charge ")
    print("---------------------------------")
    for iatom in range(p.natoms):
        symbol = wfn.molecule().flabel(iatom)
        print("{: 3d}    {:2s}    {: 5.2f}      {: 5.2f}".format(iatom, symbol, pop[iatom], wfn.molecule().Z(iatom)-pop[iatom]))
    print("")

def M_diagnostic(p,n):

    m_vals = 2*n
    if(p.HighSpin):
        m_vals[p.nbeta:p.nalpha] = n[p.nbeta:p.nalpha]

    m_diagnostic = 0

    m_vals[p.no1:p.nbeta] = 2.0 - m_vals[p.no1:p.nbeta]
    m_diagnostic += max(m_vals[p.no1:p.nbeta])

    #if(p.nsoc!=0): #This is always zero
    #    m_vals[p.nbeta:p.nalpha] = 1.0 - m_vals[p.nbeta:p.nalpha]
    #    m_diagnostic += max(m_vals[p.nbeta:p.nalpha]) 

    m_vals[p.nalpha:p.nbf5] = m_vals[p.nalpha:p.nbf5] - 0.0
    m_diagnostic += max(m_vals[p.nalpha:p.nbf5])

    m_diagnostic = 0.5*m_diagnostic

    print("")
    print("---------------------------------")
    print("   M Diagnostic: {:4.2f} ".format(m_diagnostic))
    print("---------------------------------")
    print("")

