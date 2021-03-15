#!/usr/bin/env python
# coding: utf-8

# # PNOFi
# 
# ![DoNOF](https://donof.readthedocs.io/en/latest/_images/Logo-DoNOF.jpeg)
# 
# $$\newcommand{\ket}[1]{\left|{#1}\right\rangle}$$
# $$\newcommand{\bra}[1]{\left\langle{#1}\right|}$$
# 
# En este notebook se encuentra el álgebra elemental para calcular un punto simple de energía de PNOFi (i=5,6,7)

# **Seleccionamos un PNOFi (i=5,7)** y el tipo de gradiente, $\frac{dE}{dn_i}$, para los n.umeros de ocupación.

# In[1]:


PNOFi = 7
gradient = "numerical" # analytical/numerical


# ## Introducción

# **Importamos librerías**

# In[2]:


import psi4
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from time import time


# **Seleccionamos una molécula**, y otros datos como la memoria del sistema y la base

# In[3]:


psi4.set_memory('4 GB')

mol = psi4.geometry("""
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
  symmetry c1
""")

psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})


# Construimos la función de onda y **evaluamos matrices e integrales en orbital atómico**, $S$, $T$, $V$, $H$, $(\mu\nu|\sigma\lambda)$

# In[4]:


# Wavefunction
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))

# Integrador
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap, Kinetics, Potential
S = np.asarray(mints.ao_overlap())
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V

# Integrales de Repulsión Electrónica, ERIs (mu nu | sigma lambda)
I = np.asarray(mints.ao_eri())

# Energía Nuclear
E_nuc = mol.nuclear_repulsion_energy()


# **Se declaran variables del sistema.**
# 
# El número de electrones (ne) es la suma de los electrones con spín alpha (nalpha) y los electrones con spín beta (nbeta)
# \begin{equation}
# N_e = N_{\alpha} + N_{\beta}
# \end{equation}
# 
# El número de orbitales doblemente ocupados inactivos (no1), es decir con ocupación de 1, está dado por
# \begin{equation}
# N_{o1} = \left\{
#   \begin{array}{lll}
#   \sum_{átomo}^{N_{átomos}} {orbitales\ de\ core}_{átomo}      & Default \\
#   Otro     & \mathrm{si\ se\ indica}
#   \end{array}
#   \right.
# \end{equation}
# 
# *****************************************************************************************************************
# 
# Dividiremos el espacio orbital en dos subespacios
# \begin{equation}
# \Omega = \Omega_I + \Omega_{II}
# \end{equation}
# 
# *****************************************************************************************************************
# 
# El subespacio $\Omega_{II}$ se divide en $N_{II}/2$ subespacios $\Omega_g$. Cada subespacio $\Omega_g \in \Omega_{II}$ contiene un orbital $\ket{g}$ con $g \le N_{II}/2$ y $N_g$ orbitales $\ket{p}$ con $p > N_{II}/2$, es decir
# \begin{equation}
# \Omega_g = \{\ket{g},\ket{p_1},\ket{p_2},\cdots,\ket{p_{N_{g}}}\} 
# \end{equation}
# 
# El número de orbitales fuertemente doble ocupados (ndoc) está dado por
# \begin{equation}
# N_{doc} = N_{\beta} - N_{o1} = N_{II}/2
# \end{equation}
# 
# *****************************************************************************************************************
# 
# El subespacio $\Omega_I$ se compone por $N_I$ subespacios $\Omega_g$, en este caso
# 
# El número de orbitales fuertemente ocupados con ocupación simple (nsoc) está dado por
# \begin{equation}
# N_{soc} = N_{\alpha} - N_{\beta} = N_I
# \end{equation}
# 
# *****************************************************************************************************************
# 
# El número de orbitales fuertemente ocupados (ndns) está dado por
# \begin{equation}
# N_{ndns} = N_{doc} + N_{soc} = N_{II}/2 + N_I = N_{\Omega}
# \end{equation}
# 
# El número de orbitales virtuales (nvir) es la diferencia entre los electrones $N_\alpha$ (nalpha) y el número de funciones de base $N_{bf}$ 
# \begin{equation}
# N_{vir} = N_{bf} - N_{\alpha}
# \end{equation}
# 
# El número de orbitales débilmente ocupados acoplados (ncwo) a cada orbital fuertemente doble ocupado (ndoc), y que constituye $N_g$, está dado por
# \begin{equation}
# N_{cwo} = \left\{
#   \begin{array}{lll}
#   N_{vir}/N_{doc}      & \mathrm{si\ } N_e = 2 \\
#   1 & \mathrm{si\ } N_e > 2 \mathrm{\ ó\ } N_{cwo}>N_{vir}/N_{doc}\\
#   Otro     & \mathrm{si\ se\ indica}
#   \end{array}
#   \right.
# \end{equation}
# 
# *****************************************************************************************************************
# 
# La dimensión del subespacio de orbitales activos está dada por los orbitales fuertemente doble ocupados y sus orbitales débilmente ocupados asociados
# \begin{equation}
# N_{ac} = N_{doc} + N_{doc} N_{cwo}
# \end{equation}
# 
# *****************************************************************************************************************
# 
# Los orbitals con número de ocupación distinto de cero (nbf5) son
# \begin{equation}
# N_{bf5} = N_{o1} + N_{ac} + N_{soc} 
# \end{equation}
# es decir, la suma de los orbitales de core (no1),más los fuertemente doble ocupados (ndoc) con sus orbitales débilmente ocupados acoplados (nwco) y los orbitales fuertemente ocupados con ocupación simple (nsoc)
# \begin{equation}
# N_{bf5} = N_{o1} + N_{doc} + N_{doc}N_{wco} + N_{soc}
# \end{equation}
# 
# *****************************************************************************************************************
# 
# Establecemos los orbitales a optimizar (noptorb)
# \begin{equation}
# N_{optorb} = \left\{
#   \begin{array}{lll}
#   N_{bf}      & Default \\
#   Otro     & \mathrm{si\ se\ indica}
#   \end{array}
#   \right.
# \end{equation}

# In[5]:


natoms = mol.natom()
nbf = S.shape[0]
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()
ne = nalpha + nbeta
mul = mol.multiplicity()
no1 = 0 #Number of inactive doubly occupied orbitals | Se puede variar
for i in range(natoms):
    Z = mol.Z(i)
    if ( 1<=Z and Z<=  2):
        no1 += 0           # H-He
    elif ( 3<=Z and Z<= 10):
        no1 +=  1          # Li-Ne
    elif (11<=Z and Z<= 18):
        no1 +=  5          # Na-Ar
    elif(19<=Z and Z<= 36):
        no1 +=  9          # K-Kr
    elif(37<=Z and Z<= 49):
        no1 += 18          # Rb-In
    elif(50<=Z and Z<= 54):
        no1 += 23          # Sn-Xe
    elif(55<=Z and Z<= 71):
        no1 += 27          # Cs-Lu
    elif(72<=Z and Z<= 81):
        no1 += 30          # Hf-Tl
    elif(82<=Z and Z<= 86):
        no1 += 39          # Pb-Rn
    elif(87<=Z and Z<=109):
        no1 += 43          # Fr-Mt
    
ndoc = nbeta   -   no1
nsoc = nalpha  -   nbeta
ndns = ndoc    +   nsoc
nvir = nbf     -   nalpha

ncwo = 1
if(ne==2):
    ncwo= -1
if(ndns!=0):
    if(ndoc>0):
        if(ncwo!=1):
            if(ncwo==-1 or ncwo > nvir/ndoc):
                ncwo = int(nvir/ndoc)
    else:
        ncwo = 0

nac = ndoc * (1 + ncwo)
nbf5 = no1 + nac + nsoc   #JFHLY warning: nbf must be >nbf5
no0 = nbf - nbf5

noptorb = nbf     

closed = (nbeta == (ne+mul-1)/2 and nalpha == (ne-mul+1)/2)


# Establecemos algunos parámetros

# In[6]:


maxit = 1000  # Número máximo de iteraciones de Occ-SCF
no1 = no1     # Número de orbitales inactivos con ocupación 1
thresheid = 10**-8 # Convergencia de la energía total
maxitid = 30  # Número máximo de iteraciones externas
ipnof = 5     # PNOFi a calcular
threshl = 10**-4   # Convergencia de los multiplicadores de Lagrange
threshe = 10**-6   # Convergencia de la energía
threshec = 10**-10 # Convergencia  de la energía en optimización orbital
threshen = 10**-10 # Convergencia  de la energía en optimización de ocupaciones
scaling = True     # Scaling for f
nzeros = 0
nzerosm = 4
nzerosr = 2
itziter = 10        # Iteraciones para scaling constante
diis = True         # DIIS en optimización orbital
thdiis = 10**-3     # Para iniciar DIIS
ndiis = 5           # Número de ciclos para interpolar matriz de Fock generalizada en DIIS
perdiis = True      # Aplica DIIS cada NDIIS (True) o después de NDIIS (False) 
ncwo = ncwo         # Número de orbitales débilmente ocupados acoplados a cada orbital fueremtente ocupado 
noptorb = noptorb   # Número de orbitales a optimizar Nbf5 <= Noptorb <= Nbf
scaling = True


# Obtenemos un Guess acorde a la ecuación
# 
# \begin{equation}
# HC = SC\varepsilon
# \end{equation}

# In[7]:


E_i,C = eigh(H, S)


# **Revisamos la ortonormalidad.** Asumiremos que los orbitales son ortonormales si $C^T SC$ difiere en menos de $10^{-6}$ respecto a la matriz identidad. Si los orbitales no son ortonormales, haremos hasta 3 intentos por ortonormalizarlos.

# In[8]:


# Revisa ortonormalidad
orthonormality = True

CTSC = np.matmul(np.matmul(np.transpose(C),S),C)
ortho_deviation = np.abs(CTSC - np.identity(nbf))

if (np.any(ortho_deviation > 10**-6)):
    orthonormality = False

if not orthonormality:
    print("Orthonormality violations {:d}, Maximum Violation {:f}".format((ortho_deviation > 10**-6).sum(),ortho_deviation.max()))        
else:
    print("No violations of the orthonormality")


# Se cambian de fase los obitales moleculares tal que **en cada orbital sea positivo el coeficiente de mayor magnitud**

# In[9]:


# Vuelve positivo el elemento más largo de cada MO
for j in range(nbf):
    
    #Obtiene el índice del coeficiente con mayor valor absoluto del MO
    idxmaxabsval = 0
    for i in range(nbf):
        if(abs(C[i][j])>abs(C[idxmaxabsval][j])):
            idxmaxabsval = i
    
    # Ajusta el signo del MO
    sign = np.sign(C[idxmaxabsval][j])
    C[0:nbf,j] = sign*C[0:nbf,j]


# ## Cálculo de orbitales y multiplicadores de Lagrange a partir de J, K y F

# Crearemos una función que calcule
# 
# \begin{eqnarray}
# D_{\mu\nu}^{(i)} &=& C_{\mu i} C_{\nu i}\\
# J_{\mu\nu}^{(i)} &=& \sum_{\sigma\lambda} D_{\sigma\lambda}^{(i)} (\mu\nu|\sigma\lambda)\\
# K_{\mu\sigma}^{(i)} &=& \sum_{\nu\lambda} D_{\nu\lambda}^{(i)} (\mu\nu|\sigma\lambda)
# \end{eqnarray}

# In[10]:


def computeJK(C):
    
    #denmatj    
    D = np.einsum('mi,ni->imn', C[:,0:nbf5], C[:,0:nbf5], optimize=True)
    
    #hstarj
    J = np.einsum('isl,mnsl->imn', D, I, optimize=True)    
    
    #hstark
    K = np.einsum('inl,mnsl->ims', D, I, optimize=True)    

    return J,K


# Construimos
# 
# \begin{equation}
# \lambda_{qp} = n_p H_{qp} + \int dr \frac{d V_{ee}}{d \phi_p (r)} \phi_q (r)
# \end{equation}

# Definimos una función para calcular la matriz generalizada de Fock
# \begin{equation}
# F_{\mu\nu}^{(i)} = \left\{
#   \begin{array}{lll}
#   H_{\mu\nu} + \sum_j^{N_{bf5}} J_{\mu\nu}^{(j)} C^{J}_{1j} - \sum_j^{N_{bf5}} K_{\mu\nu}^{(j)} C^{K}_{1j}      & \mathrm{si\ } i \in [1,N_{o1}]  \\
#   n_{i}(H_{\mu\nu} + J_{\mu\nu}^{(i)}) + \sum_{j \ne i}^{N_{bf5}} J_{\mu\nu}^{(j)} C^{J}_{ij} - \sum_{j \ne i}^{N_{bf5}} K_{\mu\nu}^{(j)} C^{K}_{ij}     & \mathrm{si\ } i \in (N_{o1},N_{beta}]\\
#   n_{i}H_{\mu\nu} + \sum_{j \ne i}^{N_{bf5}} J_{\mu\nu}^{(j)} C^{J}_{ij} - \sum_{j \ne i}^{N_{bf5}} K_{\mu\nu}^{(j)} C^{K}_{ij}     & \mathrm{si\ } i \in (N_{beta},N_{alpha}]\\
#   n_i(H_{\mu\nu} + J_{\mu\nu}^{(i)}) + \sum_{j \ne i}^{N_{bf5}} J_{\mu\nu}^{(j)} C^J_{ij} - \sum_{j \ne i}^{N_{bf5}} K_{\mu\nu}^{(j)} C^K_{ij}     & \mathrm{si\ } i \in (N_{\alpha},N_{bf5}]
#   \end{array}
#   \right.
# \end{equation}

# In[11]:


def computeF(J,K,n,cj12,ck12):
    
    # Matriz de Fock Generalizada                    
    F = np.zeros((nbf5,nbf,nbf))    

    ini = 0
    if(no1>1):        
        ini = no1       
        
    # nH
    F += np.einsum('i,mn->imn',n,H,optimize=True)        # i = [1,nbf5]

    # nJ
    F[ini:nbeta,:,:] += np.einsum('i,imn->imn',n[ini:nbeta],J[ini:nbeta,:,:],optimize=True)        # i = [ini,nbeta]
    F[nalpha:nbf5,:,:] += np.einsum('i,imn->imn',n[nalpha:nbf5],J[nalpha:nbf5,:,:],optimize=True)  # i = [nalpha,nbf5]
          
    # C^J J
    F += np.einsum('ij,jmn->imn',cj12,J,optimize=True)                                                # i = [1,nbf5]
    F[ini:nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:nbf5,ini:nbf5],J[ini:nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    F -= np.einsum('ij,jmn->imn',ck12,K,optimize=True)                                                # i = [1,nbf5]
    F[ini:nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:nbf5,ini:nbf5],K[ini:nbf5,:,:],optimize=True) # quita i==j
      
    return F


# Definimos una función que reciba los orbitales moleculares y la matriz de Fock y calcule los multiplicadores de lagrange según
# Generamos 
# \begin{equation}
# G_{\mu}^{(i)} = \sum_{\nu}^{N_{bf}} F_{\mu\nu}^{(i)} C_{\nu i}
# \end{equation}
# 
# \begin{equation}
# \lambda_{ij} = \sum_{\mu}^{N_{bf}} C_{\mu i} G_{\mu j}
# \end{equation}

# In[12]:


def computeLagrange(F,C):
    
    G = np.einsum('imn,ni->mi',F,C[:,0:nbf5],optimize=True)
            
    #Compute Lagrange multipliers
    elag = np.zeros((nbf,nbf))
    elag[0:noptorb,0:nbf5] = np.einsum('mi,mj->ij',C[:,0:noptorb],G,optimize=True)[0:noptorb,0:nbf5]
                
    return elag


# Definimos una función para calcular la energía electrónica
# \begin{equation}
# E = \sum_{i=1}^{N_{\beta}} \left( \lambda_{ii} + n_i \sum_{\mu\nu} C_{\mu i}H_{\mu\nu}C_{\nu i} \right) + \sum_{i=N_{\beta}+1}^{N_{\alpha}} \left( \lambda_{ii} + n_i \sum_{\mu\nu} C_{\mu i}H_{\mu\nu}C_{\nu i} \right) + \sum_{i=N_{\alpha+1}}^{N_{bf5}} \left( \lambda_{ii} + n_i \sum_{\mu\nu} C_{\mu i}H_{\mu\nu}C_{\nu i} \right)
# \end{equation}

# In[13]:


def computeE_elec(H,C,n,elag):
    #EELECTRr
    E = 0

    E = E + np.einsum('ii',elag[:nbf5,:nbf5],optimize=True)
    E = E + np.einsum('i,mi,mn,ni',n[:nbf5],C[:,:nbf5],H,C[:,:nbf5],optimize=True)
          
    return E


# Definimos una función que calcule la convergencia de los multiplicadores de lagrange

# In[14]:


def computeLagrangeConvergency(elag):
    # Convergency
    
    sumdiff = np.sum(np.abs(elag-elag.T))
    maxdiff = np.max(np.abs(elag-elag.T))

    return sumdiff,maxdiff


# Finalmente definimos una función que reciba coeficientes y haga lo siguiente
# 1. Calcule $J_{\mu\nu}^{(i)}$ y $K_{\mu\nu}^{(i)}$
# 2. Calcule la matriz $F_{\mu\nu}^{(i)}$
# 3. Calcule los multiplicadores de Lagrange $\lambda_{ij}$
# 4. Calcule la energía
# 5. Revise la convergencia de los multiplicadores de Lagrange

# In[15]:


def ENERGY1r(C,n,cj12,ck12):
    
    J,K = computeJK(C)

    F = computeF(J,K,n,cj12,ck12)

    elag = computeLagrange(F,C)

    E = computeE_elec(H,C,n,elag)

    sumdiff,maxdiff = computeLagrangeConvergency(elag)        
        
    return E,elag,sumdiff,maxdiff


# ## Iterative Diagonalizator Hartree-Fock

# Definimos valores iniciales de números de ocupación (n).
# 
# \begin{equation}
# n_{i} = \left\{
#   \begin{array}{lll}
#   1     & \mathrm{si\ } i \in [1,N_{\beta}]  \\
#   0.5   & \mathrm{si\ } i \in [N_{\beta},N_{\alpha}]  \\
#   0     & \mathrm{si\ otro\ caso} 
#   \end{array}
#   \right.
# \end{equation}
# 
# Los funcionales PNOF son del tipo
# \begin{equation}
# E = C^{J}_{ij} J_{MO} + C^{K}_{ij} K_{MO}
# \end{equation}
# 
# Donde $C^{J}_{ij}$ y $C^{K}_{ij}$ son ciertos coeficientes que se determinan para cada funcional y cumplen con ciertas reglas.
# 
# Definiremos valores iniciales para un HF
# \begin{eqnarray}
# C^J_{ij} &=& 2n_in_j\\
# C^K_{ij} &=& n_in_j
# \end{eqnarray}

# In[16]:


no1_ori = no1
no1 = nbeta

n = np.zeros((nbf5))
n[0:nbeta] = 1.0
n[nbeta:nalpha] = 0.5

cj12 = 2*np.einsum('i,j->ij',n,n)
ck12 = np.einsum('i,j->ij',n,n)


# Definimos una función que reciba los multiplicadores de Lagrange y calcule Fmiug
# 
# En la primera iteración
# \begin{equation}
# Fmiug_{ij} = \frac{\lambda_{ij}+\lambda_{ji}}{2}
# \end{equation}
# 
# En las demás
# \begin{equation}
# Fmiug_{ij} =  \theta(j-i) (\lambda_{ij}-\lambda_{ji}) + \theta(i-j) (\lambda_{ji}-\lambda_{ij})
# \end{equation}
# 
# Donde $\theta(x)$ es la función de Heaviside
# 
# Además, se aplica una técnica de escalamiento a los elementos fuera de la diagonal, tal que
# \begin{equation}
# Fmiug_{ij} = 
#   \begin{array}{lll}
#   0.1Fmiug_{ij}     & \mathrm{si\ } 10^{N_{zeros}+9-k} < |Fmiug_{ij}| < 10^{N_{zeros}+10-k}  \\
#   \end{array}
# \end{equation}
# con $k \in [0,N_{zeros}+9]$

# In[17]:


def fmiug_scaling(fmiug0,elag,i_ext,nzeros):
    
    #scaling
    fmiug = np.zeros((nbf,nbf))
    if(i_ext == 0):
        fmiug[:noptorb,:noptorb] = ((elag[:noptorb,:noptorb] + elag[:noptorb,:noptorb].T) / 2)

    else:
        fmiug[:noptorb,:noptorb] = (elag[:noptorb,:noptorb] - elag[:noptorb,:noptorb].T)
        fmiug = np.tril(fmiug,-1) + np.tril(fmiug,-1).T
        for k in range(nzeros+9+1):
            fmiug[(abs(fmiug) > 10**(9-k)) & (abs(fmiug) < 10**(10-k))] *= 0.1
        np.fill_diagonal(fmiug[:noptorb,:noptorb],fmiug0[:noptorb])

    return fmiug


# Creamos el SCF del HFIDr, el cual consiste en iteraciones externas e internas. En cada iteración interna se sigue el siguiente procedimiento:
# 1. Calcular $Fmiug$ a partir de los multiplicadores de lagrange $\varepsilon$, aplicando escalamiento de ser necesario
# 2. Diagonalizar $Fmiug$
# 3. Obtener nuevos coeficientes de orbitales moleculares
# 4. Calcular nueva matriz generalizada de Fock $F_{\mu\nu}^{(i)}$, así como nuevos multiplicadores de Lagrange $\varepsilon_{ij}$
# 5. Revisar convergencia

# In[18]:


#HFIDr

print('{:^7} {:^7} {:^14} {:^14} {:^15} {:^14}'.format("Nitext","Nitint","Eelec","Etot","Ediff","maxdiff"))

E,elag,sumdiff,maxdiff = ENERGY1r(C,n,cj12,ck12)

fmiug0 = np.zeros((nbf))
nzeros = 0

ext = True
# iteraciones externas
for i_ext in range(maxitid):
    if i_ext==0:
        maxlp = 1
    else:
        maxlp = 30
            
    # iteraciones internas             
    for i_int in range(maxlp):
        E_old = E

        if(scaling):
            fmiug = fmiug_scaling(fmiug0,elag,i_ext,nzeros)
                   
        fmiug0, W = np.linalg.eigh(fmiug)
        C = np.matmul(C,W)
        E,elag,sumdiff,maxdiff = ENERGY1r(C,n,cj12,ck12)
        
        E_diff = E-E_old                    
        if(abs(E_diff)<thresheid):
            print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))
            for i in range(nbf):
                fmiug0[i] = elag[i][i]
            ext = False
            break
    if(not ext):
        break
    print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))


# Regresamos no1 a su estado original

# In[19]:


no1 = no1_ori


# ## PNOF
# 
# A continuación generamos las $C_{ij}^{J}$ y $C_{ij}^{K}$ que definen a los funcionales PNOF, así como las derivadas para los números de ocupación. 
# 
# **Nota.** En $\frac{d C_{ij}^J}{d\gamma_k}$ y $\frac{d C_{ij}^K}{d\gamma_k}$ se aprovecha la simetría y solo se toma la mitad de la derivada, al calcular el gradiente se multiplicará por 2.

# Para PNOF5
# 
# \begin{equation}
# E^{PNOF5} = \sum_{g=1}^{N_{II}/2} \left[ \sum_{p \in \Omega_g} n_p (2H_{pp} + J_{pp}) + \sum_{q,p \in \Omega_g q \ne p} \underbrace{\Pi_{qp}^{g}}_{C^K_{pq}} K_{pq} \right] + \sum_{f \neq g}^{N_{II}/2} \sum_{p \in \Omega_f} \sum_{q \in \Omega_g} (\underbrace{2n_q n_p}_{C^J_{q p}} J_{pq}- \underbrace{n_q n_p}_{C^K_{q p}} K_{pq})
# \end{equation}
# 
# 
# \begin{equation}
# C^J_{ij} = \left\{
#   \begin{array}{lll}
#   2 n_i n_j & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#   0         & \mathrm{si\ } i,j \in \Omega_{g}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# C^K_{ij} = \left\{
#   \begin{array}{lll}
#   n_i n_j   & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#   -\Pi_{ij}       & \mathrm{si\ } i,j \in \Omega_{g}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# \Pi_{ij} = \left\{
#   \begin{array}{lll}
#    \sqrt{n_i n_j}      & \mathrm{si\ } i,j \in \Omega_{g} i=j\ \text{ó}\ i\ \text{y}\ j > \frac{N_{II}}{2}\\
#   -\sqrt{n_i n_j}      & \mathrm{si\ } i,j \in \Omega_{g} i\ \text{ó}\ j \leq \frac{N_{II}}{2}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# \frac{dC^J_{ij}}{d\gamma_k} = \left\{
#   \begin{array}{lll}
#   2 \frac{d n_i}{d \gamma_k} n_j & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#   0         & \mathrm{si\ } i,j \in \Omega_{g}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# \frac{d C^K_{ij}}{d \gamma_k} = \left\{
#   \begin{array}{lll}
#   \frac{dn_i}{d\gamma_k} n_j & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#  -\frac{d \sqrt{n_i n_j}}{d \gamma_k} = -\frac{1}{2\sqrt{n_i}} \frac{d n_i}{d \gamma_k} \sqrt{n_j} & \mathrm{si\ } i,j \in \Omega_{g} i=j\ \text{ó}\ i\ \text{y}\ j > \frac{N_{II}}{2}\\
#   \frac{d \sqrt{n_i n_j}}{d \gamma_k} =  \frac{1}{2\sqrt{n_i}} \frac{d n_i}{d \gamma_k} \sqrt{n_j} & \mathrm{si\ } i,j \in \Omega_{g} i\ \text{ó}\ j \leq \frac{N_{II}}{2}\\  \end{array}
#   \right.
# \end{equation}

# In[20]:


#CJCKD5
def CJCKD5(n):    
    
    cj12 = 2*np.einsum('i,j->ij',n,n)
    ck12 = np.einsum('i,j->ij',n,n)    
    
    for l in range(ndoc):            
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns+ncwo*(ndoc-l-1)
        ul = no1 + ndns+ncwo*(ndoc-l)

        cj12[ldx,ll:ul] = 0    
        cj12[ll:ul,ldx] = 0    
    
        cj12[ll:ul,ll:ul] = 0    
        
        ck12[ldx,ll:ul] = np.sqrt(n[ldx]*n[ll:ul])
        ck12[ll:ul,ldx] = np.sqrt(n[ldx]*n[ll:ul])

        ck12[ll:ul,ll:ul] = -np.outer(np.sqrt(n[ll:ul]),np.sqrt(n[ll:ul]))

    return cj12,ck12        
        
def der_CJCKD5(n,gamma,dn_dgamma):

    Dcj12r = 2*np.einsum('ik,j->ijk',dn_dgamma,n)    
    Dck12r = np.einsum('ik,j->ijk',dn_dgamma,n)    

    for l in range(ndoc):            
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns+ncwo*(ndoc-l-1)
        ul = no1 + ndns+ncwo*(ndoc-l)

        Dcj12r[ldx,ll:ul,:nv] = 0
        Dcj12r[ll:ul,ldx,:nv] = 0

        Dcj12r[ll:ul,ll:ul,:nv] = 0   
        
        a = n[ldx] 
        if(a<10**-12):
            a = 10**-12
        b = n[ll:ul]
        if(b<10**-12):
            b = 10**-12
        
        Dck12r[ldx,ll:ul,:nv] = 1/2 * np.sqrt(1/a)*dn_dgamma[ldx,:nv]*np.sqrt(n[ll:ul])
#        Dck12r[ll:ul,ldx,:nv] = 1/2 * np.sqrt(1/b)*dn_dgamma[ll:ul,:nv]*np.sqrt(n[ldx])
        Dck12r[ll:ul,ldx,:nv] = 1/2 * np.sqrt(1/b)*dn_dgamma[ll:ul,:nv]*np.sqrt(n[ldx])
        
        for k in range(nv):
            Dck12r[ll:ul,ll:ul,k] = -1/2 * dn_dgamma[ll:ul,k]   
                        
    return Dcj12r,Dck12r


# Para PNOF7
# 
# \begin{equation}
# E^{PNOF7} = \sum_{g=1}^{N_{II}/2} \left[ \sum_{p \in \Omega_g} n_p (2H_{pp} + J_{pp}) + \sum_{q,p \in \Omega_g q \ne p} \underbrace{\Pi_{qp}^{g}}_{C^K_{pq}} K_{pq} \right] + \sum_{f \neq g}^{N_{II}/2} \sum_{p \in \Omega_f} \sum_{q \in \Omega_g} (\underbrace{2n_q n_p}_{C^J_{q p}} J_{pq}- \underbrace{n_q n_p + \Phi_q \Phi_p}_{C^K_{q p}} K_{pq})
# \end{equation}
# 
# \begin{equation}
# \Phi_i = \sqrt{n_i(1-n_i)}
# \end{equation}
# 
# 
# \begin{equation}
# C^J_{ij} = \left\{
#   \begin{array}{lll}
#   2 n_i n_j & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#   0         & \mathrm{si\ } i,j \in \Omega_{g}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# C^K_{ij} = \left\{
#   \begin{array}{lll}
#   n_i n_j + \Phi_i \Phi_j   & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#   -\Pi_{ij}       & \mathrm{si\ } i,j \in \Omega_{g}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# \Pi_{ij} = \left\{
#   \begin{array}{lll}
#    \sqrt{n_i n_j}      & \mathrm{si\ } i,j \in \Omega_{g} i=j\ \text{ó}\ i\ \text{y}\ j > \frac{N_{II}}{2}\\
#   -\sqrt{n_i n_j}      & \mathrm{si\ } i,j \in \Omega_{g} i\ \text{ó}\ j \leq \frac{N_{II}}{2}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# \frac{d \Phi_i}{d\gamma_k} = \left\{
#   \begin{array}{lll}
#   \frac{d \sqrt{n_i(1-n_i)}}{d \gamma_k} = \frac{1}{2 \sqrt{n_i(1-n_i)}} (1 - 2n_i) \frac{d n_i}{d \gamma_i} \delta_{ik} & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#   0         & \mathrm{si\ } i,j \in \Omega_{g}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# \frac{dC^J_{ij}}{d\gamma_k} = \left\{
#   \begin{array}{lll}
#   2 \frac{d n_i}{d \gamma_k} n_j & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#   0         & \mathrm{si\ } i,j \in \Omega_{g}\\
#   \end{array}
#   \right.
# \end{equation}
# 
# \begin{equation}
# \frac{d C^K_{ij}}{d \gamma_k} = \left\{
#   \begin{array}{lll}
#   \frac{dn_i}{d\gamma_k} n_j + \frac{d\Phi_i}{d\gamma_k} \Phi_j & \mathrm{si\ } i \in \Omega_{g} j \in \Omega_{f} f \ne g\\
#  -\frac{d \sqrt{n_i n_j}}{d \gamma_k} = -\frac{1}{2\sqrt{n_i}} \frac{d n_i}{d \gamma_k} \sqrt{n_j} & \mathrm{si\ } i,j \in \Omega_{g} i=j\ \text{ó}\ i\ \text{y}\ j > \frac{N_{II}}{2}\\
#   \frac{d \sqrt{n_i n_j}}{d \gamma_k} =  \frac{1}{2\sqrt{n_i}} \frac{d n_i}{d \gamma_k} \sqrt{n_j} & \mathrm{si\ } i,j \in \Omega_{g} i\ \text{ó}\ j \leq \frac{N_{II}}{2}\\  \end{array}
#   \right.
# \end{equation}

# In[21]:


#CJCKD7
def CJCKD7(n):    
    
    fi = n*(1-n)
    fi[fi<=0] = 0
    fi = np.sqrt(fi)      
    
    cj12 = 2*np.einsum('i,j->ij',n,n)
    ck12 = np.einsum('i,j->ij',n,n)    
    ck12 += np.einsum('i,j->ij',fi,fi)
    
    for l in range(ndoc):            
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns+ncwo*(ndoc-l-1)
        ul = no1 + ndns+ncwo*(ndoc-l)

        cj12[ldx,ll:ul] = 0    
        cj12[ll:ul,ldx] = 0    
    
        cj12[ll:ul,ll:ul] = 0    
        
        ck12[ldx,ll:ul] = np.sqrt(n[ldx]*n[ll:ul])
        ck12[ll:ul,ldx] = np.sqrt(n[ldx]*n[ll:ul])

        ck12[ll:ul,ll:ul] = -np.outer(np.sqrt(n[ll:ul]),np.sqrt(n[ll:ul]))

    return cj12,ck12        
        
def der_CJCKD7(n,dn_dgamma):
    
    fi = n*(1-n)
    fi[fi<=0] = 0
    fi = np.sqrt(fi)      
            
    dfi_dgamma = np.zeros((nbf5,nv))
    for i in range(no1,nbf5):
        a = fi[i]
        if(a < 10**-12):
            a = 10**-12
        for k in range(nv):
            dfi_dgamma[i,k] = 1/(2*a)*(1-2*n[i])*dn_dgamma[i][k]
    
    Dcj12r = 2*np.einsum('ik,j->ijk',dn_dgamma,n)    
    Dck12r = np.einsum('ik,j->ijk',dn_dgamma,n)    
    Dck12r += np.einsum('ik,j->ijk',dfi_dgamma,fi)    

    for l in range(ndoc):            
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns+ncwo*(ndoc-l-1)
        ul = no1 + ndns+ncwo*(ndoc-l)

        Dcj12r[ldx,ll:ul,:nv] = 0
        Dcj12r[ll:ul,ldx,:nv] = 0

        Dcj12r[ll:ul,ll:ul,:nv] = 0   

        a = n[ldx] 
        if(a<10**-12):
            a = 10**-12
        b = n[ll:ul]
        if(b<10**-12):
            b = 10**-12        
        
        Dck12r[ldx,ll:ul,:nv] = 1/2 * np.sqrt(1/a)*dn_dgamma[ldx,:nv]*np.sqrt(n[ll:ul])
        Dck12r[ll:ul,ldx,:nv] = 1/2 * np.sqrt(1/b)*dn_dgamma[ll:ul,:nv]*np.sqrt(n[ldx])
        
        for k in range(nv):
            Dck12r[ll:ul,ll:ul,k] = - 1/2 * dn_dgamma[ll:ul,k]
                        
    return Dcj12r,Dck12r


# Creamos un seleccionador de PNOF

# In[22]:


def PNOFi_selector(PNOFi,n):
    if(PNOFi==5):
        cj12,ck12 = CJCKD5(n)
    if(PNOFi==7):
        cj12,ck12 = CJCKD7(n)
        
    return cj12,ck12

def der_PNOFi_selector(PNOFi,n,dn_dgamma):
    if(PNOFi==5):
        Dcj12r,Dck12r = der_CJCKD5(n,dn_dgamma)
    if(PNOFi==7):
        Dcj12r,Dck12r = der_CJCKD7(n,dn_dgamma)
        
    return Dcj12r,Dck12r


# ## Optimización de Ocupaciones
# 
# Declaramos el número de variables en la optimización de ocupaciones
# \begin{equation}
# N_v = N_{cwo}N_{doc}
# \end{equation}

# In[23]:


nv = ncwo*ndoc


# Definimos una función para calcular $J_{MO}$ y $K_{MO}$
# \begin{eqnarray}
# {J_{MO}}_{ij} &=& \sum_{\mu\nu} D^{(j)}_{\mu\nu} J^{(i)}_{\mu\nu}\\
# {K_{MO}}_{ij} &=& \sum_{\mu\sigma} D^{(j)}_{\mu\sigma} K^{(i)}_{\mu\sigma}\\
# {H_{core}}_{i} &=& \sum_{\mu\nu} D^{(i)}_{\mu\nu} H_{\mu\nu}
# \end{eqnarray}
# 

# In[24]:


def computeJKH_core_MO(C,H):

    #denmatj    
    D = np.einsum('mi,ni->imn', C[:,0:nbf5], C[:,0:nbf5], optimize=True)

    #QJMATm
    J = np.einsum('isl,mnsl->imn', D, I, optimize=True)    
    J_MO = np.einsum('jmn,imn->ij', D, J, optimize=True)    
        
    #QKMATm        
    K = np.einsum('inl,mnsl->ims', D, I, optimize=True)    
    K_MO = np.einsum('jms,ims->ij', D, K, optimize=True)        

    #QHMATm
    H_core = np.einsum('imn,mn->i', D, H, optimize=True)

    return J_MO,K_MO,H_core


# Definimos una función que calcule los números de ocupación y sus derivadas respecto a $\gamma$

# RO:
# \begin{equation}
# n_i = \left\{
#   \begin{array}{lll}
#    1      & \mathrm{si\ } i \in [1,N_{o1}]\\
#    \frac{1}{2} (1 + cos^2(\gamma_{i-N_{doc}}))     & \mathrm{si\ } i \in (N_{o1},N_{\beta}]\\
#    \frac{1}{2} sin^2(\gamma_{N_{doc}-i-1})     & \mathrm{si\ } i \in (N_{\alpha},N_{bf5}]\\
#   \end{array}
#   \right.
# \end{equation}
# 
# 
# DRO:
# \begin{equation}
# \frac{d n_i}{d \gamma_k} = \left\{
#   \begin{array}{lll}
#    0      & \mathrm{si\ } i \in [1,N_{o1}]\\
#    -\frac{1}{2} sin(2\gamma_{i-N_{doc}}) \delta_{ik}     & \mathrm{si\ } i \in (N_{o1},N_{\beta}]\\
#    \frac{1}{2} sin(2\gamma_{N_{doc}-i-1})     & \mathrm{si\ } i \in (N_{\alpha},N_{bf5}]\\
#   \end{array}
#   \right.
# \end{equation}

# In[25]:


def ocupacion(gamma):
        
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
            icf = nalpha+ndoc - i - 1
            n[icf] = 1/2*np.sin(gamma[i])**2
            dni_dgammai[icf]  = 1/2*np.sin(2*gamma[i])
            dn_dgamma[icf][i] = dni_dgammai[icf]

    return n,dn_dgamma


# **Generamos gamma**
# 
# \begin{equation}
# \gamma_{k} = \left\{
#   \begin{array}{lll}
#   cos^{-1}(\sqrt{2\times0.999-1})      & \mathrm{si\ } k \in N_{doc}  \\
#   sin^{-1}(\sqrt{\frac{1}{N_{cwo}-j+1}})     & \mathrm{si\ } k = N_{doc} + (i-1)(N_{cwo}-1)+j;i\in N_{doc};j\in N_{cwo}-1
#   \end{array}
#   \right.
# \end{equation}
# 
# Esto se hace considerando la relación
# \begin{equation}
# n_g = \frac{1}{2} (1 + cos^2 \gamma_g) ; g= 1,2,\cdots,N_{II}/2
# \end{equation}
# 

# In[26]:


gamma = np.zeros((nbf5))
for i in range(ndoc):
    gamma[i] = np.arccos(np.sqrt(2.0*0.999-1.0))
    for j in range(ncwo-1):
        ig = ndoc+(i)*(ncwo-1)+j
        gamma[ig] = np.arcsin(np.sqrt(1.0/(ncwo-j)))        


# **Definimos una función que recibe ($\gamma$) y calcula la energía**
# 
# \begin{eqnarray}
# E &=& \sum^{N_{o1}}_{i=1} \left[ n_i (2{H_{core}}_i + {J_{MO}}_{ii}) + \sum^{N_{bf5}}_{j \neq i} C^J_{ij} {J_{MO}}_{ji} - C^K_{ij} {K_{MO}}_{ji} \right]\\
# &+& \sum^{N_{doc}}_{i=N_{o1}+1} \left[ n_i (2{H_{core}}_i + {J_{MO}}_{ii}) + \sum^{N_{bf5}}_{j \neq i} C^J_{ij} {J_{MO}}_{ji} - C^K_{ij} {K_{MO}}_{ji} \right]\\
# &+& \sum^{(N_{doc}+1)N_{cwo}}_{i=N_{doc}+1} \left[ n_i (2{H_{core}}_i + {J_{MO}}_{ii}) + \sum^{N_{bf5}}_{j \neq i} C^J_{ij} {J_{MO}}_{ji} - C^K_{ij} {K_{MO}}_{ji} \right]\\
# \end{eqnarray}

# In[27]:


#calce
def calce(gamma,J_MO,K_MO,H_core,PNOFi):
    
    n,dn_dgamma = ocupacion(gamma)
    cj12,ck12 = PNOFi_selector(PNOFi,n)
    
    E = 0

    # 2H + J
    E = E + np.einsum('i,i',n[:nbeta],2*H_core[:nbeta]+np.diagonal(J_MO)[:nbeta]) # [0,Nbeta]
    E = E + np.einsum('i,i',n[nbeta:nalpha],2*H_core[nbeta:nalpha])               # (Nbeta,Nalpha]
    E = E + np.einsum('i,i',n[nalpha:nbf5],2*H_core[nalpha:nbf5]+np.diagonal(J_MO)[nalpha:nbf5]) # (Nalpha,Nbf5)

    #C^J JMO
    E = E + np.einsum('ij,ji',cj12,J_MO) # sum_ij
    E = E - np.einsum('ii,ii',cj12,J_MO) # Quita i=j

    #C^K KMO     
    E = E - np.einsum('ij,ji',ck12,K_MO) # sum_ij
    E = E + np.einsum('ii,ii',ck12,K_MO) # Quita i=j
            
    return E


# Definimos una función que recibe ($\gamma$) y calcula el gradiente
# 
# \begin{equation}
# \frac{dE}{d\gamma_{k}} = \sum_{i=N_{o1}+1}^{N_{\beta}} \left[ \frac{dn_i}{d \gamma_{k}} (2{H_{core}}_{i} + {J_{MO}}_{ii}) + \sum_{j \neq i}^{N_{bf5}} 2 \frac{d C^J_{ij}}{d \gamma_{k}}{J_{MO}}_{ji} - 2 \frac{d C^K_{ij}}{d \gamma_{k}}{K_{MO}}_{ji} \right] + \sum_{i=N_{\alpha}+1}^{N_{bf5}} \left[ \frac{dn_i}{d \gamma_{k}} (2{H_{core}}_{i} + {J_{MO}}_{ii}) + \sum_{j \neq i}^{N_{bf5}} 2 \frac{d C^J_{ij}}{d \gamma_{k}}{J_{MO}}_{ji} - 2 \frac{d C^K_{ij}}{d \gamma_{k}}{K_{MO}}_{ji} \right]
# \end{equation}

# In[28]:


#calcg
def calcg(gamma,J_MO,K_MO,H_core,PNOFi):
    
    grad = np.zeros((nv))

    n,dn_dgamma = ocupacion(gamma)    
    Dcj12r,Dck12r = der_PNOFi_selector(PNOFi,n,dn_dgamma)

    # dn_dgamma (2H+J)
    grad += np.einsum('ik,i->k',dn_dgamma[no1:nbeta,:nv],2*H_core[no1:nbeta]+np.diagonal(J_MO)[no1:nbeta],optimize=True) # [0,Nbeta]
    grad += np.einsum('ik,i->k',dn_dgamma[nalpha:nbf5,:nv],2*H_core[nalpha:nbf5]+np.diagonal(J_MO)[nalpha:nbf5],optimize=True) # [Nalpha,Nbf5]

    # 2 dCJ_dgamma J_MO
    grad += 2*np.einsum('ijk,ji->k',Dcj12r[no1:nbeta,:nbf5,:nv],J_MO[:nbf5,no1:nbeta],optimize=True)
    grad -= 2*np.einsum('iik,ii->k',Dcj12r[no1:nbeta,no1:nbeta,:nv],J_MO[no1:nbeta,no1:nbeta],optimize=True)

    grad += 2*np.einsum('ijk,ji->k',Dcj12r[nalpha:nbf5,:nbf5,:nv],J_MO[:nbf5,nalpha:nbf5],optimize=True)
    grad -= 2*np.einsum('iik,ii->k',Dcj12r[nalpha:nbf5,nalpha:nbf5,:nv],J_MO[nalpha:nbf5,nalpha:nbf5],optimize=True)
      
    # -2 dCK_dgamma K_MO    
    grad -= 2*np.einsum('ijk,ji->k',Dck12r[no1:nbeta,:nbf5,:nv],K_MO[:nbf5,no1:nbeta],optimize=True)
    grad += 2*np.einsum('iik,ii->k',Dck12r[no1:nbeta,no1:nbeta,:nv],K_MO[no1:nbeta,no1:nbeta],optimize=True)

    grad -= 2*np.einsum('ijk,ji->k',Dck12r[nalpha:nbf5,:nbf5,:nv],K_MO[:nbf5,nalpha:nbf5],optimize=True)
    grad += 2*np.einsum('iik,ii->k',Dck12r[nalpha:nbf5,nalpha:nbf5,:nv],K_MO[nalpha:nbf5,nalpha:nbf5],optimize=True)   

    return grad


# **Definimos una función que optimiza los numeros de ocupación utilizando la energía como función objetivo.**

# In[29]:


def occoptr(gamma,firstcall,convgdelag,elag,C,H):
        
    J_MO,K_MO,H_core = computeJKH_core_MO(C,H)
    
    if (not convgdelag):
        if(gradient=="analytical"):
            res = minimize(calce, gamma[:nv], args=(J_MO,K_MO,H_core,PNOFi), jac=calcg, method='CG')
        elif(gradient=="numerical"):
            res = minimize(calce, gamma[:nv], args=(J_MO,K_MO,H_core,PNOFi),  method='CG')
        gamma = res.x
    n,DR = ocupacion(gamma)
    cj12,ck12 = PNOFi_selector(PNOFi,n)
        
    if (firstcall):
        elag_diag = np.zeros((nbf))
               
        # RO (H_core + J)
        elag_diag[:nbeta] = np.einsum('i,i->i',n[:nbeta],H_core[:nbeta]+np.diagonal(J_MO)[:nbeta])        
        elag_diag[nbeta:nalpha] = np.einsum('i,i->i',n[nbeta:nalpha],H_core[nbeta:nalpha])        
        elag_diag[nalpha:nbf5] = np.einsum('i,i->i',n[nalpha:nbf5],H_core[nalpha:nbf5]+np.diagonal(J_MO)[nalpha:nbf5])        

        # CJ12 J_MO
        elag_diag[:nbf5] += np.einsum('ij,ji->i',cj12,J_MO)
        elag_diag[:nbf5] -= np.einsum('ii,ii->i',cj12,J_MO)

        # CK12 K_MO
        elag_diag[:nbf5] -= np.einsum('ij,ji->i',ck12,K_MO)
        elag_diag[:nbf5] += np.einsum('ii,ii->i',ck12,K_MO)
        
        for i in range(nbf):
            elag[i][i] = elag_diag[i]
    
    return gamma,elag,n,cj12,ck12


# Hacemos una primera optimización de los números de ocupación

# In[30]:


gamma,elag,n,cj12,ck12 = occoptr(gamma,True,False,elag,C,H)


# ## Optimización Orbital y (de ocupaciones)

# Definimos una función para DIIS

# In[31]:


def fmiug_diis(fk,fmiug,idiis,bdiis,maxdiff):
    
    if(maxdiff<thdiis):

        restart_diis = False
        fk[idiis,0:noptorb,0:noptorb] = fmiug[0:noptorb,0:noptorb]
        for m in range(idiis+1):
            bdiis[m][idiis] = 0
            for i in range(noptorb):
                for j in range(i):
                    bdiis[m][idiis] = bdiis[m][idiis] + fk[m][i][j]*fk[idiis][j][i]
            bdiis[idiis][m] = bdiis[m][idiis]
            bdiis[m][idiis+1] = -1
            bdiis[idiis+1][m] = -1
        bdiis[idiis+1][idiis+1] = 0  
                        
        if(idiis>=ndiis):
            
            cdiis = np.zeros((idiis+2))
            cdiis[0:idiis+1] = 0
            cdiis[idiis+1] = -1
            x = np.linalg.solve(bdiis[0:idiis+2,0:idiis+2],cdiis[0:idiis+2])
            
            for i in range(noptorb):
                for j in range(i):
                    fmiug[i][j] = 0
                    for k in range(idiis+1):
                        fmiug[i][j] = fmiug[i][j] + x[k]*fk[k][i][j]
                    fmiug[j][i] = fmiug[i][j]
                                        
            restart_diis=True
        idiis = idiis + 1    
        if(restart_diis):
            idiis = 0
            
    return fk,fmiug,idiis,bdiis


# Creamos una función para la optimización orbital

# In[32]:


def orboptr(C,n,cj12,ck12,E_old,sumdiff_old,i_ext,itlim,nzeros,fmiug0):

    convgdelag = False

    E,elag,sumdiff,maxdiff = ENERGY1r(C,n,cj12,ck12)
    E_diff = E-E_old                    
    P_CONV = abs(E_diff)
    E_old = E
    
    if(i_ext==0):
        print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,0,E,E+E_nuc,E_diff,maxdiff))        
            
    if (i_ext>=2 and i_ext >= itlim and sumdiff > sumdiff_old):
        nzeros = nzeros + 1
        itlim = (i_ext + 1) + 10#itziter
        if (nzeros>4):
            nzeros = 2
    sumdiff_old = sumdiff
    
    if(i_ext>=1 and maxdiff<threshl and P_CONV<threshe):
        convgdelag = True
        print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,0,E,E+E_nuc,E_diff,maxdiff))        
        return convgdelag,E_old,sumdiff_old,itlim,nzeros,fmiug0,C
 
    maxlp = 0        
    if i_ext==0:
        maxlp = 1
    else:
        maxlp = 30
    
    fmiug = np.zeros((noptorb,noptorb))
    fk = np.zeros((30,noptorb,noptorb))
    bdiis = np.zeros((31,31))
    cdiis = np.zeros((31))        
    iloop = 0
    idiis = 0
    
    for i_int in range(maxlp):
        iloop = iloop + 1
        E_old2 = E

        #scaling
        fmiug = fmiug_scaling(fmiug0,elag,i_ext,nzeros)        
        fk,fmiug,idiis,bdiis = fmiug_diis(fk,fmiug,idiis,bdiis,maxdiff)   
        eigval, eigvec = np.linalg.eigh(fmiug)
        fmiug0 = eigval
                
        C = np.matmul(C,eigvec)

        E,elag,sumdiff,maxdiff = ENERGY1r(C,n,cj12,ck12)

        E_diff2 = E-E_old2                    
                       
        if(abs(E_diff2)<threshec or i_int==maxlp-1):
            E_diff = E-E_old
            E_old = E
            print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext+1,i_int,E,E+E_nuc,E_diff,maxdiff))
            break    
    
    return convgdelag,E_old,sumdiff_old,itlim,nzeros,fmiug0,C


# ## SCF de PNOF

# In[33]:


iloop = 0
itlim = 1
E_old = E
sumdiff_old = 0

print('{:^7} {:^7} {:^14} {:^14} {:^14} {:^14}'.format("Nitext","Nitint","Eelec","Etot","Ediff","maxdiff"))

for i_ext in range(1000):
    
    #orboptr
    convgdelag,E_old,sumdiff_old,itlim,nzeros,fmiug0,C = orboptr(C,n,cj12,ck12,E_old,sumdiff_old,i_ext,itlim,nzeros,fmiug0)     
    
    #occopt
    gamma,elag,n,cj12,ck12 = occoptr(gamma,False,convgdelag,elag,C,H)
        
    if(convgdelag):
        break


# In[ ]:




