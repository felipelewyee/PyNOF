import numpy as np
import psi4

class param():
   
    def __init__(self,mol,basname):

        psi4.set_options({'basis': basname})

        # Paramdetros del sistema
        self.wfn = wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))

        # Parámetros 
        self.natoms = mol.natom()
        self.nbf = wfn.basisset().nbf()
        self.nbfaux = 0
        self.nalpha = wfn.nalpha()
        self.nbeta = wfn.nbeta()
        self.ne = self.nalpha + self.nbeta
        self.mul = mol.multiplicity()
        self.charge = mol.molecular_charge()
        no1 = 0 #Number of inactive doubly occupied orbitals | Se puede variar
        for i in range(self.natoms):
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
        self.ndoc = self.nbeta   -   no1
        self.nsoc = self.nalpha  -   self.nbeta
        self.ndns = self.ndoc    +   self.nsoc
        self.nvir = self.nbf     -   self.nalpha
        
        ncwo = 1
        if(self.ne==2):
            ncwo= -1
        if(self.ndns!=0):
            if(self.ndoc>0):
                if(ncwo!=1):
                    if(ncwo==-1 or ncwo > self.nvir/self.ndoc):
                        ncwo = int(self.nvir/self.ndoc)
            else:
                ncwo = 0

        self.closed = (self.nbeta == (self.ne+self.mul-1)/2 and self.nalpha == (self.ne-self.mul+1)/2)
        
        self.nac = self.ndoc * (1 + ncwo)
        self.nbf5 = no1 + self.nac + self.nsoc   #JFHLY warning: nbf must be >nbf5
        self.no0 = self.nbf - self.nbf5

        noptorb = self.nbf
  
        self.title = "pynof"
        self.maxit = 1000  # Número máximo de iteraciones de Occ-SCF
        self.no1 = no1     # Número de orbitales inactivos con ocupación 1
        self.thresheid = 10**-6#8 # Convergencia de la energía total
        self.maxitid = 30  # Número máximo de iteraciones externas en HF
        self.maxloop = 30  # Iteraciones internas en optimización orbital
        self.ipnof = 7     # PNOFi a calcular
        self.ista = 0     # PNOFi a calcular
        self.threshl = 10**-5#4   # Convergencia de los multiplicadores de Lagrange
        self.threshe = 10**-6#6   # Convergencia de la energía
        self.threshec = 10**-10 # Convergencia  de la energía en optimización orbital
        self.threshen = 10**-10 # Convergencia  de la energía en optimización de ocupaciones
        self.scaling = True     # Scaling for f
        self.nzeros = 0
        self.nzerosm = 5
        self.nzerosr = 2
        self.itziter = 10        # Iteraciones para scaling constante
        self.diis = True         # DIIS en optimización orbital
        self.thdiis = 10**-3     # Para iniciar DIIS
        self.ndiis = 5           # Número de ciclos para interpolar matriz de Fock generalizada en DIIS
        self.perdiis = True      # Aplica DIIS cada NDIIS (True) o después de NDIIS (False)
        self.ncwo = ncwo         # Número de orbitales débilmente ocupados acoplados a cada orbital fueremtente ocupado
        self.noptorb = noptorb   # Número de orbitales a optimizar Nbf5 <= Noptorb <= Nbf
        self.scaling = True
        self.nv = self.ncwo*self.ndoc
        self.gradient = "analytical"
        self.optimizer = "L-BFGS-B"
        self.gpu = False
        self.RI = False
        self.jit= False

        self.HighSpin = False
        self.MSpin = 0

        self.lamb = 0.0
        self.method = "ID"

    def autozeros(self,restart=False):
        if(restart):
            self.nzeros = abs(int(np.log10(self.threshl))) - 1
            self.nzerosr = self.nzeros
            #self.nzerosm = abs(int(np.log10(self.threshl))) + 2         
            #if(self.nzeros<3):
            #    self.nzeros = 2
            #    self.nzerosr = 2
            #    self.nzerosm = 5
        else:
            self.nzeros = 0
            self.nzerosr = 0
            #self.nzerosm = abs(int(np.log10(self.threshl))) + 2

    def set_ncwo(self,ncwo):
        if(self.ne==2):
            ncwo= -1
        if(self.ndns!=0):
            if(self.ndoc>0):
                if(ncwo!=1):
                    if(ncwo==-1 or ncwo > self.nvir/self.ndoc):
                        ncwo = int(self.nvir/self.ndoc)
            else:
                ncwo = 0

        self.ncwo = ncwo

        self.nac = self.ndoc * (1 + ncwo)
        self.nbf5 = self.no1 + self.nac + self.nsoc   #JFHLY warning: nbf must be >nbf5
        self.no0 = self.nbf - self.nbf5
        self.nv = self.ncwo*self.ndoc
