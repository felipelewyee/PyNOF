import numpy as np

def read_C(title = "pydonof"):
    C = np.load(title+"_C.npy")
    return C

def read_gamma(title = "pydonof"):
    gamma = np.load(title+"_gamma.npy")
    return gamma

def read_fmiug0(title = "pydonof"):
    fmiug0 = np.load(title+"_fmiug0.npy")
    return fmiug0

def read_all(title = "pydonof"):
    C = read_C(title)
    gamma = read_gamma(title)
    fmiug0 = read_fmiug0(title)

    return C,gamma,fmiug0
