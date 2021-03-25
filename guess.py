import numpy as np

def read_C():
    C = np.load("C.npy")
    return C

def read_gamma():
    gamma = np.load("gamma.npy")
    return gamma

def read_fmiug0():
    fmiug0 = np.load("fmiug0.npy")
    return fmiug0

def read_all():
    C = read_C()
    gamma = read_gamma()
    fmiug0 = read_fmiug0()

    return C,gamma,fmiug0
