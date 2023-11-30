import pynof
import numpy as np

def read_C(title = "pynof"):
    C = np.load(title+"_C.npy")
    return C

def read_n(title = "pynof"):
    n = np.load(title+"_n.npy")
    return n

def read_fmiug0(title = "pynof"):
    fmiug0 = np.load(title+"_fmiug0.npy")
    return fmiug0

def read_all(title = "pynof"):
    C = read_C(title)
    n = read_n(title)
    fmiug0 = read_fmiug0(title)

    return C,n,fmiug0

def order_subspaces(old_C,old_n,elag,H,I,b_mnl,p):

    C = np.zeros((p.nbf,p.nbf))
    n = np.zeros((p.nbf5))

    #Sort no1 orbitals
    elag_diag = np.diag(elag)[0:p.no1]
    sort_idx = elag_diag.argsort()
    for i in range(p.no1):
        i_old  = sort_idx[i]
        C[0:p.nbf,i] = old_C[0:p.nbf,i_old]
        n[i] = old_n[i_old]

    #Sort ndoc subspaces
    elag_diag = np.diag(elag)[p.no1:p.ndoc]
    sort_idx = elag_diag.argsort()
    for i in range(p.ndoc):
        i_old  = sort_idx[i]
        ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - i - 1)
        ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - i)

        ll_old = p.no1 + p.ndns + p.ncwo*(p.ndoc - i_old - 1)
        ul_old = p.no1 + p.ndns + p.ncwo*(p.ndoc - i_old)

        C[0:p.nbf,p.no1+i] = old_C[0:p.nbf,p.no1+i_old]
        C[0:p.nbf,ll:ul] = old_C[0:p.nbf,ll_old:ul_old]
        n[p.no1+i] = old_n[p.no1+i_old]
        n[ll:ul] = old_n[ll_old:ul_old]

    #Sort nsoc orbitals
    elag_diag = np.diag(elag)[p.no1+p.ndoc:p.no1+p.ndns]
    sort_idx = elag_diag.argsort()
    for i in range(p.nsoc):
        i_old  = sort_idx[i]
        C[0:p.nbf,p.no1+p.ndoc+i] = old_C[0:p.nbf,p.no1+p.ndoc+i_old]
        n[p.no1+p.ndoc+i] = old_n[p.no1+p.ndoc+i_old]

    C[0:p.nbf,p.nbf5:p.nbf] = old_C[0:p.nbf,p.nbf5:p.nbf]

    cj12,ck12 = pynof.PNOFi_selector(n,p)
    Etmp,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

    return C,n,elag

def order_occupations_softmax(old_C,old_gamma,H,I,b_mnl,p):

    C = old_C.copy()
    gamma = np.zeros((p.nv))

    #Sort ndoc subspaces
    gamma_tmp = np.zeros((1+p.ncwo))
    C_tmp = np.zeros((p.nbf,1+p.ncwo))
    for i in range(p.ndoc):
        old_ll = p.no1 + p.ndns + p.ncwo*(p.ndoc - i - 1)
        old_ul = p.no1 + p.ndns + p.ncwo*(p.ndoc - i)
        C_tmp[:,0] = old_C[:,p.no1+i]
        C_tmp[:,1:] = old_C[:,old_ll:old_ul]

        old_ll_x = old_ll - p.ndns + p.ndoc - p.no1
        old_ul_x = old_ul - p.ndns + p.ndoc - p.no1
        gamma_tmp[0] = old_gamma[i]
        gamma_tmp[1:] = old_gamma[old_ll_x:old_ul_x]

        sort_idx = gamma_tmp.argsort()[::-1]
        gamma_tmp = gamma_tmp[sort_idx]
        C_tmp = C_tmp[:,sort_idx]

        gamma[i] = gamma_tmp[0]
        gamma[old_ll_x:old_ul_x] = gamma_tmp[1:]
        C[:,p.no1+i] = C_tmp[:,0]
        C[:,old_ll:old_ul] = C_tmp[:,1:]

    return C,gamma

