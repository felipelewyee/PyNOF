import psi4

def molecule(mol):
    mol = psi4.geometry(mol)
    return mol


