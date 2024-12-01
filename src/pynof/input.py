import psi4

def molecule(mol):

    mol = mol + "symmetry c1"
    mol = psi4.geometry(mol)

    return mol


