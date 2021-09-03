# PyNOF

Formally readed as Python-Natural-Orbital-Functionals, PyNOF, is based on the original [DoNOF](https://github.com/DoNOF/DoNOFsw/) software written in Fortran by Mario Piris, but it take advantage of the Python capabilities such as optimizers, vectorization via numpy and gpu compatibiliy via cupy.

# Requirements

You should install the folowing libreries: psi4, numpy, scipy, numba, cupy; then, you may install pynof via pip.

# Installation

In the PyNOF folder, execute the following code
~~~
conda create -n pynof
conda install -c psi4 psi4
conda install numpy, scipy, numba, cupy, pip
cd dist
pip install PyNOF-0.1-py3-none-any.whl
~~~

# Example

A pynof input has the following parts:
- Import pynof
- A declaration of the molecule geometry
- A parameter object
- The calculation instruction
 
~~~
import pynof

mol = pynof.molecule("""
0 1
O  0.0000   0.000   0.116
H  0.0000   0.749  -0.453
H  0.0000  -0.749  -0.453
""")

p = pynof.param(mol,"cc-pVDZ")
p.autozeros()

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,p.gradient)
~~~

Minimal working examples are provided in test_energy.py and test_optimization.py.

You can edit the behavior of the calculation through the parmeter object. FOr example, if you have a GPU, you can set
~~~
p.gpu = true
~~~
