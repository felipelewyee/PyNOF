# PyNOF

| **Citation** | [![DOI](https://zenodo.org/badge/346216950.svg)](https://zenodo.org/badge/latestdoi/346216950) |
| ------------ | ---------------------------------------------------------------------------------------------- |

Formally read as Python-Natural-Orbital-Functionals, PyNOF is based on the original [DoNOF](https://github.com/DoNOF/DoNOFsw/) software written in Fortran by Prof. Mario Piris, but it takes advantage of the Python capabilities such as optimizers, vectorization via numpy and gpu compatibility via cupy.

# <img src="https://github.com/felipelewyee/PyNOF/blob/master/PyNOF.png" height=150>

# Requirements

You should have [Anaconda](https://www.anaconda.com/) installed, and the following libraries: psi4, numpy, scipy, numba, cupy; then, you may install pynof via pip.

# Installation

First, clone PyNOF from github and change to the project directory
~~~
git clone https://github.com/felipelewyee/PyNOF.git
cd PyNOF
~~~

In the PyNOF folder, execute the following code
~~~
conda create -n pynof -y
conda activate pynof
conda install -c conda-forge cupy
conda install numpy scipy numba
conda install psi4 -c conda-forge/label/libint_dev -c conda-forge 
python setup.py bdist_wheel && cd dist && pip install PyNOF-0.1-py3-none-any.whl && cd ..
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

p = pynof.param(mol,"cc-pvdz")
p.autozeros()

p.ipnof=8

p.RI = True
#p.gpu = True

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p,hfidr=True)
~~~

If everything worked, the job may be executed by
~~~
python -u test_energy.py
~~~

*Note.* The first run may be slightly slow due to jit precompilation.

<meta name="google-site-verification" content="c8fIbSDge0oLPu2RxGxupxP2Gq0GlFawiFoX9M4QCGw" />
