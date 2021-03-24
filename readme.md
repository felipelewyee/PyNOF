# PyDoNOF

Formally readed as Python-Doing-Natural-Orbital-Functionals, PyDoNOF is based on the original DoNOF code written in Fortran by Mario Piris, but it take advantage of the Python capabilities such as optimizers, vectorization via numpy and gpu compatibiliy via cupy.

# Requirements

You should install the folowing libreries
- numpy
- scipy
- psi4
- numba
- cupy

Although GPUs are not required, it is still needed to install cupy or comment the appropiate lines in integrals.py

# Example

You can edit the molecule and parameters of the calculation via the main.py file. All parameters available are found in parameters.py. To run a calculation, just execute
~~~
python main.py
~~~
