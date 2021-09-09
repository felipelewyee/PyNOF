import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyNOF",
    version="0.1",
    author="Juan Felipe Huan Lew Yee",
    author_email="felipe.lew.yee@gmail.com",
    description="Paquete para realizar funcionales PNOF en Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/felipelewyee/PyNOF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['numpy','scipy','numba'],
)

