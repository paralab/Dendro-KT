# Dendro-KT

Tree based adaptive grid for spacetime *(kairotopos)* discretizations for k-dimensional finite element computations.

The following dependencies are required to compile Dendro-KT:

* C/C++ compilers with C++14 standards and OpenMP support (e.g., g++ 6.1+, not clang)
* MPI implementation (e.g. mpich, mvapich2, openmpi)
* BLAS is optional
* LAPACK (liblapack-dev, liblapacke-dev)
* PETSc is optional but recommended (versions 3.12+)
* CMake 2.8+

The following dependencies are shipped with the source code and need no manual intervention:
* ZLib compression library (used to write \texttt{.vtu} files in binary format with compression enabled)

To compile the most recent code (develop branch), execute these commands

```
git clone https://github.com/paralab/Dendro-KT.git
cd Dendro-KT
git checkout develop
mkdir build
cd build
cmake ..
make all
```
