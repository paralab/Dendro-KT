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
The current CMakeLists.txt has default settings assuming PETSc is installed in the user's $HOME directory.
Please make sure to use appropriate CMake flags or change the top level CMakeLists.txt if PETSc is 
installed in a different directory.

If both g++ and clang are installed, make sure to check which version is picked by CMake by checking 
CMake logs and CMakeCache.txt file.

If using GNUPlot for plotting/debugging, make sure to install all the required GNUPlot packages.
Specifically check the log files which will contain errors related to
any missing packages. 
