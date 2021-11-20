# Dendro-KT
kd-tree based adaptive grid for spacetime(kairotopos) discretizations for k dimentional finite element computations

The following dependencies are required to compile Dendro-KT

* C/C++ compilers with C++11 standards and OpenMP support
* MPI implementation (e.g. openmpi, mvapich2 )
* ZLib compression library (used to write \texttt{.vtu} files in binary format with compression enabled)
* BLAS and LAPACK are optional and not needed for current version of \dendrokt~
* CMake 2.8 or higher version

To compile the code, execute these commands

```
cd <path to Dendro-KT directory >
cd build
ccmake ../ 
make all 
```
