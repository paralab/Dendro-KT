## CMAKE for Dendro-KT
cmake_minimum_required(VERSION 3.10)
project(Dendro-KT C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This lets clangd LSP know the included directories.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)

# For now we just make it compulsory to have LAPACK installed.
#Later we will make it possible if LAPACK is not present to automaticall install before compiling dendro5
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



## options for dendro

option(USE_64BIT_INDICES "Use 64-Bit indices. Reverts to 32-bit if turned off" ON)
option(ALLTOALLV_FIX "Use K-way all to all v" ON)
option(SPLITTER_SELECTION_FIX "Turn on Splitter Selection fix" ON)
option(DIM_2 "use the two dimentional sorting" OFF)
option(DENDRO_VTK_BINARY "write vtk/vtu files in binary mode " ON)
option(DENDRO_VTK_ZLIB_COMPRES "write vtk/vtu files in binary mode with zlib compression (only compatible with binary mode) " OFF)
option(BUILD_WITH_PETSC " build dendro with PETSC " ON)
option(HILBERT_ORDERING "use the Hilbert space-filling curve to order orthants" OFF)
option(BUILD_EXAMPLES "build example programs" ON)
option(USE_2TO1_GLOBAL_SORT "Use global sort instead of two-pass method for 2:1 balancing" OFF)
option(ENABLE_DOCTEST_IN_RELEASE "Enable doctest tests even in a release build" OFF)
option(HIGHFIVE_USE_BOOST "Retains the default dependence of HighFive on Boost" OFF)

set(KWAY 128 CACHE INT 128)
set(NUM_NPES_THRESHOLD 2 CACHE INT 2)

#set the build type to release by default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release " FORCE)
endif()

if ((CMAKE_BUILD_TYPE STREQUAL "Release") AND (NOT ENABLE_DOCTEST_IN_RELEASE))
    add_definitions(-DDOCTEST_CONFIG_DISABLE)
endif()


if(BUILD_WITH_PETSC)
    pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)
    add_library(PETSc ALIAS PkgConfig::PETSc)
    add_definitions(-DBUILD_WITH_PETSC)
endif()

if(USE_2TO1_GLOBAL_SORT)
  add_definitions(-DUSE_2TO1_GLOBAL_SORT)
endif()


if(DIM_2)
    add_definitions(-DDIM_2)
endif()

if(USE_64BIT_INDICES)
    add_definitions(-DUSE_64BIT_INDICES)
    #message('Configuring 64BIT indices')
endif()

if(ALLTOALLV_FIX)
    add_definitions(-DALLTOALLV_FIX)
    add_definitions(-DKWAY=${KWAY})
endif()

if(SPLITTER_SELECTION_FIX)
    add_definitions(-DSPLITTER_SELECTION_FIX)
    add_definitions(-DNUM_NPES_THRESHOLD=${NUM_NPES_THRESHOLD})
endif()

if(ALLTOALL_SPARSE)
    add_definitions(-DALLTOALL_SPARSE)
endif()

if(DENDRO_VTK_BINARY)
else()
    set(DENDRO_VTK_ZLIB_COMPRES OFF)
endif()

if(DENDRO_VTK_BINARY)
    add_definitions(-DDENDRO_VTU_BINARY)
    if(DENDRO_VTK_ZLIB_COMPRES)
        add_definitions(-DDENDRO_VTU_ZLIB)
    endif()
else()
    add_definitions(-DDENDRO_VTU_ASCII)
endif()

add_subdirectory(external/eigen)
add_subdirectory(external/c4conf)
add_subdirectory(external/fmt)

add_subdirectory(debug/zonelog)

# Add the Tracy profiler.
option(TRACY_ENABLE "" ON)
### option(TRACY_ON_DEMAND "" ON)
add_subdirectory(external/tracy)
# For targets using tracy:  target_link_libraries(<TARGET> PUBLIC TracyClient)


set(DENDRO_KT_INC include/dendro.h
                  include/dtypes.h
                  include/hcurvedata.h
                  include/KDhcurvedata_decl.h
                  include/binUtils.h
                  include/colors.h
                  include/mathUtils.h
                  include/mathUtils.tcc
                  include/octUtils.h
                  include/tnUtils.h
                  include/ompUtils.h
                  include/ompUtils.tcc
                  include/point.h
                  include/profiler.h
                  include/seqUtils.h
                  include/seqUtils.tcc
                  include/parUtils.h
                  include/parUtils.tcc
                  include/comm_hierarchy.h
                  include/templateUtils.h
                  include/tsort.h
                  include/filterFunction.h
                  include/distTree.h
                  include/oda.h
                  include/oda.tcc
                  include/nsort.h
                  include/nsort.tcc
                  include/tsearchCmpx.h
                  include/pcoord.h
                  include/treeNode.h
                  include/treeNode.tcc
                  include/asyncExchangeContex.h
                  include/sfcTreeLoop.h
                  include/sfcTreeLoop_matvec.h
                  include/sfcTreeLoop_matvec_io.h
                  include/meshLoop.h
                  include/dollar_stat.h
                  include/serialize.hpp
                  include/p2p.h
                  include/p2p.tcc
                  include/lazy.hpp
                  include/passive_shared_ptr.hpp
                  IO/zlib/inc/zlib.h
                  IO/vtk/include/oct2vtk.h
                  IO/vtk/include/checkPoint.h
                  IO/hexlist/json_hexlist.h
                  array/include/arraySlice.h
                  FEM/include/matvec.h
                  FEM/include/tensor.h
                  FEM/include/tensor_impl.h
                  FEM/include/refel.h
                  FEM/include/basis.h
                  FEM/include/lapac.h
                  FEM/include/feMat.h
                  FEM/include/feMatrix.h
                  FEM/include/feVec.h
                  FEM/include/feVector.h
                  FEM/include/interpMatrices.h
                  FEM/include/intergridTransfer.h
                  FEM/include/coarseToFine.hpp
                  FEM/include/surrogate_cell_transfer.hpp
                  FEM/include/setDiag.h
                  FEM/include/gmgMat.h
                  FEM/include/mg.hpp
                  FEM/include/solver_utils.hpp
                  FEM/include/hybrid_mat_wrapper.hpp
                  )

set(DENDRO_KT_SRC src/binUtils.cpp
                  src/octUtils.cpp
                  src/parUtils.cpp
                  src/comm_hierarchy.cpp
                  src/point.cpp
                  src/profiler.cpp
                  src/KDhcurvedata.cpp
                  src/KDhcurvedata_DATA.cpp
                  src/tsort.cpp
                  src/nsort.cpp
                  src/distTree.cpp
                  src/tsearchCmpx.cpp
                  src/pcoord.cpp
                  src/treeNode.cpp
                  src/dollar_stat.cpp
                  IO/zlib/src/adler32.c
                  IO/vtk/src/oct2vtk.cpp
                  IO/vtk/src/checkPoint.cpp
                  IO/hexlist/json_hexlist.cpp
                  src/oda.cpp
                  src/p2p.cpp
                  FEM/src/tensor.cpp
                  FEM/src/tensor_special_float.cpp
                  FEM/src/tensor_special_double.cpp
                  FEM/src/tensor_special_longlong.cpp
                  FEM/src/tensor_special_int.cpp
                  FEM/src/tensor_special_unsigned.cpp
                  FEM/src/tensor_special_char.cpp
                  FEM/src/refel.cpp
                  FEM/src/basis.cpp
                  FEM/src/hybrid_mat_wrapper.cpp
                  )


add_library(dendroKT ${DENDRO_KT_INC} ${DENDRO_KT_SRC})
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/dollar-master)
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/IO/vtk/include)
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/IO/zlib/inc)
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/array/include)
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/FEM/include)
target_link_libraries(dendroKT PUBLIC zonelog)
target_link_libraries(dendroKT PRIVATE eigen)
target_link_libraries(dendroKT PRIVATE fmt::fmt)
target_link_libraries(dendroKT PRIVATE m)
target_link_libraries(dendroKT PRIVATE MPI::MPI_CXX)

if(BUILD_WITH_PETSC)
  target_link_libraries(dendroKT PUBLIC PETSc)
endif()

if(HILBERT_ORDERING)
  add_definitions(-DHILBERT_ORDERING)
endif()


add_subdirectory(test)
add_subdirectory(test_doctests)

add_subdirectory(utility)


## tsort_bench (./tsortBench)
## -----------
set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/bench/include/tsort_bench.h ${CMAKE_CURRENT_SOURCE_DIR}/bench/src/tsort_bench.cpp)
add_executable(tsortBench ${SRC_FILES})
target_include_directories(tsortBench PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/bench/include)
target_link_libraries(tsortBench dendroKT ${MPI_LIBRARIES} m)

## matvec_bench (./matvecBench)
## ------------
set(SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/include/matvec_bench.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/src/matvec_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/heatMat.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/heatVec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/heatVec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/heatMat.cpp)
add_executable(matvecBench ${SRC_FILES})
target_include_directories(matvecBench PUBLIC ${MPI_INCLUDE_PATH}
                                              ${CMAKE_CURRENT_SOURCE_DIR}/bench/include
                                              ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
target_link_libraries(matvecBench dendroKT ${MPI_LIBRARIES} m)

## matvec_bench_adaptive (./matvecBenchAdaptive)
## ---------------------
set(SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/include/matvec_bench.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/src/matvec_bench_adaptive.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/heatMat.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/heatVec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/heatVec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/heatMat.cpp)
add_executable(matvecBenchAdaptive ${SRC_FILES} external/tracy/public/TracyClient.cpp)
target_include_directories(matvecBenchAdaptive PUBLIC ${MPI_INCLUDE_PATH}
                                              ${CMAKE_CURRENT_SOURCE_DIR}/bench/include
                                              ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
target_link_libraries(matvecBenchAdaptive dendroKT ${MPI_LIBRARIES} m)
target_link_libraries(matvecBenchAdaptive TracyClient)


## matvec_bench_channel (./matvecBenchChannel)
## ---------------------
set(SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/include/matvec_bench.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/src/matvec_bench_channel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/include/genChannelPoints.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/src/genChannelPoints.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonMat.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonVec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonVec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonMat.cpp)
add_executable(matvecBenchChannel ${SRC_FILES})
target_include_directories(matvecBenchChannel PUBLIC ${MPI_INCLUDE_PATH}
                                              ${CMAKE_CURRENT_SOURCE_DIR}/bench/include
                                              ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
target_link_libraries(matvecBenchChannel dendroKT ${MPI_LIBRARIES} m)

## uniform_faces (./bench_uniform_faces)
## -----------
set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/bench/src/uniform_faces.cpp)
add_executable(bench_uniform_faces ${SRC_FILES})
target_include_directories(bench_uniform_faces PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/bench/include)
target_link_libraries(bench_uniform_faces dendroKT ${MPI_LIBRARIES} m)





set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/matrix.h
              ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/matrix.cpp)
add_executable(tstMatAssembly ${SRC_FILES})
set_property(TARGET tstMatAssembly PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
target_include_directories(tstMatAssembly PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
target_link_libraries(tstMatAssembly dendroKT MPI::MPI_CXX m)





## Examples
if(BUILD_EXAMPLES)
  ## Heat equation.
  set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/heatMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/heatVec.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/heatVec.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/heatMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/heatEq.cpp)
  add_executable(exHeatEq ${SRC_FILES})
  set_property(TARGET exHeatEq PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
  target_include_directories(exHeatEq PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
  target_link_libraries(exHeatEq dendroKT ${MPI_LIBRARIES} m)

  ## Poisson equation.
  set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonVec.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonVec.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonEq.cpp)
  add_executable(exPoissonEq ${SRC_FILES})
  set_property(TARGET exPoissonEq PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
  target_include_directories(exPoissonEq PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
  target_link_libraries(exPoissonEq dendroKT ${MPI_LIBRARIES} m)

  ## Poisson equation with gmg.
  set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonVec.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonVec.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonGMG.cpp)
  add_executable(exPoissonGMG ${SRC_FILES})
  set_property(TARGET exPoissonGMG PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
  target_include_directories(exPoissonGMG PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
  target_link_libraries(exPoissonGMG dendroKT ${MPI_LIBRARIES} m)

  ## Test the poisson elemental operators are consistent.
  set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/test/testPoissonOperator.cpp)
  add_executable(tstPoissonOperator ${SRC_FILES})
  set_property(TARGET tstPoissonOperator PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
  target_include_directories(tstPoissonOperator PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
  target_link_libraries(tstPoissonOperator dendroKT ${MPI_LIBRARIES} m)


  ## Poisson equation in a series of doctest test cases.
  set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonVec.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonVec.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/doctest_poisson.cpp)
  # future: use a petsc_main.cpp to have PetscInitialize()
  add_executable(doctest_poisson ${SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/test_doctests/main.cpp)
  set_property(TARGET doctest_poisson PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
  target_include_directories(doctest_poisson PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
  target_link_libraries(doctest_poisson dendroKT ${MPI_LIBRARIES} m)

  ## GMG as preconditioner to solve Poisson (doctest test cases).
  set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonVec.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/hybridPoissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonVec.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/hybridPoissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/doctest_gmg_poisson.cpp)
  add_executable(doctest_gmg_poisson ${SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/test_doctests/main_with_petsc.cpp)
  set_property(TARGET doctest_gmg_poisson PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
  target_include_directories(doctest_gmg_poisson PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
  target_link_libraries(doctest_gmg_poisson dendroKT MPI::MPI_CXX m eigen)



add_subdirectory(external/HighFive)

  ## Parameterized GMG to study convergence behavior.
  set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/poissonVec.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include/hybridPoissonMat.h
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonVec.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/poissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/hybridPoissonMat.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/src/gmg_stat.cpp)
  # future: use a petsc_main.cpp to have PetscInitialize()
  add_executable(gmg_stat ${SRC_FILES})
  set_property(TARGET gmg_stat PROPERTY RUNTIME_OUTPUT_DIRECTORY Examples/FEM)
  target_include_directories(gmg_stat PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include)
  target_link_libraries(gmg_stat dendroKT MPI::MPI_CXX m eigen HighFive c4conf)


endif(BUILD_EXAMPLES)


