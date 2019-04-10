## CMAKE for Dendro-KT airtight subdirectory
cmake_minimum_required(VERSION 2.8)
project(DKT-airtight)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(DKT_AIRT_INCLUDE
  )

set(DKT_AIRT_SRC
  )

add_library(dktat ${DKT_AIRT_INCLUDE} ${DKT_AIRT_SRC})
target_include_directories(dendroKT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dktat dendroKT ${MPI_LIBRARIES} m)

## ----------------------------------------------------------

set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/main/tryConstruction.cpp)
add_executable(tryConstruction ${SRC_FILES})
target_include_directories(tryConstruction  PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(tryConstruction dktat ${MPI_LIBRARIES} m)

set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/main/trySorting.cpp)
add_executable(trySorting ${SRC_FILES})
target_include_directories(trySorting  PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(trySorting dktat ${MPI_LIBRARIES} m)
