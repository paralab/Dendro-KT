
## https://stackoverflow.com/a/65133324
add_custom_target(copy_assets
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/copy-assets.cmake
)

configure_file(directories.h.in directories.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(doctest_hello_world hello_world.cpp main.cpp)
target_include_directories(doctest_hello_world PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(doctest_hello_world dendroKT ${MPI_LIBRARIES} m)


add_executable(doctest_coarsening coarsening.cpp main.cpp)
add_dependencies(doctest_coarsening copy_assets)
target_include_directories(doctest_coarsening PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(doctest_coarsening dendroKT ${MPI_LIBRARIES} m)

add_executable(doctest_refine_many refine_many.cpp main.cpp)
add_dependencies(doctest_refine_many copy_assets)
target_include_directories(doctest_refine_many PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(doctest_refine_many dendroKT ${MPI_LIBRARIES} m)

add_executable(doctest_octree_adler32 octree_adler32.cpp main.cpp)
add_dependencies(doctest_octree_adler32 copy_assets)
target_include_directories(doctest_octree_adler32 PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(doctest_octree_adler32 dendroKT ${MPI_LIBRARIES} m)

add_executable(doctest_partition_unweighted partition_unweighted.cpp main.cpp)
target_include_directories(doctest_partition_unweighted PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(doctest_partition_unweighted dendroKT ${MPI_LIBRARIES} m)
