
## https://stackoverflow.com/a/65133324
add_custom_target(copy_assets
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/copy-assets.cmake
)

configure_file(directories.h.in directories.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(doctest_hello_world hello_world.cpp main.cpp)
set_property(TARGET doctest_hello_world PROPERTY RUNTIME_OUTPUT_DIRECTORY test_doctests)
target_include_directories(doctest_hello_world PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(doctest_hello_world dendroKT ${MPI_LIBRARIES} m)


add_executable(doctest_coarsening coarsening.cpp main.cpp)
add_dependencies(doctest_coarsening copy_assets)
set_property(TARGET doctest_coarsening PROPERTY RUNTIME_OUTPUT_DIRECTORY test_doctests)
target_include_directories(doctest_coarsening PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(doctest_coarsening dendroKT ${MPI_LIBRARIES} m)
