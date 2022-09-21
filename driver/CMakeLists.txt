add_executable(triangle_mesh triangle_mesh.cpp)
target_include_directories(triangle_mesh PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(triangle_mesh dendroKT ${MPI_LIBRARIES} m)
