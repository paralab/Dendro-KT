
## da_vs_push_face_neighbor (./da_vs_push_face_neighbor)
## ------------------------
set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/da_vs_push_face_neighbor.cpp)
add_executable(da_vs_push_face_neighbor ${SRC_FILES})
target_include_directories(da_vs_push_face_neighbor PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(da_vs_push_face_neighbor dendroKT ${MPI_LIBRARIES} m)


