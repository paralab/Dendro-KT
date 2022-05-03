add_executable(treesort-hexlist treesort-hexlist.cpp)
target_include_directories(treesort-hexlist PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(treesort-hexlist dendroKT ${MPI_LIBRARIES} m)
