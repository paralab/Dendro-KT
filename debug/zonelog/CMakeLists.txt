#future: remove this section to be a library of larger project,
#        or make it optional for standalone project, but then FIND(fmtlib)
  cmake_minimum_required(VERSION 3.10)
  project(libzonelog C CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  add_subdirectory(../../external/fmt external/fmt)

add_library(zonelog zonelog.h zonelog.cpp)
target_sources(zonelog PRIVATE zonelog_call_trie.h)
target_sources(zonelog PRIVATE zonelog_call_trie.cpp)
target_include_directories(zonelog PUBLIC .)
target_link_libraries(zonelog PRIVATE fmt::fmt)

add_executable(zonelog_example zonelog_example.cpp)
target_link_libraries(zonelog_example zonelog)
