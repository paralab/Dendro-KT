add_library(zonelog zonelog.h zonelog.cpp)
target_sources(zonelog PRIVATE zonelog_call_trie.h)
target_sources(zonelog PRIVATE zonelog_call_trie.cpp)
target_include_directories(zonelog PUBLIC .)
target_link_libraries(zonelog PRIVATE fmt::fmt)

add_executable(zonelog_example zonelog_example.cpp)
target_link_libraries(zonelog_example zonelog)
