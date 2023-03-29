
/**
 * @author Masado ishii
 * @date   2023-03-13
 * @brief  Collect references to names in library objects to find test cases.
 */

#include "doctest/extensions/doctest_mpi.h"
#include "include/corner_set.hpp"
#include "include/neighborhood.hpp"
#include "include/neighbor_sets.hpp"
#include "include/neighbors_to_nodes.hpp"

namespace lib_tests
{
  void references()
  {
    link_corner_set_tests();
    link_neighborhood_tests();
    link_neighbor_sets_tests();
    link_neighbors_to_nodes_tests();
  }
}



// Link this file with test_doctests/main.cpp in the CMake config.
