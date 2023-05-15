
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
#include "include/leaf_sets.hpp"
/// #include "include/partition_border.hpp"
#include "include/contextual_hyperface.hpp"
#include "include/da_p2p.hpp"
#include "include/ghost_exchange.hpp"
#include "subdivision_search.hpp"

namespace lib_tests
{
  void references()
  {
    link_corner_set_tests();
    link_neighborhood_tests();
    link_neighbor_sets_tests();
    link_neighbors_to_nodes_tests();
    link_leaf_sets_tests();
    /// link_partition_border_tests();
    link_contextual_hyperface_tests();
    link_da_p2p_tests();
    link_ghost_exchange_tests();
    link_subdivision_search_tests();
  }
}



// Link this file with test_doctests/main.cpp in the CMake config.
