
/**
 * @author Masado Ishii
 * @date   2023-03-27
 * @brief  Build set of same-level neighbors for octants (+ parents + children)
 */

#ifndef DENDRO_KT_NEIGHBOR_SETS_HPP
#define DENDRO_KT_NEIGHBOR_SETS_HPP

// Function declaration for linkage purposes.
inline void link_neighbor_sets_tests() {};

#include "doctest/doctest.h"

#include "include/tsort.h"
#include "include/treeNode.h"
#include "include/neighborhood.hpp"

#include <iostream>


// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  template <unsigned dim>
  std::pair< std::vector<TreeNode<uint32_t, dim>>,
             std::vector<Neighborhood<dim>> >
  neighbor_sets(const std::vector<TreeNode<uint32_t, dim>> &leaf_set);

  DOCTEST_TEST_SUITE("Neighbor sets")
  {
    DOCTEST_TEST_CASE("Empty set mapped to empty set")
    {
      _InitializeHcurve(3);
      auto result = neighbor_sets<3>({});
      CHECK( result.first.empty() );
      CHECK( result.second.empty() );
      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("Singleton set mapped to one occupied key and 3^dim-1 neighbors")
    {
      _InitializeHcurve(3);
      const auto some_octant = TreeNode<uint32_t, 3>().getChildMorton(0)
                                                      .getChildMorton(7)
                                                      .getChildMorton(7)
                                                      .getChildMorton(7);
      auto result = neighbor_sets<3>({some_octant});
      CHECK( result.first.size() == result.second.size() );
      CHECK( result.second.size() == 27 );
      CHECK( std::count_if(result.second.begin(), result.second.end(),
            [](auto nbhd) { return nbhd.center_occupied(); })
          == 1 );
      for (auto nbhd: result.second)
      {
        CHECK( nbhd.any() );
      }
      _DestroyHcurve();
    }


    DOCTEST_TEST_CASE("Singleton set on non-periodic corner mapped to one occupied key and 2^dim-1 neighbors")
    {
      _InitializeHcurve(3);
      for (int extreme : {0, 7})
      {
        const auto some_octant = TreeNode<uint32_t, 3>().getChildMorton(extreme)
                                                        .getChildMorton(extreme);
        auto result = neighbor_sets<3>({some_octant});
        CHECK( result.first.size() == result.second.size() );
        CHECK( result.second.size() == 8 );
        CHECK( std::count_if(result.second.begin(), result.second.end(),
              [](auto nbhd) { return nbhd.center_occupied(); })
            == 1 );
        for (auto nbhd: result.second)
        {
          CHECK( nbhd.any() );
        }
      }
      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("Singleton set on periodic corner mapped to one occupied key and 3^dim-1 neighbors")
    {
      _InitializeHcurve(3);
      for (int axis = 0; axis < 3; ++axis)
        periodic::PCoord<uint32_t, 3>::period(axis, (1u << m_uiMaxDepth));

      const auto some_octant = TreeNode<uint32_t, 3>().getChildMorton(0)
                                                      .getChildMorton(0)
                                                      .getChildMorton(0)
                                                      .getChildMorton(0);
      auto result = neighbor_sets<3>({some_octant});
      CHECK( result.first.size() == result.second.size() );
      CHECK( result.second.size() == 27 );
      CHECK( std::count_if(result.second.begin(), result.second.end(),
            [](auto nbhd) { return nbhd.center_occupied(); })
          == 1 );
      for (auto nbhd: result.second)
      {
        CHECK( nbhd.any() );
      }
      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("Multiple levels overlap")
    {
      _InitializeHcurve(3);
      const auto some_parent = TreeNode<uint32_t, 3>().getChildMorton(0)
                                                      .getChildMorton(7)
                                                      .getChildMorton(7)
                                                      .getChildMorton(7);
      const auto octantA = some_parent.getChildMorton(0);
      const auto octantB = some_parent.getChildMorton(1);

      std::vector<TreeNode<uint32_t, 3>> octlist;
      octlist.push_back(octantA);
      for (int c = 0; c < 8; ++c)
        octlist.push_back(octantB.getChildMorton(c));

      auto result = neighbor_sets<3>(octlist);
      CHECK( result.second.size() == (27 + 64) );

      CHECK( std::count_if(result.second.begin(), result.second.end(),
            [](auto nbhd) { return nbhd.center_occupied(); })
          == 9 );

      _DestroyHcurve();
    }

    //nextpath:
    // - [ ] count nodes (using local implementation).
    // - [ ] generate nodes in a single pass.
    // - [ ] test: do an input matvec on an adaptive grid, ensuring all nonzero.
    // - [ ] distributed version.
    //   - [ ] find boundary octants
    //   - [ ] compute scattermap
    //   - [ ] compute gathermap
  }


}



// =============================================================================
// Implementation
// =============================================================================

namespace ot
{
  //future: filter out the keys that do not occupy their own neighborhood,
  //        unless a parent or child does.
  template <unsigned dim>
  std::pair< std::vector<TreeNode<uint32_t, dim>>,
             std::vector<Neighborhood<dim>> >
  neighbor_sets(const std::vector<TreeNode<uint32_t, dim>> &leaf_set)
  {
    std::vector<TreeNode<uint32_t, dim>> keys;
    std::vector<Neighborhood<dim>> neighbor_sets;

    keys.reserve(leaf_set.size());
    neighbor_sets.reserve(leaf_set.size());

    // Initialize with solitary neighborhood for each leaf.
    const auto self_set = Neighborhood<dim>::solitary();
    for (const auto leaf: leaf_set)
    {
      keys.push_back(leaf);
      neighbor_sets.push_back(self_set);
    }

    // For each axis, inform face-neighbors of partial neighborhoods.
    // This is a map-reduce by sorting on octant keys. Accumulate each pass.
    std::vector<TreeNode<uint32_t, dim>> new_keys;
    std::vector<Neighborhood<dim>> new_values;
    for (int axis = 0; axis < dim; ++axis)
    {
      new_keys.clear();
      new_values.clear();
      new_keys.reserve(3 * keys.size());
      new_values.reserve(3 * keys.size());

      // Map (multiple outputs)
      const size_t n_keys = keys.size();
      for (size_t i = 0; i < n_keys; ++i)
      {
        const auto key = keys[i];
        const auto neighborhood = neighbor_sets[i];

        // Identity
        new_keys.push_back(key);
        new_values.push_back(neighborhood);

        const uint32_t cell_size = key.range().side();

        // Send up: Current key's neighborhood as viewed by upward neighbor.
        auto key_up = key;
        key_up.setX(axis, key.getX(axis) + cell_size);
        if (TreeNode<uint32_t, dim>().isAncestorInclusive(key_up))
        {
          new_keys.push_back(key_up);
          new_values.push_back(neighborhood.shifted_down(axis));
        }

        // Send down: Current key's neighborhood as viewed by downward neighbor.
        auto key_down = key;
        key_down.setX(axis, key.getX(axis) - cell_size);
        if (TreeNode<uint32_t, dim>().isAncestorInclusive(key_down))
        {
          new_keys.push_back(key_down);
          new_values.push_back(neighborhood.shifted_up(axis));
        }
      }

      // Reduce (segment reduction)
      SFC_Tree<uint32_t, dim>::locTreeSort(new_keys, new_values);

      // For each set of equal keys, reduce the corresponding set of neighborhoods.
      // Note that parents and children are handled separately.
      // The final key set likely includes ancestors that overlaps leafs.

      size_t n_written = 0;
      for (size_t i = 0; i < new_keys.size(); ++i)
      {
        const Neighborhood<dim> value = new_values[i];
        const bool equals_prev = i > 0 and new_keys[i - 1] == new_keys[i];
        if (equals_prev)
        {
          new_values[n_written - 1] |= value;
        }
        else
        {
          new_values[n_written] = value;
          ++n_written;
        }
      }
      new_values.erase(new_values.begin() + n_written, new_values.end());

      // For each set of equal keys, keep one of them.
      new_keys.erase(std::unique(new_keys.begin(), new_keys.end()), new_keys.end());

      std::swap(keys, new_keys);
      std::swap(neighbor_sets, new_values);
    }
    new_keys.clear();
    new_values.clear();

    return {keys, neighbor_sets};
  }


}



#endif//DENDRO_KT_NEIGHBOR_SETS_HPP
