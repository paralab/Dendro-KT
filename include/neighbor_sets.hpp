
/**
 * @author Masado Ishii
 * @date   2023-03-27
 * @brief  Build set of same-level neighbors for octants (+ parents + children)
 */

//future: Rename to share_neighbors.hpp, share_neighbors()

#ifndef DENDRO_KT_NEIGHBOR_SETS_HPP
#define DENDRO_KT_NEIGHBOR_SETS_HPP

// Function declaration for linkage purposes.
inline void link_neighbor_sets_tests() {};

#include "include/tsort.h"
#include "include/treeNode.h"
#include "include/neighborhood.hpp"

#include <iostream>


// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  template <int dim>
  struct NeighborSetDict
  {
    std::vector<TreeNode<uint32_t, dim>> keys;
    std::vector<Neighborhood<dim>> neighborhoods;

    inline void concat(const NeighborSetDict &other);
    inline void reduce();
  };

  /**
   * neighbor_sets()
   * @brief Create a sorted dictionary from octants to (3^dim) neighbors.
   * @param leaf_set: Leafs cannot overlap unless exact duplicates. Unsorted ok.
   */
  template <unsigned dim>
  NeighborSetDict<dim> neighbor_sets(
      const std::vector<TreeNode<uint32_t, dim>> &leaf_set);


  /**
   * filter_occupied()
   * @brief Keep entries relevant to center_occupied()-entries (parents + kids).
   * @param dict The dictionary to modify.
   * @param begin The ith relevant entry will be retained only if i >= begin.
   * @param end The ith relevant entry will be retained only if i < end.
   */
  template <int dim>
  void filter_occupied(NeighborSetDict<dim> &dict,
      size_t begin = 0, size_t end = -(size_t)1);


#ifdef DOCTEST_LIBRARY_INCLUDED
  DOCTEST_TEST_SUITE("Neighbor sets")
  {
    DOCTEST_TEST_CASE("Empty set mapped to empty set")
    {
      _InitializeHcurve(3);
      auto result = neighbor_sets<3>({});
      CHECK( result.keys.empty() );
      CHECK( result.neighborhoods.empty() );
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
      CHECK( result.keys.size() == result.neighborhoods.size() );
      CHECK( result.neighborhoods.size() == 27 );
      CHECK( std::count_if(result.neighborhoods.begin(), result.neighborhoods.end(),
            [](auto nbhd) { return nbhd.center_occupied(); })
          == 1 );
      for (auto nbhd: result.neighborhoods)
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
        CHECK( result.keys.size() == result.neighborhoods.size() );
        CHECK( result.neighborhoods.size() == 8 );
        CHECK( std::count_if(result.neighborhoods.begin(), result.neighborhoods.end(),
              [](auto nbhd) { return nbhd.center_occupied(); })
            == 1 );
        for (auto nbhd: result.neighborhoods)
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
      CHECK( result.keys.size() == result.neighborhoods.size() );
      CHECK( result.neighborhoods.size() == 27 );
      CHECK( std::count_if(result.neighborhoods.begin(), result.neighborhoods.end(),
            [](auto nbhd) { return nbhd.center_occupied(); })
          == 1 );
      for (auto nbhd: result.neighborhoods)
      {
        CHECK( nbhd.any() );
      }
      periodic::PCoord<uint32_t, 3>::reset_periods();
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
      CHECK( result.neighborhoods.size() == (27 + 64) );

      CHECK( std::count_if(result.neighborhoods.begin(), result.neighborhoods.end(),
            [](auto nbhd) { return nbhd.center_occupied(); })
          == 9 );

      _DestroyHcurve();
    }

    //nextpath:
    // - [x] count nodes (using local implementation).
    // - [x] generate nodes in a single pass.
    // - [ ] test: do an input matvec on an adaptive grid, ensuring all nonzero.
    // - [ ] distributed version.
    //   - [x] find boundary octants
    //   - [ ] compute scattermap
    //   - [ ] compute gathermap
  }
#endif//DOCTEST_LIBRARY_INCLUDED


}



// =============================================================================
// Implementation
// =============================================================================

namespace ot
{
  namespace detail
  {
    template <int dim>
    void translate(NeighborSetDict<dim> &in, int axis);

    template <int dim>
    void reduce(NeighborSetDict<dim> &dict);
  }


  // NeighborSetDict::concat()
  template <int dim>
  inline void NeighborSetDict<dim>::concat(const NeighborSetDict &other)
  {
    this->keys.insert( this->keys.begin(),
        other.keys.cbegin(), other.keys.cend() );

    this->neighborhoods.insert( this->neighborhoods.begin(),
        other.neighborhoods.cbegin(), other.neighborhoods.cbegin() );
  }

  // NeighborSetDict::reduce()
  template <int dim>
  inline void NeighborSetDict<dim>::reduce()
  {
    detail::reduce(*this);
  }



  // neighbor_sets()
  template <unsigned dim>
  NeighborSetDict<dim> neighbor_sets(
      const std::vector<TreeNode<uint32_t, dim>> &leaf_set)
  {
    NeighborSetDict<dim> dict;

    dict.keys.reserve(leaf_set.size());
    dict.neighborhoods.reserve(leaf_set.size());

    // Initialize with solitary neighborhood for each leaf.
    const auto self_set = Neighborhood<dim>::solitary();
    for (const auto leaf: leaf_set)
    {
      dict.keys.push_back(leaf);
      dict.neighborhoods.push_back(self_set);
    }

    // For each axis, inform face-neighbors of partial neighborhoods.
    // This is a map-reduce by sorting on octant keys. Accumulate each pass.
    for (int axis = 0; axis < dim; ++axis)
    {
      detail::translate<dim>(dict, axis);
      detail::reduce<dim>(dict);
    }

    return dict;
  }

  namespace detail
  {
    // translate()
    template <int dim>
    void translate(NeighborSetDict<dim> &in, int axis)
    {
      // future: memory pools
      static NeighborSetDict<dim> next;

      next.keys.clear();
      next.neighborhoods.clear();
      next.keys.reserve(3 * in.keys.size());
      next.neighborhoods.reserve(3 * in.keys.size());

      // Map (multiple outputs)
      const size_t n_keys = in.keys.size();
      for (size_t i = 0; i < n_keys; ++i)
      {
        const auto key = in.keys[i];
        const auto neighborhood = in.neighborhoods[i];

        // Identity
        next.keys.push_back(key);
        next.neighborhoods.push_back(neighborhood);

        const uint32_t cell_size = key.range().side();

        // Send up: Current key's neighborhood as viewed by upward neighbor.
        auto key_up = key;
        key_up.setX(axis, key.getX(axis) + cell_size);
        if (TreeNode<uint32_t, dim>().isAncestorInclusive(key_up))
        {
          next.keys.push_back(key_up);
          next.neighborhoods.push_back(neighborhood.shifted_down(axis));
        }

        // Send down: Current key's neighborhood as viewed by downward neighbor.
        auto key_down = key;
        key_down.setX(axis, key.getX(axis) - cell_size);
        if (TreeNode<uint32_t, dim>().isAncestorInclusive(key_down))
        {
          next.keys.push_back(key_down);
          next.neighborhoods.push_back(neighborhood.shifted_up(axis));
        }
      }

      std::swap(next, in);
    }

    template <int dim>
    void reduce(NeighborSetDict<dim> &dict)
    {
      // Reduce (segment reduction)
      SFC_Tree<uint32_t, dim>::locTreeSort(dict.keys, dict.neighborhoods);

      // For each set of equal keys, reduce the corresponding set of neighborhoods.
      // Note that parents and children are handled separately.
      // The final key set likely includes ancestors that overlaps leafs.

      size_t n_written = 0;
      const size_t input_size = dict.keys.size();
      for (size_t i = 0; i < input_size; ++i)
      {
        const Neighborhood<dim> value = dict.neighborhoods[i];
        const bool equals_prev = i > 0 and dict.keys[i - 1] == dict.keys[i];
        if (equals_prev)
        {
          dict.neighborhoods[n_written - 1] |= value;
        }
        else
        {
          dict.neighborhoods[n_written] = value;
          ++n_written;
        }
      }
      dict.neighborhoods.erase(
          dict.neighborhoods.begin() + n_written,
          dict.neighborhoods.end());

      // For each set of equal keys, keep one of them.
      dict.keys.erase(
          std::unique(dict.keys.begin(), dict.keys.end()),
          dict.keys.end());
    }
  }


  // filter_occupied()
  template <int dim>
  void filter_occupied(NeighborSetDict<dim> &dict, size_t slice_begin, size_t slice_end)
  {
    using Coordinate = periodic::PCoord<uint32_t, dim>;
    std::vector<Coordinate>        parents_by_level(m_uiMaxDepth + 1);
    std::vector<uint8_t>           parents_written(m_uiMaxDepth + 1, false);
    std::vector<Neighborhood<dim>> neighborhoods_by_level(m_uiMaxDepth + 1);

    // This streaming algorithm does not look ahead (by more than one entry),
    // and any look-back is stored on the above stacks.

    const size_t input_size = dict.keys.size();
    size_t output_size = 0;
    size_t count_occupied = 0;
    for (size_t i = 0, end = input_size; i < end; ++i)
    {
      const auto key = dict.keys[i];
      const int key_level = key.getLevel();
      const int child_number = key.getMortonIndex();

      const auto self_neighborhood = dict.neighborhoods[i];

      auto parent_neighborhood = Neighborhood<dim>::empty();
      if (TreeNode<uint32_t, dim>(
            parents_by_level[key_level - 1], key_level - 1).isAncestor(key))
        parent_neighborhood = neighborhoods_by_level[key_level - 1];

      const bool self_occupied = self_neighborhood.center_occupied();
      const bool parent_occupied = parent_neighborhood.center_occupied();
      bool emitted_self = false;

      const auto emit = [&dict, &output_size](
          const TreeNode<uint32_t, dim> &t, const Neighborhood<dim> &n)
      {
        dict.keys[output_size] = t;
        dict.neighborhoods[output_size] = n;
        ++output_size;
      };

      // Modify dict
      assert(output_size <= i);

      // Emit parent if it is purely a parent of a self-occupied entry,
      // and this is its first self-occupied child.
      if (self_occupied
          and (slice_begin <= count_occupied and count_occupied < slice_end)
          and not parent_occupied and parent_neighborhood.any()
          and not parents_written[key_level - 1])
      {
        emit(key.getParent(), parent_neighborhood);
        parents_written[key_level - 1] = true;
      }

      // Emit self if parent is occupied or self is occupied.
      if ((self_occupied
            and (slice_begin <= count_occupied and count_occupied < slice_end))
          or (parent_occupied and parents_written[key_level - 1]))
      {
        emit(key, self_neighborhood);
        emitted_self = true;
      }

      if (self_occupied)
        ++count_occupied;

      if (i + 1 < end and key.isAncestor(dict.keys[i + 1]))
      {
        parents_by_level[key_level] = key.coords();
        neighborhoods_by_level[key_level] = self_neighborhood;
        parents_written[key_level] = emitted_self;
      }
    }

    dict.keys.erase(
        dict.keys.begin() + output_size, dict.keys.end() );

    dict.neighborhoods.erase(
        dict.neighborhoods.begin() + output_size, dict.neighborhoods.end() );
  }

}



#endif//DENDRO_KT_NEIGHBOR_SETS_HPP
