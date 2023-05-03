
/**
 * @author Masado Ishii
 * @date   2023-03-28
 * @brief  Make node sets out of neighborhood flags.
 */

#ifndef DENDRO_KT_NEIGHBORS_TO_NODES_HPP
#define DENDRO_KT_NEIGHBORS_TO_NODES_HPP

// Function declaration for linkage purposes.
inline void link_neighbors_to_nodes_tests() {};

#include "include/corner_set.hpp"
#include "include/neighborhood.hpp"
#include "include/neighbor_sets.hpp"
#include "include/sfc_search.h"
#include "include/nested_for.hpp"

#include <vector>

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  // node_set()
  // parent_neighborhood and children_neighborhood are made relative to current.
  template <int dim, typename Policy>
  std::vector<TreeNode<uint32_t, dim>> node_set(
      const std::vector<TreeNode<uint32_t, dim>> &octant_keys,
      const std::vector<Neighborhood<dim>> &neighborhoods,
      const int degree,
      Policy &&policy);

  template <int dim>
  inline Neighborhood<dim> priority_neighbors();

  template <int dim>
  inline Neighborhood<dim> pure_corners();


  // In general the node ownership policy could be a function of the
  // same-level, coarser, and finer neighborhoods, and the SFC ordering.

  // A simplistic policy that uses the same-level and coarser neighborhoods,
  // and emits the nonhanging nodes.

  template <int dim>
  inline
  std::pair< size_t, size_t >   // range begin and end for vertices of this cell
  neighborhood_to_nonhanging(
      const TreeNode<uint32_t, dim> &self_key,
      Neighborhood<dim> self_neighborhood,
      Neighborhood<dim> parent_neighborhood,
      Neighborhood<dim> children_neighborhood,
      const int degree,
      std::vector<TreeNode<uint32_t, dim>> &output )
  {
    const static Neighborhood<dim> priority = priority_neighbors<dim>();

    const size_t range_begin = output.size();

    // Append any nodes who prefer no proper neighbor over the current cell.
    if (self_neighborhood.center_occupied())
    {
      // Policy for now: If a node touches any coarser cell, then it is either
      //       ___       hanging or owned by the coarser cell, so do not emit
      //     _|   |_     from the fine cell. There is no directional priority;
      //   _|_|___|_|_   just filter for neighbors that can touch the vertex.

      //future: Maybe parent should inspect neighbors of children.

      // Parent neighbors are restricted to those visible from this child.
      // Restrict self neighbors to those that this cell may borrow from.

      const Neighborhood<dim> unowned =
          (parent_neighborhood | (self_neighborhood & priority)).spread_out();
      const Neighborhood<dim> owned = ~unowned;

      // Map numerators (node indices) and denominator (degree) to coordinate.
      const auto create_node = [](
          TreeNode<uint32_t, dim> octant, std::array<int, dim> idxs, int degree)
        -> TreeNode<uint32_t, dim>
      {
        periodic::PCoord<uint32_t, dim> node_pt = {};
        const uint32_t side = octant.range().side();
        for (int d = 0; d < dim; ++d)
          node_pt.coord(d, side * idxs[d] / degree);
        node_pt += octant.range().min();
        return TreeNode<uint32_t, dim>(node_pt, octant.getLevel());
      };

      // Emit nodes for all hyperfaces that are owned and nonhanging.
      tmp::nested_for<dim>(0, degree + 1, [&](auto...idx_pack)
      {
        std::array<int, dim> idxs = {idx_pack...};  // 0..degree per axis.
        // Map node index to hyperface index.
        int stride = 1;
        int hyperface_idx = 0;
        for (int d = 0; d < dim; ++d, stride *= 3)
          hyperface_idx += ((idxs[d] > 0) + (idxs[d] == degree)) * stride;
        if(owned.test_flat(hyperface_idx))
        {
          output.push_back(create_node(self_key, idxs, degree));
        }
      });
    }

    const size_t range_end = output.size();

    return {range_begin, range_end};
  }


  // A simplistic policy that uses the same-level, coarser, and finer
  // neighborhoods to emit the nonhanging and hanging nodes.

  template <int dim>
  inline
  std::pair< size_t, size_t >   // range begin and end for vertices of this cell
  neighborhood_to_all_vertices(
      const TreeNode<uint32_t, dim> &self_key,
      Neighborhood<dim> self_neighborhood,
      Neighborhood<dim> parent_neighborhood,
      Neighborhood<dim> children_neighborhood,
      const int degree,
      std::vector<TreeNode<uint32_t, dim>> &output )
  {
    assert(degree == 1);  //this policy only emits vertices

    const size_t range_begin = output.size();

    // Append any vertices who prefer no proper neighbor over the current cell.
    if (self_neighborhood.center_occupied())
    {
      const static Neighborhood<dim> priority = priority_neighbors<dim>();
      const Neighborhood<dim> unowned =
          (parent_neighborhood | (self_neighborhood & priority)).spread_out();
      const Neighborhood<dim> owned = ~unowned;
      const Neighborhood<dim> split = children_neighborhood.spread_out();

      // Map numerators (node indices) and denominator (span) to coordinate.
      const auto create_node = [](
          TreeNode<uint32_t, dim> octant, std::array<int, dim> idxs, int span)
        -> TreeNode<uint32_t, dim>
      {
        periodic::PCoord<uint32_t, dim> node_pt = {};
        const uint32_t side = octant.range().side();
        for (int d = 0; d < dim; ++d)
          node_pt.coord(d, side * idxs[d] / span);
        node_pt += octant.range().min();
        return TreeNode<uint32_t, dim>(node_pt, octant.getLevel());
      };

      // Emit nodes for owned hyperfaces that either are split or are vertices.
      const Neighborhood<dim> emit = owned & (split | pure_corners<dim>());
      tmp::nested_for<dim>(0, 3, [&](auto...idx_pack)
      {
        std::array<int, dim> idxs = {idx_pack...};  // 0..2 per axis.
        // Map node index to hyperface index.
        int stride = 1;
        int hyperface_idx = 0;
        for (int d = 0; d < dim; ++d, stride *= 3)
          hyperface_idx += ((idxs[d] > 0) + (idxs[d] == 2)) * stride;
        if(emit.test_flat(hyperface_idx))
        {
          output.push_back(create_node(self_key, idxs, 2));
        }
      });
    }

    const size_t range_end = output.size();

    return {range_begin, range_end};
  }
}


// =============================================================================
// Tests
// =============================================================================
#ifdef DOCTEST_LIBRARY_INCLUDED
#include "include/tnUtils.h"
namespace ot
{
  DOCTEST_TEST_SUITE("neighbors to nodes")
  {
    DOCTEST_TEST_CASE("Count vertices on uniform 2D grid")
    {
      _InitializeHcurve(2);
      std::vector<TreeNode<uint32_t, 2>> uniform_grid = { TreeNode<uint32_t, 2>() };
      std::vector<TreeNode<uint32_t, 2>> queue;
      const int max_depth = 2;
      for (int level = 1; level <= max_depth; ++level)
      {
        queue.clear();
        for (auto oct: uniform_grid)
          for (int child = 0; child < nchild(2); ++child)
            queue.push_back(oct.getChildMorton(child));
        std::swap(uniform_grid, queue);
      }
      REQUIRE( uniform_grid.size() == (1u << (2 * max_depth)) );
      SFC_Tree<uint32_t, 2>::locTreeSort(uniform_grid);
      REQUIRE( uniform_grid.size() == (1u << (2 * max_depth)) );

      const auto neighbor_sets_pair = neighbor_sets(uniform_grid);
      const std::vector<TreeNode<uint32_t, 2>> &octant_keys = neighbor_sets_pair.keys;
      const std::vector<Neighborhood<2>> &neighborhoods = neighbor_sets_pair.neighborhoods;
      REQUIRE( std::count_if(neighborhoods.begin(), neighborhoods.end(),
                   [](auto nh){return nh.count() == 4;})  ==  4 );
      REQUIRE( std::count_if(neighborhoods.begin(), neighborhoods.end(),
                   [](auto nh){return nh.count() == 6;})  ==  (((1u << max_depth) - 2) * 4)  );
      REQUIRE( std::count_if(neighborhoods.begin(), neighborhoods.end(),
                   [](auto nh){return nh.count() == 9;})  ==
                              (((1u << max_depth) - 2) * ((1u << max_depth) - 2)) );

      const int degree = 1;
      std::vector<TreeNode<uint32_t, 2>> nodes =
          node_set<2>(
              octant_keys, neighborhoods, degree, neighborhood_to_all_vertices<2>);

      const int verts_per_side = (1u << max_depth) + 1;
      CHECK( nodes.size() == (verts_per_side * verts_per_side) );
      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("Count vertices on uniform 3D grid")
    {
      _InitializeHcurve(3);
      std::vector<TreeNode<uint32_t, 3>> uniform_grid = { TreeNode<uint32_t, 3>() };
      std::vector<TreeNode<uint32_t, 3>> queue;
      const int max_depth = 2;
      for (int level = 1; level <= max_depth; ++level)
      {
        queue.clear();
        for (auto oct: uniform_grid)
          for (int child = 0; child < nchild(3); ++child)
            queue.push_back(oct.getChildMorton(child));
        std::swap(uniform_grid, queue);
      }
      REQUIRE( uniform_grid.size() == (1u << (3 * max_depth)) );
      SFC_Tree<uint32_t, 3>::locTreeSort(uniform_grid);
      REQUIRE( uniform_grid.size() == (1u << (3 * max_depth)) );

      const auto neighbor_sets_pair = neighbor_sets(uniform_grid);
      const std::vector<TreeNode<uint32_t, 3>> &octant_keys = neighbor_sets_pair.keys;
      const std::vector<Neighborhood<3>> &neighborhoods = neighbor_sets_pair.neighborhoods;

      const size_t corner_cell_mid = 1;
      const size_t edge_cell_mid   = ((1u << max_depth) - 2);
      const size_t face_cell_mid   = edge_cell_mid * edge_cell_mid;
      const size_t volume_cell_mid = edge_cell_mid * edge_cell_mid * edge_cell_mid;

      REQUIRE( std::count_if(neighborhoods.begin(), neighborhoods.end(),
                   [](auto nh){return nh.count() == 8;})  ==  8 * corner_cell_mid );
      REQUIRE( std::count_if(neighborhoods.begin(), neighborhoods.end(),
                   [](auto nh){return nh.count() == 12;})  ==  12 * edge_cell_mid  );
      REQUIRE( std::count_if(neighborhoods.begin(), neighborhoods.end(),
                   [](auto nh){return nh.count() == 18;})  ==  6 * face_cell_mid );
      REQUIRE( std::count_if(neighborhoods.begin(), neighborhoods.end(),
                   [](auto nh){return nh.count() == 27;})  ==  1 * volume_cell_mid );

      const int degree = 1;
      std::vector<TreeNode<uint32_t, 3>> nodes =
          node_set<3>(
              octant_keys, neighborhoods, degree, neighborhood_to_all_vertices<3>);

      const int verts_per_side = (1u << max_depth) + 1;
      CHECK( nodes.size() == (verts_per_side * verts_per_side * verts_per_side) );
      _DestroyHcurve();
    }


    DOCTEST_TEST_CASE("Parent-neighbor interference")
    {
      _InitializeHcurve(2);

      const auto descendant = [](
          TreeNode<uint32_t, 2> node, std::initializer_list<int> lineage) {
        return morton_lineage(node, lineage);
      };

      // Incomplete tree with gap between coarse and fine cells.
      std::vector<TreeNode<uint32_t, 2>> grid;
      const auto root = TreeNode<uint32_t, 2>();
      grid.push_back(descendant(root, {0, 1}));
      grid.push_back(descendant(root, {1, 1, 0}));
      grid.push_back(descendant(root, {1, 1, 1}));
      grid.push_back(descendant(root, {1, 1, 2}));
      grid.push_back(descendant(root, {1, 1, 3}));

      const auto neighbor_sets_pair = neighbor_sets(grid);
      const std::vector<TreeNode<uint32_t, 2>> &octant_keys = neighbor_sets_pair.keys;
      const std::vector<Neighborhood<2>> &neighborhoods = neighbor_sets_pair.neighborhoods;
      const int degree = 1;
      std::vector<TreeNode<uint32_t, 2>> nodes =
          node_set<2>(
              octant_keys, neighborhoods, degree, neighborhood_to_nonhanging<2>);
      CHECK( nodes.size() == 4 + 9 );

      _DestroyHcurve();
    }


    DOCTEST_TEST_CASE("Count hanging vertices and nonhanging nodes on nonuniform 2D grid")
    {
      //  Case 1 _ _ _ _      Case 2  _ _ _ _
      //        |_|_|_|_|            |+|+|+|+|
      //        |_|+|+|_|            |+|_|_|+|
      //        |_|+|+|_|            |+|_|_|+|
      //        |_|_|_|_|            |+|+|+|+|
      //
      constexpr int dim = 2;

      const auto node_shell = [](int dim, size_t outer_cells, int degree = 1) {
        size_t outer_ball = 1;
        size_t inner_ball = 1;
        while (dim > 0)
        {
          outer_ball *= outer_cells * degree + 1;
          inner_ball *= (outer_cells - 2) * degree + 1;
          --dim;
        }
        return outer_ball - inner_ball;
      };

      const auto inner_node_shell = [node_shell](int dim, size_t outer_cells, int degree) {
        size_t outer_ball = 1;
        size_t inner_ball = 1;
        while (dim > 0)
        {
          outer_ball *= outer_cells * degree - 1;
          inner_ball *= (outer_cells - 2) * degree - 1;
          --dim;
        }
        return outer_ball - inner_ball;
      };

      const auto case_1_vertices = [node_shell](int dim, int max_depth) {
        return 1 + node_shell(dim, 2) + node_shell(dim, 4) * (max_depth - 2 + 1);
      };

      const auto case_1_nonhanging = [node_shell, inner_node_shell](
          int dim, int max_depth, int degree) {
        return 1 + node_shell(dim, 2, degree) - node_shell(dim, 2 * degree, 1)
               +  inner_node_shell(dim, 4, degree) * (max_depth - 2 + 1)
               + node_shell(dim, 4 * degree, 1);
      };

      const auto case_2_vertices = [node_shell](int dim, int max_depth) {
        size_t vertices = 1;
        for (int cells = 4; cells < (1 << max_depth); cells = cells * 2 + 4)
          vertices += node_shell(dim, cells);
        vertices += node_shell(dim, (1 << max_depth) - 2);
        vertices += node_shell(dim, (1 << max_depth));
        return vertices;
      };

      const auto case_2_nonhanging = [node_shell](int dim, int max_depth, int degree) {
        size_t vertices = 1;
        for (int cells = 2; cells < (1 << max_depth); cells = cells * 2 + 2)
          vertices += node_shell(dim, cells, degree);
        vertices += node_shell(dim, (1 << max_depth), degree);
        return vertices;
      };


      _InitializeHcurve(dim);

      for (int max_depth = 2; max_depth <= 5; ++max_depth)
      {
        // Case 1
        {
          // Grid.
          std::vector<TreeNode<uint32_t, dim>> grid = { TreeNode<uint32_t, dim>() };
          std::vector<TreeNode<uint32_t, dim>> queue;
          for (int level = 1; level <= max_depth; ++level)
          {
            queue.clear();
            const auto middle = TreeNode<uint32_t, dim>().getChildMorton(0).range().max();
            for (auto oct: grid)
            {
              // Case 1: Refine the center.
              if (oct.range().closedContains(middle))
                for (int child = 0; child < nchild(dim); ++child)
                  queue.push_back(oct.getChildMorton(child));
              else
                queue.push_back(oct);
            }
            std::swap(grid, queue);
          }

          // Neighbors.
          const auto neighbor_sets_pair = neighbor_sets(grid);
          const std::vector<TreeNode<uint32_t, dim>> &octant_keys = neighbor_sets_pair.keys;
          const std::vector<Neighborhood<dim>> &neighborhoods = neighbor_sets_pair.neighborhoods;

          // Nodes.
          const int degree = 1;
          std::vector<TreeNode<uint32_t, dim>> vertices =
              node_set<dim>(
                  octant_keys, neighborhoods, degree, neighborhood_to_all_vertices<dim>);
          CHECK_MESSAGE( vertices.size() == case_1_vertices(dim, max_depth),
              "dim==", dim, " degree==", degree, "  max_depth==", max_depth);

          for (int degree: {1, 2, 3})
          {
            std::vector<TreeNode<uint32_t, dim>> nodes =
                node_set<dim>(
                    octant_keys, neighborhoods, degree, neighborhood_to_nonhanging<dim>);

            CHECK_MESSAGE( nodes.size() == case_1_nonhanging(dim, max_depth, degree),
                "dim==", dim, " degree==", degree, "  max_depth==", max_depth);
          }
        }

        // Case 2
        {
          // Grid.
          std::vector<TreeNode<uint32_t, dim>> grid = { TreeNode<uint32_t, dim>() };
          std::vector<TreeNode<uint32_t, dim>> queue;
          for (int level = 1; level <= max_depth; ++level)
          {
            queue.clear();
            const uint32_t maximum = TreeNode<uint32_t, dim>().range().side();
            for (auto oct: grid)
            {
              // Case 2: Refine the cube surface.
              const std::array<uint32_t, dim> min = oct.range().min();
              const std::array<uint32_t, dim> max = oct.range().max();
              if (*(std::min_element(min.begin(), min.end())) == 0 or
                  *(std::max_element(max.begin(), max.end())) == maximum)
                for (int child = 0; child < nchild(dim); ++child)
                  queue.push_back(oct.getChildMorton(child));
              else
                queue.push_back(oct);
            }
            std::swap(grid, queue);
          }

          // Neighbors.
          const auto neighbor_sets_pair = neighbor_sets(grid);
          const std::vector<TreeNode<uint32_t, dim>> &octant_keys = neighbor_sets_pair.keys;
          const std::vector<Neighborhood<dim>> &neighborhoods = neighbor_sets_pair.neighborhoods;

          // Nodes.
          const int degree = 1;
          std::vector<TreeNode<uint32_t, dim>> vertices =
              node_set<dim>(
                  octant_keys, neighborhoods, degree, neighborhood_to_all_vertices<dim>);
          CHECK_MESSAGE( vertices.size() == case_2_vertices(dim, max_depth),
              "dim==", dim, " degree==", degree, "  max_depth==", max_depth);

          for (int degree: {1, 2})
          {
            std::vector<TreeNode<uint32_t, dim>> nodes =
                node_set<dim>(
                    octant_keys, neighborhoods, degree, neighborhood_to_nonhanging<dim>);
            CHECK_MESSAGE( nodes.size() == case_2_nonhanging(dim, max_depth, degree),
                "dim==", dim, " degree==", degree, "  max_depth==", max_depth);
          }
        }
      }

      _DestroyHcurve();
    }

  }
}
#endif//DOCTEST_LIBRARY_INCLUDED


// =============================================================================
// Implementation
// =============================================================================

namespace ot
{
  namespace detail
  {
    template <int dim>
    inline std::array<Neighborhood<dim>, dim> neighbors_not_down();

    template <int dim>
    inline std::array<Neighborhood<dim>, dim> neighbors_not_up();
  }

  // ForLeafNeighborhoods()
  // parent_neighborhood and children_neighborhood are made relative to current.
  // self_neighborhood only has center filled if that is the case in dict.
  template <int dim>
  class ForLeafNeighborhoods
  {
    public:
      inline ForLeafNeighborhoods(
        const NeighborSetDict<dim> &dict,
        const std::vector<TreeNode<uint32_t, dim>> &leaf_queries);

      struct Env;
      inline Env operator*() const;
      inline ForLeafNeighborhoods & operator++();

      struct End { };
      struct MaybeEnd
      {
        ForLeafNeighborhoods &ref;
        bool end;
        Env operator*() const { return *ref; }
        MaybeEnd & operator++() { ++ref; return *this; }
        bool operator!=(const MaybeEnd &end) const { return not end.end or ref != End(); }
        size_t query_idx() const { return ref.query_idx; }
      };
      inline MaybeEnd begin() { return {*this, false}; }
      inline MaybeEnd end()   { return {*this, true}; }

      inline bool operator!=(End end) const;

    private:
      static const std::array<std::array<Neighborhood<dim>, dim>, 2> & directions();
      static Neighborhood<dim> combine_on_corner(int corner);

    private:
      const NeighborSetDict<dim> &dict;
      const std::vector<TreeNode<uint32_t, dim>> &leaf_queries;

      mutable std::vector<Neighborhood<dim>>                neighborhoods_by_level;
      mutable std::vector<periodic::PCoord<uint32_t, dim>>  parents_by_level;

      size_t dict_size = 0;
      size_t n_queries = 0;

      mutable size_t entry_idx = 0;
      size_t entries_end = 0;
      size_t query_idx = 0;
  };


  template <int dim>
  const std::array<std::array<Neighborhood<dim>, dim>, 2> &
    ForLeafNeighborhoods<dim>::directions() 
  {
    const static std::array<std::array<Neighborhood<dim>, dim>, 2>
        directions = { detail::neighbors_not_up<dim>(),
                       detail::neighbors_not_down<dim>() };
    return directions;
  }

  template <int dim>
  Neighborhood<dim> ForLeafNeighborhoods<dim>::combine_on_corner(int corner)
  {
    //                            1 1 1     0 1 1      0 1 1
    // NorthEast = North & East = 1 1 1  &  0 1 1  ==  0 1 1
    //                            0 0 0     0 1 1      0 0 0
    auto neighborhood = Neighborhood<dim>::full();
    for (int axis = 0; axis < dim; ++axis)
      neighborhood &= directions()[((corner >> axis) & 1)][axis];
    return neighborhood;
  };

  template <int dim>
  ForLeafNeighborhoods<dim>::ForLeafNeighborhoods(
      const NeighborSetDict<dim> &dict,
      const std::vector<TreeNode<uint32_t, dim>> &leaf_queries)
  : dict(dict), leaf_queries(leaf_queries),
    neighborhoods_by_level(m_uiMaxDepth + 1),
    parents_by_level(m_uiMaxDepth + 1)
  {
    // The dictionary contains linearized multi-level data, where parent and
    // child entries can be far separated. Because of this, we must visit the
    // parent of a query key before emitting. In theory, two binary searches
    // per query would be enough (one search for the query key and one search
    // for the parent). But, if we expect the queries to be dense in a sublist
    // of the dictionary, it's better to make a single pass over the sublist.

    this->dict_size = dict.keys.size();
    this->n_queries = leaf_queries.size();
    if (n_queries == 0)
      return;

    // Find range in which queries can be filled.
    const size_t entries_begin = sfc_binary_search<uint32_t, dim>(
        leaf_queries.front(), dict.keys.data(),
        0, dict_size, RankType::exclusive);
    this->entries_end = sfc_binary_search<uint32_t, dim>(
        leaf_queries.back(), dict.keys.data(),
        entries_begin, dict_size, RankType::inclusive);

    // Linear search up to entries_begin. This initializes the parent stack.
    this->entry_idx = 0;
    const TreeNode<uint32_t, dim> initial_query_key = leaf_queries[0];
    for (entry_idx = 0; entry_idx < entries_begin; ++entry_idx)
    {
      const auto entry_key = dict.keys[entry_idx];
      if (entry_key.isAncestor(initial_query_key))
      {
        const int level = entry_key.getLevel();
        parents_by_level[level] = entry_key.coords();
        neighborhoods_by_level[level] = dict.neighborhoods[entry_idx];
      }
    }

    this->query_idx = 0;
  }

  template <int dim>
  bool ForLeafNeighborhoods<dim>::operator!=(ForLeafNeighborhoods<dim>::End) const
  {
    return this->query_idx < this->n_queries;
  }

  template <int dim>
  ForLeafNeighborhoods<dim> & ForLeafNeighborhoods<dim>::operator++()
  {
    ++this->query_idx;
    return *this;
  }

  template <int dim>
  struct ForLeafNeighborhoods<dim>::Env
  {
    size_t                   query_idx;
    TreeNode<uint32_t, dim>  query_key;
    Neighborhood<dim>        self_neighborhood;
    Neighborhood<dim>        parent_neighborhood;
    Neighborhood<dim>        children_neighborhood;
  };

  // For each query, linear search until reach rank of query_key in entries.
  template <int dim>
  typename ForLeafNeighborhoods<dim>::Env ForLeafNeighborhoods<dim>::operator*() const
  {
    const size_t query_idx = this->query_idx;
    const TreeNode<uint32_t, dim> query_key = leaf_queries[query_idx];

    // Linear search, updating parent stack.
    while (entry_idx < entries_end
        and sfc_compare<uint32_t, dim>(
          dict.keys[entry_idx], query_key, {}).first < 0)
    {
      const TreeNode<uint32_t, dim> entry_key = dict.keys[entry_idx];
      if (entry_key.isAncestor(query_key))
      {
        const int level = entry_key.getLevel();
        parents_by_level[level] = entry_key.coords();
        neighborhoods_by_level[level] = dict.neighborhoods[entry_idx];
      }
      ++entry_idx;
    }

    // At this point, the current entry is either
    // - equal to query_key .......... fill self and children neighborhoods
    // - a child of query_key ........ fill just children neighborhood
    // - unrelated to query_key ...... no further fill
    // - nonexistent (past the end) .. no further fill

    const int query_level = query_key.getLevel();
    const int query_corner = query_key.getMortonIndex();

    // Parent neighborhood
    auto parent_neighborhood = Neighborhood<dim>::empty();
    const TreeNode<uint32_t, dim>
        maybe_parent(parents_by_level[query_level - 1], query_level - 1);
    if (maybe_parent.isAncestor(query_key))
    {
      parent_neighborhood =
          neighborhoods_by_level[query_level - 1]
          & combine_on_corner(query_corner);
    }

    // Self neighborhood
    auto self_neighborhood = Neighborhood<dim>::empty();
    if (entry_idx < entries_end and dict.keys[entry_idx] == query_key)
    {
      self_neighborhood = dict.neighborhoods[entry_idx];
      ++entry_idx;
    }

    // Children neighborhood
    auto children_neighborhood = Neighborhood<dim>::empty();
    const size_t child_limit_idx = std::min(dict_size, entry_idx + nchild(dim));
    while (entry_idx < child_limit_idx
        and query_key.isAncestor(dict.keys[entry_idx]))
    {
      const auto child_key = dict.keys[entry_idx];
      const auto child_neighborhood = dict.neighborhoods[entry_idx];
      assert(child_key.getLevel() == query_key.getLevel() + 1);
      assert(not child_neighborhood.center_occupied());
      children_neighborhood |=
          child_neighborhood & combine_on_corner(child_key.getMortonIndex());
      ++entry_idx;
    }

    return {query_idx, query_key,
        self_neighborhood, parent_neighborhood, children_neighborhood};
  }





  // node_set()
  // parent_neighborhood and children_neighborhood are made relative to current.
  template <int dim, typename Policy>
  std::vector<TreeNode<uint32_t, dim>> node_set(
      const std::vector<TreeNode<uint32_t, dim>> &octant_keys,
      const std::vector<Neighborhood<dim>> &neighborhoods,
      const int degree,
      Policy &&policy)
  {
    const static std::array<Neighborhood<dim>, dim>
        directions[2] = { detail::neighbors_not_up<dim>(),
                          detail::neighbors_not_down<dim>() };
    const auto combine_on_corner = [&](int corner) -> Neighborhood<dim>
    {
      //                            1 1 1     0 1 1      0 1 1
      // NorthEast = North & East = 1 1 1  &  0 1 1  ==  0 1 1
      //                            0 0 0     0 1 1      0 0 0
      auto neighborhood = Neighborhood<dim>::full();
      for (int axis = 0; axis < dim; ++axis)
        neighborhood &= directions[((corner >> axis) & 1)][axis];
      return neighborhood;
    };

    using Coordinate = periodic::PCoord<uint32_t, dim>;
    std::vector<Coordinate>        parents_by_level(m_uiMaxDepth + 1);
    std::vector<Neighborhood<dim>> neighborhoods_by_level(m_uiMaxDepth + 1);

    std::vector<TreeNode<uint32_t, dim>> nodes;

    for (size_t i = 0, end = octant_keys.size(); i < end; ++i)
    {
      const auto key = octant_keys[i];
      const int key_level = key.getLevel();
      const int child_number = key.getMortonIndex();

      const auto self_neighborhood = neighborhoods[i];

      // Examine stack of parents.
      auto parent_neighborhood = Neighborhood<dim>::empty();
      if (TreeNode<uint32_t, dim>(
            parents_by_level[key_level - 1], key_level - 1).isAncestor(key))
      {
        parent_neighborhood = neighborhoods_by_level[key_level - 1];
        parent_neighborhood &= combine_on_corner(child_number);
      }

      // Look ahead (by no more than nchild(dim)) for any neighbors of children.
      auto children_neighborhood = Neighborhood<dim>::empty();
      for (size_t j = i + 1; j < end; ++j)
      {
        const auto child_key = octant_keys[j];
        if (not key.isAncestor(child_key))
          break;
        if (child_key.getLevel() > key.getLevel() + 1)
          break;
        const auto child_neighborhood = neighborhoods[j];

        if (self_neighborhood.center_occupied())
        {
          assert(child_key.getLevel() == key.getLevel() + 1);
          assert(not child_neighborhood.center_occupied());
        }

        children_neighborhood |=
            child_neighborhood & combine_on_corner(child_key.getMortonIndex());
      }

      policy(key, self_neighborhood, parent_neighborhood, children_neighborhood, degree, nodes);

      if (i + 1 < end and key.isAncestor(octant_keys[i + 1]))
      {
        parents_by_level[key_level] = key.coords();
        neighborhoods_by_level[key_level] = self_neighborhood;
      }
    }

    return nodes;
  }

  namespace detail
  {

    template <int dim>
    inline std::array<Neighborhood<dim>, dim> neighbors_not_down()
    {
      std::array<Neighborhood<dim>, dim> neighborhoods = {};
      for (int axis = 0; axis < dim; ++axis)
        neighborhoods[axis] = Neighborhood<dim>::not_down(axis);
      return neighborhoods;
    }

    template <int dim>
    inline std::array<Neighborhood<dim>, dim> neighbors_not_up()
    {
      std::array<Neighborhood<dim>, dim> neighborhoods = {};
      for (int axis = 0; axis < dim; ++axis)
        neighborhoods[axis] = Neighborhood<dim>::not_up(axis);
      return neighborhoods;
    }
  }

  template <int dim>
  inline Neighborhood<dim> priority_neighbors()
  {
    // Prioritization: Lexicographic predecessors.
    constexpr int N = Neighborhood<dim>::n_neighbors();
    return Neighborhood<dim>::where([N](int i) { return i < N/2; });
    //future: lexicographic successors
  }

  template <int dim>
  inline Neighborhood<dim> pure_corners()
  {
    // {0, 2}^dim, since 1 is for sides.
    Neighborhood<dim> result = Neighborhood<dim>::full();
    for (int axis = 0; axis < dim; ++axis)
      result &= ~Neighborhood<dim>::center_slab_mask(axis);
    return result;
  }
}



#endif//DENDRO_KT_NEIGHBORS_TO_NODES_HPP
