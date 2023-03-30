
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
#include "include/nested_for.hpp"

#include "doctest/doctest.h"

#include <vector>

// debug
#include "include/octUtils.h"

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  // node_set()
  template <int dim, typename Policy>
  std::vector<TreeNode<uint32_t, dim>> node_set(
      const std::vector<TreeNode<uint32_t, dim>> &octant_keys,
      const std::vector<Neighborhood<dim>> &neighborhoods,
      const int degree,
      Policy &&policy);


  namespace detail
  {
    template <int dim>
    inline std::array<Neighborhood<dim>, dim> neighbors_not_down();

    template <int dim>
    inline std::array<Neighborhood<dim>, dim> neighbors_not_up();
  }

  template <int dim>
  inline Neighborhood<dim> priority_neighbors();

  template <int dim>
  inline std::array<Neighborhood<dim>, nverts(dim)> corner_neighbors();

  template <int dim>
  inline std::array<Neighborhood<dim>, nverts(dim)> vertex_preferred_neighbors();

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
      const int child_number,
      const int degree,
      std::vector<TreeNode<uint32_t, dim>> &output )
  {
    const static std::array<Neighborhood<dim>, dim>
        directions[2] = { detail::neighbors_not_up<dim>(),
                          detail::neighbors_not_down<dim>() };
    const static Neighborhood<dim>
        priority = priority_neighbors<dim>();

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

    const auto combine_on_facet = [&](auto...facet_idxs) -> Neighborhood<dim>
    {
      //                               1 1 1     0 1 1     1 1 0      0 1 0
      // TrueNorth = North & (E & W) = 1 1 1  &  0 1 1  &  1 1 0  ==  0 1 0
      //                               0 0 0     0 1 1     1 1 0      0 0 0
      const std::array<int, dim> idxs = {facet_idxs...};
      auto neighborhood = Neighborhood<dim>::full();
      for (int axis = 0; axis < dim; ++axis)
      {
        if (idxs[axis] < 2)
          neighborhood &= directions[0][axis];
        if (idxs[axis] > 0)
          neighborhood &= directions[1][axis];
      }
      return neighborhood;
    };

    /// const static std::array<Neighborhood<dim>, nverts(dim)>
    ///   preferred_neighbors = vertex_preferred_neighbors<dim>();
    ///   // Ideally constexpr, but Neighborhood uses std::bitset.
    /// const static std::array<Neighborhood<dim>, nverts(dim)>
    ///   corner_relevant = corner_neighbors<dim>();

    const size_t range_begin = output.size();

    // Append any nodes who prefer no proper neighbor over the current cell.
    if (self_neighborhood.center_occupied())
    {
      // Policy for now: If a node touches any coarser cell, then it is either
      //       ___       hanging or owned by the coarser cell, so do not emit
      //     _|   |_     from the fine cell. There is no directional priority;
      //   _|_|___|_|_   just filter for neighbors that can touch the vertex.

      //future: Maybe parent should inspect neighbors of children.

      // Restrict parent neighbors to those visible from this child.
      // Restrict self neighbors to those that this cell may borrow from.
      const Neighborhood<dim> greedy_neighbors =
          (parent_neighborhood & combine_on_corner(child_number))
          | (self_neighborhood & priority);

      // Classify facets.
      Neighborhood<dim> facets_nonhanging_owned;
      int facet_idx = 0;
      tmp::nested_for<dim>(0, 3, [&](auto...idx_pack)
      {
        if ((greedy_neighbors & combine_on_facet(idx_pack...)).none())
          facets_nonhanging_owned.set_flat(facet_idx);
        ++facet_idx;
      });
 
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

      // Emit nodes for all facets that are owned and nonhanging.
      tmp::nested_for<dim>(0, degree + 1, [&](auto...idx_pack)
      {
        std::array<int, dim> idxs = {idx_pack...};  // 0..degree per axis.
        // Map node index to facet index.
        int stride = 1;
        int facet_idx = 0;
        for (int d = 0; d < dim; ++d, stride *= 3)
          facet_idx += ((idxs[d] > 0) + (idxs[d] == degree)) * stride;
        if(facets_nonhanging_owned.test_flat(facet_idx))
        {
          output.push_back(create_node(self_key, idxs, degree));
        }
      });

      /// /// CornerSet<dim> vertices_hanging;
      /// /// CornerSet<dim> vertices_borrowed;

      /// // Combine same-level directional priority with coarseness-priority.
      /// for (int vertex = 0; vertex < nverts(dim); ++vertex)
      /// {
      ///   bool hanging = (parent_neighborhood & corner_relevant[vertex]).any();
      ///   bool borrowed = (self_neighborhood & preferred_neighbors[vertex]).any();
      ///   /// if (hanging)
      ///   ///   vertices_hanging.set_flat(vertex);
      ///   /// if (borrowed)
      ///   ///   vertices_borrowed.set_flat(vertex);

      ///   if (degree == 1 and ((not hanging) & (not borrowed)))
      ///   {
      ///     auto vertex_pt = self_key.range().min();
      ///     for (int d = 0; d < dim; ++d)
      ///       if (bool(vertex & (1 << d)))
      ///         vertex_pt.coord(d, self_key.range().max(d));
      ///     output.push_back(TreeNode<uint32_t, dim>(vertex_pt, self_key.getLevel()));
      ///   }
      /// }

      /// if (degree > 1)
      /// {
      ///   /// // Infer edge/face hanging and ownership from status of incident vertices.
      ///   /// // Iterate over pow(3, dim) facets.
      ///   /// // Add pow(degree - 1, k) nodes per k-dimensional facet.
      ///   ///
      ///   /// // Classify facets.
      ///   /// Neighborhood<dim> facets_owned;
      ///   /// int facet_idx = 0;
      ///   /// tmp::nested_for<dim>(0, 3, [&](auto...idx_pack)
      ///   /// {
      ///   ///   do
      ///   ///   {
      ///   ///     const std::array<int, dim> idxs = {idx_pack...};  // 0..2 per axis.
      ///   ///     // A facet is hanging if the vertex nearest cell siblings is hanging.
      ///   ///     int hanging_detector_vertex = 0;
      ///   ///     for (int d = 0; d < dim; ++d)
      ///   ///     {
      ///   ///       // 0 (low)-> 0;  2 (high)-> 1;  1 (middle)-> opposite of child bit.
      ///   ///       bool bit = (idxs[d] == 2) | (idxs[d] == 1) & (not ((child_number >> d) & 1));
      ///   ///       hanging_detector_vertex |= (bit << d);
      ///   ///     }
      ///   ///     if (vertices_hanging.test_flat(hanging_detector_vertex))
      ///   ///       continue;
      ///   ///
      ///   ///     // A facet is owned if the most-likely-owned vertex is owned.
      ///   ///     // Depends on lexicographic priority in vertex_preferred_neighbors.
      ///   ///     int borrow_detector_vertex = 0;
      ///   ///     for (int d = 0; d < dim; ++d)
      ///   ///     {
      ///   ///       // 0 (low)-> 0;  2 (high)-> 1;  1 (middle)-> 1 (go high when can)
      ///   ///       bool bit = idxs[d] > 0;
      ///   ///       borrow_detector_vertex |= (bit << d);
      ///   ///     }
      ///   ///     if (vertices_borrowed.test_flat(borrow_detector_vertex))
      ///   ///       continue;
      ///   ///
      ///   ///     // If above checks passed, mark the facet as owned and nonhanging.
      ///   ///     facets_owned.set_flat(facet_idx);
      ///   ///   } while (false); // Allows continue to skip
      ///   ///   ++facet_idx;
      ///   /// }); //end classify facets
      ///   ///
      ///   /// // Emit nodes for all facets that are owned and nonhanging.
      ///   /// int node_idx = 0;
      ///   /// tmp::nested_for<dim>(0, degree + 1, [&](auto...idx_pack)
      ///   /// {
      ///   ///   std::array<int, dim> idxs = {idx_pack...};  // 0..degree per axis.
      ///   ///   int stride = 1;
      ///   ///   int facet_idx = 0;
      ///   ///   for (int d = 0; d < dim; ++d, stride *= 3)
      ///   ///     facet_idx += ((idxs[d] > 0) + (idxs[d] == degree)) * stride;
      ///   ///   const bool owned_and_nonhanging = facets_owned.test_flat(facet_idx);
      ///   ///
      ///   ///   if (owned_and_nonhanging)
      ///   ///   {
      ///   ///     output.push_back(create_node(self_key, idxs, degree));
      ///   ///   }
      ///   ///   ++node_idx;
      ///   /// }); //end emit nodes

      /// }//end degree > 1
    }

    const size_t range_end = output.size();

    return {range_begin, range_end};
  }


  // A simplistic policy that uses the same-level and coarser neighborhoods,
  // and emits the nonhanging and hanging nodes.

  template <int dim>
  inline
  std::pair< size_t, size_t >   // range begin and end for vertices of this cell
  neighborhood_to_all_vertices(
      const TreeNode<uint32_t, dim> &self_key,
      Neighborhood<dim> self_neighborhood,
      Neighborhood<dim> parent_neighborhood,
      const int child_number,
      const int degree,
      std::vector<TreeNode<uint32_t, dim>> &output )
  {
    assert(degree == 1);  //this policy only emits vertices

    const static std::array<Neighborhood<dim>, nverts(dim)>
      preferred_neighbors = vertex_preferred_neighbors<dim>();
      // Ideally constexpr, but Neighborhood uses std::bitset.
    const static std::array<Neighborhood<dim>, nverts(dim)>
      corner_relevant = corner_neighbors<dim>();

    const size_t range_begin = output.size();

    // Append any vertices who prefer no proper neighbor over the current cell.
    if (self_neighborhood.center_occupied())
    {
      // Policy for now: Parent's neighbor owns a vertex shared with child,
      //       ___       but one of the children owns a hanging node.
      //     _|   |_     Only one vertex on the child could be shared with a
      //   _|_|___|_|_   parent's neighbor, which it also shares with parent.

      //future: Maybe parent should inspect neighbors of children.

      const bool skip_shared_vertex =
        (parent_neighborhood & corner_relevant[child_number]).any();

      // Except for the shared vertex, proceed as with same-level neighbors.
      for (int vertex = 0; vertex < nverts(dim); ++vertex)
      {
        if (skip_shared_vertex and (vertex == child_number))
          continue;
        if ((self_neighborhood & preferred_neighbors[vertex]).none())
        {
          auto vertex_pt = self_key.range().min();
          for (int d = 0; d < dim; ++d)
            if (bool(vertex & (1 << d)))
              vertex_pt.coord(d, self_key.range().max(d));
          output.push_back(TreeNode<uint32_t, dim>(vertex_pt, self_key.getLevel()));
        }
      }
    }

    const size_t range_end = output.size();

    return {range_begin, range_end};
  }



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
      const std::vector<TreeNode<uint32_t, 2>> &octant_keys = neighbor_sets_pair.first;
      const std::vector<Neighborhood<2>> &neighborhoods = neighbor_sets_pair.second;
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
      const std::vector<TreeNode<uint32_t, 3>> &octant_keys = neighbor_sets_pair.first;
      const std::vector<Neighborhood<3>> &neighborhoods = neighbor_sets_pair.second;

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
          const std::vector<TreeNode<uint32_t, dim>> &octant_keys = neighbor_sets_pair.first;
          const std::vector<Neighborhood<dim>> &neighborhoods = neighbor_sets_pair.second;

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
          const std::vector<TreeNode<uint32_t, dim>> &octant_keys = neighbor_sets_pair.first;
          const std::vector<Neighborhood<dim>> &neighborhoods = neighbor_sets_pair.second;

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

// =============================================================================
// Implementation
// =============================================================================

namespace ot
{
  // node_set()
  template <int dim, typename Policy>
  std::vector<TreeNode<uint32_t, dim>> node_set(
      const std::vector<TreeNode<uint32_t, dim>> &octant_keys,
      const std::vector<Neighborhood<dim>> &neighborhoods,
      const int degree,
      Policy &&policy)
  {
    std::vector<Neighborhood<dim>> neighborhoods_by_level(m_uiMaxDepth + 1);

    std::vector<TreeNode<uint32_t, dim>> nodes;

    for (size_t i = 0, end = octant_keys.size(); i < end; ++i)
    {
      const auto key = octant_keys[i];
      const int key_level = key.getLevel();
      const int child_number = key.getMortonIndex();

      const auto self_neighborhood = neighborhoods[i];
      const auto parent_neighborhood = neighborhoods_by_level[key_level - 1];

      policy(key, self_neighborhood, parent_neighborhood, child_number, degree, nodes);

      if (i + 1 < end and key.isAncestor(octant_keys[i + 1]))
        neighborhoods_by_level[key_level] = self_neighborhood;
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
  }

  template <int dim>
  inline std::array<Neighborhood<dim>, nverts(dim)> corner_neighbors()
  {
    constexpr int V = nverts(dim);
    std::array<Neighborhood<dim>, V> neighborhoods = {};
    neighborhoods.fill(Neighborhood<dim>::full());
    //future: static variable

    for (int v = 0; v < V; ++v)
      for (int axis = 0; axis < dim; ++axis)
        if (bool(v & (1 << axis)))//up
          neighborhoods[v] &= Neighborhood<dim>::not_down(axis);
        else//down
          neighborhoods[v] &= Neighborhood<dim>::not_up(axis);

    return neighborhoods;
  }


  template <int dim>
  std::array<Neighborhood<dim>, nverts(dim)> vertex_preferred_neighbors()
  {
    //     Relevant           Priority              Preferred
    //  o|o|_   _|o|o      _|_|_    _|_|_        _|_|_    _|_|_
    //  o|x|_   _|x|o      o|x|_    o|x|_        o|x|_    _|x|_
    //  _|_|_   _|_|_      o|o|o    o|o|o        _|_|_    _|_|_
    //                  &                    =
    //  _|_|_   _|_|_      _|_|_    _|_|_        _|_|_    _|_|_
    //  o|x|_   _|x|o      o|x|_    o|x|_        o|x|_    _|x|_
    //  o|o|_   _|o|o      o|o|o    o|o|o        o|o|_    _|o|o
    constexpr int V = nverts(dim);
    std::array<Neighborhood<dim>, V> preferred_neighbors =
      corner_neighbors<dim>();

    Neighborhood<dim> priority = priority_neighbors<dim>();
    for (int vertex = 0; vertex < V; ++vertex)
      preferred_neighbors[vertex] &= priority;

    return preferred_neighbors;
  }


  DOCTEST_TEST_CASE("vertex_preferred_neighbors 2D")
  {
    const std::array<Neighborhood<2>, nverts(2)> preferred =
        vertex_preferred_neighbors<2>();

    CHECK( std::to_string(preferred[0])
        == std::string("0 0 0\n"
                       "1 0 0\n"
                       "1 1 0") );

    CHECK( std::to_string(preferred[1])
        == std::string("0 0 0\n"
                       "0 0 0\n"
                       "0 1 1") );

    CHECK( std::to_string(preferred[2])
        == std::string("0 0 0\n"
                       "1 0 0\n"
                       "0 0 0") );

    CHECK( std::to_string(preferred[3])
        == std::string("0 0 0\n"
                       "0 0 0\n"
                       "0 0 0") );
  }

  DOCTEST_TEST_CASE("vertex_preferred_neighbors 3D")
  {
    const std::array<Neighborhood<3>, nverts(3)> preferred =
        vertex_preferred_neighbors<3>();

    CHECK( ("\n\n" + std::to_string(preferred[0]))
        == std::string("\n\n" "0 0 0  0 0 0  0 0 0\n"
                              "1 1 0  1 0 0  0 0 0\n"
                              "1 1 0  1 1 0  0 0 0") );

    CHECK( ("\n\n" + std::to_string(preferred[1]))
        == std::string("\n\n" "0 0 0  0 0 0  0 0 0\n"
                              "0 1 1  0 0 0  0 0 0\n"
                              "0 1 1  0 1 1  0 0 0") );

    // ...

    CHECK( ("\n\n" + std::to_string(preferred[6]))
        == std::string("\n\n" "0 0 0  0 0 0  0 0 0\n"
                              "0 0 0  1 0 0  0 0 0\n"
                              "0 0 0  0 0 0  0 0 0") );

    CHECK( ("\n\n" + std::to_string(preferred[7]))
        == std::string("\n\n" "0 0 0  0 0 0  0 0 0\n"
                              "0 0 0  0 0 0  0 0 0\n"
                              "0 0 0  0 0 0  0 0 0") );
  }


}






#endif//DENDRO_KT_NEIGHBORS_TO_NODES_HPP
