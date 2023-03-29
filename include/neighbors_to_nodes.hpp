
/**
 * @author Masado Ishii
 * @date   2023-03-28
 * @brief  Make node sets out of neighborhood flags.
 */

#ifndef DENDRO_KT_NEIGHBORS_TO_NODES_HPP
#define DENDRO_KT_NEIGHBORS_TO_NODES_HPP

// Function declaration for linkage purposes.
inline void link_neighbors_to_nodes_tests() {};

#include "include/neighborhood.hpp"
#include "include/neighbor_sets.hpp"

#include "doctest/doctest.h"

#include <vector>

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
  std::pair< typename std::vector<TreeNode<uint32_t, dim>>::iterator,    // range begin
             typename std::vector<TreeNode<uint32_t, dim>>::iterator >   // range end
  neighborhood_to_nonhanging(
      const TreeNode<uint32_t, dim> &self_key,
      Neighborhood<dim> self_neighborhood,
      Neighborhood<dim> parent_neighborhood,
      const int child_number,
      const int degree,
      std::vector<TreeNode<uint32_t, dim>> &output )
  {
    assert(degree == 1);  //future: support quadratic and high-order.

    auto range_begin = output.end();

    // insert...

    throw std::logic_error("Not implemented: neighborhood_to_nonhanging()");

    auto range_end = output.end();
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
      child_relevant = corner_neighbors<dim>();

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
        (parent_neighborhood & child_relevant[child_number]).any();

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
      _DestroyHcurve;
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
      _DestroyHcurve;
    }

    DOCTEST_TEST_CASE("Count vertices on nonuniform 2D grid")
    {
      //  Case 1 _ _ _ _      Case 2  _ _ _ _
      //        |_|_|_|_|            |+|+|+|+|
      //        |_|+|+|_|            |+|_|_|+|
      //        |_|+|+|_|            |+|_|_|+|
      //        |_|_|_|_|            |+|+|+|+|
      //

      const auto shell = [](int dim, size_t outer_side) {
        size_t outer_ball = 1;
        size_t inner_ball = 1;
        while (dim > 0)
        {
          outer_ball *= outer_side;
          inner_ball *= (outer_side - 2);
          --dim;
        }
        return outer_ball - inner_ball;
      };

      const auto case_1_vertices = [shell](int dim, int max_depth) {
        return 1 + shell(dim, 3) + shell(dim, 5) * (max_depth - 2 + 1);
      };

      const auto case_2_vertices = [shell](int dim, int max_depth) {
        size_t vertices = 1;
        for (int side = 5; side < (1 << max_depth) + 1; side = side * 2 + 3)
          vertices += shell(dim, side);
        vertices += shell(dim, (1 << max_depth) + 1 - 2);
        vertices += shell(dim, (1 << max_depth) + 1);
        return vertices;
      };

      _InitializeHcurve(2);

      for (int max_depth = 2; max_depth <= 5; ++max_depth)
      {
        // Case 1
        {
          // Grid.
          std::vector<TreeNode<uint32_t, 2>> grid = { TreeNode<uint32_t, 2>() };
          std::vector<TreeNode<uint32_t, 2>> queue;
          for (int level = 1; level <= max_depth; ++level)
          {
            queue.clear();
            const auto middle = TreeNode<uint32_t, 2>().getChildMorton(0).range().max();
            for (auto oct: grid)
            {
              // Case 1: Refine the center.
              if (oct.range().closedContains(middle))
                for (int child = 0; child < nchild(2); ++child)
                  queue.push_back(oct.getChildMorton(child));
              else
                queue.push_back(oct);
            }
            std::swap(grid, queue);
          }

          // Neighbors.
          const auto neighbor_sets_pair = neighbor_sets(grid);
          const std::vector<TreeNode<uint32_t, 2>> &octant_keys = neighbor_sets_pair.first;
          const std::vector<Neighborhood<2>> &neighborhoods = neighbor_sets_pair.second;

          // Nodes.
          const int degree = 1;
          std::vector<TreeNode<uint32_t, 2>> nodes =
              node_set<2>(
                  octant_keys, neighborhoods, degree, neighborhood_to_all_vertices<2>);

          CHECK( nodes.size() == case_1_vertices(2, max_depth) );
        }

        // Case 2
        {
          // Grid.
          std::vector<TreeNode<uint32_t, 2>> grid = { TreeNode<uint32_t, 2>() };
          std::vector<TreeNode<uint32_t, 2>> queue;
          for (int level = 1; level <= max_depth; ++level)
          {
            queue.clear();
            const uint32_t maximum = TreeNode<uint32_t, 2>().range().side();
            for (auto oct: grid)
            {
              // Case 2: Refine the cube surface.
              const std::array<uint32_t, 2> min = oct.range().min();
              const std::array<uint32_t, 2> max = oct.range().max();
              if (*(std::min_element(min.begin(), min.end())) == 0 or
                  *(std::max_element(max.begin(), max.end())) == maximum)
                for (int child = 0; child < nchild(2); ++child)
                  queue.push_back(oct.getChildMorton(child));
              else
                queue.push_back(oct);
            }
            std::swap(grid, queue);
          }

          // Neighbors.
          const auto neighbor_sets_pair = neighbor_sets(grid);
          const std::vector<TreeNode<uint32_t, 2>> &octant_keys = neighbor_sets_pair.first;
          const std::vector<Neighborhood<2>> &neighborhoods = neighbor_sets_pair.second;

          // Nodes.
          const int degree = 1;
          std::vector<TreeNode<uint32_t, 2>> nodes =
              node_set<2>(
                  octant_keys, neighborhoods, degree, neighborhood_to_all_vertices<2>);

          CHECK( nodes.size() == case_2_vertices(2, max_depth) );
        }
      }

      _DestroyHcurve;
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


  template <int dim>
  inline std::array<Neighborhood<dim>, nverts(dim)> corner_neighbors()
  {
    constexpr int V = nverts(dim);
    std::array<Neighborhood<dim>, V> neighbors = {};
    //future: static variable

    // Middle-high, or low-middle, neighbors for high, or low, corners.
    neighbors[V - 1] =
      Neighborhood<dim>::full() & ~Neighborhood<dim>::solitary();
    for (int codim = dim-1; codim >= 0; --codim)
      for (int i = nverts(codim) - 1; i < V; i += nverts(codim + 1))
      {
        int j = i + nverts(codim);
        neighbors[i] = neighbors[j].cleared_up(codim);
        neighbors[j] = neighbors[j].cleared_down(codim);
      }
    return neighbors;
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

    // Prioritization: Lexicographic predecessors.
    constexpr int N = Neighborhood<dim>::n_neighbors();
    auto priority = Neighborhood<dim>::where([N](int i) { return i < N/2; });
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
