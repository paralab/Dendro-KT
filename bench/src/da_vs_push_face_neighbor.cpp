/**
 * @author Masado Ishii
 * @date   2023-03-29
 * @brief  Single-core benchmark comparing node sets via old DA and new method.
 */

#include "dollar-master/dollar.hpp"
#include "include/dollar_stat.h"
#include "include/treeNode.h"
#include "include/tsort.h"
#include "include/oda.h"
#include "include/neighbors_to_nodes.hpp"

#include <vector>
#include <string>
#include <sstream>

#include <mpi.h>


template <int dim>
std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_central(int max_depth);
template <int dim>
std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_edges(int max_depth);


template <int dim>
struct ProxyNewDA
{
  void construct(
      const std::vector<ot::TreeNode<uint32_t, dim>> &grid,
      int degree);

  size_t n_owned_nodes() const;
  long long unsigned n_global_nodes() const;

  std::vector<ot::TreeNode<uint32_t, dim>> nodal_points;
};


int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  constexpr int dim = 3;
  _InitializeHcurve(dim);

  using namespace ot;

  std::stringstream logging;

  std::random_device rd;
  std::mt19937 g(rd());

  enum Algorithm { old_da, new_push_face_neighbors };
  std::vector<Algorithm> algorithm_choice;

  // Parameters
  enum Pattern { central, edges };
  std::vector<int> degrees[2],  depths[2],  runs[2];
  //
  degrees[central] = {1, 1, 2, 2};
  depths[central]  = {20, 29, 20, 29};
  runs[central]    = {100, 100, 10, 10};
  //
  degrees[edges] = {1, 1, /*1,*/    2, 2, /*2*/};
  depths[edges]  = {4, 7, /*11,*/   4, 7, /*11*/};
  runs[edges]    = {100, 20, /*3,*/ 100, 20, /*1*/};

  for (int pattern: {central, edges})
  {
    for (size_t i = 0; i < runs[pattern].size(); ++i)
    {
      const std::string parameters = std::string()
          + "pattern=" + std::to_string(pattern)
          + " degree=" + std::to_string(degrees[pattern][i])
          + " depth=" + std::to_string(depths[pattern][i]);
      std::cerr << "Begin:  " << parameters << "\n";
      DOLLAR(parameters);
      const std::vector<TreeNode<uint32_t, dim>> grid =
          pattern == central ? grid_pattern_central<dim>(depths[pattern][i])
          :                    grid_pattern_edges<dim>(depths[pattern][i]);

      auto grid_copy = grid;
      dollar::clobber();
      const DistTree<uint32_t, dim> dtree(grid_copy, MPI_COMM_WORLD);
      dollar::unclobber();

      DA<dim> da(degrees[pattern][i]);
      ProxyNewDA<dim> proxy_new_da;

      long long unsigned n_cells = 0;
      long long unsigned n_nodes_old = 0;
      long long unsigned n_nodes_new = 0;

      algorithm_choice.clear();
      algorithm_choice.insert(algorithm_choice.end(), runs[pattern][i], old_da);
      algorithm_choice.insert(algorithm_choice.end(), runs[pattern][i], new_push_face_neighbors);
      std::shuffle(algorithm_choice.begin(), algorithm_choice.end(), g);

      for (Algorithm alg: algorithm_choice)
      {
        switch (alg)
        {
          case old_da:
          {
            DOLLAR("old_da");
            dollar::clobber();
            da.construct(dtree, MPI_COMM_WORLD, degrees[pattern][i], {}, 0.3);
            dollar::unclobber();
            volatile auto _ = da.getLocalNodalSz();
          }
          n_cells = da.getGlobalElementSz();
          n_nodes_old = da.getGlobalNodeSz();
          break;

          case new_push_face_neighbors:
          {
            DOLLAR("new_push_face_neighbors");
            proxy_new_da.construct(grid, degrees[pattern][i]);
            volatile auto _ = proxy_new_da.n_owned_nodes();
          }
          n_nodes_new = proxy_new_da.n_global_nodes();
          break;
        }
      }

      logging << parameters
        << "\t grid.size=" << n_cells
        << "\t nodes.size.old=" << n_nodes_old
        << "\t nodes.size.new=" << n_nodes_new
        << "\n";
      std::cerr << "\n";
    }
  }

  // Case 2 grid pattern

  std::cout << logging.str() << "\n";

  dollar::DollarStat dollar_stat(MPI_COMM_WORLD);
  dollar_stat.tsv(std::cout);

  if (false)
  {
    std::cout << "Max Depth \t Case 1 sizes" << "\n";
    std::cout << "--------- \t ------------" << "\n";
    for (int max_depth : {20, 30})
      std::cout << "  " << max_depth << "        \t "
                << grid_pattern_central<dim>(max_depth).size() << "\n";

    std::cout << "\n";

    std::cout << "Max Depth \t Case 2 sizes" << "\n";
    std::cout << "--------- \t ------------" << "\n";
    for (int max_depth : {4, 7, 11})
      std::cout << "  " << max_depth << "        \t "
                << grid_pattern_edges<dim>(max_depth).size() << "\n";
  }


  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}


// construct()
template <int dim>
void ProxyNewDA<dim>::construct(
    const std::vector<ot::TreeNode<uint32_t, dim>> &grid,
    int degree)
{
  using namespace ot;

  // Neighbors.
  const auto neighbor_sets_pair = neighbor_sets(grid);
  const std::vector<TreeNode<uint32_t, dim>> &octant_keys = neighbor_sets_pair.first;
  const std::vector<Neighborhood<dim>> &neighborhoods = neighbor_sets_pair.second;

  /// DOLLAR("node_set()");

  // Nodes.
  this->nodal_points = node_set<dim>(
          octant_keys, neighborhoods, degree,
          neighborhood_to_nonhanging<dim>);
}


template <int dim>
size_t ProxyNewDA<dim>::n_owned_nodes( ) const
{
  return nodal_points.size();
}

template <int dim>
long long unsigned ProxyNewDA<dim>::n_global_nodes( ) const
{
  return n_owned_nodes();//future: use global number if have comm
}





// =======================================
//  Case 1 _ _ _ _      Case 2  _ _ _ _
//        |_|_|_|_|            |+|+|+|+|
//        |_|+|+|_|            |+|_|_|+|
//        |_|+|+|_|            |+|_|_|+|
//        |_|_|_|_|            |+|+|+|+|
//   "Central"              "Edges"
//   linear in max_depth    exponential in max_depth
// =======================================

//
// grid_pattern_central()
//
template <int dim>
std::vector<ot::TreeNode<uint32_t, dim>>
  grid_pattern_central(int max_depth)
{
  using namespace ot;
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
  return grid;
}

//
// grid_pattern_edges()
//
template <int dim>
std::vector<ot::TreeNode<uint32_t, dim>>
  grid_pattern_edges(int max_depth)
{
  using namespace ot;
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
  return grid;
}


