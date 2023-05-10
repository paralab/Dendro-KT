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
#include "include/da_p2p.hpp"

#include <vector>
#include <string>
#include <sstream>

#include <mpi.h>

#include "include/debug.hpp"

template <int dim>
std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_central(int max_depth, MPI_Comm comm);
template <int dim>
std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_edges(int max_depth, MPI_Comm comm);



int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  dbg::wait_for_debugger(comm);
  DendroScopeBegin();
  constexpr int dim = 3;
  _InitializeHcurve(dim);

  using namespace ot;

  std::stringstream logging;

  std::random_device rd;
  unsigned int seed = (par::mpi_comm_rank(comm) == 0? rd() : 0);
  MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, comm);
  std::mt19937 g(seed);

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
      dollar::clobber();
      const std::vector<TreeNode<uint32_t, dim>> grid =
          pattern == central ? grid_pattern_central<dim>(depths[pattern][i], comm)
          :                    grid_pattern_edges<dim>(depths[pattern][i], comm);
      dollar::unclobber();

      auto grid_copy = grid;
      dollar::clobber();
      const DistTree<uint32_t, dim> dtree(grid_copy, comm);
      dollar::unclobber();

      using OldDA = slow_da::DA<dim>;
      using NewDA = da_p2p::DA_Wrapper<dim>;
      OldDA da(degrees[pattern][i]);
      NewDA new_da;

      long long unsigned n_cells = 0;
      long long unsigned n_nodes_old = 0;
      long long unsigned n_nodes_new = 0;

      algorithm_choice.clear();
      algorithm_choice.insert(algorithm_choice.end(), runs[pattern][i], old_da);
      algorithm_choice.insert(algorithm_choice.end(), runs[pattern][i], new_push_face_neighbors);
      std::shuffle(algorithm_choice.begin(), algorithm_choice.end(), g);

      size_t choice_idx = 0;
      for (Algorithm alg: algorithm_choice)
      {
        switch (alg)
        {
          case old_da:
          {
            DOLLAR("old_da");
            dollar::clobber();
            da.construct(dtree, comm, degrees[pattern][i], {}, 0.3);
            dollar::unclobber();
            volatile auto _ = da.getLocalNodalSz();
          }
          n_cells = da.getGlobalElementSz();
          n_nodes_old = da.getGlobalNodeSz();
          break;

          case new_push_face_neighbors:
          {
            DOLLAR("new_push_face_neighbors");
            new_da = NewDA(dtree, comm, degrees[pattern][i]);
            volatile auto _ = new_da.getLocalNodalSz();
          }
          n_nodes_new = new_da.getGlobalNodeSz();
          break;
        }
        ++choice_idx;
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

  const dollar::DollarStat dollar_stat(comm);
  const dollar::DollarStat stat_mean = dollar_stat.mpi_reduce_mean();
  /// const dollar::DollarStat stat_min = dollar_stat.mpi_reduce_min();
  /// const dollar::DollarStat stat_max = dollar_stat.mpi_reduce_max();
  if (par::mpi_comm_rank(comm) == 0)
  {
    std::cout << "\nMean\n";
    stat_mean.tsv(std::cout);
  }

  if (false)
  {
    std::cout << "Max Depth \t Case 1 sizes" << "\n";
    std::cout << "--------- \t ------------" << "\n";
    for (int max_depth : {20, 30})
      std::cout << "  " << max_depth << "        \t "
                << grid_pattern_central<dim>(max_depth, comm).size() << "\n";

    std::cout << "\n";

    std::cout << "Max Depth \t Case 2 sizes" << "\n";
    std::cout << "--------- \t ------------" << "\n";
    for (int max_depth : {4, 7, 11})
      std::cout << "  " << max_depth << "        \t "
                << grid_pattern_edges<dim>(max_depth, comm).size() << "\n";
  }


  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
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
  grid_pattern_central(int max_depth, MPI_Comm comm)
{
  using namespace ot;
  std::vector<TreeNode<uint32_t, dim>> grid;
  if (par::mpi_comm_rank(comm) == 0)
    grid = { TreeNode<uint32_t, dim>() };
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
  SFC_Tree<uint32_t, dim>::distTreeSort(grid, 0.3, comm);
  return grid;
}

//
// grid_pattern_edges()
//
template <int dim>
std::vector<ot::TreeNode<uint32_t, dim>>
  grid_pattern_edges(int max_depth, MPI_Comm comm)
{
  using namespace ot;
  std::vector<TreeNode<uint32_t, dim>> grid;
  if (par::mpi_comm_rank(comm) == 0)
    grid = { TreeNode<uint32_t, dim>() };
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
  SFC_Tree<uint32_t, dim>::distTreeSort(grid, 0.3, comm);
  return grid;
}


