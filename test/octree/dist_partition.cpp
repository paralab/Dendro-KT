/// #include <dollar.hpp>
/// #include "dollar_stat.h"

#include <vector>
#include <array>

#include "distTree.h"
#include "filterFunction.h"
#include "octUtils.h"
#include "tnUtils.h"
#include "treeNode.h"
#include "oda.h"

#include "./gaussian.hpp"

using uint = unsigned int;
using LLU = long long unsigned;
constexpr int DIM = 2;

// --------------------------------------------------------------------
using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;
using DistTree = ot::DistTree<uint, DIM>;

size_t size(const OctList &octList);
const OctList & octList(const DistTree &dtree);

template <typename T, unsigned dim>
bool isPartitioned(std::vector<ot::TreeNode<T, dim>> octants, MPI_Comm comm);

int maxLevel(const OctList &octList);
int maxLevel(const DistTree &dtree);

template <typename T>
T mpi_max(T x, MPI_Comm comm);

DendroIntL mpi_sum(DendroIntL x, MPI_Comm comm);
bool mpi_and(bool x, MPI_Comm comm);

// --------------------------------------------------------------------

//
// main()
//
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank, comm_size;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  m_uiMaxDepth = 15;
  /// m_uiMaxDepth = 9;

  const auto new_oct = [=](std::array<uint, DIM> coords, int lev)
  {
    lev = m_uiMaxDepth;  // override
    const uint mask = (1u << m_uiMaxDepth) - (1u << m_uiMaxDepth - lev);
    for (int d = 0; d < DIM; ++d)
      coords[d] &= mask;
    return Oct(coords, lev);
  };

  /// const LLU Ng = 10e6;  // oops 10e6 == 1e7
  const LLU Ng = 10e4;   // 10e4 == 1e5
  const auto N_begin = [=](int rank) { return Ng * rank / comm_size; };
  const LLU Nl = N_begin(comm_rank + 1) - N_begin(comm_rank);

  OctList octants = test::gaussian<uint, DIM>(N_begin(comm_rank), Nl, new_oct);
  const int max_level = mpi_max(maxLevel(octants), comm);

  /// ot::quadTreeToGnuplot(octants, max_level, "points.pre", comm);

  const double sfc_tol = 0.1;
  std::vector<ot::TNPoint<uint, DIM>> xs;
  for (const Oct &oct : octants)
    xs.emplace_back(oct.getX(), oct.getLevel());

  ot::distTreePartition_kway(comm, octants, xs, sfc_tol);

  OctList ys;
  for (const auto &pt : xs)
    ys.emplace_back(pt);

  if (ys != octants)
    printf("[%-2d] " RED "values not partitioned with keys!\n" NRM, comm_rank);

  /// ot::quadTreeToGnuplot(octants, max_level, "points.post", comm);

  const bool partitioned = isPartitioned(octants, comm);
  if (comm_rank == 0)
    if (not partitioned)
      printf(RED "Edges unsorted\n" NRM);

  ot::distTreePartition_kway(comm, octants, sfc_tol);

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}

// ---------------------------------------------------------------------



// size()
size_t size(const OctList &octList)
{
  return octList.size();
}

// octList()
const OctList & octList(const DistTree &dtree)
{
  return dtree.getTreePartFiltered();
}

// maxLevel(OctList)
int maxLevel(const OctList &octList)
{
  int maxLevel = 0;
  for (const Oct &oct : octList)
    if (maxLevel < oct.getLevel())
      maxLevel = oct.getLevel();
  return maxLevel;
}

// maxLevel(DistTree)
int maxLevel(const DistTree &dtree)
{
  return mpi_max(maxLevel(octList(dtree)), dtree.getComm());
}

// mpi_sum()
DendroIntL mpi_sum(DendroIntL x, MPI_Comm comm)
{
  DendroIntL sum = 0;
  par::Mpi_Allreduce(&x, &sum, 1, MPI_SUM, comm);
  return sum;
}

// mpi_and()
bool mpi_and(bool x_, MPI_Comm comm)
{
  int x = x_, global = true;
  par::Mpi_Allreduce(&x, &global, 1, MPI_LAND, comm);
  return bool(global);
}

template <typename T>
T mpi_max(T x, MPI_Comm comm)
{
  T global;
  par::Mpi_Allreduce(&x, &global, 1, MPI_MAX, comm);
  return global;
}



template <typename T, unsigned dim>
bool isPartitioned(std::vector<ot::TreeNode<T, dim>> octants, MPI_Comm comm)
{
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  ot::SFC_Tree<T, dim>::locTreeSort(octants);

  ot::TreeNode<T, dim> edges[2] = {{}, {}};
  if (octants.size() > 0)
  {
    edges[0] = octants.front();
    edges[1] = octants.back();
  }
  int local_edges = octants.size() > 0 ? 2 : 0;
  int global_edges = 0;
  int scan_edges = 0;
  par::Mpi_Allreduce(&local_edges, &global_edges, 1, MPI_SUM, comm);
  par::Mpi_Scan(&local_edges, &scan_edges, 1, MPI_SUM, comm);
  scan_edges -= local_edges;

  std::vector<int> count_edges(comm_size, 0);
  std::vector<int> displ_edges(comm_size, 0);
  std::vector<ot::TreeNode<T, dim>> gathered(global_edges);

  par::Mpi_Allgather(&local_edges, &(*count_edges.begin()), 1, comm);
  par::Mpi_Allgather(&scan_edges, &(*displ_edges.begin()), 1, comm);
  par::Mpi_Allgatherv(edges, local_edges,
      &(*gathered.begin()), &(*count_edges.begin()), &(*displ_edges.begin()), comm);

  return ot::isLocallySorted(gathered);
}

