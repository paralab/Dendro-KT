#include <dollar.hpp>
#include "dollar_stat.h"

#include "treeNode.h"
#include "distTree.h"
#include "oda.h"
#include "octUtils.h"

#include <vector>
#include <array>
#include <petsc.h>

#include "test/octree/multisphere.h"

constexpr int DIM = 2;
using uint = unsigned int;
using DofT = double;

using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;

ot::DistTree<uint, DIM> make_dist_tree(size_t grain, double sfc_tol, MPI_Comm comm);
void print_dollars(MPI_Comm comm);

/// std::vector<double> local_vector(const ot::DA<DIM> &da);
/// std::vector<double> ghosted_vector(const ot::DA<DIM> &da);

template <typename X>  X*       ptr(std::vector<X> &v)             { return v.data(); }
template <typename X>  const X* const_ptr(const std::vector<X> &v) { return v.data(); }
template <typename X>  size_t   size(const std::vector<X> &v)      { return v.size(); }

/// template <typename X>
/// class View
/// {
///   X *begin;
///   size_t size;
/// };

//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank, comm_size;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  const double sfc_tol = 0.1;
  const size_t grain = 1e3;
  const int degree = 1;
  ot::DistTree<uint, DIM> dtree = make_dist_tree(grain, sfc_tol, comm);
  /// ot::DistTree<uint, DIM> dtree = ot::DistTree<uint, DIM>::constructSubdomainDistTree(3, comm, sfc_tol);

  ot::DA<DIM> da(dtree, comm, degree, int{}, sfc_tol);
  printf("[%d] da size (e:%lu n:%lu)\n", comm_rank, da.getLocalElementSz(), da.getLocalNodalSz());

  ot::quadTreeToGnuplot(dtree.getTreePartFiltered(), 8, "octree", comm);

  std::vector<int> increase;
  increase.reserve(dtree.getTreePartFiltered().size());
  for (const Oct &oct : dtree.getTreePartFiltered())
    increase.push_back(oct.getIsOnTreeBdry() ? 3 : 0);
  ot::DistTree<uint, DIM> dtree2;
  dtree.distRefine(dtree, std::move(increase), dtree2, sfc_tol);

  ot::quadTreeToGnuplot(dtree2.getTreePartFiltered(), 10, "newTree", comm);

  /// ot::quadTreeToGnuplot(da.getTNVec(), 8, "da", comm);

  print_dollars(comm);

  _DestroyHcurve();
  DendroScopeEnd();
  PetscFinalize();
  return 0;
}

// ============================================================



// ============================================================


// make_dist_tree()
ot::DistTree<uint, DIM> make_dist_tree(size_t grain, double sfc_tol, MPI_Comm comm)
{
  test::SphereSet<DIM> sphereSet;
  sphereSet.carveSphere(0.10, {0.15, 0.6, 0.5});
  sphereSet.carveSphere(0.10, {0.7, 0.4, 0.5});
  sphereSet.carveSphere(0.210, {0.45, 0.55, 0.5});
  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::minimalSubdomainDistTreeGrain(
          grain, sphereSet, comm, sfc_tol);
  return distTree;
}

// print_dollars()
void print_dollars(MPI_Comm comm)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  dollar::DollarStat dollar_stat(comm);
  dollar::clear();
  dollar::DollarStat dollar_mean = dollar_stat.mpi_reduce_mean();
  if (comm_rank == 0)
  {
    std::cout << "\n" << "[Mean]\n";
    dollar_mean.text(std::cout);
  }
}

