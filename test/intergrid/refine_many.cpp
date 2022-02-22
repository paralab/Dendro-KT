#include <dollar.hpp>
#include "dollar_stat.h"

#include "treeNode.h"
#include "distTree.h"
#include "oda.h"
#include "octUtils.h"
#include "lerp.hpp"

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

std::vector<DofT> local_vector(const ot::DA<DIM> *da, int dofs);
/// std::vector<DofT> ghosted_vector(const ot::DA<DIM> *da, int dofs);

void fill_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                 const ot::DA<DIM> *da,
                 std::vector<DofT> &local);

size_t check_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                    const ot::DA<DIM> *da,
                    const std::vector<DofT> &local);

/// template <typename X>  X*       ptr(std::vector<X> &v)             { return v.data(); }
/// template <typename X>  const X* const_ptr(const std::vector<X> &v) { return v.data(); }
/// template <typename X>  size_t   size(const std::vector<X> &v)      { return v.size(); }

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

  using DistTree = ot::DistTree<uint, DIM>;
  using DA = ot::DA<DIM>;
  DistTree coarse_dtree, fine_dtree;
  DA *coarse_da, *fine_da;

  const double sfc_tol = 0.1;
  const size_t grain = 5e1;
  const int degree = 1;
  {DOLLAR("coarse_dtree")
    coarse_dtree = make_dist_tree(grain, sfc_tol, comm);
    /// coarse_dtree = ot::DistTree<uint, DIM>::constructSubdomainDistTree(3, comm, sfc_tol);
  }
  {DOLLAR("coarse_da")
    coarse_da = new DA(coarse_dtree, comm, degree, int{}, sfc_tol);
  }
    printf("[%d] da size (e:%lu n:%lu)\n", comm_rank, coarse_da->getLocalElementSz(), coarse_da->getLocalNodalSz());
  ot::quadTreeToGnuplot(coarse_dtree.getTreePartFiltered(), 8, "coarse.tree", comm);
  /// ot::quadTreeToGnuplot(coarse_da->getTNVec(), 8, "coarse.da", comm);

  std::vector<int> increase;
  increase.reserve(coarse_dtree.getTreePartFiltered().size());
  for (const Oct &oct : coarse_dtree.getTreePartFiltered())
    increase.push_back(oct.getIsOnTreeBdry() ? 1 : 0);
  {DOLLAR("Refine")
    coarse_dtree.distRefine(coarse_dtree, std::move(increase), fine_dtree, sfc_tol);
  }
  {DOLLAR("fine_da")
    fine_da = new DA(fine_dtree, comm, degree, int{}, sfc_tol);
  }
  printf("[%d] refined size (e:%lu n:%lu)\n", comm_rank, fine_da->getLocalElementSz(), fine_da->getLocalNodalSz());
  ot::quadTreeToGnuplot(fine_dtree.getTreePartFiltered(), 10, "fine.tree", comm);

    const int singleDof = 1;
  std::vector<DofT> coarse_local = local_vector(coarse_da, singleDof);
  std::vector<DofT> fine_local = local_vector(fine_da, singleDof);
  fill_xpyp1(coarse_dtree, coarse_da, coarse_local);

  {DOLLAR("lerp")
    ot::lerp(
        coarse_dtree, coarse_da, singleDof, coarse_local,
        fine_dtree, fine_da, fine_local);
  }

  const size_t misses = check_xpyp1(fine_dtree, fine_da, fine_local);
  printf("[%d] misses: %s%lu/%lu (%.0f%%)%s\n", comm_rank,
      (misses == 0 ? GRN : RED),
      misses, fine_da->getLocalNodalSz(), 100.0 * misses / fine_da->getLocalNodalSz(),
      NRM);

  print_dollars(comm);

  _DestroyHcurve();
  DendroScopeEnd();
  PetscFinalize();
  return 0;
}

// ============================================================

// accumulate_sum()
template <typename It>
auto accumulate_sum(It begin, It end) -> decltype(0 + *begin)
{
  decltype(0 + *begin) sum = {};
  while (begin != end) { sum += *begin; ++begin; }
  return sum;
}

// local_vector()
std::vector<DofT> local_vector(const ot::DA<DIM> *da, int ndofs)
{
  std::vector<DofT> local;
  da->createVector(local, false, false, ndofs);
  return local;
}

// fill_xpyp1()
void fill_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                 const ot::DA<DIM> *da,
                 std::vector<DofT> &local)
{
  const int degree = da->getElementOrder();
  const size_t local_sz = da->getLocalNodalSz();
  const size_t local_begin = da->getLocalNodeBegin();
  const ot::TreeNode<uint, DIM> *tn_coords = da->getTNCoords();
  for (size_t i = 0; i < local_sz; ++i)
  {
    std::array<double, DIM> coords;
    ot::treeNode2Physical(tn_coords[local_begin + i], degree, coords.data());
    local[i] = 1 + accumulate_sum(coords.begin(), coords.end());
  }
}

// check_xpyp1()
size_t check_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                   const ot::DA<DIM> *da,
                   const std::vector<DofT> &local)
{
  const int degree = da->getElementOrder();
  const size_t local_sz = da->getLocalNodalSz();
  const size_t local_begin = da->getLocalNodeBegin();
  const ot::TreeNode<uint, DIM> *tn_coords = da->getTNCoords();
  std::vector<const char*> colors(local.size(), NRM);
  size_t misses = 0;
  for (size_t i = 0; i < local_sz; ++i)
  {
    std::array<double, DIM> coords;
    ot::treeNode2Physical(tn_coords[local_begin + i], degree, coords.data());
    const DofT sum = 1 + accumulate_sum(coords.begin(), coords.end());
    if (local[i] != sum)
    {
      ++misses;
      colors[i] = RED;
    }
  }
  ot::printNodes(da->getTNCoords() + da->getLocalNodeBegin(),
                 da->getTNCoords() + da->getLocalNodeBegin() + da->getLocalNodalSz(),
                 local.data(),
                 colors.data(),
                 degree);
  return misses;
}


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

