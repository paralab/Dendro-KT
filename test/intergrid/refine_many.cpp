#include <dollar.hpp>
#include "dollar_stat.h"

#include "treeNode.h"
#include "distTree.h"
#include "oda.h"
#include "octUtils.h"
#include "coarseToFine.hpp"

#include <vector>
#include <array>
#include <petsc.h>

#include "test/octree/multisphere.h"

constexpr int DIM = 3;
using uint = unsigned int;
using DofT = double;

using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;

ot::DistTree<uint, DIM> make_dist_tree(size_t grain, double sfc_tol, MPI_Comm comm);
void print_dollars(MPI_Comm comm);

std::vector<DofT> local_vector(const ot::DA<DIM> *da, int ndofs);
std::vector<DofT> cell_vector(const ot::DistTree<uint, DIM> &dtree, int ndofs);

int coarsest_level(const OctList &octants);
int coarsest_level(const ot::DistTree<uint, DIM> &dtree)
    { return coarsest_level(dtree.getTreePartFiltered()); }

void fill_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                 const ot::DA<DIM> *da,
                 const int ndofs,
                 std::vector<DofT> &local);

size_t check_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                    const ot::DA<DIM> *da,
                    const int ndofs,
                    const std::vector<DofT> &local);

void fill_cell_xpypl(
    const ot::DistTree<uint, DIM> &dtree,
    const int ndofs,
    std::vector<DofT> &cell_dofs);

size_t check_cell_xpypl(
    const ot::DistTree<uint, DIM> &dtree,
    const int ndofs,
    const std::vector<DofT> &cell_dofs);

void oldIntergridTransfer(
    const ot::DistTree<uint, DIM> &from_dtree,
    const ot::DA<DIM> *from_da,
    const int ndofs,
    const std::vector<DofT> &from_local,
    const ot::DistTree<uint, DIM> &to_dtree,
    const ot::DA<DIM> *to_da,
    std::vector<DofT> &to_local);

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

  const double sfc_tol = 0.05;
  const size_t grain = 1e2;
  const int degree = 1;
  {DOLLAR("coarse_dtree")
    coarse_dtree = make_dist_tree(grain, sfc_tol, comm);
    /// coarse_dtree = ot::DistTree<uint, DIM>::constructSubdomainDistTree(3, comm, sfc_tol);
  }
  {DOLLAR("coarse_da")
    coarse_da = new DA(coarse_dtree, comm, degree, int{}, sfc_tol);
  }
    printf("[%d] da size (e:%lu n:%lu)\n", comm_rank, coarse_da->getLocalElementSz(), coarse_da->getLocalNodalSz());
  /// ot::quadTreeToGnuplot(coarse_dtree.getTreePartFiltered(), 8, "coarse.tree", comm);
  /// ot::quadTreeToGnuplot(coarse_da->getTNVec(), 8, "coarse.da", comm);

  std::vector<int> increase;
  const int amount = 2;
  const bool refine_all = false;
  increase.reserve(coarse_dtree.getTreePartFiltered().size());
  for (const Oct &oct : coarse_dtree.getTreePartFiltered())
    increase.push_back(oct.getIsOnTreeBdry() or refine_all ? amount : 0);
  {DOLLAR("Refine")
    coarse_dtree.distRefine(coarse_dtree, std::move(increase), fine_dtree, sfc_tol);
  }
  {DOLLAR("fine_da")
    fine_da = new DA(fine_dtree, comm, degree, int{}, sfc_tol);
  }
  printf("[%d] refined size (e:%lu n:%lu)\n", comm_rank, fine_da->getLocalElementSz(), fine_da->getLocalNodalSz());
  /// ot::quadTreeToGnuplot(fine_dtree.getTreePartFiltered(), 10, "fine.tree", comm);

  /// const int local_coarsest = coarsest_level(fine_dtree);
  /// const int global_coarsest = par::mpi_min(local_coarsest, comm);

  const int ndofs = 1;
  std::vector<DofT> coarse_local = local_vector(coarse_da, ndofs);
  std::vector<DofT> fine_local = local_vector(fine_da, ndofs);
  fill_xpyp1(coarse_dtree, coarse_da, ndofs, coarse_local);

  const int cell_ndofs = 1;
  std::vector<DofT> coarse_cell_dofs = cell_vector(coarse_dtree, cell_ndofs);
  std::vector<DofT> fine_cell_dofs = cell_vector(fine_dtree, cell_ndofs);
  std::fill(fine_cell_dofs.begin(), fine_cell_dofs.end(), 0.0f);
  fill_cell_xpypl(coarse_dtree, cell_ndofs, coarse_cell_dofs);

  {DOLLAR("lerp")
    fem::coarse_to_fine(
        coarse_dtree, coarse_da, ndofs, cell_ndofs, coarse_local, coarse_cell_dofs,
        fine_dtree, fine_da, fine_local, fine_cell_dofs);
  }

  const size_t node_misses = check_xpyp1(fine_dtree, fine_da, ndofs, fine_local);
  const size_t cell_misses = check_cell_xpypl(
      fine_dtree, cell_ndofs, fine_cell_dofs);

  printf("[%d] node_misses: %s%lu/%lu (%.0f%%)%s \t "
              "cell misses: %s%lu/%lu (%.0f%%)%s\n",
      comm_rank,
      (node_misses == 0 ? GRN : RED),
      node_misses,
      fine_da->getLocalNodalSz() * ndofs,
      100.0 * node_misses / (ndofs ? fine_da->getLocalNodalSz() * ndofs : 1),
      NRM,
      (cell_misses == 0 ? GRN : RED),
      cell_misses,
      fine_dtree.getTreePartFiltered().size() * cell_ndofs,
      100.0 * cell_misses / (cell_ndofs ? fine_dtree.getTreePartFiltered().size() * cell_ndofs : 1),
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

// cell_vector()
std::vector<DofT> cell_vector(const ot::DistTree<uint, DIM> &dtree, int ndofs)
{
  return std::vector<DofT>(dtree.getTreePartFiltered().size() * ndofs);
}

// coarsest_level();
int coarsest_level(const OctList &octants)
{
  return (octants.size() == 0 ? m_uiMaxDepth :
      std::min_element(octants.begin(), octants.end(),
        [](const Oct &a, const Oct &b) { return a.getLevel() < b.getLevel(); })
      -> getLevel());
}

// fill_xpyp1()
void fill_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                 const ot::DA<DIM> *da,
                 const int ndofs,
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
    const DofT sum = 1 + accumulate_sum(coords.begin(), coords.end());
    for (int dof = 0; dof < ndofs; ++dof)
      local[i * ndofs + dof] = sum + dof;
  }
}

// check_xpyp1()
size_t check_xpyp1( const ot::DistTree<uint, DIM> &dtree,
                   const ot::DA<DIM> *da,
                   const int ndofs,
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
    for (int dof = 0; dof < ndofs; ++dof)
      if (fabs(local[i * ndofs + dof] - (sum + dof)) > 1e-12)
      {
        ++misses;
        colors[i] = RED;
      }
  }
  // Note: The p2c interpolation matrices can introduce tiny errors, O(1e-16),
  //       even for the case degree=1. The matrices are formed in refel.cpp
  //       using a linear solve (lapack_DGESV() -> ip_1D_0 and ip_1D_1).
  //       Tested on degree=1 by rounding the matrix elements to the
  //       nearest 0.5, and the tiny errors went away.
  return misses;
}


// fill_cell_xpypl()
void fill_cell_xpypl(
    const ot::DistTree<uint, DIM> &dtree,
    const int ndofs,
    std::vector<DofT> &cell_dofs)
{
  const OctList &octants = dtree.getTreePartFiltered();
  for (size_t i = 0; i < octants.size(); ++i)
  {
    const Oct oct = octants[i];
    const int level = oct.getLevel();
    std::array<double, DIM> coords;
    double dummy;
    ot::treeNode2Physical(oct, coords.data(), dummy);
    const DofT sum = DIM * level + accumulate_sum(coords.begin(), coords.end());
    for (int dof = 0; dof < ndofs; ++dof)
      cell_dofs[i * ndofs + dof] = sum;
  }
}

// check_cell_xpypl()
size_t check_cell_xpypl(
    const ot::DistTree<uint, DIM> &dtree,
    const int ndofs,
    const std::vector<DofT> &cell_dofs)
{
  const OctList &octants = dtree.getTreePartFiltered();
  size_t misses = 0;
  for (size_t i = 0; i < octants.size(); ++i)
  {
    const int level = cell_dofs[i * ndofs + 0] / DIM;
    assert(level >= 0);
    const Oct oct = octants[i].getAncestor(level);
    std::array<double, DIM> coords;
    double dummy;
    ot::treeNode2Physical(oct, coords.data(), dummy);
    const DofT sum = DIM * level + accumulate_sum(coords.begin(), coords.end());
    for (int dof = 0; dof < ndofs; ++dof)
      misses += (cell_dofs[i * ndofs + dof] != sum);
  }
  return misses;
}




// ============================================================

#include "intergridTransfer.h"

void oldIntergridTransfer(
    const ot::DistTree<uint, DIM> &from_dtree,
    const ot::DA<DIM> *from_da,
    const int ndofs,
    const std::vector<DofT> &from_local,
    const ot::DistTree<uint, DIM> &to_dtree,
    const ot::DA<DIM> *to_da,
    std::vector<DofT> &to_local)
{
  // Surrogate octree: Coarse by fine
  OctList surrogateOctree = ot::SFC_Tree<uint, DIM>::getSurrogateGrid(
      ot::SurrogateOutByIn,
      to_dtree.getTreePartFiltered(),    // this partition
      from_dtree.getTreePartFiltered(),  // this grid
      from_dtree.getComm());
  ot::DistTree<uint, DIM> surr_dtree(
      surrogateOctree, from_dtree.getComm(), ot::DistTree<uint, DIM>::NoCoalesce);
  surr_dtree.filterTree(from_dtree.getDomainDecider());

  /// ot::quadTreeToGnuplot(surr_dtree.getTreePartFiltered(), 8, "surr", from_dtree.getComm());

  // Surrogate DA
  ot::DA<DIM> *surr_da = new ot::DA<DIM>(
      surr_dtree,
      from_dtree.getComm(),
      from_da->getElementOrder());

  // DistShiftNodes: Coarse --> Surrogate
  std::vector<DofT> fine_ghost,  surr_ghost;
  to_da->createVector(fine_ghost, false, true, ndofs);
  surr_da->createVector(surr_ghost, false, true, ndofs);
  std::fill(fine_ghost.begin(), fine_ghost.end(), 0);
  std::fill(surr_ghost.begin(), surr_ghost.end(), 0);
  ot::distShiftNodes(
      *from_da, from_local.data(),
      *surr_da, surr_ghost.data() + surr_da->getLocalNodeBegin() * ndofs,
      ndofs);

  // Ghost read in surrogate
  surr_da->readFromGhostBegin(surr_ghost.data(), ndofs);
  surr_da->readFromGhostEnd(surr_ghost.data(), ndofs);

  // 2-loop, output to ghost
  fem::MeshFreeInputContext<DofT, Oct>
      inctx{ surr_ghost.data(),
             surr_da->getTNCoords(),
             surr_da->getTotalNodalSz(),
             surr_dtree.getTreePartFiltered().data(),
             surr_dtree.getTreePartFiltered().size(),
             ot::dummyOctant<DIM>(),
             ot::dummyOctant<DIM>() };
  fem::MeshFreeOutputContext<DofT, Oct>
      outctx{fine_ghost.data(),
             to_da->getTNCoords(),
             to_da->getTotalNodalSz(),
             to_dtree.getTreePartFiltered().data(),
             to_dtree.getTreePartFiltered().size(),
             ot::dummyOctant<DIM>(),
             ot::dummyOctant<DIM>() };

  // Note: Old versions required outDirty and writeToGhosts,
  // but now not needed because node partition respects element partition.

  /// std::vector<char> dirty;
  /// to_da->createVector(dirty, false, true, 1);
  /// std::fill(dirty.begin(), dirty.end(), 0);
  /// fem::locIntergridTransfer(inctx, outctx, ndofs, to_da->getReferenceElement(), dirty.data());
  /// to_da->writeToGhostsBegin(fine_ghost.data(), ndofs, dirty.data());
  /// to_da->writeToGhostsEnd(fine_ghost.data(), ndofs, false, dirty.data());

  fem::locIntergridTransfer(inctx, outctx, ndofs, to_da->getReferenceElement());

  // Extract ghost --> local
  to_da->ghostedNodalToNodalVec(fine_ghost, to_local, true, ndofs);

  delete surr_da;
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

