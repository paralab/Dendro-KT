#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro
#include "directories.h"

#include <include/treeNode.h>
#include <include/distTree.h>
#include <IO/hexlist/json_hexlist.h>
#include <include/octUtils.h>
#include <test/octree/multisphere.h>

#include <FEM/include/coarseToFine.hpp>
#include <FEM/include/surrogate_cell_transfer.hpp>

#include <vector>
#include <array>
#include <petsc.h>

const static char * bin_dir = DENDRO_KT_DOCTESTS_BIN_DIR;

constexpr int DIM = 2;
using uint = unsigned int;
using DofT = double;

using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;

ot::DistTree<uint, DIM> make_dist_tree(size_t grain, double sfc_tol, MPI_Comm comm);

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

void testOldIntergridCells(
    const ot::DistTree<uint, DIM> &coarse_dtree,
    const ot::DA<DIM> *coarse_da,
    const int ndofs,
    const ot::DistTree<uint, DIM> &fine_dtree,
    const ot::DA<DIM> *fine_da);


void test_refine_many(MPI_Comm test_comm);

MPI_TEST_CASE("refine many", 1) { test_refine_many(test_comm); }
MPI_TEST_CASE("refine many", 2) { test_refine_many(test_comm); }
MPI_TEST_CASE("refine many", 10) { test_refine_many(test_comm); }
MPI_TEST_CASE("refine many", 30) { test_refine_many(test_comm); }

void test_refine_many(MPI_Comm test_comm)
{
  /// PetscInitialize(&argc, &argv, NULL, NULL);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = test_comm;
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
  coarse_dtree = make_dist_tree(grain, sfc_tol, comm);
  /// coarse_dtree = ot::DistTree<uint, DIM>::constructSubdomainDistTree(3, comm, sfc_tol);
  coarse_da = new DA(coarse_dtree, comm, degree, int{}, sfc_tol);

  printf("[%d] da size (e:%lu n:%lu dest:%d src:%d)\n", comm_rank, coarse_da->getLocalElementSz(), coarse_da->getLocalNodalSz(),
      coarse_da->getNumDestNeighbors(), coarse_da->getNumSrcNeighbors());

  std::vector<int> increase;
  const int amount = 2;
  const bool refine_all = false;
  increase.reserve(coarse_dtree.getTreePartFiltered().size());
  for (const Oct &oct : coarse_dtree.getTreePartFiltered())
    increase.push_back(oct.getIsOnTreeBdry() or refine_all ? amount : 0);
  coarse_dtree.distRefine(coarse_dtree, std::move(increase), fine_dtree, sfc_tol);
  fine_dtree = std::move(fine_dtree).repartitioned(sfc_tol);
  fine_da = new DA(fine_dtree, comm, degree, int{}, sfc_tol);

  printf("[%d] refined size (e:%lu n:%lu dest:%d src:%d)\n", comm_rank, fine_da->getLocalElementSz(), fine_da->getLocalNodalSz(),
      fine_da->getNumDestNeighbors(), fine_da->getNumSrcNeighbors());

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

  fem::coarse_to_fine(
      coarse_dtree, coarse_da, ndofs, cell_ndofs, coarse_local, coarse_cell_dofs,
      fine_dtree, fine_da, fine_local, fine_cell_dofs);

  const size_t node_misses = check_xpyp1(fine_dtree, fine_da, ndofs, fine_local);
  const size_t cell_misses = check_cell_xpypl(
      fine_dtree, cell_ndofs, fine_cell_dofs);

  CHECK(node_misses == 0);
  CHECK(cell_misses == 0);

  _DestroyHcurve();
  DendroScopeEnd();
  /// PetscFinalize();
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


