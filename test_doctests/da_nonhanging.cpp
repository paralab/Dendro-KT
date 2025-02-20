//
// Created by masado on 9/08/22.
//

#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro

/// #include "test/octree/multisphere.h"

#include <include/distTree.h> // for convenient uniform grid partition
#include <include/oda.h>

#include <test/testAdaptiveExamples.h>

#include <vector>
#include <limits>
#include <algorithm>


using uint = unsigned int;
using LLU = long long unsigned;

template <int dim>
using Oct = ot::TreeNode<uint, dim>;

struct SfcTableScope
{
  SfcTableScope(int dim) { _InitializeHcurve(dim); }
  ~SfcTableScope() { _DestroyHcurve(); }
};




// =============================================================================
// Test case
// =============================================================================
MPI_TEST_CASE("Uniform grid API sequential global node index vector is 1 to n", 1)
{
  MPI_Comm comm = test_comm;  // test_comm is a parameter supplied by test case
  DendroScopeBegin();
  constexpr int DIM = 3;
  using Oct = Oct<DIM>;
  const SfcTableScope sfc_table_scope = {DIM};

  const int refinement_level = 7;

  std::cout << "DIM == " << DIM << ". "
            << "refinement_level == " << refinement_level << ".\n";
  std::cout << "pow(pow(2, refinement_level), DIM) == "
            << LLU(std::pow(2, refinement_level * DIM)) << ".\n";
  std::cout << "pow(pow(2, refinement_level) + 1, DIM) == "
            << LLU(std::pow(std::pow(2, refinement_level) + 1, DIM)) << ".\n";

  const double sfc_tolerance = 0.3;
  const auto tree = ot::DistTree<uint, DIM>::constructSubdomainDistTree(
      refinement_level, comm, sfc_tolerance);
  const auto da = ot::DA<DIM>(
      tree, /*stratum=*/0, comm, /*order=*/1, {}, sfc_tolerance, /*version=*/1);

  const auto & global_node_ids = da.getNodeLocalToGlobalMap();
  std::cout << "global_node_ids.size() == " << global_node_ids.size() << "\n";
  for (LLU node = 0; node < global_node_ids.size(); ++node)
  {
    CHECK(global_node_ids[node] == node);
  }
  DendroScopeEnd();
}


// =============================================================================
// Test case
// =============================================================================
MPI_TEST_CASE("Nonuniform grid API sequential global node index vector is 1 to n", 1)
{
  MPI_Comm comm = test_comm;  // test_comm is a parameter supplied by test case
  DendroScopeBegin();
  constexpr int DIM = 3;
  using Oct = Oct<DIM>;
  const SfcTableScope sfc_table_scope = {DIM};

  const int refinement_level = 7;

  std::cout << "DIM == " << DIM << ". "
            << "refinement_level == " << refinement_level << " (nonuniform).\n";
  std::cout << "Bound points ["
            << Example3<DIM>::num_points(refinement_level, /*Q1*/1) << " .. "
            << Example3<DIM>::num_points(refinement_level, /*Q2*/2) << "].\n";

  std::vector<Oct> tree_nodes;
  Example3<DIM>::fill_tree(refinement_level, tree_nodes);

  const double sfc_tolerance = 0.3;
  const auto tree = ot::DistTree<uint, DIM>(tree_nodes, comm);
  const auto da = ot::DA<DIM>(
      tree, /*stratum=*/0, comm, /*order=*/1, {}, sfc_tolerance, /*version=*/1);

  const auto & global_node_ids = da.getNodeLocalToGlobalMap();
  std::cout << "global_node_ids.size() == " << global_node_ids.size() << "\n";
  for (LLU node = 0; node < global_node_ids.size(); ++node)
  {
    CHECK(global_node_ids[node] == node);
  }
  DendroScopeEnd();
}





#include <include/matRecord.h>
#include <FEM/include/feMatrix.h>

template <int dim>
struct DummyMatrix : public feMatrix<DummyMatrix<dim>, dim>
{
  DummyMatrix(
      const ot::DA<dim>* da,
      const std::vector<ot::TreeNode<unsigned int, dim>> *octList,
      unsigned int dof)
    :
      feMatrix<DummyMatrix<dim>, dim>(da, octList, dof)
  { }

  void getElementalMatrix(
      std::vector<ot::MatRecord> &records,
      const double *coords,
      bool isElementBoundary)
  {
    constexpr int max_nodes_per_element = intPow(3, dim);
    const int ndofs = this->ndofs();
    const int n = ndofs * max_nodes_per_element;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        records.push_back(ot::MatRecord(i/ndofs, j/ndofs, i%ndofs, j%ndofs, 0.0));
  }
};

// =============================================================================
// Test case
// =============================================================================
MPI_TEST_CASE("Nonuniform grid matrix sequential indices in range", 1)
{
  MPI_Comm comm = test_comm;  // test_comm is a parameter supplied by test case
  DendroScopeBegin();
  constexpr int DIM = 3;
  using Oct = Oct<DIM>;
  const SfcTableScope sfc_table_scope = {DIM};

  const int refinement_level = 7;

  std::cout << "DIM == " << DIM << ". "
            << "refinement_level == " << refinement_level << " (nonuniform).\n";
  std::cout << "Bound points ["
            << Example3<DIM>::num_points(refinement_level, /*Q1*/1) << " .. "
            << Example3<DIM>::num_points(refinement_level, /*Q2*/2) << "].\n";

  std::cout << "Create mesh.\n";
  std::vector<Oct> tree_nodes;
  Example3<DIM>::fill_tree(refinement_level, tree_nodes);
  const double sfc_tolerance = 0.3;
  const auto tree = ot::DistTree<uint, DIM>(tree_nodes, comm);
  const auto da = ot::DA<DIM>(
      tree, /*stratum=*/0, comm, /*order=*/1, {}, sfc_tolerance, /*version=*/1);

  std::cout << "da.getGlobalNodeSz() == " << da.getGlobalNodeSz() << "\n";

  const int ndofs = 3;
  const LLU global_n_dofs = ndofs * da.getGlobalNodeSz();

  std::cout << "Create dummy unassembled matrix.\n";

  DummyMatrix<DIM> dummy_matrix(&da, &tree.getTreePartFiltered(), ndofs);
  LLU minimum_index = std::numeric_limits<decltype(global_n_dofs)>::max();
  LLU maximum_index = std::numeric_limits<decltype(global_n_dofs)>::min();
  dummy_matrix.collectMatrixEntries( [&] (
        const std::vector<PetscInt>& rowIdxBuffer,
        const std::vector<PetscScalar> & colValBuffer )
      {
        for (PetscInt idx : rowIdxBuffer)
        {
          minimum_index = std::min<LLU>(minimum_index, idx);
          maximum_index = std::max<LLU>(maximum_index, idx);
        }
      });

  std::cout << "global_n_dofs == " << global_n_dofs << ".\n";
  std::cout << "minimum_index == " << minimum_index << ".\n";
  std::cout << "maximum_index == " << maximum_index << ".\n";
  CHECK(minimum_index < global_n_dofs);
  CHECK(maximum_index < global_n_dofs);
  DendroScopeEnd();
}


