//
// Created by masado on 9/08/22.
//

#include <doctest/extensions/doctest_mpi.h> // include doctest before dendro

#include "test/octree/multisphere.h"

#include <include/treeNode.h>
#include <include/tsort.h>
#include <include/distTree.h> // for convenient uniform grid partition

#include <vector>

// -----------------------------
// Typedefs
// -----------------------------
using uint = unsigned int;
using LLU = long long unsigned;

template <int dim>
using Oct = ot::TreeNode<uint, dim>;

// -----------------------------
// Helper classes
// -----------------------------
struct SfcTableScope
{
  SfcTableScope(int dim) { _InitializeHcurve(dim); }
  ~SfcTableScope() { _DestroyHcurve(); }
};

template <typename T>
class LoadBalance
{
private:
  T m_local_load;
  T m_global_min;
  T m_global_sum;
  T m_global_max;
  int m_comm_size;
  int m_comm_rank;

public:
  LoadBalance(T local_load, MPI_Comm comm) : m_local_load(local_load)
  {
    MPI_Comm_size(comm, &m_comm_size);
    MPI_Comm_rank(comm, &m_comm_rank);
    par::Mpi_Allreduce(&local_load, &m_global_min, 1, MPI_MIN, comm);
    par::Mpi_Allreduce(&local_load, &m_global_sum, 1, MPI_SUM, comm);
    par::Mpi_Allreduce(&local_load, &m_global_max, 1, MPI_MAX, comm);
  }

  double ideal_load() const { return double(m_global_sum) / m_comm_size; }
  double overload_ratio() const { return m_global_max / ideal_load(); }
  double underload_ratio() const { return m_global_min / ideal_load(); }
  double local_ratio() const { return m_local_load / ideal_load(); }
};

// =============================================================================
// Test case
// =============================================================================
MPI_TEST_CASE("load balance 2D sphere-refine 5 process", 3)
{
  MPI_Comm comm = test_comm; // test_comm is a parameter supplied by test case

  constexpr int DIM = 2;
  using Oct = Oct<DIM>;
  const auto sfc_table_scope = SfcTableScope(DIM);

  // Define a refinement pattern using three overlapping spheres.
  test::SphereSet<DIM> sphereSet;
  sphereSet.carveSphere(0.10, {0.15, 0.6});
  sphereSet.carveSphere(0.10, {0.7, 0.4});
  sphereSet.carveSphere(0.210, {0.45, 0.55});
  const auto refine_octant = [&](const Oct &oct)
  {
    std::array<double, DIM> coords;
    double size;
    ot::treeNode2Physical(oct, coords.data(), size);
    return sphereSet(coords.data(), size) == ibm::INTERCEPTED;
  };

  // Create an input tree over the unit cube using the above refinement pattern.
  // Two subcases have different initial partitions but same refinement pattern.
  const int finest_level = 8;
  const int initial_level = 4;
  const double initial_sfc_tol = 0.3;
  std::vector<Oct> prev_input, final_input;

  // Initial distributed tree, before refinement.
  SUBCASE("initially uniform at level 4, partition=uniform with sfc_tol=0.3")
  {
    final_input = ot::DistTree<uint, DIM>::constructSubdomainDistTree(
                      initial_level, comm, initial_sfc_tol)
                      .getTreePartFiltered();
  }

  // Initial distributed tree, before refinement.
  SUBCASE("initially uniform at level 4, partition=gathered to process 0")
  {
    const int initial_level = 4;
    const double initial_sfc_tol = 0.3;

    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    if (comm_rank == 0)
      final_input = ot::DistTree<uint, DIM>::constructSubdomainDistTree(
                        initial_level, MPI_COMM_SELF, initial_sfc_tol)
                        .getTreePartFiltered();
  }

  // Refine the tree until finest_level.
  for (int level = 0; level < finest_level; ++level)
  {
    std::swap(prev_input, final_input);
    final_input.clear();
    for (Oct &oct : prev_input)
      if (refine_octant(oct))
        for (ot::sfc::ChildNum c(0); c < ot::nchild(DIM); ++c)
          final_input.push_back(oct.getChildMorton(c));
      else
        final_input.push_back(oct);
  }

  // (Sort) and re-partition the tree by sfc_tol.
  std::vector<Oct> sorted = final_input;
  const double sfc_tol = 0.3;
  ot::SFC_Tree<uint, DIM>::distTreeSort(sorted, sfc_tol, comm);
  REQUIRE(
      ot::checksum_octree(sorted, comm) ==
      ot::checksum_octree(final_input, comm));

  // Both endpoints of a partition could be fudged by sfc_tol, so multiply by 2.
  const double max_ratio = 1 + 2 * sfc_tol;
  const double min_ratio = 1 - 2 * sfc_tol;

  // Measure load imbalance and report to doctest.
  const LoadBalance<LLU> load_balance(sorted.size(), comm);
  CHECK(load_balance.local_ratio() <= max_ratio);
  CHECK(load_balance.local_ratio() >= min_ratio);
  WARN_FALSE(
      LoadBalance<LLU>(final_input.size(), comm)
          .overload_ratio() <= max_ratio);

  // Optional visualization with gnuplot.
#if 0
  static int subcase_id_hack = 0;
  ot::quadTreeToGnuplot(final_input, finest_level,
      "_output/case_" + std::to_string(subcase_id_hack) + "_input", comm);
  ot::quadTreeToGnuplot(sorted, finest_level,
      "_output/case_" + std::to_string(subcase_id_hack) + "_sorted", comm);
  ++subcase_id_hack;
#endif
}
