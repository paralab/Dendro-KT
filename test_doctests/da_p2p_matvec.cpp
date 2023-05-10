//
// Created by masado on 10/13/22.
//

#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro

/// #include <include/treeNode.h>
/// #include <include/tsort.h>
/// /// #include <include/distTree.h> // for convenient uniform grid partition
/// #include <include/sfc_search.h>

#include "include/parUtils.h"

#include "include/oda.h"
#include "include/da_p2p.hpp"
#include "FEM/include/matvec.h"

#include "test/octree/gaussian.hpp"
#include "test_doctests/constructors.hpp"

/// #include <vector>

namespace test
{
  template <int dim, typename T, typename Elemental>
  void matvec_ghosted(
      const ot::DistTree<uint32_t, dim> &dtree,
      const ot::DA<dim> &da,
      int ndofs,
      std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental);

  template <int dim, typename T, typename Elemental>
  void matvec_ghosted(
      const ot::DA_P2P<dim> &da,
      int ndofs,
      std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental);

  template <int dim, typename T, typename Elemental>
  void matvec_owned(
      const ot::DistTree<uint32_t, dim> &dtree,
      const ot::DA<dim> &da,
      int ndofs,
      const std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental);

  template <int dim, typename T, typename Elemental>
  void matvec_owned(
      const ot::DA_P2P<dim> &da,
      int ndofs,
      const std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental);
}



// =============================================================================
// Test cases
// =============================================================================
TEST_SUITE("Compare old and new DA matvec")
{
  template <int dim>
  ot::DistTree<uint32_t, dim> gaussian_grid(DendroLLU n_global_generators, MPI_Comm comm);

  // Ellipsoid definitions

  constexpr double radius(int axis)
  {
    if (axis == 0)
      return 0.35;
    else
      return 0.61 * radius(axis - 1);
  }

  template <int dim>
  constexpr double normalized_coordinate(double x, int axis)
  {
    return (x - 0.5) / radius(axis);
  }

  template <int dim>
  constexpr double distance_squared(const double *x)
  {
    double sum = 0.0;
    for (int d = 0; d < dim; ++d)
      sum += x[d] * x[d];
    return sum;
  }

  template <int dim>
  constexpr ibm::Partition ellipsoid(const double *x, double side)
  {
    // If all corners are exclusively within the ellipsoid, return OUT.
    bool corners_within = true;
    for (int corner = 0; corner < (1u << dim); ++corner)
    {
      double y[dim];
      for (int d = 0; d < dim; ++d)
      {
        double c_x = x[d] + side * bool((corner >> d) & 1);
        y[d] = normalized_coordinate<dim>(c_x, d);
      }
      corners_within &= (distance_squared<dim>(y) < 1.0);
    }
    if (corners_within)
      return ibm::Partition::OUT;

    // Check for intersection.
    // https://stackoverflow.com/a/4579192
    double dmin = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      const double x_minus_nrm = normalized_coordinate<dim>(x[d], d);
      const double x_plus_nrm = normalized_coordinate<dim>(x[d] + side, d);
      if (0.0 < x_minus_nrm)
        dmin += x_minus_nrm * x_minus_nrm;
      else if (0.0 > x_plus_nrm)
        dmin += x_plus_nrm * x_plus_nrm;
    }
    if (dmin <= 1.0)
      return ibm::Partition::INTERCEPTED;

    // Otherwise totally outside of the ellipsoid.
    return ibm::Partition::IN;
  };

  MPI_TEST_CASE("Quadratic matvec on gaussian grid", 3)
  {
    MPI_Comm comm = test_comm;
    const int comm_rank = par::mpi_comm_rank(comm);

    constexpr int dim = 3;
    using Octant = ot::TreeNode<uint32_t, dim>;
    _InitializeHcurve(dim);
    const double sfc_tol = 0.3;

    const DendroLLU n_global_generators = 1000;
    ot::DistTree<uint32_t, dim> dtree = gaussian_grid<dim>(n_global_generators, comm);

    std::string plot_name = "dtree.matvec";
    SUBCASE("complete tree")
    {
      plot_name += ".complete";
    }
    SUBCASE("incomplete tree")
    {
      dtree.filterTree(ellipsoid<dim>);
      plot_name += ".incomplete";
    }

    /// if (dim == 2)
    ///   ot::quadTreeToGnuplot(dtree.getTreePartFiltered(), 10, plot_name, comm);

    const int degree = 2;
    ot::DA<dim> old_da(dtree, comm, degree);
    ot::DA_P2P<dim> new_da(dtree, comm, degree);

    /// if (comm_rank == 0)
    /// {
    ///   DOCTEST_MESSAGE("getGlobalElementSz() == ", new_da.getGlobalElementSz());
    ///   DOCTEST_MESSAGE("getGlobalNodeSz() == ", new_da.getGlobalNodeSz());
    /// }

    const int ndofs = 1;
    std::vector<double> old_field,  old_result;
    std::vector<double> new_field,  new_result;
    old_da.createVector(old_field, false, true, ndofs);
    old_da.createVector(old_result, false, true, ndofs);
    new_da.createVector(new_field, false, true, ndofs);
    new_da.createVector(new_result, false, true, ndofs);
    auto old_result_local = ot::local_range(old_da, ndofs, old_result.data());
    auto new_result_local = ot::local_range(new_da, ndofs, new_result.data());

    /// // Simple fill with ones.
    /// std::fill(old_field.begin(), old_field.end(), 1.0);
    /// std::fill(new_field.begin(), new_field.end(), 1.0);

    const auto linear = [](const double *x, double *out) -> void {
      double sum = 0.0;
      for (int d = 0; d < dim; ++d)
        sum += x[d];
      *out = sum;
    };
    old_da.template setVectorByFunction<double>(old_field.data(), linear, false, true, ndofs);
    new_da.template setVectorByFunction<double>(new_field.data(), linear, false, true, ndofs);

    // Elemental operator.
    const auto identity = [npe = old_da.getNumNodesPerElement()](
        const double *in, double *out, unsigned ndofs, const double *x,
        double, bool) -> void
    {
      for (size_t i = 0; i < npe * ndofs; ++i)
        out[i] = in[i];
    };

    test::matvec_ghosted<dim>(dtree,
                              old_da, ndofs, old_field, old_result, identity);
    test::matvec_ghosted<dim>(new_da, ndofs, new_field, new_result, identity);

    const double old_sum = par::mpi_sum(std::accumulate(
          old_result_local.begin(), old_result_local.end(), double(0.0)), comm);
    const double new_sum = par::mpi_sum(std::accumulate(
          new_result_local.begin(), new_result_local.end(), double(0.0)), comm);

    /// if (comm_rank == 0)
    /// {
    ///   MESSAGE("new_sum == ", new_sum);
    /// }

    MPI_CHECK(0, new_sum == old_sum);
  }


  // gaussian_grid()
  template <int dim>
  ot::DistTree<uint32_t, dim> gaussian_grid(DendroLLU n_global_generators, MPI_Comm comm)
  {
    const double sfc_tol = 0.3;

    const int comm_size = par::mpi_comm_size(comm);
    const int comm_rank = par::mpi_comm_rank(comm);

    const DendroLLU local_begin = n_global_generators * comm_rank / comm_size;
    const DendroLLU local_end = n_global_generators * (comm_rank + 1) / comm_size;
    const DendroLLU local_octants = local_end - local_begin;

    std::vector<ot::TreeNode<uint32_t, dim>> generators =
      test::gaussian<uint32_t, dim>(
        local_begin, local_octants, test::Constructors<ot::TreeNode<uint32_t, dim>>{});

    ot::SFC_Tree<uint32_t, dim>::distTreeSort(generators, sfc_tol, comm);
    ot::SFC_Tree<uint32_t, dim>::distRemoveDuplicates(
        generators, sfc_tol, ot::SFC_Tree<uint32_t, dim>::RM_DUPS_AND_ANC, comm);

    std::vector<ot::TreeNode<uint32_t, dim>> octree;
    ot::SFC_Tree<uint32_t, dim>::distTreeBalancing(
        generators, octree, 1, sfc_tol, comm);

    ot::DistTree<uint32_t, dim> dtree(octree, comm);
    return dtree;
  }


}
// =============================================================================


namespace test
{
  // matvec_ghosted()  (DA, T *)
  template <int dim, typename T, typename Elemental>
  void matvec_ghosted(
      const ot::DistTree<uint32_t, dim> &dtree,
      const ot::DA<dim> &da,
      int ndofs,
      T *ghost_in,
      T *ghost_out,
      Elemental elemental)
  {
    // <--- preMatVec(ghost_in + local) here.
    da.readFromGhostBegin(ghost_in, ndofs);
    da.readFromGhostEnd(ghost_in, ndofs);

    fem::matvec<T, ot::TreeNode<uint32_t, dim>>(
        ghost_in, ghost_out, ndofs,
        da.getTNCoords(), da.getTotalNodalSz(), 
        dtree.getTreePartFiltered().data(),
        dtree.getTreePartFiltered().size(),
        {}, {},
        elemental,
        {}, da.getReferenceElement());

    da.writeToGhostsBegin(ghost_out, ndofs);
    da.writeToGhostsEnd(ghost_out, ndofs);
    // <--- postMatVec(ghost_out + local) here.
  }

  // matvec_ghosted()  (DA_P2P, T *)
  template <int dim, typename T, typename Elemental>
  void matvec_ghosted(
      const ot::DA_P2P<dim> &da,
      int ndofs,
      T *ghost_in,
      T *ghost_out,
      Elemental elemental)
  {
    // <--- preMatVec(ghost_in + local) here.
    da.readFromGhostBegin(ghost_in, ndofs);
    da.readFromGhostEnd(ghost_in, ndofs);

    fem::matvec<T, ot::TreeNode<uint32_t, dim>>(
        ghost_in, ghost_out, ndofs,
        da.getTNCoords(), da.getTotalNodalSz(), 
        da.local_cell_list().begin(),
        da.local_cell_list().size(),
        {}, {},
        elemental,
        {}, da.getReferenceElement());

    da.writeToGhostsBegin(ghost_out, ndofs);
    da.writeToGhostsEnd(ghost_out, ndofs);
    // <--- postMatVec(ghost_out + local) here.
  }

  // ---------------------------------------------------------------------------

  // matvec_owned()  (DA, T *)
  template <int dim, typename T, typename Elemental>
  void matvec_owned(
      const ot::DistTree<uint32_t, dim> &dtree,
      const ot::DA<dim> &da,
      int ndofs,
      const T *in,
      T *out,
      Elemental elemental)
  {
    std::vector<T> ghost_in, ghost_out;
    da.nodalVecToGhostedNodal(in, ghost_in.data(), false, ndofs);
    // <--- preMatVec(ghost_in + local) here.
    da.readFromGhostBegin(ghost_in.data(), ndofs);
    da.createVector(ghost_out, false, true, ndofs);
    da.readFromGhostEnd(ghost_in.data(), ndofs);

    fem::matvec<T, ot::TreeNode<uint32_t, dim>>(
        ghost_in.data(), ghost_out.data(), ndofs,
        da.getTNCoords(), da.getTotalNodalSz(), 
        dtree.getTreePartFiltered().data(),
        dtree.getTreePartFiltered().size(),
        {}, {},
        elemental,
        {}, da.getReferenceElement());

    da.writeToGhostsBegin(ghost_out.data(), ndofs);
    da.writeToGhostsEnd(ghost_out.data(), ndofs);
    // <--- postMatVec(ghost_out + local) here.
    da.ghostedNodalToNodalVec(ghost_out.data(), out, true, ndofs);
  }

  // matvec_owned()  (DA_P2P, T *)
  template <int dim, typename T, typename Elemental>
  void matvec_owned(
      const ot::DA_P2P<dim> &da,
      int ndofs,
      const T *in,
      T *out,
      Elemental elemental)
  {
    std::vector<T> ghost_in, ghost_out;
    da.nodalVecToGhostedNodal(in, ghost_in.data(), false, ndofs);
    // <--- preMatVec(ghost_in + local) here.
    da.readFromGhostBegin(ghost_in.data(), ndofs);
    da.createVector(ghost_out, false, true, ndofs);
    da.readFromGhostEnd(ghost_in.data(), ndofs);

    fem::matvec<T, ot::TreeNode<uint32_t, dim>>(
        ghost_in.data(), ghost_out.data(), ndofs,
        da.getTNCoords(), da.getTotalNodalSz(), 
        da.local_cell_list().begin(),
        da.local_cell_list().size(),
        {}, {},
        elemental,
        {}, da.getReferenceElement());

    da.writeToGhostsBegin(ghost_out.data(), ndofs);
    da.writeToGhostsEnd(ghost_out.data(), ndofs);
    // <--- postMatVec(ghost_out + local) here.
    da.ghostedNodalToNodalVec(ghost_out.data(), out, true, ndofs);
  }


  // ---------------------------------------------------------------------------

  // matvec_ghosted()  (DA, vector<T>&)
  template <int dim, typename T, typename Elemental>
  void matvec_ghosted(
      const ot::DistTree<uint32_t, dim> &dtree,
      const ot::DA<dim> &da,
      int ndofs,
      std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental)
  {
    matvec_ghosted<dim, T, Elemental>(dtree, da, ndofs, in.data(), out.data(), elemental);
  }

  // matvec_ghosted()  (DA_P2P, vector<T>&)
  template <int dim, typename T, typename Elemental>
  void matvec_ghosted(const ot::DA_P2P<dim> &da, int ndofs,
      std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental)
  {
    matvec_ghosted<dim, T, Elemental>(da, ndofs, in.data(), out.data(), elemental);
  }

  // ---------------------------------------------------------------------------

  // matvec_owned()  (DA, vector<T>&)
  template <int dim, typename T, typename Elemental>
  void matvec_owned(
      const ot::DistTree<uint32_t, dim> &dtree,
      const ot::DA<dim> &da,
      int ndofs,
      const std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental)
  {
    matvec_owned<dim, T, Elemental>(dtree, da, ndofs, in.data(), out.data(), elemental);
  }
 
  // matvec_owned()  (DA_P2P, vector<T>&)
  template <int dim, typename T, typename Elemental>
  void matvec_owned(
      const ot::DA_P2P<dim> &da,
      int ndofs,
      const std::vector<T> &in,
      std::vector<T> &out,
      Elemental elemental)
  {
    matvec_owned<dim, T, Elemental>(da, ndofs, in.data(), out.data(), elemental);
  }

  // ---------------------------------------------------------------------------


}



