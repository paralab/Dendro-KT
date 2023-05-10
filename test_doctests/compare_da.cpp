
#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro

#include "include/oda.h"
#include "include/da_p2p.hpp"

DOCTEST_TEST_SUITE("Compare old and new DA")
{
  // Helper routines
  template <int dim>
  std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_central(int max_depth);
  template <int dim>
  std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_edges(int max_depth);

  auto make_next_comm_size(int max_comm_size) {
    auto &cout = std::cout;
    return [max_comm_size, &cout](int np) {
      if (np < max_comm_size)
      {
        if (((np-1) & np) == 0)                   // np is a power of 2
          np += 1;                                // -> power of 2 plus 1
        else if ((((np + 1)-1) & (np + 1)) == 0)  // np+1 is power of 2
          np += 1;                                // -> power of 2
        else                                      // np-1 is power of 2
          np = 2*(np - 1) - 1;                    // -> next power of 2 minus 1
        if (np > max_comm_size)                   // clamp
          np = max_comm_size;
        return np;
      }
      else
        return np + 1;
    };
  }

  DOCTEST_MPI_TEST_CASE("Number of nodes", 3)
  {
    constexpr int dim = 2;
    _InitializeHcurve(dim);
    const double sfc_tol = 0.3;
    using namespace ot;

    const auto next_comm_size = make_next_comm_size(test_nb_procs);
    for (int np = 1; np <= test_nb_procs; np = next_comm_size(np))
    {
      MPI_Comm comm;
      MPI_Comm_split(test_comm, test_rank < np, 0, &comm);
      if (test_rank >= np)
      {
        MPI_Comm_free(&comm);
        continue;
      }

      INFO("mpi_size=", np, "  mpi_rank=", par::mpi_comm_rank(comm));
      const bool is_root = par::mpi_comm_rank(comm) == 0;

      enum Pattern { central, edges };
      for (Pattern grid_pattern : {central, edges})
      {
        INFO("grid_pattern=", std::string(grid_pattern == central? "central" : "edges"));
        for (int max_depth = 2; max_depth <= 5; ++max_depth)
        {
          // Grid.
          std::vector<TreeNode<uint32_t, dim>> grid;
          if (is_root and grid_pattern == central)
            grid = grid_pattern_central<dim>(max_depth);
          else if (is_root and grid_pattern == edges)
            grid = grid_pattern_edges<dim>(max_depth);
          SFC_Tree<uint32_t, dim>::distTreeSort(grid, sfc_tol, comm);
          DistTree<uint32_t, dim> dtree(grid, comm);

          for (int degree: {1, 2, 3})
          {
            slow_da::DA<dim> old_da(dtree, comm, degree);
            DA_P2P<dim> new_da(dtree, comm, degree);

            INFO("max_depth=", max_depth, "  degree=", degree);

            CHECK( new_da.getReferenceElement()->getOrder()
                == old_da.getReferenceElement()->getOrder() );
            CHECK( new_da.getReferenceElement()->getDim()
                == old_da.getReferenceElement()->getDim() );
            CHECK( new_da.getReferenceElement()->get1DNumInterpolationPoints()
                == old_da.getReferenceElement()->get1DNumInterpolationPoints() );

            CHECK( new_da.getNumNodesPerElement() == old_da.getNumNodesPerElement() );
            CHECK( new_da.getElementOrder() == old_da.getElementOrder() );

            CHECK( new_da.getLocalElementSz() == old_da.getLocalElementSz() );
            CHECK( new_da.getGlobalElementSz() == old_da.getGlobalElementSz() );
            CHECK( new_da.getGlobalElementBegin() == old_da.getGlobalElementBegin() );

            CHECK( new_da.getLocalNodalSz() == old_da.getLocalNodalSz() );
            CHECK( new_da.getTotalNodalSz() == old_da.getTotalNodalSz() );

            CHECK( (new_da.getTotalNodalSz() - new_da.getLocalNodalSz())
                == (old_da.getTotalNodalSz() - old_da.getLocalNodalSz()) );
              // ^ May fail for hilbert curve ordering; then, make other test.

            CHECK( new_da.getGlobalNodeSz() == old_da.getGlobalNodeSz() );
            CHECK( new_da.getGlobalRankBegin() == old_da.getGlobalRankBegin() );

            // future: compare some kind of matvec
          }
        }
      }
      MPI_Comm_free(&comm);
    }

    _DestroyHcurve();
  }


  DOCTEST_MPI_TEST_CASE("Consistent ghost exchange", 3)
  {
    dbg::wait_for_debugger(test_comm);
    constexpr int dim = 2;
    _InitializeHcurve(dim);
    const double sfc_tol = 0.3;
    using namespace ot;

    const auto next_comm_size = make_next_comm_size(test_nb_procs);
    for (int np = 1; np <= test_nb_procs; np = next_comm_size(np))
    {
      MPI_Comm comm;
      MPI_Comm_split(test_comm, test_rank < np, 0, &comm);
      if (test_rank >= np)
      {
        MPI_Comm_free(&comm);
        continue;
      }

      INFO("mpi_size=", np, "  mpi_rank=", par::mpi_comm_rank(comm));
      const bool is_root = par::mpi_comm_rank(comm) == 0;

      enum Pattern { central, edges };
      for (Pattern grid_pattern : {central, edges})
      {
        INFO("grid_pattern=", std::string(grid_pattern == central? "central" : "edges"));
        for (int max_depth = 2; max_depth <= 5; ++max_depth)
        {
          // Grid.
          std::vector<TreeNode<uint32_t, dim>> grid;
          if (is_root and grid_pattern == central)
            grid = grid_pattern_central<dim>(max_depth);
          else if (is_root and grid_pattern == edges)
            grid = grid_pattern_edges<dim>(max_depth);
          SFC_Tree<uint32_t, dim>::distTreeSort(grid, sfc_tol, comm);
          DistTree<uint32_t, dim> dtree(grid, comm);

          /// for (int degree: {1, 2, 3})  // node ordering different for 2+.
          for (int degree: {1})
          {
            slow_da::DA<dim> old_da(dtree, comm, degree);
            DA_P2P<dim> new_da(dtree, comm, degree);

            INFO("max_depth=", max_depth, "  degree=", degree);

            std::vector<int> old_ghost_read, old_ghost_write;
            std::vector<int> new_ghost_read, new_ghost_write;
            const int unit = intPow(100, par::mpi_comm_rank(comm));
            for (int ndofs: {1, 2})
            {
              old_da.createVector(old_ghost_read, false, true, ndofs);
              old_da.createVector(old_ghost_write, false, true, ndofs);
              new_da.createVector(new_ghost_read, false, true, ndofs);
              new_da.createVector(new_ghost_write, false, true, ndofs);
              for (auto *v: {&old_ghost_read, &old_ghost_write,
                             &new_ghost_read, &new_ghost_write})
                std::generate(v->begin(), v->end(),
                    [=, i=0]() mutable { return unit * (i++); });

              old_da.readFromGhostBegin(old_ghost_read.data(), ndofs);
              old_da.readFromGhostEnd(old_ghost_read.data(), ndofs);
              old_da.writeToGhostsBegin(old_ghost_write.data(), ndofs);
              old_da.writeToGhostsEnd(old_ghost_write.data(), ndofs);
              new_da.readFromGhostBegin(new_ghost_read.data(), ndofs);
              new_da.readFromGhostEnd(new_ghost_read.data(), ndofs);
              new_da.writeToGhostsBegin(new_ghost_write.data(), ndofs);
              new_da.writeToGhostsEnd(new_ghost_write.data(), ndofs);

              CHECK( new_ghost_read == old_ghost_read );
              CHECK( new_ghost_write == old_ghost_write );
            }

            // isDirtyOut   //future: Deprecate this feature
            for (int ndofs: {1, 2})
            {
              old_da.createVector(old_ghost_write, false, true, ndofs);
              new_da.createVector(new_ghost_write, false, true, ndofs);
              for (auto *v: {&old_ghost_write, &new_ghost_write})
                std::generate(v->begin(), v->end(),
                    [=, i=0]() mutable { return unit * (i++); });

              std::vector<char> write_odd(old_ghost_write.size(), false);
              for (size_t i = 1; i < write_odd.size(); i += 2)
                write_odd[i] = true;

              old_da.writeToGhostsBegin(old_ghost_write.data(), ndofs, write_odd.data());
              old_da.writeToGhostsEnd(old_ghost_write.data(), ndofs, true, write_odd.data());
              new_da.writeToGhostsBegin(new_ghost_write.data(), ndofs, write_odd.data());
              new_da.writeToGhostsEnd(new_ghost_write.data(), ndofs, write_odd.data());
              CHECK( new_ghost_write == old_ghost_write );
            }
          }
        }
      }
      MPI_Comm_free(&comm);
    }

    _DestroyHcurve();
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
}

