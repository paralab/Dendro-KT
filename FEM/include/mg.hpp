
/**
 * @author Masado Ishii
 * @date 2023-05-24
 * @brief Functions related to multigrid
 */

#ifndef DENDRO_KT_FEM_MG_HPP
#define DENDRO_KT_FEM_MG_HPP

#include "include/oda.h"

#include "FEM/include/intergridTransfer.h"

#include <map>
#include <iostream>

#include <petsc.h>

namespace detail
{
  template <typename X>
  class VectorPool
  {
    public:
      VectorPool() = default;

      ~VectorPool();
      void free_all();
      std::vector<X> checkout(size_t size);
      void checkin(std::vector<X> &&vec);
        //idea: multimap find, assign moved, erase empty, return assigned; if not found, return new.

      VectorPool(VectorPool && other) : m_idle(std::move(other.m_idle)) {}
      VectorPool & operator=(VectorPool &&other) { m_idle = std::move(other.m_idle); return *this; }
      VectorPool(const VectorPool &) = delete;
      VectorPool & operator=(const VectorPool &) = delete;

      void print(bool print) { m_print = print; }
      bool print() const { return m_print; }

    private:
      std::multimap<int, std::vector<X>> m_idle;
      long long unsigned m_log_checkouts = 0;
      long long unsigned m_log_allocs = 0;
      bool m_print = false;
  };

  // matvec_base_in()
  template <typename X, unsigned dim>
  ot::MatvecBaseIn<dim, X> matvec_base_in(
      const ot::DA<dim> *da, int ndofs, const X *ghosted, int extra_depth = 0);

  // matvec_base_ou_accumulatet()
  template <typename X, unsigned dim>
  ot::MatvecBaseOut<dim, X, true> matvec_base_out_accumulate(
      const ot::DA<dim> *da, int ndofs, int extra_depth = 0);
}


namespace mg
{
  // DA_Pair
  template <int dim>
  struct DA_Pair
  {
    const ot::DA<dim> *primary;
    const ot::DA<dim> *surrogate;
  };

  // CycleSettings
  struct CycleSettings
  {
    int n_grids()        const { return m_n_grids; }
    int pre_smooth()     const { return m_pre_smooth; }
    int post_smooth()    const { return m_post_smooth; }
    double damp_smooth() const { return m_damp_smooth; }
    bool print()         const { return m_print; }

    void n_grids(int n_grids)          { m_n_grids = n_grids; }
    void pre_smooth(int pre_smooth)      { m_pre_smooth = pre_smooth; }
    void post_smooth(int post_smooth)    { m_post_smooth = post_smooth; }
    void damp_smooth(double damp_smooth) { m_damp_smooth = damp_smooth; }
    void print(bool print)               { m_print = print; }

    int m_n_grids = 2;
    int m_pre_smooth = 1;
    int m_post_smooth = 1;
    double m_damp_smooth = 1.0;
    bool m_print = false;
  };


  // restrict_fine_to_coarse()
  template <int dim, typename PostRestriction>
  void restrict_fine_to_coarse(
      DA_Pair<dim> fine_da, const VECType *fine_vec_ghosted,
      DA_Pair<dim> coarse_da, VECType *coarse_vec_ghosted,
      PostRestriction post_restriction,  // on local vector
      int ndofs,
      detail::VectorPool<VECType> &vector_pool);

  // prolongate_coarse_to_fine()
  template <int dim, typename PreProlongation>
  void prolongate_coarse_to_fine(
      DA_Pair<dim> coarse_da, const VECType *coarse_vec_ghosted,
      DA_Pair<dim> fine_da, VECType *fine_vec_ghosted,
      PreProlongation pre_prolongation,  // on local vector
      int ndofs,
      detail::VectorPool<VECType> &vector_pool);


  // VCycle
  template <typename MatType>
  struct VCycle
  {
    public:
      static constexpr int dim()
      {
        return std::remove_pointer_t<decltype(std::declval<MatType>().da())>::template_dim;
      }

    // Temporary data
    public:
      // Inputs
      MatType * *mats = nullptr;
      CycleSettings settings = {};
      int ndofs = 1;

      std::vector<const ot::DA<dim()> *> surrogate_das;

      // PETSc
      Mat coarse_mat;
      Vec coarse_u, coarse_rhs;
      KSP coarse_ksp;
      PC coarse_pc;

      // Memory resources
      detail::VectorPool<double> vector_pool;
      std::vector<std::vector<double>> e_ghosted;
      std::vector<std::vector<double>> r_ghosted;
      std::vector<std::vector<double>> u_ghosted;
      std::vector<std::vector<double>> a_diag_ghosted;

    // Public methods
    public:
      VCycle(const VCycle &) = delete;
      VCycle(VCycle &&) = default;

      int n_grids() const { return settings.n_grids(); }

      VCycle(
          const std::vector<DA_Pair<dim()>> &da_pairs,
          MatType * *mats, CycleSettings settings, int ndofs)
        : mats(mats), settings(settings), ndofs(ndofs),
          r_ghosted(settings.n_grids()),
          u_ghosted(settings.n_grids()),
          e_ghosted(settings.n_grids()),
          a_diag_ghosted(settings.n_grids())
      {
        // Save pointers to surrogate DAs. (Primary DAs already stored in mats).
        surrogate_das.reserve(n_grids());
        for (DA_Pair<dim()> pair: da_pairs)
          surrogate_das.push_back(pair.surrogate);
        assert(surrogate_das.size() == n_grids());
        assert(surrogate_das[0] == nullptr);

        // Allocate temporary ghosted vectors.
        for (int g = 0; g < n_grids(); ++g)
        {
          const auto *da = mats[g]->da();
          u_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
          r_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
          e_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
        }

        // Initialilze a_diag_ghosted.
        for (int g = 0; g < n_grids(); ++g)
        {
          const auto *da = mats[g]->da();
          a_diag_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
          mats[g]->setDiag(a_diag_ghosted[g].data() + da->getLocalNodeBegin() * ndofs);
          da->readFromGhostBegin(a_diag_ghosted[g].data(), ndofs);
          da->readFromGhostEnd(a_diag_ghosted[g].data(), ndofs);
        }

        const MPI_Comm fine_comm = mats[0]->da()->getCommActive();
        if (par::mpi_comm_rank(fine_comm) == 0)
          vector_pool.print(true);

        // ===========================================================================
        // Try KSPSolve() with Coarse grid system.
        // ===========================================================================
        const auto *coarse_da = mats[n_grids() - 1]->da();

        if (coarse_da->isActive())
        {
          /// auto dbg_region = debug::global_comm_log->declare_region(coarse_da->getCommActive(), COMMLOG_CONTEXT);

          // Assemble the coarse grid matrix (assuming one-time assembly).
          coarse_da->createMatrix(coarse_mat, MATAIJ, ndofs);
          mats[n_grids() - 1]->getAssembledMatrix(&coarse_mat, {});
          MatAssemblyBegin(coarse_mat, MAT_FINAL_ASSEMBLY);
          MatAssemblyEnd(coarse_mat, MAT_FINAL_ASSEMBLY);

          // Placeholder Vec's to which user array will be bound in coarse grid solve.
          const MPI_Comm coarse_comm = coarse_da->getCommActive();
          const size_t coarse_local_size = coarse_da->getLocalNodalSz() * ndofs;
          const size_t coarse_global_size = coarse_da->getGlobalNodeSz() * ndofs;
          VecCreateMPIWithArray(coarse_comm, 1, coarse_local_size, coarse_global_size, nullptr, &coarse_u);
          VecCreateMPIWithArray(coarse_comm, 1, coarse_local_size, coarse_global_size, nullptr, &coarse_rhs);

          std::vector<int> coarse_global_ids_of_local_boundary_nodes(coarse_da->getBoundaryNodeIndices().size());
          std::copy(coarse_da->getBoundaryNodeIndices().cbegin(),
                    coarse_da->getBoundaryNodeIndices().cend(),
                    coarse_global_ids_of_local_boundary_nodes.begin());
          for (int &id : coarse_global_ids_of_local_boundary_nodes)
            id += coarse_da->getGlobalRankBegin();
          // If ndofs != 1 then need to duplicate and zip.

          MatZeroRows(
              coarse_mat,
              coarse_global_ids_of_local_boundary_nodes.size(),
              coarse_global_ids_of_local_boundary_nodes.data(),
              1.0,
              NULL, NULL);

          /// MatView(coarse_mat, PETSC_VIEWER_STDOUT_(coarse_da->getCommActive()));

          // Coarse solver setup with PETSc.
          KSPCreate(coarse_da->getCommActive(), &coarse_ksp);
          KSPSetOperators(coarse_ksp, coarse_mat, coarse_mat);
          KSPSetTolerances(coarse_ksp, 1.0e-14, 0.0, 10.0, 50);
          /// KSPSetType(coarse_ksp, KSPPREONLY);  // Do not use iteration.
          KSPGetPC(coarse_ksp, &coarse_pc);
          PCSetType(coarse_pc, PCGAMG);  // "Direct" solver choice.
          KSPSetUp(coarse_ksp);
        }
      }

      ~VCycle()
      {
        if (mats[n_grids()-1]->da()->isActive())
        {
          /// auto dbg_region = debug::global_comm_log->declare_region(mats[n_grids()-1]->da()->getCommActive(), COMMLOG_CONTEXT);

          VecDestroy(&coarse_u);
          VecDestroy(&coarse_rhs);
          KSPDestroy(&coarse_ksp);
          MatDestroy(&coarse_mat);
        }
      }

      int coarse_solver(double *u_local, const double *rhs_local)
      {
        if (not mats[n_grids()-1]->da()->isActive())
          return 0;

        /// auto dbg_region = debug::global_comm_log->declare_region(mats[n_grids()-1]->da()->getCommActive(), COMMLOG_CONTEXT);

        // Bind u and rhs to Vec
        VecPlaceArray(coarse_u, u_local);
        VecPlaceArray(coarse_rhs, rhs_local);

        // Solve.
        KSPSolve(coarse_ksp, coarse_rhs, coarse_u);

        // Debug for solution magnitude.
        PetscReal sol_norm = 0.0;
        VecNorm(coarse_u, NORM_INFINITY, &sol_norm);

        // Unbind from Vec.
        VecResetArray(coarse_u);
        VecResetArray(coarse_rhs);

        // Get residual norm and number of iterations.
        int iterations = 0;
        PetscReal res_norm = 0.0;
        KSPGetIterationNumber(coarse_ksp, &iterations);
        KSPGetResidualNorm(coarse_ksp, &res_norm);
        const int rank = par::mpi_comm_rank(this->mats[0]->da()->getCommActive());
        if (this->settings.print() and rank == 0)
        {
          std::cout
            << "\t"
            << "iterations=" << iterations
            << "  res=" << res_norm
            << "  sol=" << sol_norm << "\n";
        }
        return iterations;
      }

      void pre_smoother(int height, double *u_ghosted, double *r_ghosted)
      {
        jacobi(
            this->mats[height],
            u_ghosted,
            r_ghosted,
            this->settings.damp_smooth(),
            this->a_diag_ghosted[height].data());
      }

      void post_smoother(int height, double *u_ghosted, double *r_ghosted)
      {
        jacobi(
            this->mats[height],
            u_ghosted,
            r_ghosted,
            this->settings.damp_smooth(),
            this->a_diag_ghosted[height].data());
      }

      void vcycle(double *u_local, double *r_local)
      {
        const auto &base_da = *this->mats[0]->da();
        const int ndofs = this->ndofs;
        const int n_grids = this->n_grids();
        const int pre_smooth = this->settings.pre_smooth();
        const int post_smooth = this->settings.post_smooth();
        const double damp = this->settings.damp_smooth();

        std::vector<std::vector<double>> &r_ghosted = this->r_ghosted;
        std::vector<std::vector<double>> &u_ghosted = this->u_ghosted;
        MatType * *mats = this->mats;

        // Multigrid v-cycle
        //  Parameters: [in-out] u  initial guess and final guess
        //              [in-out] r  residual of initial guess and of final guess

        // Copy u, r to u_ghosted, r_ghosted.
        base_da.nodalVecToGhostedNodal(r_local, r_ghosted[0].data(), ndofs);
        base_da.nodalVecToGhostedNodal(u_local, u_ghosted[0].data(), ndofs);
        //
        base_da.readFromGhostBegin(r_ghosted[0].data(), ndofs);
        base_da.readFromGhostBegin(u_ghosted[0].data(), ndofs);
        base_da.readFromGhostEnd(r_ghosted[0].data(), ndofs);
        base_da.readFromGhostEnd(u_ghosted[0].data(), ndofs);

        // reset initial guess for coarser grids to 0.
        for (int height = 1; height < n_grids; ++height)
          u_ghosted[height].assign(u_ghosted[height].size(), 0);

        for (int height = 0; height < n_grids - 1; ++height)
        {
          // pre-smoothing (on ghosted vectors)
          for (int i = 0; i < pre_smooth; ++i)
            this->pre_smoother(height, u_ghosted[height].data(), r_ghosted[height].data());

          // restriction (fine-to-coarse) (on ghosted vectors)
          restrict_fine_to_coarse<dim()>(
              {mats[height]->da(), this->surrogate_das[height]}, r_ghosted[height].data(),
              {mats[height+1]->da(), this->surrogate_das[height+1]}, r_ghosted[height+1].data(),
              [mat=mats[height+1]](double *vec) { mat->postMatVec(vec, vec); },
              ndofs, vector_pool);
        }

        // Coarse solve
        // Direct method.
        const int coarse_steps = this->coarse_solver(
            u_ghosted[n_grids-1].data() + mats[n_grids-1]->da()->getLocalNodeBegin() * ndofs,
            r_ghosted[n_grids-1].data() + mats[n_grids-1]->da()->getLocalNodeBegin() * ndofs);

        mats[n_grids-1]->da()->readFromGhostBegin(u_ghosted[n_grids-1].data(), ndofs);
        mats[n_grids-1]->da()->readFromGhostEnd(u_ghosted[n_grids-1].data(), ndofs);

        for (int height = n_grids - 1; height > 0; --height)
        {
          // prolongation (coarse-to-fine) (on ghosted vectors)
          prolongate_coarse_to_fine<dim()>(
              {mats[height]->da(), this->surrogate_das[height]}, u_ghosted[height].data(),
              {mats[height-1]->da(), this->surrogate_das[height-1]}, e_ghosted[height-1].data(),
              [mat=mats[height]](double *vec) { mat->preMatVec(vec, vec); },
              ndofs, vector_pool);

          // Accumulate into u[h-1] and r[h-1]
          for (size_t i = 0; i < mats[height-1]->da()->getTotalNodalSz() * ndofs; ++i)
            u_ghosted[height-1][i] += e_ghosted[height-1][i];
          //future: matVecGhosted directly on fresh ghosted data
          mats[height-1]->matVec(
              e_ghosted[height-1].data() + mats[height-1]->da()->getLocalNodeBegin() * ndofs,
              e_ghosted[height-1].data() + mats[height-1]->da()->getLocalNodeBegin() * ndofs);
          for (size_t i = 0; i < mats[height-1]->da()->getTotalNodalSz() * ndofs; ++i)
            r_ghosted[height-1][i] -= e_ghosted[height-1][i];

          // post-smoothing (on ghosted vectors)
          for (int i = 0; i < post_smooth; ++i)
            this->post_smoother(height - 1, u_ghosted[height-1].data(), r_ghosted[height-1].data());
        }

        // Copy u_ghosted, r_ghosted to u, r.
        base_da.ghostedNodalToNodalVec(r_ghosted[0].data(), r_local, ndofs);
        base_da.ghostedNodalToNodalVec(u_ghosted[0].data(), u_local, ndofs);
      }


    // Private methods
    private:

      // Jacobi relaxation (one iteration)
      void jacobi(MatType *mat, double *u_ghost, double *r_ghost, const double damp, const double *diag_ghosted)
      {
        const auto *da = mat->da();
        std::vector<double> Dinv_r = this->vector_pool.checkout(da->getTotalNodalSz() * this->ndofs);

        for (size_t i = 0; i < da->getTotalNodalSz() * this->ndofs; ++i)
        {
          const double update = damp / diag_ghosted[i] * r_ghost[i];
          Dinv_r[i] = update;
          u_ghost[i] += update;
        }

        // future: matVecGhosted directly on fresh ghosted data
        mat->matVec(
            Dinv_r.data() + da->getLocalNodeBegin() * this->ndofs,
            Dinv_r.data() + da->getLocalNodeBegin() * this->ndofs);
        auto & ADinv_r = Dinv_r;
        da->readFromGhostBegin(ADinv_r.data(), this->ndofs);
        da->readFromGhostEnd(ADinv_r.data(), this->ndofs);

        for (size_t i = 0; i < da->getTotalNodalSz() * this->ndofs; ++i)
          r_ghost[i] -= ADinv_r[i];

        this->vector_pool.checkin(std::move(Dinv_r));
      };



  };//class VCycle

}//namespace mg



namespace mg
{
  // restrict_fine_to_coarse()
  template <int dim, typename PostRestriction>
  void restrict_fine_to_coarse(
      DA_Pair<dim> fine_da, const VECType *fine_vec_ghosted,
      DA_Pair<dim> coarse_da, VECType *coarse_vec_ghosted,
      PostRestriction post_restriction,
      int ndofs,
      detail::VectorPool<VECType> &vector_pool)
  {
    const unsigned int nPe = fine_da.primary->getNumNodesPerElement();

    std::vector<VECType> coarse_surr_ghosted = vector_pool.checkout(
        coarse_da.surrogate->getTotalNodalSz() * ndofs);

    std::vector<VECType> leafBuffer = vector_pool.checkout(ndofs * nPe);
    leafBuffer.assign(leafBuffer.size(), 42);

    // Surrogate is coarse grid partitioned by fine
    // Interpolate^T from the fine primary grid to coarse surrogate.

    // readFromGhost*(fine_vec_ghosted) must precede restrict_fine_to_coarse().

    // Fine ghosted elemental owners.
    using OwnershipT = DendroIntL;
    const OwnershipT * ownersGhostedPtr = fine_da.primary->getNodeOwnerElements();

    if (fine_da.primary->getLocalNodalSz() > 0)
    {
      /// auto dbg_region = debug::global_comm_log->declare_region(fine_da.primary->getCommActive(), COMMLOG_CONTEXT);

      // Index fine grid elements as we loop.
      OwnershipT globElementId = fine_da.primary->getGlobalElementBegin();

      unsigned eleOrder = fine_da.primary->getElementOrder();

      // Fine and coarse element-to-node loops.
      ot::MatvecBaseIn<dim, OwnershipT>
          loopOwners = detail::matvec_base_in(fine_da.primary, ndofs, ownersGhostedPtr, 0);
      ot::MatvecBaseIn<dim, VECType>
          loopFine = detail::matvec_base_in(fine_da.primary, ndofs, fine_vec_ghosted, 0);
      ot::MatvecBaseOut<dim, VECType, true>
          loopCoarse = detail::matvec_base_out_accumulate<VECType>(coarse_da.surrogate, ndofs, 1);

      // Traverse fine and coarse grids simultaneously.
      while (!loopFine.isFinished())
      {
        // Depth controlled by fine.
        if (loopFine.isPre() && loopFine.subtreeInfo().isLeaf())
        {
          const VECType * fineLeafIn = loopFine.subtreeInfo().readNodeValsIn();
          const OwnershipT * fineOwners = loopOwners.subtreeInfo().readNodeValsIn();
          for (size_t nIdx = 0; nIdx < nPe; ++nIdx)
          {
            if (loopFine.subtreeInfo().readNodeNonhangingIn()[nIdx])
            {
              // Only transfer a node to parent from the owning element.
              if (fineOwners[nIdx] == globElementId)
              {
                for (int dof = 0; dof < ndofs; ++dof)
                  leafBuffer[ndofs * nIdx + dof] = fineLeafIn[ndofs * nIdx + dof];
              }
              else
              {
                for (int dof = 0; dof < ndofs; ++dof)
                  leafBuffer[ndofs * nIdx + dof] = 0;
              }
            }
            else
            {
              for (int dof = 0; dof < ndofs; ++dof)
                leafBuffer[ndofs * nIdx + dof] = 0.0f;
            }
          }

          const size_t written = loopCoarse.subtreeInfo().overwriteNodeValsOut(leafBuffer.data());

          loopFine.next();
          loopCoarse.next();
          loopOwners.next();

          globElementId++;
        }
        else
        {
          loopFine.step();
          loopCoarse.step();
          loopOwners.step();
        }
      }
      const size_t writtenSz = loopCoarse.finalize(coarse_surr_ghosted.data());
    }

    // Coarse ghost write.
    {
      /// auto dbg_region = debug::global_comm_log->declare_region(coarse_da.surrogate->getCommActive(), COMMLOG_CONTEXT);

      coarse_da.surrogate->writeToGhostsBegin(coarse_surr_ghosted.data(), ndofs);
      coarse_da.surrogate->writeToGhostsEnd(coarse_surr_ghosted.data(), ndofs);
    }

    // Shift in the coarse grid from surrogate to primary.
    {
      /// auto dbg_region = debug::global_comm_log->declare_region(
      ///     {coarse_da.surrogate->getCommActive(),
      ///     coarse_da.primary->getCommActive()},
      ///     COMMLOG_CONTEXT);

      ot::distShiftNodes(
          *coarse_da.surrogate,
          coarse_surr_ghosted.data() + coarse_da.surrogate->getLocalNodeBegin() * ndofs,
          *coarse_da.primary,
          coarse_vec_ghosted + coarse_da.primary->getLocalNodeBegin() * ndofs,
          ndofs);
    }

    post_restriction(coarse_vec_ghosted + coarse_da.primary->getLocalNodeBegin() * ndofs);

    {
      /// auto dbg_region = debug::global_comm_log->declare_region(coarse_da.primary->getCommActive(), COMMLOG_CONTEXT);

      coarse_da.primary->readFromGhostBegin(coarse_vec_ghosted, ndofs);
      coarse_da.primary->readFromGhostEnd(coarse_vec_ghosted, ndofs);
    }

    vector_pool.checkin(std::move(leafBuffer));
    vector_pool.checkin(std::move(coarse_surr_ghosted));
  }


  // prolongate_coarse_to_fine()
  template <int dim, typename PreProlongation>
  void prolongate_coarse_to_fine(
      DA_Pair<dim> coarse_da, const VECType *coarse_vec_ghosted,
      DA_Pair<dim> fine_da, VECType *fine_vec_ghosted,
      PreProlongation pre_prolongation,  // on local vector
      int ndofs,
      detail::VectorPool<VECType> &vector_pool)
  {
    const unsigned int nPe = fine_da.primary->getNumNodesPerElement();

    std::vector<VECType> coarse_surr_ghosted = vector_pool.checkout(
        coarse_da.surrogate->getTotalNodalSz() * ndofs);

    std::vector<VECType> leafBuffer = vector_pool.checkout(ndofs * nPe);
    leafBuffer.assign(leafBuffer.size(), 42);

    std::vector<VECType> coarse_copy_ghosted = vector_pool.checkout(
        coarse_da.primary->getTotalNodalSz() * ndofs);
    std::copy_n(coarse_vec_ghosted, coarse_copy_ghosted.size(), coarse_copy_ghosted.begin());
    pre_prolongation(coarse_copy_ghosted.data() + coarse_da.primary->getLocalNodeBegin() * ndofs);
    coarse_vec_ghosted = coarse_copy_ghosted.data();

    // Shift in the coarse grid from primary to surrogate.
    ot::distShiftNodes(
        *coarse_da.primary,
        coarse_vec_ghosted + coarse_da.primary->getLocalNodeBegin() * ndofs,
        *coarse_da.surrogate,
        coarse_surr_ghosted.data() + coarse_da.surrogate->getLocalNodeBegin() * ndofs,
        ndofs);
    coarse_da.surrogate->readFromGhostBegin(coarse_surr_ghosted.data(), ndofs);
    coarse_da.surrogate->readFromGhostEnd(coarse_surr_ghosted.data(), ndofs);

    // Surrogate is coarse grid partitioned by fine
    // Interpolate from the coarse surrogate grid to fine primary.

    using TN = ot::TreeNode<uint, dim>;

    fem::MeshFreeInputContext<VECType, TN>
        inctx{ coarse_surr_ghosted.data(),
               coarse_da.surrogate->getTNCoords(),
               (unsigned) coarse_da.surrogate->getTotalNodalSz(),
               coarse_da.surrogate->dist_tree()->getTreePartFiltered(coarse_da.surrogate->stratum()).data(),
               coarse_da.surrogate->dist_tree()->getTreePartFiltered(coarse_da.surrogate->stratum()).size(),
               *coarse_da.surrogate->getTreePartFront(),
               *coarse_da.surrogate->getTreePartBack() };

    fem::MeshFreeOutputContext<VECType, TN>
        outctx{fine_vec_ghosted,
               fine_da.primary->getTNCoords(),
               (unsigned) fine_da.primary->getTotalNodalSz(),
               fine_da.primary->dist_tree()->getTreePartFiltered(fine_da.primary->stratum()).data(),
               fine_da.primary->dist_tree()->getTreePartFiltered(fine_da.primary->stratum()).size(),
               *fine_da.primary->getTreePartFront(),
               *fine_da.primary->getTreePartBack() };

    const RefElement * refel = fine_da.primary->getReferenceElement();

    std::vector<char> outDirty(fine_da.primary->getTotalNodalSz(), 0);
    fem::locIntergridTransfer(inctx, outctx, ndofs, refel, &(*outDirty.begin()));
    // The outDirty array is needed when wrwiteToGhosts useAccumulation==false (hack).
    fine_da.primary->template writeToGhostsBegin<VECType>(fine_vec_ghosted, ndofs, &(*outDirty.cbegin()));
    fine_da.primary->template writeToGhostsEnd<VECType>(fine_vec_ghosted, ndofs, false, &(*outDirty.cbegin()));

    fine_da.primary->readFromGhostBegin(fine_vec_ghosted, ndofs);
    fine_da.primary->readFromGhostEnd(fine_vec_ghosted, ndofs);

    vector_pool.checkin(std::move(leafBuffer));
    vector_pool.checkin(std::move(coarse_surr_ghosted));
    vector_pool.checkin(std::move(coarse_copy_ghosted));
  }


}//namespace mg




namespace detail
{
  template <typename X>
  VectorPool<X>::~VectorPool()
  {
    if (this->print())
      std::cerr << "checkouts==" << m_log_checkouts
                << " \tallocs==" << m_log_allocs << "\n";
  }

  template <typename X>
  void VectorPool<X>::free_all()
  {
    m_idle.clear();
  }

  template <typename X>
  std::vector<X> VectorPool<X>::checkout(size_t size)
  {
    ++m_log_checkouts;
    auto it = m_idle.find(size);
    if (it == m_idle.end())
    {
      ++m_log_allocs;
      return std::vector<X>(size);
    }
    std::vector<X> extracted = std::move(it->second);
    m_idle.erase(it);
    return extracted;
  }

  template <typename X>
  void VectorPool<X>::checkin(std::vector<X> &&vec)
  {
    m_idle.insert({vec.size(), std::move(vec)});
  }


  // matvec_base_in()
  template <typename X, unsigned dim>
  ot::MatvecBaseIn<dim, X> matvec_base_in(
      const ot::DA<dim> *da, int ndofs, const X *ghosted, int extra_depth)
  {
    return ot::MatvecBaseIn<dim, X>(
          da->getTotalNodalSz(), ndofs, da->getElementOrder(),
          extra_depth > 0, extra_depth,
          da->getTNCoords(), ghosted,
          da->dist_tree()->getTreePartFiltered(da->stratum()).data(),
          da->dist_tree()->getTreePartFiltered(da->stratum()).size(),
          *da->getTreePartFront(),
          *da->getTreePartBack());
  }

  // matvec_base_ou_accumulatet()
  template <typename X, unsigned dim>
  ot::MatvecBaseOut<dim, X, true> matvec_base_out_accumulate(
      const ot::DA<dim> *da, int ndofs, int extra_depth)
  {
    const bool accumulate = true;
    return ot::MatvecBaseOut<dim, X, accumulate>(
        da->getTotalNodalSz(), ndofs, da->getElementOrder(),
        extra_depth > 0, extra_depth,
        da->getTNCoords(),
        da->dist_tree()->getTreePartFiltered(da->stratum()).data(),
        da->dist_tree()->getTreePartFiltered(da->stratum()).size(),
        *da->getTreePartFront(),
        *da->getTreePartBack());
  }



}



#endif//DENDRO_KT_FEM_MG_HPP
