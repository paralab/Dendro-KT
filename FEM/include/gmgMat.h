/**
 * @author Masado Ishii
 * @date 2020-02-14
 * @brief Abstract class for geometric multigrid smoother & residual.
 *
 * @note When used with petsc, this interface ignores some of
 *       the smoother options passed through petsc, and assumes that the
 *       application smoother makes the right choice.
 */

#ifndef DENDRO_KT_GMG_MAT_H
#define DENDRO_KT_GMG_MAT_H

#include "oda.h"
#include "intergridTransfer.h"
#include "point.h"
#include <stdexcept>
#ifdef BUILD_WITH_PETSC
#include "petscpc.h"
#include "petscksp.h"
#include "petscdmda.h"
#endif


// =================================
// Class forward declarations
// =================================
template <unsigned int dim, class LeafClass>
struct gmgMatStratumWrapper;

template <unsigned int dim, class LeafClass>
class gmgMat;



// =================================
// gmgMatStratumWrapper
// =================================
template <unsigned int dim, class LeafClass>
struct gmgMatStratumWrapper
{
  gmgMat<dim, LeafClass> * m_gmgMat;
  unsigned int m_stratum;
};


// =================================
// gmgMat
// =================================
template <unsigned int dim, class LeafClass>
class gmgMat {

protected:
    static constexpr unsigned int m_uiDim = dim;

    /**@brief: pointer to OCT DA*/
    ot::MultiDA<dim> * m_multiDA;
    ot::MultiDA<dim> * m_surrogateMultiDA;
    unsigned int m_numStrata;
    unsigned int m_ndofs;

    /**@brief problem domain min point*/
    Point<dim> m_uiPtMin;

    /**@brief problem domain max point*/
    Point<dim> m_uiPtMax;

    std::vector<gmgMatStratumWrapper<dim, LeafClass>> m_stratumWrappers;

#ifdef BUILD_WITH_PETSC
    Vec * m_stratumWorkRhs;
    Vec * m_stratumWorkX;
    Vec * m_stratumWorkR;
#endif

public:
    /**@brief: gmgMat constructor
      * @par[in] daType: type of the DA
      * @note Does not own da.
    **/
    gmgMat(ot::MultiDA<dim>* mda, ot::MultiDA<dim> *smda, unsigned int ndofs)
      : m_multiDA(mda), m_surrogateMultiDA(smda), m_ndofs(ndofs)
    {
      assert(mda != nullptr);
      assert(smda != nullptr);
      assert(mda->size() == smda->size());  // Did you generate DA from surrogate tree?

      m_numStrata = mda->size();
      for (int ii = 0; ii < m_numStrata; ++ii)
        m_stratumWrappers.emplace_back(gmgMatStratumWrapper<dim, LeafClass>{this, ii});


#ifdef BUILD_WITH_PETSC
      m_stratumWorkRhs = new Vec[m_numStrata];
      m_stratumWorkX = new Vec[m_numStrata];
      m_stratumWorkR = new Vec[m_numStrata];

      (*m_multiDA)[0].petscCreateVector(m_stratumWorkR[0], false, false, m_ndofs);
      for (int ii = 1; ii < m_numStrata; ++ii)
      {
        (*m_multiDA)[ii].petscCreateVector(m_stratumWorkRhs[ii], false, false, m_ndofs);
        (*m_multiDA)[ii].petscCreateVector(m_stratumWorkX[ii], false, false, m_ndofs);
        (*m_multiDA)[ii].petscCreateVector(m_stratumWorkR[ii], false, false, m_ndofs);
      }
#endif
    }

    /**@brief Destructor*/
    virtual ~gmgMat()
    {
#ifdef BUILD_WITH_PETSC
      (*m_multiDA)[0].petscDestroyVec(m_stratumWorkR[0]);
      for (int ii = 1; ii < m_numStrata; ++ii)
      {
        (*m_multiDA)[ii].petscDestroyVec(m_stratumWorkRhs[ii]);
        (*m_multiDA)[ii].petscDestroyVec(m_stratumWorkX[ii]);
        (*m_multiDA)[ii].petscDestroyVec(m_stratumWorkR[ii]);
      }

      delete [] m_stratumWorkRhs;
      delete [] m_stratumWorkX;
      delete [] m_stratumWorkR;
#endif
    }

    LeafClass & asConcreteType()
    {
      return static_cast<LeafClass &>(*this);
    }


    // Design note (static polymorphism)
    //   The gmgMat does not impose a mass matrix or smoothing operator.
    //   The leaf derived type is responsible to implement those as
    //   leafMatVec() and leafSmooth().
    //   Suggestion:
    //     Define a leaf derived type from gmgMat that contains
    //     (via class composition) an feMat instance per level.
    //     The feMat class knows how to do a matvec and extract the diagonal,
    //     provided that the elemental versions are implemented in a leaf
    //     derived type of feMat.

    /**@brief Computes the LHS of the weak formulation, normally the stiffness matrix times a given vector.
     * @param [in] in input vector u
     * @param [out] out output vector Ku
     * @param [in] stratum Which coarse grid to evaluate on, 0 (default) being the finest.
     * @param [in] default parameter scale vector by scale*Ku
     * */
    void matVec(const VECType *in, VECType *out, unsigned int stratum = 0, double scale=1.0)//=0
    {
      asConcreteType().leafMatVec(in, out, stratum, scale);
    }


    /**
     * @brief Update the vector 'u' by applying one or more SOR smoothing steps.
     * @param [in out] u Approximate solution to be improved.
     * @param [in] rhs Righthand side of the linear system.
     * @param [in] omega Relaxation factor.
     * @param [in] iters Number of (global) iterations.
     * @param [in] localIters Number of local iterations.
     * @note Total number of smoothing steps should be (iters*localIters).
     */
    void smooth(VECType *u, const VECType *rhs, double omega, int iters, int localIters, unsigned int stratum = 0)//=0
    {
      asConcreteType().leafSmooth(u, rhs, omega, iters, localIters, stratum);
    }



    /**@brief set the problem dimension*/
    inline void setProblemDimensions(const Point<dim>& pt_min, const Point<dim>& pt_max)
    {
      m_uiPtMin=pt_min;
      m_uiPtMax=pt_max;
    }


    void restriction(const VECType *fineRes, VECType *coarseRes, unsigned int fineStratum = 0)
    {
      /// using namespace std::placeholders;   // Convenience for std::bind().

      ot::DA<dim> &fineDA = (*this->m_multiDA)[fineStratum];
      ot::DA<dim> &surrDA = (*this->m_surrogateMultiDA)[fineStratum+1];
      ot::DA<dim> &coarseDA = (*this->m_multiDA)[fineStratum+1];

      // Static buffers for ghosting. Check/increase size.
      static std::vector<VECType> fineGhosted, surrGhosted;
      fineDA.template createVector<VECType>(fineGhosted, false, true, m_ndofs);
      surrDA.template createVector<VECType>(surrGhosted, false, true, m_ndofs);
      std::fill(surrGhosted.begin(), surrGhosted.end(), 0);
      VECType *fineGhostedPtr = fineGhosted.data();
      VECType *surrGhostedPtr = surrGhosted.data();

      // 1. Copy input data to ghosted buffer.
      fineDA.template nodalVecToGhostedNodal<VECType>(fineRes, fineGhostedPtr, true, m_ndofs);

      // code import note: There was prematvec here.

      using TN = ot::TreeNode<typename ot::DA<dim>::C, dim>;

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.start();
#endif

      // 2. Upstream->downstream ghost exchange.
      fineDA.template readFromGhostBegin<VECType>(fineGhostedPtr, m_ndofs);
      fineDA.template readFromGhostEnd<VECType>(fineGhostedPtr, m_ndofs);

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.stop();
#endif

      // code import note: There was binding elementalMatvec here.

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_gmg_loc_restrict.start();
#endif

      fem::MeshFreeInputContext<VECType, TN>
          inctx{ fineGhostedPtr,
                 fineDA.getTNCoords(),
                 (unsigned) fineDA.getTotalNodalSz(),
                 *fineDA.getTreePartFront(),
                 *fineDA.getTreePartBack() };

      fem::MeshFreeOutputContext<VECType, TN>
          outctx{surrGhostedPtr,
                 surrDA.getTNCoords(),
                 (unsigned) surrDA.getTotalNodalSz(),
                 *surrDA.getTreePartFront(),
                 *surrDA.getTreePartBack() };

      const RefElement * refel = fineDA.getReferenceElement();

      fem::locIntergridTransfer(inctx, outctx, m_ndofs, refel);

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_gmg_loc_restrict.start();
#endif


#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.start();
#endif

      // 4. Downstream->Upstream ghost exchange.
      surrDA.template writeToGhostsBegin<VECType>(surrGhostedPtr, m_ndofs);
      surrDA.template writeToGhostsEnd<VECType>(surrGhostedPtr, m_ndofs);

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.stop();
#endif

      // 5. Copy output data from ghosted buffer.
      ot::distShiftNodes(surrDA,    surrGhostedPtr + surrDA.getLocalNodeBegin(),
                         coarseDA,  coarseRes,
                         m_ndofs);
                         
      // code import note: There was postmatvec here.
    }


    void prolongation(const VECType *coarseCrx, VECType *fineCrx, unsigned int fineStratum = 0)
    {
      /// using namespace std::placeholders;   // Convenience for std::bind().

      ot::DA<dim> &fineDA = (*this->m_multiDA)[fineStratum];
      ot::DA<dim> &surrDA = (*this->m_surrogateMultiDA)[fineStratum+1];
      ot::DA<dim> &coarseDA = (*this->m_multiDA)[fineStratum+1];

      // Static buffers for ghosting. Check/increase size.
      static std::vector<VECType> fineGhosted, surrGhosted;
      fineDA.template createVector<VECType>(fineGhosted, false, true, m_ndofs);
      surrDA.template createVector<VECType>(surrGhosted, false, true, m_ndofs);
      std::fill(fineGhosted.begin(), fineGhosted.end(), 0);
      VECType *fineGhostedPtr = fineGhosted.data();
      VECType *surrGhostedPtr = surrGhosted.data();

      // 1. Copy input data to ghosted buffer.
      ot::distShiftNodes(coarseDA,   coarseCrx,
                         surrDA,     surrGhostedPtr + surrDA.getLocalNodeBegin(),
                         m_ndofs);

      // code import note: There was prematvec here.

      using TN = ot::TreeNode<typename ot::DA<dim>::C, dim>;

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.start();
#endif

      // 2. Upstream->downstream ghost exchange.
      surrDA.template readFromGhostBegin<VECType>(surrGhostedPtr, m_ndofs);
      surrDA.template readFromGhostEnd<VECType>(surrGhostedPtr, m_ndofs);

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.stop();
#endif

      // code import note: There was binding elementalMatvec here.

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_gmg_loc_restrict.start();
#endif

      fem::MeshFreeInputContext<VECType, TN>
          inctx{ surrGhostedPtr,
                 surrDA.getTNCoords(),
                 (unsigned) surrDA.getTotalNodalSz(),
                 *surrDA.getTreePartFront(),
                 *surrDA.getTreePartBack() };

      fem::MeshFreeOutputContext<VECType, TN>
          outctx{fineGhostedPtr,
                 fineDA.getTNCoords(),
                 (unsigned) fineDA.getTotalNodalSz(),
                 *fineDA.getTreePartFront(),
                 *fineDA.getTreePartBack() };

      const RefElement * refel = fineDA.getReferenceElement();

      fem::locIntergridTransfer(inctx, outctx, m_ndofs, refel);

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_gmg_loc_restrict.start();
#endif


#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.start();
#endif

      // 4. Downstream->Upstream ghost exchange.
      fineDA.template writeToGhostsBegin<VECType>(fineGhostedPtr, m_ndofs);
      fineDA.template writeToGhostsEnd<VECType>(fineGhostedPtr, m_ndofs);

#ifdef DENDRO_KT_GMG_BENCH_H
      bench::t_ghostexchange.stop();
#endif

      // 5. Copy output data from ghosted buffer.
      fineDA.template ghostedNodalToNodalVec<VECType>(fineGhostedPtr, fineCrx, true, m_ndofs);

      // code import note: There was postmatvec here.
    }



#ifdef BUILD_WITH_PETSC
    // all PETSC function should go here.

    // -----------------------------------
    // Operators accepting Petsc Vec type.
    // -----------------------------------

    //
    // matVec() [Petsc Vec]
    //
    void matVec(const Vec &in, Vec &out, unsigned int stratum = 0, double scale=1.0)
    {
      const PetscScalar * inArry = nullptr;
      PetscScalar * outArry = nullptr;

      VecGetArrayRead(in, &inArry);
      VecGetArray(out, &outArry);

      matVec(inArry, outArry, stratum, scale);

      VecRestoreArrayRead(in, &inArry);
      VecRestoreArray(out, &outArry);
    }

    //
    // smooth() [Petsc Vec]
    //
    void smooth(Vec &u, const Vec &rhs, PetscReal omega, PetscInt iters, PetscInt localIters, unsigned int stratum = 0)
    {
      PetscScalar * uArry = nullptr;
      const PetscScalar * fArry = nullptr;

      VecGetArray(u, &uArry);
      VecGetArrayRead(rhs, &fArry);

      smooth(uArry, fArry, (double) omega, (int) iters, (int) localIters, stratum);

      VecRestoreArray(u, &uArry);
      VecRestoreArrayRead(rhs, &fArry);
    }

    //
    // restriction() [Petsc Vec]
    //
    void restriction(const Vec &fineRes, Vec &coarseRes, unsigned int stratum = 0)
    {
      const PetscScalar * inArry = nullptr;
      PetscScalar * outArry = nullptr;

      VecGetArrayRead(fineRes, &inArry);
      VecGetArray(coarseRes, &outArry);

      restriction(inArry, outArry, stratum);

      VecRestoreArrayRead(fineRes, &inArry);
      VecRestoreArray(coarseRes, &outArry);
    }

    //
    // prolongation() [Petsc Vec]
    //
    void prolongation(const Vec &coarseCrx, Vec &fineCrx, unsigned int stratum = 0)
    {
      const PetscScalar * inArry = nullptr;
      PetscScalar * outArry = nullptr;

      VecGetArrayRead(coarseCrx, &inArry);
      VecGetArray(fineCrx, &outArry);

      prolongation(inArry, outArry, stratum);

      VecRestoreArrayRead(coarseCrx, &inArry);
      VecRestoreArray(fineCrx, &outArry);
    }


    // -------------------------
    // Mat Shell wrappers
    // -------------------------

    // Note that Dendro-KT indexing of levels is inverted w.r.t. PETSc indexing.
    // In Dendro-KT, the finest level is 0 and the coarsest level is numStrata-1.
    // In PETSc, the finest level is numStrata-1 and the coarsest level is 0.
    // The interface of gmgMat assumes the Dendro-KT indexing.
    // The conversion from Dendro-KT to PETSc indexing is done in petscCreateGMG().


    /// void petscCreateGMG(int smoothUp, int smoothDown, MPI_Comm comm)
    KSP petscCreateGMG(MPI_Comm comm)
    {
      KSP gmgKSP;
      PC  gmgPC;

      KSPCreate(comm, &gmgKSP);
      KSPSetType(gmgKSP, KSPRICHARDSON);  // standalone solver, not preconditioner.

      KSPGetPC(gmgKSP, &gmgPC);
      PCSetType(gmgPC, PCMG);

      /// PCMGSetLevels(gmgPC, (int) m_numStrata, ???comms???);

      PCMGSetType(gmgPC, PC_MG_MULTIPLICATIVE); // MGMULTIPLICATIVE,MGADDITIVE,MGFULL,MGCASCADE

      PCMGSetLevels(gmgPC, (int) m_numStrata, PETSC_NULL); // PETSC_NULL indicates don't use separate comm for each level.

      PCMGSetCycleType(gmgPC, PC_MG_CYCLE_V);

      /// MGSetNumberSmoothUp(gmgPC, smoothUp);      // Outdated petsc interface
      /// MGSetNumberSmoothDown(gmgPC, smoothDown);

      // Set smoothers.
      for (int s = 0; s < m_numStrata-1; ++s)  //0<petscLevel<nlevels
      {
        const PetscInt petscLevel = m_numStrata-1 - s;

        Mat matrixFreeSmoothMat;
        this->petscMatCreateShellSmooth(matrixFreeSmoothMat, s);

        KSP gmgSmootherKSP;
        PCMGGetSmoother(gmgPC, petscLevel, &gmgSmootherKSP);
        KSPSetOperators(gmgSmootherKSP, matrixFreeSmoothMat, matrixFreeSmoothMat);

        // Slight confusion here..
        //  The manual says, to set the matrix that defines the smoother on level 1, do
        //      PCMGGetSmoother(pc, 1, &ksp);
        //      KSPSetOperators(ksp, A1, A1);
        //  On the other hand, to set SOR as the smoother to use on level 1, do
        //      PCMGGetSmoother(pc, 1, &ksp);
        //      KSPGetPC(ksp, &pc);
        //      PCSetType(pc, PCSOR);
        //  I want to use SOR, but how could petsc know how to apply SOR
        //  without us giving it the matrix? So I am going with the first option.
      }

      // Set coarse solver.
      {
        Mat matrixFreeSmoothMat;
        this->petscMatCreateShellSmooth(matrixFreeSmoothMat, m_numStrata-1);

        KSP gmgSmootherKSP;
        PCMGGetCoarseSolve(gmgPC, &gmgSmootherKSP);
        KSPSetOperators(gmgSmootherKSP, matrixFreeSmoothMat, matrixFreeSmoothMat);
      }

      // Set workspace vectors.
      PCMGSetR(gmgPC, m_numStrata-1, m_stratumWorkR[0]);
      for (int s = 1; s < m_numStrata; ++s)  // All but finest
      {
        const PetscInt petscLevel = m_numStrata-1 - s;

        PCMGSetRhs(gmgPC, petscLevel, m_stratumWorkRhs[s]);
        PCMGSetX(gmgPC, petscLevel, m_stratumWorkX[s]);
        PCMGSetR(gmgPC, petscLevel, m_stratumWorkR[s]);
      }

      // Set restriction/prolongation
      for (int s = 0; s < m_numStrata-1; ++s)  //0<petscLevel<nlevels
      {
        const PetscInt petscLevel = m_numStrata-1 - s;

        Mat matrixFreeRestrictionMat;
        Mat matrixFreeProlongationMat;
        this->petscMatCreateShellRestriction(matrixFreeRestrictionMat, s);
        this->petscMatCreateShellProlongation(matrixFreeProlongationMat, s);

        PCMGSetRestriction(gmgPC, petscLevel, matrixFreeRestrictionMat);
        PCMGSetInterpolation(gmgPC, petscLevel, matrixFreeProlongationMat);
      }

      // Set residual functions.
      //
      for (int s = 0; s < m_numStrata-1; ++s)  //0<petscLevel<nlevels
      {
        const PetscInt petscLevel = m_numStrata-1 - s;

        Mat matrixFreeOperatorMat;
        this->petscMatCreateShellMatVec(matrixFreeOperatorMat, s);

        // PETSC_NULL indicates that the default petsc residual method will
        // be used, which will take A (our matvec), u, and rhs,
        // and compute rhs - A*u.
        PCMGSetResidual(gmgPC, petscLevel, PETSC_NULL, matrixFreeOperatorMat);
      }

      return gmgKSP;
    }


    /**
     * @brief Calls MatCreateShell and MatShellSetOperation to create a matrix-free matrix usable by petsc, e.g. in KSPSetOperators().
     * @param [out] matrixFreeMat Petsc shell matrix, representing a matrix-free matrix that uses this instance.
     */
    void petscMatCreateShellMatVec(Mat &matrixFreeMat, unsigned int stratum = 0)
    {
      PetscInt localM = (*m_multiDA)[stratum].getLocalNodalSz();
      PetscInt globalM = (*m_multiDA)[stratum].getGlobalNodeSz();
      MPI_Comm comm = (*m_multiDA)[stratum].getGlobalComm();

      // MATOP_MULT is for registering a multiply.
      MatCreateShell(comm, localM, localM, globalM, globalM, &m_stratumWrappers[stratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_MULT, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserMultMatVec);
    }

    void petscMatCreateShellSmooth(Mat &matrixFreeMat, unsigned int stratum)
    {
      PetscInt localM = (*m_multiDA)[stratum].getLocalNodalSz();
      PetscInt globalM = (*m_multiDA)[stratum].getGlobalNodeSz();
      MPI_Comm comm = (*m_multiDA)[stratum].getGlobalComm();

      // MATOP_SOR is for providing a smoother.
      MatCreateShell(comm, localM, localM, globalM, globalM, &m_stratumWrappers[stratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_SOR, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserSmooth);
    }

    void petscMatCreateShellRestriction(Mat &matrixFreeMat, unsigned int fineStratum)
    {
      PetscInt localM_in = (*m_multiDA)[fineStratum].getLocalNodalSz();
      PetscInt globalM_in = (*m_multiDA)[fineStratum].getGlobalNodeSz();

      PetscInt localM_out = (*m_multiDA)[fineStratum+1].getLocalNodalSz();
      PetscInt globalM_out = (*m_multiDA)[fineStratum+1].getGlobalNodeSz();

      MPI_Comm comm = (*m_multiDA)[fineStratum].getGlobalComm();

      // MATOP_MULT is for registering a multiply.
      MatCreateShell(comm, localM_out, localM_in, globalM_out, globalM_in, &m_stratumWrappers[fineStratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_MULT, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserMultRestriction);
    }

    void petscMatCreateShellProlongation(Mat &matrixFreeMat, unsigned int fineStratum)
    {
      PetscInt localM_in = (*m_multiDA)[fineStratum+1].getLocalNodalSz();
      PetscInt globalM_in = (*m_multiDA)[fineStratum+1].getGlobalNodeSz();

      PetscInt localM_out = (*m_multiDA)[fineStratum].getLocalNodalSz();
      PetscInt globalM_out = (*m_multiDA)[fineStratum].getGlobalNodeSz();

      MPI_Comm comm = (*m_multiDA)[fineStratum].getGlobalComm();

      // MATOP_MULT is for registering a multiply.
      MatCreateShell(comm, localM_out, localM_in, globalM_out, globalM_in, &m_stratumWrappers[fineStratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_MULT, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserMultProlongation);
    }


    /**
     * @brief The 'user defined' matvec we give to petsc to make a matrix-free matrix. Don't call this directly.
     * Must follow the same interface as petsc MatMult().
     */
    static void petscUserMultMatVec(Mat mat, Vec x, Vec y_out)
    {
      gmgMatStratumWrapper<dim, LeafClass> *gmgMatWrapper;
      MatShellGetContext(mat, &gmgMatWrapper);
      const unsigned int stratum = gmgMatWrapper->m_stratum;
      gmgMatWrapper->m_gmgMat->matVec(x, y_out, stratum);
    };

    /**
     * @brief The 'user defined' sor routine we give to petsc to make a matrix-free smoother. Don't call this directly.
     * Must follow the same interface as petsc MatSOR().
     */
    static void petscUserSmooth(Mat mat,
                                Vec rhs,
                                PetscReal omega,
                                MatSORType sorTypeFlag,
                                PetscReal diagShift,
                                PetscInt iters,
                                PetscInt localIters,
                                Vec x_out)
    {
      // Note: Only forwards omega, iters, and localIters,
      //       ignores sorTypeFlag and diagShift.
      gmgMatStratumWrapper<dim, LeafClass> *gmgMatWrapper;
      MatShellGetContext(mat, &gmgMatWrapper);
      const unsigned int stratum = gmgMatWrapper->m_stratum;
      gmgMatWrapper->m_gmgMat->smooth(x_out, rhs, omega, iters, localIters, stratum);
    };

    /**
     * @brief The 'user defined' matvec we give to petsc to make matrix-free restriction. Don't call this directly.
     * Must follow the same interface as petsc MatMult().
     */
    static void petscUserMultRestriction(Mat mat, Vec x, Vec y_out)
    {
      gmgMatStratumWrapper<dim, LeafClass> *gmgMatWrapper;
      MatShellGetContext(mat, &gmgMatWrapper);
      const unsigned int fineStratum = gmgMatWrapper->m_stratum;
      gmgMatWrapper->m_gmgMat->restriction(x, y_out, fineStratum);
    };

    /**
     * @brief The 'user defined' matvec we give to petsc to make matrix-free prolongation. Don't call this directly.
     * Must follow the same interface as petsc MatMult().
     */
    static void petscUserMultProlongation(Mat mat, Vec x, Vec y_out)
    {
      gmgMatStratumWrapper<dim, LeafClass> *gmgMatWrapper;
      MatShellGetContext(mat, &gmgMatWrapper);
      const unsigned int fineStratum = gmgMatWrapper->m_stratum;
      gmgMatWrapper->m_gmgMat->prolongation(x, y_out, fineStratum);
    };



#endif


};

#endif //DENDRO_KT_GMG_MAT_H
