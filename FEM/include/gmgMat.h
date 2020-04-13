/**
 * @author Masado Ishii
 * @date 2020-02-14
 * @brief Abstract class for geometric multigrid smoother & residual.
 */

#ifndef DENDRO_KT_GMG_MAT_H
#define DENDRO_KT_GMG_MAT_H

#include "oda.h"
#include "intergridTransfer.h"
#include "point.h"
#include <stdexcept>
#ifdef BUILD_WITH_PETSC
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
    unsigned int m_ndofs;

    /**@brief problem domain min point*/
    Point<dim> m_uiPtMin;

    /**@brief problem domain max point*/
    Point<dim> m_uiPtMax;

    std::vector<gmgMatStratumWrapper<dim, LeafClass>> m_stratumWrappers;

    // I don't know what this is
/// #ifdef BUILD_WITH_PETSC
///     /**@brief: petsc DM*/
///     DM m_uiPETSC_DA;
/// #endif

public:
    /**@brief: gmgMat constructor
      * @par[in] daType: type of the DA
      * @note Does not own da.
    **/
    gmgMat(ot::MultiDA<dim>* mda, ot::MultiDA<dim> *smda, unsigned int ndofs)
      : m_multiDA{mda}, m_surrogateMultiDA{smda}, m_ndofs{ndofs}
    {
      assert(mda != nullptr);
      assert(smda != nullptr);
      assert(mda->size() == smda->size());  // Did you generate DA from surrogate tree?

      const unsigned int numStrata = mda->size();
      for (int ii = 0; ii < numStrata; ++ii)
        m_stratumWrappers.emplace_back(this, ii);
    }

    /**@brief deconstructor*/
    ~gmgMat()
    {

    }

    LeafClass & asConcreteType()
    {
      return static_cast<LeafClass &>(*this);
    }


    // Design note (static polymorphism)
    //   The gmgMat does not impose a mass matrix or smoothing operator.
    //   The leaf derived type is responsible to implement those.
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
      static bool reentrant = false;
      if (reentrant)
        throw std::logic_error{"matVec() not implemented by LeafClass"};
      reentrant = true;
      {
        asConcreteType().matVec(in, out, stratum, scale);
      }
      reentrant = false;
    }


    void smooth(VECType *u, const VECType *f, unsigned int stratum = 0)//=0
    {
      static bool reentrant = false;
      if (reentrant)
        throw std::logic_error{"smooth() not implemented by LeafClass"};
      reentrant = true;
      {
        asConcreteType().smooth(u, f, stratum);
      }
      reentrant = false;
    }


    void residual(const VECType *x, const VECType *f, VECType *r, unsigned int stratum = 0)//=0
    {
      static bool reentrant = false;
      if (reentrant)
        throw std::logic_error{"residual() not implemented by LeafClass"};
      reentrant = true;
      {
        asConcreteType().residual(x, f, r, stratum);
      }
      reentrant = false;
    }


    /**@brief set the problem dimension*/
    inline void setProblemDimensions(const Point<dim>& pt_min, const Point<dim>& pt_max)
    {
      m_uiPtMin=pt_min;
      m_uiPtMax=pt_max;
    }


    void restriction(const VECType *fineErr, VECType *coarseErr, unsigned int fineStratum = 0)
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
      fineDA.template nodalVecToGhostedNodal<VECType>(fineErr, fineGhostedPtr, true, m_ndofs);

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
                 fineDA.getTotalNodalSz(),
                 *fineDA.getTreePartFront(),
                 *fineDA.getTreePartBack() };

      fem::MeshFreeOutputContext<VECType, TN>
          outctx{surrGhostedPtr,
                 surrDA.getTNCoords(),
                 surrDA.getTotalNodalSz(),
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
                         coarseDA,  coarseErr,
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
                 surrDA.getTotalNodalSz(),
                 *surrDA.getTreePartFront(),
                 *surrDA.getTreePartBack() };

      fem::MeshFreeOutputContext<VECType, TN>
          outctx{fineGhostedPtr,
                 fineDA.getTNCoords(),
                 fineDA.getTotalNodalSz(),
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
    void smooth(Vec &u, const Vec &f, unsigned int stratum = 0)
    {
      PetscScalar * uArry = nullptr;
      const PetscScalar * fArry = nullptr;

      VecGetArray(u, &uArry);
      VecGetArrayRead(f, &fArry);

      smooth(uArry, fArry, stratum);

      VecRestoreArray(u, &uArry);
      VecRestoreArrayRead(f, &fArry);
    }

    //
    // residual() [Petsc Vec]
    //
    void residual(const Vec &x, const Vec &f, Vec &r, unsigned int stratum = 0)
    {
      const PetscScalar * xArry = nullptr;
      const PetscScalar * fArry = nullptr;
      PetscScalar * rArry = nullptr;

      VecGetArrayRead(x, &xArry);
      VecGetArrayRead(f, &fArry);
      VecGetArray(r, &rArry);

      residual(xArry, fArry, rArry, stratum);

      VecRestoreArrayRead(x, &xArry);
      VecRestoreArrayRead(f, &fArry);
      VecRestoreArray(r, &rArry);
    }


    // -------------------------
    // Mat Shell wrappers
    // -------------------------

    /**
     * @brief Calls MatCreateShell and MatShellSetOperation to create a matrix-free matrix usable by petsc, e.g. in KSPSetOperators().
     * @param [out] matrixFreeMat Petsc shell matrix, representing a matrix-free matrix that uses this instance.
     */
    void petscMatCreateShellMatVec(Mat &matrixFreeMat, unsigned int stratum = 0)
    {
      PetscInt localM = (*m_multiDA[stratum]).getLocalNodalSz();
      PetscInt globalM = (*m_multiDA[stratum]).getGlobalNodeSz();
      MPI_Comm comm = (*m_multiDA[stratum]).getGlobalComm();

      MatCreateShell(comm, localM, localM, globalM, globalM, &m_stratumWrappers[stratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_MULT, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserMultMatVec);
    }

    void petscMatCreateShellSmooth(Mat &matrixFreeMat, unsigned int stratum = 0)
    {
      PetscInt localM = (*m_multiDA[stratum]).getLocalNodalSz();
      PetscInt globalM = (*m_multiDA[stratum]).getGlobalNodeSz();
      MPI_Comm comm = (*m_multiDA[stratum]).getGlobalComm();

      MatCreateShell(comm, localM, localM, globalM, globalM, &m_stratumWrappers[stratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_MULT, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserMultSmooth);
    }


    //TODO what user mult interface do the other PETSC multigrid operators need?
    // where do restriction and prolongation go?


    /** @brief The 'user defined' matvec we give to petsc to make a matrix-free matrix. Don't call this directly. */
    static void petscUserMultMatVec(Mat mat, Vec x, Vec y)
    {
      gmgMatStratumWrapper<dim, LeafClass> *gmgMatWrapper;
      MatShellGetContext(mat, &gmgMatWrapper);
      const unsigned int stratum = gmgMatWrapper->m_stratum;
      gmgMatWrapper->m_gmgMat->matVec(x, y, stratum);
    };

    //TODO what user mult interface do the other PETSC multigrid operators need?
    // petscUserMultSmooth


#endif


};

#endif //DENDRO_KT_GMG_MAT_H
