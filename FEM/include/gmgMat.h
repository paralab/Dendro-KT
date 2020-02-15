/**
 * @author Masado Ishii
 * @date 2020-02-14
 * @brief Abstract class for geometric multigrid smoother & residual.
 */

#ifndef DENDRO_KT_GMG_MAT_H
#define DENDRO_KT_GMG_MAT_H

#include "oda.h"
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


//TODO change DA, m_uiOctDA, da  to multi-level da, if we go that route.

// =================================
// gmgMat
// =================================
template <unsigned int dim, class LeafClass>
class gmgMat {

protected:
    static constexpr unsigned int m_uiDim = dim;

    /**@brief: pointer to OCT DA*/
    ot::DA<dim>* m_uiOctDA;

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
    gmgMat(ot::DA<dim>* da)
    {
      m_uiOctDA=da;

      // TODO get the appropriate numStrata from da/multi-level da.
      const unsigned int numStrata = m_uiMaxDepth + 1;
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
        throw std::logic_error{"smooth() not implemented by LeafClass"};
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
      //TODO get the node size at given stratum.
      PetscInt localM = m_uiOctDA->getLocalNodalSz();
      PetscInt globalM = m_uiOctDA->getGlobalNodeSz();
      MPI_Comm comm = m_uiOctDA->getGlobalComm();

      MatCreateShell(comm, localM, localM, globalM, globalM, &m_stratumWrappers[stratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_MULT, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserMultMatVec);
    }

    void petscMatCreateShellSmooth(Mat &matrixFreeMat, unsigned int stratum = 0)
    {
      //TODO get the node size at given stratum.
      PetscInt localM = m_uiOctDA->getLocalNodalSz();
      PetscInt globalM = m_uiOctDA->getGlobalNodeSz();
      MPI_Comm comm = m_uiOctDA->getGlobalComm();

      MatCreateShell(comm, localM, localM, globalM, globalM, &m_stratumWrappers[stratum], &matrixFreeMat);
      MatShellSetOperation(matrixFreeMat, MATOP_MULT, (void(*)(void)) gmgMat<dim, LeafClass>::petscUserMultSmooth);
    }


    //TODO what user mult interface do the other PETSC multigrid operators need?


    /** @brief The 'user defined' matvec we give to petsc to make a matrix-free matrix. Don't call this directly. */
    static void petscUserMultMatVec(Mat mat, Vec x, Vec y)
    {
      gmgMatStratumWrapper<dim, LeafClass> *gmgMatWrapper;
      MatShellGetContext(mat, &gmgMatWrapper);
      const unsigned int stratum = gmgMatWrapper->m_stratum;
      gmgMatWrapper->m_gmgMat->matVec(x, y, stratum);
    };

    //TODO what user mult interface do the other PETSC multigrid operators need?


#endif


};

#endif //DENDRO_KT_GMG_MAT_H
