//
// Created by masado on 04/02/21
//

#ifndef DENDRO_KT_POISSON_GMG_H
#define DENDRO_KT_POISSON_GMG_H

// Dendro
#include "distTree.h"
#include "oda.h"
#include "gmgMat.h"

// Examples defining Poisson elemental operators
#include "poissonMat.h"
#include "poissonVec.h"

namespace PoissonEq
{
  //
  // PoissonGMGMat:
  //
  // Derived class of gmgMat containing poissonMat
  // and bookkeeping for Jacobi smoother.
  //
  template <unsigned int dim>
  class PoissonGMGMat : public gmgMat<dim, PoissonGMGMat<dim>>
  {
    // References to base class members for convenience.
    using BaseT = gmgMat<dim, PoissonGMGMat<dim>>;
    const ot::MultiDA<dim> * & m_multiDA = BaseT::m_multiDA;
    /// ot::MultiDA<dim> * & m_surrogateMultiDA = BaseT::m_surrogateMultiDA;
    unsigned int & m_numStrata = BaseT::m_numStrata;
    unsigned int & m_ndofs = BaseT::m_ndofs;
    using BaseT::m_uiPtMin;
    using BaseT::m_uiPtMax;

    public:
      PoissonGMGMat(const ot::DistTree<unsigned, dim> *distTree,
                    ot::MultiDA<dim> *mda,
                    const ot::DistTree<unsigned, dim> *surrDistTree,
                    ot::MultiDA<dim> *smda,
                    ot::GridAlignment gridAlignment,
                    unsigned int ndofs)
        : BaseT(distTree, mda, surrDistTree, smda, gridAlignment, ndofs)
      {
        m_gridOperators.resize(m_numStrata);
        for (int s = 0; s < m_numStrata; ++s)
        {
          m_gridOperators[s] = new PoissonMat<dim>(&getMDA()[s], &distTree->getTreePartFiltered(s), ndofs);
          m_gridOperators[s]->setProblemDimensions(m_uiPtMin, m_uiPtMax);
        }

        m_tmpRes.resize(ndofs * getMDA()[0].getLocalNodalSz());

        m_rcp_diags.resize(m_numStrata);
        for (int s = 0; s < m_numStrata; ++s)
        {
          const double scale = 1.0;  // Set appropriately.
          m_rcp_diags[s].resize(ndofs * getMDA()[s].getLocalNodalSz(), 0.0);
          m_gridOperators[s]->setDiag(m_rcp_diags[s].data(), scale);

          for (auto &a : m_rcp_diags[s])
          {
            a = 1.0f / a;
          }
        }
      }

      virtual ~PoissonGMGMat()
      {
        for (int s = 0; s < m_numStrata; ++s)
          delete m_gridOperators[s];
      }


      void setProblemDimensions(const Point<dim>& pt_min, const Point<dim>& pt_max)
      {
        BaseT::setProblemDimensions(pt_min, pt_max);
        for (int s = 0; s < m_numStrata; ++s)
          m_gridOperators[s]->setProblemDimensions(m_uiPtMin, m_uiPtMax);
      }

      // You need to define leafMatVec() and leafApplySmoother()
      // with the same signatures as gmgMat::matVec and gmgMat::smooth(),
      // in order to create a concrete class (static polymorphism).
      //
      // Do not declare matVec() and smooth(), or else some overloads of
      // gmgMat::matVec() and gmgMat::smooth() could be hidden from name lookup.

      void leafMatVec(const VECType *in, VECType *out, unsigned int stratum = 0, double scale=1.0)
      {
        m_gridOperators[stratum]->matVec(in, out, scale);  // Global matvec.
      }

      void leafApplySmoother(const VECType *res, VECType *resLeft, unsigned int stratum)
      {
        fprintf(stdout, "Jacobi %d\n", int(stratum));
        const size_t nNodes = getMDA()[stratum].getLocalNodalSz();
        const VECType * rcp_diag = m_rcp_diags[stratum].data();

        // Jacobi
        for (int ndIdx = 0; ndIdx < m_ndofs * nNodes; ++ndIdx)
          resLeft[ndIdx] = res[ndIdx] * rcp_diag[ndIdx];
      }

      void leafPreRestriction(const VECType* in, VECType* out, double scale, unsigned int fineStratum)
      {
        for (size_t bidx : getMDA()[fineStratum].getBoundaryNodeIndices())
          for (int dof = 0; dof < this->getNdofs(); ++dof)
            out[bidx * this->getNdofs() + dof] = 0;
      }

      /** @brief Apply post-restriction condition to coarse residual. */
      void leafPostRestriction(const VECType* in, VECType* out, double scale, unsigned int fineStratum)
      {
        for (size_t bidx : getMDA()[fineStratum + 1].getBoundaryNodeIndices())
          for (int dof = 0; dof < this->getNdofs(); ++dof)
            out[bidx * this->getNdofs() + dof] = 0;
      }

      /** @brief Apply pre-prolongation condition to coarse error. */
      void leafPreProlongation(const VECType* in, VECType* out, double scale, unsigned int fineStratum)
      {
        for (size_t bidx : getMDA()[fineStratum + 1].getBoundaryNodeIndices())
          for (int dof = 0; dof < this->getNdofs(); ++dof)
            out[bidx * this->getNdofs() + dof] = 0;
      }

      /** @brief Apply post-prolongation condition to fine error. */
      void leafPostProlongation(const VECType* in, VECType* out, double scale, unsigned int fineStratum)
      {
        for (size_t bidx : getMDA()[fineStratum].getBoundaryNodeIndices())
          for (int dof = 0; dof < this->getNdofs(); ++dof)
            out[bidx * this->getNdofs() + dof] = 0;
      }

#ifdef BUILD_WITH_AMAT
      /** @brief Call aMat set_element_matrix() for each element in each grid level. */
      template<typename AMATType>
      bool getAssembledAMatStrata(AMATType** J)
      {
        for (int ii = 0; ii < this->getNumStrata(); ++ii)
          m_gridOperators[ii]->getAssembledAMat(J[ii]);
        return 0;
      }
#endif

    protected:
      std::vector<PoissonMat<dim> *> m_gridOperators;
      std::vector<VECType> m_tmpRes;
      std::vector<std::vector<VECType>> m_rcp_diags;

      // Convenience protected accessor
      inline const ot::MultiDA<dim> & getMDA() { return *m_multiDA; }
  };

}//namespace PoissonEq


#endif//DENDRO_KT_POISSON_GMG_H
