//
// Created by milinda on 11/22/18.
//

#ifndef DENDRO_KT_POISSON_VEC_H
#define DENDRO_KT_POISSON_VEC_H

#include "oda.h"
#include "feVector.h"

namespace PoissonEq
{
    template <unsigned int dim>
    class PoissonVec : public feVector<PoissonVec<dim>,dim> {

    private:

        double * imV[dim-1];

        ot::DA<dim> * &m_uiOctDA = feVec<dim>::m_uiOctDA;
        Point<dim> &m_uiPtMin = feVec<dim>::m_uiPtMin;
        Point<dim> &m_uiPtMax = feVec<dim>::m_uiPtMax;

        static constexpr unsigned int m_uiDim = dim;

        void getImPtrs(const double * fromPtrs[], double * toPtrs[], const double *in, double *out) const
        {
          fromPtrs[0] = in;
          toPtrs[dim-1] = out;
          for (unsigned int d = 0; d < dim-1; d++)
          {
            toPtrs[d] = imV[d];
            fromPtrs[d+1] = toPtrs[d];
          }
        }


    public:
        PoissonVec(ot::DA<dim>* da,unsigned int dof=1);
        ~PoissonVec();

        /**@biref elemental compute vec for rhs*/
        virtual void elementalComputeVec(const VECType* in,VECType* out, unsigned int ndofs, double*coords=NULL,double scale=1.0) override;


        bool preComputeVec(const VECType* in,VECType* out, double scale=1.0);

        bool postComputeVec(const VECType* in,VECType* out, double scale=1.0);


        /**@brief octree grid xyz to domanin xyz*/
        double gridX_to_X(unsigned int d, double x) const;
        /**@brief octree grid xyz to domanin xyz*/
        Point<dim> gridX_to_X(Point<dim> x) const;
    };

    template class PoissonVec<2u>;
    template class PoissonVec<3u>;
    template class PoissonVec<4u>;
}



#endif//DENDRO_KT_POISSON_VEC_H
