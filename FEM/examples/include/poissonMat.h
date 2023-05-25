//
// Created by milinda on 11/21/18.
//

#ifndef DENDRO_KT_POISSON_MAT_H
#define DENDRO_KT_POISSON_MAT_H

#include "oda.h"
#include "feMatrix.h"

namespace PoissonEq
{
    template <unsigned int dim>
    class PoissonMat : public feMatrix<PoissonMat<dim>, dim>{

    private:
        // some additional work space variables to perform elemental MatVec
        double * imV[dim-1] = {};
        double * Qx[dim] = {};

        double * phi_i = {};   //
        double * ematBuf = {}; // Needed for assembly.

        static constexpr unsigned int m_uiDim = dim;

        bool m_zero_boundary = false;

    private:
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
        PoissonMat() = default;

        /**@brief: constructor. Matrix-free matrix depends on spatial structure represented by ODA.*/
        PoissonMat(const ot::DA<dim>* da, const std::vector<ot::TreeNode<unsigned int, dim>> *octList, unsigned int dof=1);

        PoissonMat(const PoissonMat &) = delete;
        PoissonMat & operator=(const PoissonMat &) = delete;

        PoissonMat(PoissonMat &&);
        PoissonMat & operator=(PoissonMat &&);

        /**@brief default destructor*/
        ~PoissonMat();

        /**@brief elemental matvec*/
        virtual void elementalMatVec(const VECType* in,VECType* out, unsigned int ndofs, const double*coords ,double scale, bool isElementBoundary);

        void elementalSetDiag(VECType *out, unsigned int ndofs, const double *coords, double scale = 1.0);

        void getElementalMatrix(std::vector<ot::MatRecord> &records, const double *coords, bool isElementBoundary);

        bool zero_boundary() const { return m_zero_boundary; }
        void zero_boundary(bool enable) { m_zero_boundary = enable; }

        /**@brief things need to be performed before matvec (i.e. coords transform)*/
        bool preMatVec(const VECType* in,VECType* out,double scale=1.0);

        /**@brief things need to be performed after matvec (i.e. coords transform)*/
        bool postMatVec(const VECType* in,VECType* out,double scale=1.0);

        /**@brief octree grid xyz to domanin xyz*/
        double gridX_to_X(unsigned int d, double x) const;
        /**@brief octree grid xyz to domanin xyz*/
        Point<dim> gridX_to_X(Point<dim> x) const;

        int cgSolve(double * x ,double * b,int max_iter, double& tol,unsigned int var=0);
    };

}




#endif//DENDRO_KT_POISSON_MAT_H
