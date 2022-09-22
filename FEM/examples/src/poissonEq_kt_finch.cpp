//
// Created by milinda on 11/21/18.
// Modified by masado on 04/24/19.
// Modified by aadesh on 09/22/2022




//
// Created by milinda on 11/21/18.
//

// NOTE: #include "poissonMat.h"

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
        double * imV[dim-1];
        double * Qx[dim];

        double * phi_i;   //
        double * ematBuf; // Needed for assembly.

        double * ghostedBoundaryDofs = nullptr; // owned

        // References for convenient access to base class members.
        const ot::DA<dim> * &m_uiOctDA = feMat<dim>::m_uiOctDA;
        Point<dim> &m_uiPtMin = feMat<dim>::m_uiPtMin;
        Point<dim> &m_uiPtMax = feMat<dim>::m_uiPtMax;

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

    protected:
        PoissonMat();

    public:
        /**@brief: constructor. Matrix-free matrix depends on spatial structure represented by ODA.*/
        PoissonMat(const ot::DA<dim>* da, const std::vector<ot::TreeNode<unsigned int, dim>> *octList, unsigned int dof=1, const double * ghostedBoundaryValues = nullptr);

        PoissonMat(PoissonMat &&other);

        PoissonMat & operator=(PoissonMat &&other);

        /**@brief default destructor*/
        ~PoissonMat();

        /**@brief elemental matvec*/
        virtual void elementalMatVec(const VECType* in,VECType* out, unsigned int ndofs, const double*coords ,double scale, bool isElementBoundary);

        void elementalSetDiag(VECType *out, unsigned int ndofs, const double *coords, double scale = 1.0);

        void getElementalMatrix(std::vector<ot::MatRecord> &records, const double *coords, bool isElementBoundary);

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

// NOTE : "poissonMat.cpp"
// Created by milinda on 11/21/18.
//

//#include "poissonMat.h"
#include "mathUtils.h"

namespace PoissonEq
{

template <unsigned int dim>
PoissonMat<dim>::PoissonMat()
  : feMatrix<PoissonMat<dim>,dim>()
{
  for (unsigned int d = 0; d < dim-1; d++)
    imV[d] = nullptr;
  for (unsigned int d = 0; d < dim; d++)
    Qx[d] = nullptr;
  phi_i = nullptr;
  ematBuf = nullptr;
}

template <unsigned int dim>
PoissonMat<dim>::PoissonMat(
    const ot::DA<dim>* da,
    const std::vector<ot::TreeNode<unsigned int, dim>> *octList,
    unsigned int dof,
    const double * ghostedBoundaryValues)
  :
    feMatrix<PoissonMat<dim>,dim>(da, octList, dof)
{
    const unsigned int nPe=m_uiOctDA->getNumNodesPerElement();
    for (unsigned int d = 0; d < dim-1; d++)
      imV[d] = new double[dof*nPe];

    for (unsigned int d = 0; d < dim; d++)
      Qx[d] = new double[dof*nPe];

    phi_i = new double[(dof*nPe)];
    ematBuf = new double[(dof*nPe) * (dof*nPe)];

    if (ghostedBoundaryValues != nullptr)
    {
      const size_t numBdryDofs = dof * da->numGhostedBoundaryNodeIndices();
      this->ghostedBoundaryDofs = new double[numBdryDofs];
      std::copy_n(ghostedBoundaryValues, numBdryDofs, this->ghostedBoundaryDofs);
    }
}

template <unsigned int dim>
PoissonMat<dim>::PoissonMat(PoissonMat &&other)
  : PoissonMat()
{
  this->operator=(std::forward<PoissonMat&&>(other));
}

template <unsigned int dim>
PoissonMat<dim> & PoissonMat<dim>::operator=(PoissonMat &&other)
{
  feMatrix<PoissonMat<dim>, dim>::operator=(std::forward<PoissonMat&&>(other));
  std::swap(this->imV, other.imV);
  std::swap(this->Qx, other.Qx);
  std::swap(this->phi_i, other.phi_i);
  std::swap(this->ematBuf, other.ematBuf);
  std::swap(this->ghostedBoundaryDofs, other.ghostedBoundaryDofs);
  return *this;
}

template <unsigned int dim>
PoissonMat<dim>::~PoissonMat()
{
    for (unsigned int d = 0; d < dim-1; d++)
    {
      if (imV[d] != nullptr)
        delete [] imV[d];
      imV[d] = nullptr;
    }

    for (unsigned int d = 0; d < dim; d++)
    {
      if (Qx[d] != nullptr)
        delete [] Qx[d];
      Qx[d] = nullptr;
    }

    if (phi_i != nullptr)
      delete [] phi_i;
    phi_i = nullptr;

    if (ematBuf != nullptr)
      delete [] ematBuf;
    ematBuf = nullptr;

    if (ghostedBoundaryDofs != nullptr)
      delete [] ghostedBoundaryDofs;
}

template <unsigned int dim>
void PoissonMat<dim>::elementalMatVec(const VECType* in,VECType* out, unsigned int ndofs, const double*coords,double scale, bool isElementBoundary)
{
    if (ndofs != 1)
      throw "PoissonMat elementalMatVec() assumes scalar data, but called on non-scalar data.";

    // 1D operators.

    const RefElement* refEl=m_uiOctDA->getReferenceElement();

    const double * Q1d=refEl->getQ1d();
    const double * QT1d=refEl->getQT1d();
    const double * Dg=refEl->getDg1d();
    const double * DgT=refEl->getDgT1d();
    const double * W1d=refEl->getWgq();

    const double * mat1dPtrs[dim];


    const unsigned int eleOrder=refEl->getOrder();
    const unsigned int nPe=intPow(eleOrder+1, dim);
    const unsigned int nrp=eleOrder+1;

    Point<dim> eleMin(&coords[0*m_uiDim]);
    Point<dim> eleMax(&coords[(nPe-1)*m_uiDim]);

    const double refElSz=refEl->getElementSz();

    // Phase 1
    // Evaluate derivatives at quadrature points.
    for (unsigned int d = 0; d < dim; d++)
      mat1dPtrs[d] = Q1d;
    for (unsigned int d = 0; d < dim; d++)
    {
      const double * imFromPtrs[dim];
      double * imToPtrs[dim];

      mat1dPtrs[d] = Dg;
      getImPtrs(imFromPtrs, imToPtrs, in, Qx[d]);
      KroneckerProduct<dim, double, true>(nrp, mat1dPtrs, imFromPtrs, imToPtrs, ndofs);
      mat1dPtrs[d] = Q1d;
    }

    //Backup
    /// //x derivative
    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,Dg,in,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,Q1d,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,Q1d,imV2,Qx);

    /// //y derivative
    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,Q1d,in,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,Dg,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,Q1d,imV2,Qy);

    /// //z derivative
    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,Q1d,in,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,Q1d,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,Dg,imV2,Qz);


    const Point<dim> sz = gridX_to_X(eleMax) - gridX_to_X(eleMin);
    const Point<dim> J = sz * (1.0 / refElSz);
    // For us the Jacobian is diagonal.
    //   dx^a/du_a = J[a]
    //   det(J) = Product_a{ J[a] }
    //
    // To take the physical-space derivative, need to divide by J[a].
    // We are multiplying d{phi_i} by d{phi_j}, so need to divide by J[a] twice.
    //
    // Also the quadrature weights need to be scaled by det(J).
    //
    // scaleDQD[a] = det(J) / (J[a])*(J[a])
    //             = Product_b{ (J[b])^((-1)^delta(a,b)) }.

    double scaleDQD[dim];
    for (unsigned int d = 0; d < dim; d++)
    {
      scaleDQD[d] = 1.0;
      for (unsigned int dd = 0; dd < dim; dd++)
        scaleDQD[d] *= (dd == d ? (1.0 / J.x(dd)) : J.x(dd));
    }

    // Phase 2
    // Quadrature for each basis function.
    for (unsigned int d = 0; d < dim; d++)
      SymmetricOuterProduct<double, dim>::applyHadamardProduct(eleOrder+1, Qx[d], W1d, scaleDQD[d]);  //TODO SymmetricOuterProduct does not support ndofs.

    // Backup.
    /// for(unsigned int k=0;k<(eleOrder+1);k++)
    ///     for(unsigned int j=0;j<(eleOrder+1);j++)
    ///         for(unsigned int i=0;i<(eleOrder+1);i++)
    ///         {
    ///             Qx[k*(eleOrder+1)*(eleOrder+1)+j*(eleOrder+1)+i]*=( ((Jy*Jz)/Jx)*W1d[i]*W1d[j]*W1d[k]);
    ///             Qy[k*(eleOrder+1)*(eleOrder+1)+j*(eleOrder+1)+i]*=( ((Jx*Jz)/Jy)*W1d[i]*W1d[j]*W1d[k]);
    ///             Qz[k*(eleOrder+1)*(eleOrder+1)+j*(eleOrder+1)+i]*=( ((Jx*Jy)/Jz)*W1d[i]*W1d[j]*W1d[k]);
    ///         }


    // Phase 3
    // Back to regular-spaced points.
    for (unsigned int d = 0; d < dim; d++)
      mat1dPtrs[d] = QT1d;
    for (unsigned int d = 0; d < dim; d++)
    {
      const double * imFromPtrs[dim];
      double * imToPtrs[dim];

      mat1dPtrs[d] = DgT;
      getImPtrs(imFromPtrs, imToPtrs, Qx[d], Qx[d]);
      KroneckerProduct<dim, double, true>(nrp, mat1dPtrs, imFromPtrs, imToPtrs, ndofs);
      mat1dPtrs[d] = QT1d;
    }

    // Backup.
    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,DgT,Qx,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,QT1d,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,QT1d,imV2,Qx);

    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,QT1d,Qy,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,DgT,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,QT1d,imV2,Qy);

    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,QT1d,Qz,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,QT1d,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,DgT,imV2,Qz);

    for(unsigned int i=0;i<nPe;i++)  //TODO ndofs
    {
      out[i] = Qx[0][i];
      for (unsigned int d = 1; d < dim; d++)
        out[i]+=Qx[d][i];
    }
}



/*
template<unsigned int dim>
void PoissonMat<dim>::elementalSetDiag(VECType *out, unsigned int ndofs, const double *coords, double scale)
{
  static std::vector<ot::MatRecord> records;
  records.clear();
  this->getElementalMatrix(records, coords, false);
  #warning elementalSetDiag should also accept isElementBoundary
  for (const ot::MatRecord &rec : records)
    if (rec.getRowID() == rec.getColID() && rec.getRowDim() == rec.getColDim())
      out[ndofs * rec.getRowID() + rec.getRowDim()] = rec.getMatVal();
}
*/



template <unsigned int dim>
void PoissonMat<dim>::elementalSetDiag(VECType *out, unsigned int ndofs, const double *coords, double scale)
{
    if (ndofs != 1)
      throw "PoissonMat elementalSetDiag() assumes scalar data, but called on non-scalar data.";

    // For each basis function phi_i,
    // compute Integral_elem{ grad(phi_i) \cdot grad(phi_i) }
    //
    // = Sum_axis=a
    //   {
    //     Sum_gausspt=g
    //     {
    //       vol_scale * qweight_g * ((d_a phi_i)|eval(g))^2 / (J[a])^2
    //     }
    //   }
    //
    // For a fixed i, the factors 
    //     (d_a phi_i)|eval(g)
    // are elements of a column in one of the tensorized derivative eval matrices.
    //
    // We can compute the sum for each axis by performing a matrix-vector product,
    // where
    //   - the matrix is the entrywise square of the derivative-eval matrix
    //   - the vector is the sequence of scaled quadrature weights.
    // We need to go from indices over quadrature points to indices
    // over regular points, so we need to use DgT and QT1d .


    // 1D operators.

    const RefElement* refEl=m_uiOctDA->getReferenceElement();

    // Could avoid storing the entrywise squares if simplify KroneckerProduct
    // to allow squaring on the fly.
    const double * QT1d_sq=refEl->getQT1d_hadm2();
    const double * DgT_sq=refEl->getDgT1d_hadm2();

    const double * W1d=refEl->getWgq();

    const double * mat1dPtrs[dim];

    const unsigned int eleOrder=refEl->getOrder();
    const unsigned int nPe=intPow(eleOrder+1, dim);
    const unsigned int nrp=eleOrder+1;

    Point<dim> eleMin(&coords[0*m_uiDim]);
    Point<dim> eleMax(&coords[(nPe-1)*m_uiDim]);

    const double refElSz=refEl->getElementSz();

    const Point<dim> sz = gridX_to_X(eleMax) - gridX_to_X(eleMin);
    const Point<dim> J = sz * (1.0 / refElSz);
    // For us the Jacobian is diagonal.
    //   dx^a/du_a = J[a]
    //   det(J) = Product_a{ J[a] }
    //
    // To take the physical-space derivative, need to divide by J[a].
    // We are multiplying d{phi_i} by d{phi_j}, so need to divide by J[a] twice.
    //
    // Also the quadrature weights need to be scaled by det(J).
    //
    // scaleDQD[a] = det(J) / (J[a])*(J[a])
    //             = Product_b{ (J[b])^((-1)^delta(a,b)) }.

    double scaleDQD[dim];
    for (unsigned int d = 0; d < dim; d++)
    {
      scaleDQD[d] = 1.0;
      for (unsigned int dd = 0; dd < dim; dd++)
        scaleDQD[d] *= (dd == d ? (1.0 / J.x(dd)) : J.x(dd));
    }

    // Quadrature weights for each Gauss point.
    for (unsigned int d = 0; d < dim; d++)
    {
      for (int nIdx = 0; nIdx < nPe; nIdx++)
        Qx[d][nIdx] = 1.0f;

      SymmetricOuterProduct<double, dim>::applyHadamardProduct(eleOrder+1, Qx[d], W1d, scaleDQD[d]);
    }

    // Quadrature of square of derivative of each basis function.
    // Same sequence as third phase of elemental matvec,
    // except QT1d-->QT1d_sq and DgT-->DgT_sq.
    for (unsigned int d = 0; d < dim; d++)
      mat1dPtrs[d] = QT1d_sq;
    for (unsigned int d = 0; d < dim; d++)
    {
      const double * imFromPtrs[dim];
      double * imToPtrs[dim];

      mat1dPtrs[d] = DgT_sq;
      getImPtrs(imFromPtrs, imToPtrs, Qx[d], Qx[d]);
      KroneckerProduct<dim, double, true>(nrp, mat1dPtrs, imFromPtrs, imToPtrs, ndofs);
      mat1dPtrs[d] = QT1d_sq;
    }

    for(unsigned int i=0;i<nPe;i++)
    {
      double sum = 0.0;
      for (unsigned int d = 0; d < dim; d++)
        sum += Qx[d][i];
      out[i] = sum;
    }
}


template <unsigned int dim>
void PoissonMat<dim>::getElementalMatrix(std::vector<ot::MatRecord> &records, const double *coords, bool isElementBoundary)
{
  const RefElement* refEl=m_uiOctDA->getReferenceElement();
  const unsigned int eleOrder=refEl->getOrder();
  const unsigned int nPe=intPow(eleOrder+1, dim);
  const unsigned int nrp=eleOrder+1;
  const unsigned int ndofs = 1;

  // Populate workspace ematBuf by doing (ndofs*nPe) matvecs.

  // Zero the buffer.
  for (int ij = 0; ij < (ndofs*nPe)*(ndofs*nPe); ij++)
    ematBuf[ij] = 0;

  // Zero phi_i.
  for (int i = 0; i < (ndofs*nPe); i++)
    phi_i[i] = 0;

  const double scale = 1.0;  // Removed default scale parameter, so make it up here.
  //TODO this should be set by something.

  // To populate using matvec, need to assume column-major ordering.
  for (int j = 0; j < (ndofs*nPe); j++)
  {
    phi_i[j] = 1;  // jth basis vector.
    this->elementalMatVec(phi_i, &ematBuf[j*(ndofs*nPe)], ndofs, coords, scale, isElementBoundary );
    phi_i[j] = 0; // back to zero.
  }

  // But actually we want to store row-major, so transpose.
  for (int i = 0; i < (ndofs*nPe); i++)
    for (int j = i+1; j < (ndofs*nPe); j++)
    {
      const int idx_ij = (i*(ndofs*nPe)+j);
      const int idx_ji = (j*(ndofs*nPe)+i);
      std::swap(ematBuf[idx_ij], ematBuf[idx_ji]);
    }

  // Copy the matrix into MatRecord vector.
  for (int i = 0; i < (ndofs*nPe); i++)
    for (int j = 0; j < (ndofs*nPe); j++)
    {
      records.emplace_back(i, j, 0, 0, ematBuf[i*(ndofs*nPe)+j]);
    }
}


template <unsigned int dim>
bool PoissonMat<dim>::preMatVec(const VECType* in,VECType* out,double scale)
{
    // apply boundary conditions.
    const std::vector<size_t> &bdyIndex = m_uiOctDA->getBoundaryNodeIndices();
    const std::vector<size_t> &ghostBdyIndex = m_uiOctDA->getGhostedBoundaryNodeIndices();
    const size_t ndofs = this->ndofs();

    const size_t localBegin = m_uiOctDA->getLocalNodeBegin();
    const size_t localEnd = m_uiOctDA->getLocalNodeEnd();

    if (this->ghostedBoundaryDofs != nullptr)
      for(unsigned int i = 0; i < ghostBdyIndex.size(); i++)
        if (ghostBdyIndex[i] >= localBegin && ghostBdyIndex[i] < localEnd)
        {
          for (int dof = 0; dof < ndofs; ++dof)
            out[(ghostBdyIndex[i]-localBegin) * ndofs + dof] = this->ghostedBoundaryDofs[i * ndofs + dof];
        }
    else
      for(unsigned int i = 0; i < bdyIndex.size(); i++)
        for (int dof = 0; dof < ndofs; ++dof)
          out[bdyIndex[i] * ndofs + dof] = 0.0;  // Default 0 Dirichlet bdry

    return true;
}

template <unsigned int dim>
bool PoissonMat<dim>::postMatVec(const VECType* in,VECType* out,double scale) {

    // apply boundary conditions.
    const std::vector<size_t> &bdyIndex = m_uiOctDA->getBoundaryNodeIndices();
    const size_t ndofs = this->ndofs();

    for(unsigned int i = 0; i < bdyIndex.size(); i++)
      for (int dof = 0; dof < ndofs; ++dof)
        out[bdyIndex[i] + dof] = 0.0;  // should be 0 for any Dirichlet bdry

    return true;
}

template <unsigned int dim>
double PoissonMat<dim>::gridX_to_X(unsigned int d, double x) const
{
  double Rg=1.0;
  return (((x)/(Rg))*((m_uiPtMax.x(d)-m_uiPtMin.x(d)))+m_uiPtMin.x(d));
}

template <unsigned int dim>
Point<dim> PoissonMat<dim>::gridX_to_X(Point<dim> x) const
{
  double newCoords[dim];
  for (unsigned int d = 0; d < dim; d++)
    newCoords[d] = gridX_to_X(d, x.x(d));
  return Point<dim>(newCoords);
}

template <unsigned int dim>
int PoissonMat<dim>::cgSolve(double * x ,double * b,int max_iter, double& tol,unsigned int var)
{
    double resid,alpha,beta,rho,rho_1;
    int status=1; // 0 indicates it has solved the system within the specified max_iter, 1 otherwise.

    const unsigned int local_dof=m_uiOctDA->getLocalNodalSz();

    MPI_Comm globalComm=m_uiOctDA->getGlobalComm();

    if(m_uiOctDA->isActive())
    {

        int activeRank=m_uiOctDA->getRankActive();
        int activeNpes=m_uiOctDA->getNpesActive();

        MPI_Comm activeComm=m_uiOctDA->getCommActive();

        double* p;
        double* z;
        double* q;
        double* Ax;
        double* Ap;
        double* r0;
        double* r1;

        m_uiOctDA->createVector(p);
        m_uiOctDA->createVector(z);
        m_uiOctDA->createVector(q);

        m_uiOctDA->createVector(Ax);
        m_uiOctDA->createVector(Ap);
        m_uiOctDA->createVector(r0);
        m_uiOctDA->createVector(r1);

        double normb = normLInfty(b,local_dof,activeComm);
        par::Mpi_Bcast(&normb,1,0,activeComm);

        if(!activeRank)
            std::cout<<"normb = "<<normb<<std::endl;

        this->matVec(x,Ax);

        /*char fPrefix[256];
        sprintf(fPrefix,"%s_%d","cg",0);
        const char * varNames[]={"U"};
        const double * var[]={Ax};
        io::vtk::mesh2vtuFine(mesh,fPrefix,0,NULL,NULL,1,varNames,var);
*/
        for(unsigned int i=0;i<local_dof;i++)
        {
            r0[i]=b[i]-Ax[i];
            p[i]=r0[i];
        }


        if (normb == 0.0)
            normb = 1;

        double normr=normLInfty(r0,local_dof,activeComm);
        par::Mpi_Bcast(&normr,1,0,activeComm);
        if(!activeRank) std::cout<<"initial residual : "<<(normr/normb)<<std::endl;

        if ((resid = normr / normb) <= tol) {
            tol = resid;
            max_iter = 0;

            m_uiOctDA->destroyVector(p);
            m_uiOctDA->destroyVector(z);
            m_uiOctDA->destroyVector(q);

            m_uiOctDA->destroyVector(Ax);
            m_uiOctDA->destroyVector(Ap);
            m_uiOctDA->destroyVector(r0);
            m_uiOctDA->destroyVector(r1);

            status=0;
        }

        if(status!=0)
        {

            for(unsigned int i=1;i<=max_iter;i++)
            {

                this->matVec(p,Ap);

                alpha=(dot(r0,r0,local_dof,activeComm)/dot(p,Ap,local_dof,activeComm));
                par::Mpi_Bcast(&alpha,1,0,activeComm);

                //if(!activeRank) std::cout<<"rank: " <<activeRank<<" alpha: "<<alpha<<std::endl;
                for(unsigned int e=0;e<local_dof;e++)
                {
                    x[e]+=alpha*p[e];
                    r1[e]=r0[e]-alpha*Ap[e];
                }

                normr=normLInfty(r1,local_dof,activeComm);
                par::Mpi_Bcast(&normr,1,0,activeComm);

                if((!activeRank) && (i%10==0)) std::cout<<" iteration : "<<i<<" residual : "<<resid<<std::endl;

                if ((resid = normr / normb) <= tol) {

                    if((!activeRank)) std::cout<<" iteration : "<<i<<" residual : "<<resid<<std::endl;
                    tol = resid;
                    m_uiOctDA->destroyVector(p);
                    m_uiOctDA->destroyVector(z);
                    m_uiOctDA->destroyVector(q);

                    m_uiOctDA->destroyVector(Ax);
                    m_uiOctDA->destroyVector(Ap);
                    m_uiOctDA->destroyVector(r0);
                    m_uiOctDA->destroyVector(r1);

                    status=0;
                    break;
                }

                beta=(dot(r1,r1,local_dof,activeComm)/dot(r0,r0,local_dof,activeComm));
                par::Mpi_Bcast(&beta,1,0,activeComm);

                //if(!activeRank) std::cout<<"<r_1,r_1> : "<<dot(r1+nodeLocalBegin,r1+nodeLocalBegin,local_dof,activeComm)<<" <r_0,r_0>: "<<dot(r0+nodeLocalBegin,r0+nodeLocalBegin,local_dof,activeComm)<<" beta "<<beta<<std::endl;



                for(unsigned int e=0;e<local_dof;e++)
                {
                    p[e]=r1[e]+beta*p[e];
                    r0[e]=r1[e];
                }


            }

            if(status!=0)
            {
                tol = resid;
                m_uiOctDA->destroyVector(p);
                m_uiOctDA->destroyVector(z);
                m_uiOctDA->destroyVector(q);

                m_uiOctDA->destroyVector(Ax);
                m_uiOctDA->destroyVector(Ap);
                m_uiOctDA->destroyVector(r0);
                m_uiOctDA->destroyVector(r1);
                status=1;

            }



        }


    }


    // bcast act as a barrier for active and inactive meshes.
    par::Mpi_Bcast(&tol,1,0,globalComm);
    return status;
}

// Template instantiations.
template class PoissonMat<2u>;
template class PoissonMat<3u>;
template class PoissonMat<4u>;

}//namespace PoissonEq


// NOTE: #include "poissonVec.h"

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

    protected:
        PoissonVec();

    public:
        PoissonVec(ot::DA<dim>* da, const std::vector<ot::TreeNode<unsigned int, dim>> *octList,unsigned int dof=1);

        PoissonVec(PoissonVec &&other);

        PoissonVec & operator=(PoissonVec &&other);

        ~PoissonVec();

        /**@biref elemental compute vec for rhs*/
        virtual void elementalComputeVec(const VECType* in,VECType* out, unsigned int ndofs, const double*coords,double scale, bool isElementBoundary) override;


        bool preComputeVec(const VECType* in,VECType* out, double scale=1.0);

        bool postComputeVec(const VECType* in,VECType* out, double scale=1.0);


        /**@brief octree grid xyz to domanin xyz*/
        double gridX_to_X(unsigned int d, double x) const;
        /**@brief octree grid xyz to domanin xyz*/
        Point<dim> gridX_to_X(Point<dim> x) const;
    };

}



#endif//DENDRO_KT_POISSON_VEC_H

// NOTE : poissonVec.cpp
// Created by milinda on 11/22/18.
//

//#include "poissonVec.h"

namespace PoissonEq {

template <unsigned int dim>
PoissonVec<dim>::PoissonVec()
  : feVector<PoissonVec<dim>,dim>()
{
  for (unsigned int d = 0; d < dim-1; d++)
    imV[d] = nullptr;
}


template <unsigned int dim>
PoissonVec<dim>::PoissonVec(
    ot::DA<dim>* da,
    const std::vector<ot::TreeNode<unsigned int, dim>> *octList,
    unsigned int dof)
  :
    feVector<PoissonVec<dim>, dim>(da, octList, dof)
{
    const unsigned int nPe=m_uiOctDA->getNumNodesPerElement();
    for (unsigned int d = 0; d < dim-1; d++)
      imV[d] = new double[dof*nPe];
}

template <unsigned int dim>
PoissonVec<dim>::PoissonVec(PoissonVec &&other)
  : PoissonVec()
{
  this->operator=(std::forward<PoissonVec&&>(other));
}

template <unsigned int dim>
PoissonVec<dim> & PoissonVec<dim>::operator=(PoissonVec &&other)
{
  feVector<PoissonVec<dim>, dim>::operator=(std::forward<PoissonVec&&>(other));
  std::swap(this->imV, other.imV);
  return *this;
}


template <unsigned int dim>
PoissonVec<dim>::~PoissonVec()
{
    for (unsigned int d = 0; d < dim-1; d++)
    {
      if (imV[d] != nullptr)
        delete [] imV[d];
      imV[d] = nullptr;
    }
}

// NOTE: Maths to be generated by Finch 
template <unsigned int dim>
void PoissonVec<dim>::elementalComputeVec(const VECType* in,VECType* out, unsigned int ndofs, const double*coords,double scale, bool isElementBoundary)
{
    const RefElement* refEl=m_uiOctDA->getReferenceElement();
    const double * Q1d=refEl->getQ1d();
    const double * QT1d=refEl->getQT1d();
    const double * Dg=refEl->getDg1d();
    const double * W1d=refEl->getWgq();

    const double * mat1dPtrs[dim];

    const unsigned int eleOrder=refEl->getOrder();
    const unsigned int nPe=intPow(eleOrder+1, dim);
    const unsigned int nrp=eleOrder+1;

    Point<dim> eleMin(&coords[0*m_uiDim]);
    Point<dim> eleMax(&coords[(nPe-1)*m_uiDim]);

    // Pointers to define chains of intermediate variables.
    const double * imFromPtrs[dim];
    double * imToPtrs[dim];

    const double refElSz=refEl->getElementSz();

    // interpolate to quadrature points.
    getImPtrs(imFromPtrs, imToPtrs, in, out);
    for (unsigned int d = 0; d < dim; d++)
      mat1dPtrs[d] = Q1d;
    KroneckerProduct<dim, double, true>(nrp, mat1dPtrs, imFromPtrs, imToPtrs, ndofs);

    // Backup
    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,Q1d,in,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,Q1d,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,Q1d,imV2,out);

    // NOTE: Get From Data Structure state to co-ordinate state; Dendro does not help here:(
    const Point<dim> sz = gridX_to_X(eleMax) - gridX_to_X(eleMin);
    const Point<dim> J = sz * (1.0 / refElSz);

    double J_product = 1.0;
    for (unsigned int d = 0; d < dim; d++)
      J_product *= J.x(d);

    SymmetricOuterProduct<double, dim>::applyHadamardProduct(eleOrder+1, out, W1d, J_product);

    // apply transpose operator
    getImPtrs(imFromPtrs, imToPtrs, out, out);
    for (unsigned int d = 0; d < dim; d++)
      mat1dPtrs[d] = QT1d;
    KroneckerProduct<dim, double, true>(nrp, mat1dPtrs, imFromPtrs, imToPtrs, ndofs);

    // Backup
    /// DENDRO_TENSOR_IIAX_APPLY_ELEM(nrp,QT1d,out,imV1);
    /// DENDRO_TENSOR_IAIX_APPLY_ELEM(nrp,QT1d,imV1,imV2);
    /// DENDRO_TENSOR_AIIX_APPLY_ELEM(nrp,QT1d,imV2,out);
}




template <unsigned int dim>
bool PoissonVec<dim>::preComputeVec(const VECType* in,VECType* out, double scale)
{
    // Don't change f.

    return true;
}

template <unsigned int dim>
bool PoissonVec<dim>::postComputeVec(const VECType* in,VECType* out, double scale)
{
    // apply boundary conditions.
    const std::vector<size_t> &bdyIndex = m_uiOctDA->getBoundaryNodeIndices();
    const size_t ndofs = this->ndofs();

    for(unsigned int i = 0; i < bdyIndex.size(); i++)
      for (int dof = 0; dof < ndofs; ++dof)
        out[bdyIndex[i] + dof] = 0.0;  // should be 0 for any Dirichlet bdry

    return true;
}


template <unsigned int dim>
double PoissonVec<dim>::gridX_to_X(unsigned int d, double x) const
{
  double Rg=1.0;
  return (((x)/(Rg))*((m_uiPtMax.x(d)-m_uiPtMin.x(d)))+m_uiPtMin.x(d));
}

template <unsigned int dim>
Point<dim> PoissonVec<dim>::gridX_to_X(Point<dim> x) const
{
  double newCoords[dim];
  for (unsigned int d = 0; d < dim; d++)
    newCoords[d] = gridX_to_X(d, x.x(d));
  return Point<dim>(newCoords);
}

template class PoissonVec<2u>;
template class PoissonVec<3u>;
template class PoissonVec<4u>;

}//namespace PoissonEq


// NOTE: Core Example code Main Driver without PETSC
// NOTE: Dendro-KT
#include "treeNode.h"
#include "tsort.h"
#include "dendro.h"
#include "octUtils.h"
#include "refel.h"   // NOTE: Pull outsie of the dendro-KT; generate our own

// NOTE: Extra
#include "functional"
//#include "poissonMat.h"
//#include "poissonVec.h"
#include "mpi.h"

//#ifdef BUILD_WITH_PETSC
//  #include <petsc.h>
//  #include <petscvec.h>
//  #include <petscksp.h>
//#endif

// =======================================================
// Parameters: Change these and the options in get_args().
// =======================================================

// NOTE: Command line params; variable; affect accuracy and load balancing
struct Parameters
{
  unsigned int dim;
  unsigned int maxDepth;
  double waveletTol;
  double partitionTol;
  unsigned int eleOrder;
};
// =======================================================


// ==============================================================
// main_(): Implementation after parsing, getting dimension, etc.
// ==============================================================
template <unsigned int dim>
int main_ (Parameters &pm, MPI_Comm comm)
{   // NOTE: Setting up command line parameters 
    const unsigned int m_uiDim = dim;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    m_uiMaxDepth = pm.maxDepth;
    const double wavelet_tol = pm.waveletTol;
    const double partition_tol = pm.partitionTol;
    const unsigned int eOrder = pm.eleOrder;

    // NOTE: Setup the FEM matrix parent to child; matrix for gauss point; matrix for computing grading; it is put into Refel class like what milinda did
    double tBegin = 0, tEnd = 10, th = 0.01;
    RefElement refEl(m_uiDim,eOrder);

    /// Point<dim> grid_min(0, 0, 0);
    /// Point<dim> grid_max(1, 1, 1);

    /// Point<dim> domain_min(-0.5,-0.5,-0.5);
    /// Point<dim> domain_max(0.5,0.5,0.5);

    /// double Rg_x=(grid_max.x()-grid_min.x());
    /// double Rg_y=(grid_max.y()-grid_min.y());
    /// double Rg_z=(grid_max.z()-grid_min.z());

    /// double Rd_x=(domain_max.x()-domain_min.x());
    /// double Rd_y=(domain_max.y()-domain_min.y());
    /// double Rd_z=(domain_max.z()-domain_min.z());

    /// const Point<dim> d_min=domain_min;
    /// const Point<dim> d_max=domain_max;

    /// const Point<dim> g_min=grid_min;
    /// const Point<dim> g_max=grid_max;

    // For now must be anisotropic.
    
    // NOTE: It;s cube with displacement and stretching; simplest Mapping example; affine mapping
    double g_min = 0.0;
    double g_max = 1.0;
    double d_min = -0.5;
    double d_max =  0.5;
    double Rg = g_max - g_min;
    double Rd = d_max - d_min;
    const Point<dim> domain_min(d_min, d_min, d_min);
    const Point<dim> domain_max(d_max, d_max, d_max);

    /// std::function<void(const double *, double*)> f_rhs =[d_min,d_max,g_min,g_max,Rg_x,Rg_y,Rg_z,Rd_x,Rd_y,Rd_z](const double *x, double* var){
    ///     var[0]=(-12*M_PI*M_PI*sin(2*M_PI*(((x[0]-g_min.x())/(Rg_x))*(Rd_x)+d_min.x()))*sin(2*M_PI*(((x[1]-g_min.y())/(Rg_y))*(Rd_y)+d_min.y()))*sin(2*M_PI*(((x[2]-g_min.z())/(Rg_z))*(Rd_z)+d_min.z())));
    ///     //var[1]=(-12*M_PI*M_PI*sin(2*M_PI*(((x[0]-g_min.x())/(Rg_x))*(Rd_x)+d_min.x()))*sin(2*M_PI*(((x[1]-g_min.y())/(Rg_y))*(Rd_y)+d_min.y()))*sin(2*M_PI*(((x[2]-g_min.z())/(Rg_z))*(Rd_z)+d_min.z())));
    ///     //var[2]=(-12*M_PI*M_PI*sin(2*M_PI*(((x[0]-g_min.x())/(Rg_x))*(Rd_x)+d_min.x()))*sin(2*M_PI*(((x[1]-g_min.y())/(Rg_y))*(Rd_y)+d_min.y()))*sin(2*M_PI*(((x[2]-g_min.z())/(Rg_z))*(Rd_z)+d_min.z())));
    /// };

    // NOTE: Analytic function for forcing term F in the diff eq
    std::function<void(const double *, double*)> f_rhs = [d_min, d_max, g_min, g_max, Rg, Rd](const double *x, double *var)
    {
      var[0] = -12*M_PI*M_PI;
      for (unsigned int d = 0; d < dim; d++)
        var[0] *= sin(2*M_PI*(((x[d]-g_min)/Rg)*Rd+d_min));
    };
    
    // NOTE: unknown 
    std::function<void(const double *, double*)> f_init =[/*d_min,d_max,g_min,g_max,Rg_x,Rg_y,Rg_z,Rd_x,Rd_y,Rd_z*/](const double *x, double *var){
        var[0]=0;//(-12*M_PI*M_PI*sin(2*M_PI*(((x[0]-g_min.x())/(Rg_x))*(Rd_x)+d_min.x()))*sin(2*M_PI*(((x[1]-g_min.y())/(Rg_y))*(Rd_y)+d_min.y()))*sin(2*M_PI*(((x[2]-g_min.z())/(Rg_z))*(Rd_z)+d_min.z())));
        //var[1]=0;
        //var[2]=0;
    };

   // NOTE: creating a dist octree
    ot::DistTree<unsigned, dim> distTree =
        ot::DistTree<unsigned, dim>::constructDistTreeByFunc(f_rhs, 1, comm, eOrder, wavelet_tol, partition_tol);
    // NOTE: DA is dist Array for dist mesh; holds the co-ordinates of nodes
    ot::DA<dim> *octDA = new ot::DA<dim>(distTree, comm, eOrder, 100, partition_tol);
    // NOTE: unkown 
    const std::vector<ot::TreeNode<unsigned, dim>> &treePart = distTree.getTreePartFiltered();
    assert(treePart.size() > 0);


//#ifndef BUILD_WITH_PETSC
    //
    // Non-PETSc version.
    //

    // There are three vectors that happen to have the same sizes but are logically separate.

    // NOTE: We create three vectors 
    std::vector<double> ux, frhs, Mfrhs;
    // NOTE: ux is approx solution. frsh is evaluating function F at grid point. Mfrsh is right hand side of discrete weak form
    octDA->createVector(ux, false, false, 1);
    octDA->createVector(frhs, false, false, 1);
    octDA->createVector(Mfrhs, false, false, 1);
   
    // NOTE: contain info about the operator 
    PoissonEq::PoissonMat<dim> poissonMat(octDA, &treePart,1);
    poissonMat.setProblemDimensions(domain_min,domain_max);

    PoissonEq::PoissonVec<dim> poissonVec(octDA, &treePart,1);
    poissonVec.setProblemDimensions(domain_min,domain_max);
   
   // NOTE: init the variables
    octDA->setVectorByFunction(ux.data(),f_init,false,false,1);
    octDA->setVectorByFunction(Mfrhs.data(),f_init,false,false,1);
    octDA->setVectorByFunction(frhs.data(),f_rhs,false,false,1);

    // NOTE: compute Mfrsh from frsh using operators
    poissonVec.computeVec(&(*frhs.cbegin()), &(*Mfrhs.begin()), 1.0);


    double tol=1e-6;
    unsigned int max_iter=1000;
    // NOTE: custom solver; should be added with example cgSolve!
    poissonMat.cgSolve(&(*ux.begin()), &(*Mfrhs.begin()), max_iter, tol);

    // TODO
    // octDA->vecTopvtu(...);

    octDA->destroyVector(ux);
    octDA->destroyVector(frhs);
    octDA->destroyVector(Mfrhs);

/*#else
    //
    // PETSc version.
    //

    // There are three vectors that happen to have the same sizes but are logically separate.
    Vec ux, frhs, Mfrhs;
    octDA->petscCreateVector(ux, false, false, 1);
    octDA->petscCreateVector(frhs, false, false, 1);
    octDA->petscCreateVector(Mfrhs, false, false, 1);

    PoissonEq::PoissonMat<dim> poissonMat(octDA, &treePart,1);
    poissonMat.setProblemDimensions(domain_min,domain_max);

    PoissonEq::PoissonVec<dim> poissonVec(octDA, &treePart,1);
    poissonVec.setProblemDimensions(domain_min,domain_max);

    octDA->petscSetVectorByFunction(ux, f_init, false, false, 1);
    octDA->petscSetVectorByFunction(Mfrhs, f_init, false, false, 1);
    octDA->petscSetVectorByFunction(frhs, f_rhs, false, false, 1);

    poissonVec.computeVec(frhs, Mfrhs, 1.0);

    double tol=1e-6;
    unsigned int max_iter=1000;

    Mat matrixFreeMat;
    poissonMat.petscMatCreateShell(matrixFreeMat);

    // PETSc solver context.
    KSP ksp;
    PetscInt numIterations;

    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, matrixFreeMat, matrixFreeMat);
    KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_iter);

    KSPSolve(ksp, Mfrhs, ux);
    KSPGetIterationNumber(ksp, &numIterations);

    if (!rank)
      std::cout << " finished at iteration " << numIterations << " ...\n";

    KSPDestroy(&ksp);

    // Now that we have an approximate solution, test convergence by evaluating the residual.
    Vec residual;
    octDA->petscCreateVector(residual, false, false, 1);
    poissonMat.matVec(ux, residual);
    VecAXPY(residual, -1.0, Mfrhs);
    PetscScalar normr, normb;
    VecNorm(Mfrhs, NORM_INFINITY, &normb);
    VecNorm(residual, NORM_INFINITY, &normr);
    PetscScalar rel_resid_err = normr / normb;

    if (!rank)
      std::cout << "Final relative residual error == " << rel_resid_err << ".\n";

    // TODO
    // octDA->vecTopvtu(...);

    octDA->petscDestroyVec(ux);
    octDA->petscDestroyVec(frhs);
    octDA->petscDestroyVec(Mfrhs);
    octDA->petscDestroyVec(residual);

#endif  */

    if(!rank)
        std::cout<<" end of poissonEq: "<<std::endl;

    delete octDA;

    /// MPI_Finalize();
    return 0;
}
// ==============================================================


//
// get_args()
//
bool get_args(int argc, char * argv[], Parameters &pm, MPI_Comm comm)
{
  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  // ========================
  // Set up accepted options.
  // ========================
  enum CmdOptions                           { progName, opDim, opMaxDepth, opWaveletTol, opPartitionTol, opEleOrder, NUM_CMD_OPTIONS };
  const char *cmdOptions[NUM_CMD_OPTIONS] = { argv[0], "dim", "maxDepth", "waveletTol", "partitionTol", "eleOrder", };
  const unsigned int firstOptional = NUM_CMD_OPTIONS;  // All required.
  // ========================

  // Fill argVals.
  std::array<const char *, NUM_CMD_OPTIONS> argVals;
  argVals.fill("");
  for (unsigned int op = 0; op < argc; op++)
    argVals[op] = argv[op];

  // Check if we have the required arguments.
  if (argc < firstOptional)
  {
    if (!rProc)
    {
      std::cerr << "Usage: ";
      unsigned int op = 0;
      for (; op < firstOptional; op++)
        std::cerr << cmdOptions[op] << " ";
      for (; op < NUM_CMD_OPTIONS; op++)
        std::cerr << "[" << cmdOptions[op] << "] ";
      std::cerr << "\n";
    }
    return false;
  }

  // ================
  // Parse arguments.
  // ================
  pm.dim      = static_cast<unsigned int>(strtoul(argVals[opDim], NULL, 0));
  pm.maxDepth = static_cast<unsigned int>(strtoul(argVals[opMaxDepth], NULL, 0));
  pm.eleOrder = static_cast<unsigned int>(strtoul(argVals[opEleOrder], NULL, 0));
  pm.waveletTol   = strtod(argVals[opWaveletTol], NULL);
  pm.partitionTol = strtod(argVals[opPartitionTol], NULL);
  // ================

  // Replay arguments.
  constexpr bool replayArguments = true;
  if (replayArguments && !rProc)
  {
    for (unsigned int op = 1; op < NUM_CMD_OPTIONS; op++)
      std::cout << YLW << cmdOptions[op] << "==" << argVals[op] << NRM << " \n";
    std::cout << "\n";
  }

  return true;
}


//
// main()
//
int main(int argc, char * argv[])
{
//#ifndef BUILD_WITH_PETSC
  MPI_Init(&argc, &argv);
//#else
//  PetscInitialize(&argc, &argv, NULL, NULL);
//#endif
  int returnCode = 1;
  DendroScopeBegin();

  int rProc, nProc;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  Parameters pm;
  unsigned int &dim = pm.dim;
  if (get_args(argc, argv, pm, comm))  // NOTE: populates pm using argv and argc which are the command line args 
  {
    int synchronize;
    MPI_Bcast(&synchronize, 1, MPI_INT, 0, comm);
    // NOTE: setup lookup table needed interanally in dendro; unknow ; space filling curv 
    _InitializeHcurve(dim);

    // Convert dimension argument to template parameter.
    switch(dim)
    {
      case 2: returnCode = main_<2>(pm, comm); break;
      case 3: returnCode = main_<3>(pm, comm); break;
      case 4: returnCode = main_<4>(pm, comm); break;
      default:
        if (!rProc)
          std::cerr << "Dimension " << dim << " not currently supported.\n";
    }

    _DestroyHcurve();
  }

//#ifndef BUILD_WITH_PETSC
  MPI_Finalize();
//#else
//  PetscFinalize();
//#endif
  DendroScopeEnd();

  return returnCode;
}
