//
// Created by milinda on 10/30/18.
//
/**
 * @brief class that derived from abstract class feMat
 * LHS computation of the weak formulation
 * */
#ifndef DENDRO_KT_FEMATRIX_H
#define DENDRO_KT_FEMATRIX_H

#include "feMat.h"
#include "matvec.h"
#include "refel.h"

template <typename LeafT, unsigned int dim>
class feMatrix : public feMat<dim> {
  //TODO I don't really get why we use LeafT and not just virtual methods.

protected:
         static constexpr unsigned int m_uiDim = dim;

         /**@brief number of dof*/
         unsigned int m_uiDof;

         /**@brief element nodal vec in */
         VECType * m_uiEleVecIn;

         /***@brief element nodal vecOut */
         VECType * m_uiEleVecOut;

         /** elemental coordinates */
         double * m_uiEleCoords;

    public:
        /**
         * @brief constructs an FEM stiffness matrix class.
         * @param[in] da: octree DA
         * */
        feMatrix(ot::DA<dim>* da,unsigned int dof=1);

        feMatrix(feMatrix &&other);

        ~feMatrix();

        /**@brief Computes the LHS of the weak formulation, normally the stifness matrix times a given vector.
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] default parameter scale vector by scale*Ku
        * */
        virtual void matVec(const VECType* in,VECType* out, double scale=1.0);


        /**@brief Computes the elemental matvec
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] scale vector by scale*Ku
        **/
        virtual void elementalMatVec(const VECType *in, VECType *out, unsigned int ndofs, const double *coords, double scale) = 0;



#ifdef BUILD_WITH_PETSC

        /**@brief Computes the LHS of the weak formulation, normally the stifness matrix times a given vector.
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] default parameter scale vector by scale*Ku
        * */
        virtual void matVec(const Vec& in,Vec& out, double scale=1.0);

        /**
         * @brief Performs the matrix assembly.
         * @param [in/out] J: Matrix assembled
         * @param [in] mtype: Matrix type
         * when the function returns, J is set to assembled matrix
         **/
        virtual bool getAssembledMatrix(Mat *J, MatType mtype);


#endif

        /**@brief static cast to the leaf node of the inheritance*/
        LeafT& asLeaf() { return static_cast<LeafT&>(*this);}


        /**
         * @brief executed just before  the matVec loop in matvec function
         * @param[in] in : input Vector
         * @param[out] out: output vector
         * @param[in] scale: scalaing factror
         **/

        bool preMatVec(const VECType* in, VECType* out,double scale=1.0) {
            // If this is asLeaf().preMatVec(), i.e. there is not an override, don't recurse.
            static bool entered = false;
            bool ret = false;
            if (!entered)
            {
              entered = true;
              ret = asLeaf().preMatVec(in,out,scale);
              entered = false;
            }
            return ret;
        }


        /**@brief executed just after the matVec loop in matvec function
         * @param[in] in : input Vector
         * @param[out] out: output vector
         * @param[in] scale: scalaing factror
         * */

        bool postMatVec(const VECType* in, VECType* out,double scale=1.0) {
            // If this is asLeaf().postMatVec(), i.e. there is not an override, don't recurse.
            static bool entered = false;
            bool ret = false;
            if (!entered)
            {
              entered = true;
              ret = asLeaf().postMatVec(in,out,scale);
              entered = false;
            }
            return ret;
        }

        /**@brief executed before the matrix assembly */
        bool preMat() {
            // If this is asLeaf().preMat(), i.e. there is not an override, don't recurse.
            static bool entered = false;
            bool ret = false;
            if (!entered)
            {
              entered = true;
              ret = asLeaf().preMat();
              entered = false;
            }
            return ret;
        }

        /**@brief executed after the matrix assembly */
        bool postMat() {
            // If this is asLeaf().postMat(), i.e. there is not an override, don't recurse.
            static bool entered = false;
            bool ret = false;
            if (!entered)
            {
              entered = true;
              ret = asLeaf().postMat();
              entered = false;
            }
            return ret;
        }

        /// /**
        ///  * @brief Compute the elemental Matrix.
        ///  * @param[in] eleID: element ID
        ///  * @param[out] records: records corresponding to the elemental matrix.
        ///  * */
        /// void getElementalMatrix(unsigned int eleID, std::vector<ot::MatRecord>& records)
        /// {
        ///     return asLeaf().getElementalMatrix(eleID,records);
        /// }


};

template <typename LeafT, unsigned int dim>
feMatrix<LeafT,dim>::feMatrix(ot::DA<dim>* da,unsigned int dof) : feMat<dim>(da)
{
    m_uiDof=dof;
    const unsigned int nPe=feMat<dim>::m_uiOctDA->getNumNodesPerElement();
    m_uiEleVecIn = new  VECType[m_uiDof*nPe];
    m_uiEleVecOut = new VECType[m_uiDof*nPe];

    m_uiEleCoords= new double[m_uiDim*nPe];

}

template <typename LeafT, unsigned int dim>
feMatrix<LeafT, dim>::feMatrix(feMatrix &&other)
  : feMat<dim>(std::forward<feMat<dim>>(other)),
    m_uiDof{other.m_uiDof},
    m_uiEleVecIn{other.m_uiEleVecIn},
    m_uiEleVecOut{other.m_uiEleVecOut},
    m_uiEleCoords{other.m_uiEleCoords}
{
  other.m_uiEleVecIn = nullptr;
  other.m_uiEleVecOut = nullptr;
  other.m_uiEleCoords = nullptr;
}


template <typename LeafT, unsigned int dim>
feMatrix<LeafT,dim>::~feMatrix()
{
    if (m_uiEleVecIn != nullptr)
      delete [] m_uiEleVecIn;
    if (m_uiEleVecOut != nullptr)
      delete [] m_uiEleVecOut;
    if (m_uiEleCoords != nullptr)
      delete [] m_uiEleCoords;

    m_uiEleVecIn=NULL;
    m_uiEleVecOut=NULL;
    m_uiEleCoords=NULL;
}

template <typename LeafT, unsigned int dim>
void feMatrix<LeafT,dim>::matVec(const VECType *in, VECType *out, double scale)
{
  using namespace std::placeholders;   // Convenience for std::bind().

  // Shorter way to refer to our member DA.
  ot::DA<dim> * &m_oda = feMat<dim>::m_uiOctDA;

  // Static buffers for ghosting. Check/increase size.
  static std::vector<VECType> inGhosted, outGhosted;
  m_oda->template createVector<VECType>(inGhosted, false, true, m_uiDof);
  m_oda->template createVector<VECType>(outGhosted, false, true, m_uiDof);
  VECType *inGhostedPtr = inGhosted.data();
  VECType *outGhostedPtr = outGhosted.data();

  // 1. Copy input data to ghosted buffer.
  m_oda->template nodalVecToGhostedNodal<VECType>(in, inGhostedPtr, true, m_uiDof);

  // 1.a. Override input data with pre-matvec initialization.
  preMatVec(in, inGhostedPtr + m_oda->getLocalNodeBegin(), scale);
  // TODO what is the return value supposed to represent?

#ifdef DENDRO_KT_MATVEC_BENCH_H
  bench::t_ghostexchange.start();
#endif

  // 2. Upstream->downstream ghost exchange.
  m_oda->template readFromGhostBegin<VECType>(inGhostedPtr, m_uiDof);
  m_oda->template readFromGhostEnd<VECType>(inGhostedPtr, m_uiDof);

#ifdef DENDRO_KT_MATVEC_BENCH_H
  bench::t_ghostexchange.stop();
#endif

  // 3. Local matvec().
  const auto * tnCoords = m_oda->getTNCoords();
  std::function<void(const VECType *, VECType *, unsigned int, const double *, double)> eleOp =
      std::bind(&feMatrix<LeafT,dim>::elementalMatVec, this, _1, _2, _3, _4, _5);

#ifdef DENDRO_KT_MATVEC_BENCH_H
  bench::t_matvec.start();
#endif
  fem::matvec(inGhostedPtr, outGhostedPtr, m_uiDof, tnCoords, m_oda->getTotalNodalSz(),
      *m_oda->getTreePartFront(), *m_oda->getTreePartBack(),
      eleOp, scale, m_oda->getReferenceElement());
  //TODO I think refel won't always be provided by oda.

#ifdef DENDRO_KT_MATVEC_BENCH_H
  bench::t_matvec.stop();
#endif


#ifdef DENDRO_KT_MATVEC_BENCH_H
  bench::t_ghostexchange.start();
#endif

  // 4. Downstream->Upstream ghost exchange.
  m_oda->template writeToGhostsBegin<VECType>(outGhostedPtr, m_uiDof);
  m_oda->template writeToGhostsEnd<VECType>(outGhostedPtr, m_uiDof);

#ifdef DENDRO_KT_MATVEC_BENCH_H
  bench::t_ghostexchange.stop();
#endif

  // 5. Copy output data from ghosted buffer.
  m_oda->template ghostedNodalToNodalVec<VECType>(outGhostedPtr, out, true, m_uiDof);

  // 5.a. Override output data with post-matvec re-initialization.
  postMatVec(outGhostedPtr + m_oda->getLocalNodeBegin(), out, scale);
  // TODO what is the return value supposed to represent?
}

#ifdef BUILD_WITH_PETSC

template <typename LeafT, unsigned int dim>
void feMatrix<LeafT,dim>::matVec(const Vec &in, Vec &out, double scale)
{

    const PetscScalar * inArry=NULL;
    PetscScalar * outArry=NULL;

    VecGetArrayRead(in,&inArry);
    VecGetArray(out,&outArry);

    matVec(inArry,outArry,scale);

    VecRestoreArrayRead(in,&inArry);
    VecRestoreArray(out,&outArry);

}



template <typename LeafT, unsigned int dim>
bool feMatrix<LeafT,dim>::getAssembledMatrix(Mat *J, MatType mtype)
{
    // todo we can skip this part for sc. 
    // Octree part ...
    /*char matType[30];
    PetscBool typeFound;
    PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-fullJacMatType", matType, 30, &typeFound);
    if(!typeFound) {
        std::cout<<"[Error]"<<__func__<<" I need a MatType for the full Jacobian matrix!"<<std::endl;
        MPI_Finalize();
        exit(0);
    }

    m_uiOctDA->createMatrix(*J, matType, 1);
    MatZeroEntries(*J);
    std::vector<ot::MatRecord> records;

    preMat();

    for(m_uiOctDA->init<ot::DA_FLAGS::WRITABLE>(); m_uiOctDA->curr() < m_uiOctDA->end<ot::DA_FLAGS::WRITABLE>();m_uiOctDA->next<ot::DA_FLAGS::WRITABLE>()) {
        getElementalMatrix(m_uiOctDA->curr(), records);
        if(records.size() > 500) {
            m_uiOctDA->petscSetValuesInMatrix(*J, records, 1, ADD_VALUES);
        }
    }//end writable
    m_uiOctDA->petscSetValuesInMatrix(*J, records, 1, ADD_VALUES);

    postMat();

    MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);

    PetscFunctionReturn(0);*/
}


#endif



#endif //DENDRO_KT_FEMATRIX_H
