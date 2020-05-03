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
#include "setDiag.h"
#include "matRecord.h"

#include <exception>

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

        virtual void setDiag(VECType *out, double scale = 1.0);


        /**@brief Computes the elemental matvec
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] scale vector by scale*Ku
        **/
        virtual void elementalMatVec(const VECType *in, VECType *out, unsigned int ndofs, const double *coords, double scale) = 0;

        /**@brief Sets the diagonal of the elemental matrix.
         * @param [out] out output vector diag(K)
         * @param [in] in coords physical space coordinates of the element.
         * @param [in] in scale diagonal by scale*diag(K).
         * Leaf class responsible to implement (static polymorphism).
         */
        void elementalSetDiag(VECType *out, unsigned int ndofs, const double *coords, double scale)
        {
          static bool reentrant = false;
          if (reentrant)
            throw std::logic_error{"elementalSetDiag() not implemented by feMatrix leaf derived class"};
          reentrant = true;
          {
            asLeaf().elementalSetDiag(out, ndofs, coords, scale);
          }
          reentrant = false;
        }



        /**
         * @brief Collect all matrix entries relative to current rank.
         * 
         * If you need to do a few rows at a time, use this method as a pattern.
         */
        ot::MatCompactRows collectMatrixEntries();


#ifdef BUILD_WITH_PETSC

        /**@brief Computes the LHS of the weak formulation, normally the stifness matrix times a given vector.
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] default parameter scale vector by scale*Ku
        * */
        virtual void matVec(const Vec& in,Vec& out, double scale=1.0);

        virtual void setDiag(Vec& out, double scale = 1.0);


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
        const LeafT& asLeaf() const { return static_cast<const LeafT&>(*this);}


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

        /**
         * @brief Call application method to build the elemental matrix.
         * @param[in] coords : elemental coordinates
         * @param[out] records: records corresponding to the elemental matrix.
         * */
        void getElementalMatrix(std::vector<ot::MatRecord> &records, const double *coords, const ot::RankI *globNodeIds)
        {
          // If this IS asLeaf().getElementalMatrix(), i.e. there is not an override, don't recurse.
          static bool entered = false;
          if (!entered)
          {
            entered = true;
            asLeaf().getElementalMatrix(records, coords, globNodeIds);
            entered = false;
          }
          else
            throw std::logic_error("Application didn't override feMatrix::getElementalMatrix().");
        }


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
  std::fill(outGhosted.begin(), outGhosted.end(), 0);
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


template <typename LeafT, unsigned int dim>
void feMatrix<LeafT,dim>::setDiag(VECType *out, double scale)
{
  using namespace std::placeholders;   // Convenience for std::bind().

  // Shorter way to refer to our member DA.
  ot::DA<dim> * &m_oda = feMat<dim>::m_uiOctDA;

  // Static buffers for ghosting. Check/increase size.
  static std::vector<VECType> outGhosted;
  m_oda->template createVector<VECType>(outGhosted, false, true, m_uiDof);
  std::fill(outGhosted.begin(), outGhosted.end(), 0);
  VECType *outGhostedPtr = outGhosted.data();

  // Local setDiag().
  const auto * tnCoords = m_oda->getTNCoords();
  std::function<void(VECType *, unsigned int, const double *, double)> eleSet =
      std::bind(&feMatrix<LeafT,dim>::elementalSetDiag, this, _1, _2, _3, _4);

#ifdef DENDRO_KT_GMG_BENCH_H
  bench::t_matvec.start();
#endif
  fem::locSetDiag(outGhostedPtr, m_uiDof, tnCoords, m_oda->getTotalNodalSz(),
      *m_oda->getTreePartFront(), *m_oda->getTreePartBack(),
      eleSet, scale, m_oda->getReferenceElement());

#ifdef DENRO_KT_GMG_BENCH_H
  bench::t_matvec.stop();
#endif


#ifdef DENRO_KT_GMG_BENCH_H
  bench::t_ghostexchange.start();
#endif

  // Downstream->Upstream ghost exchange.
  m_oda->template writeToGhostsBegin<VECType>(outGhostedPtr, m_uiDof);
  m_oda->template writeToGhostsEnd<VECType>(outGhostedPtr, m_uiDof);

#ifdef DENRO_KT_GMG_BENCH_H
  bench::t_ghostexchange.stop();
#endif

  // 5. Copy output data from ghosted buffer.
  m_oda->template ghostedNodalToNodalVec<VECType>(outGhostedPtr, out, true, m_uiDof);
}



template <typename LeafT, unsigned int dim>
ot::MatCompactRows feMatrix<LeafT, dim>::collectMatrixEntries()
{
  const ot::DA<dim> &m_oda = *feMat<dim>::m_uiOctDA;
  const unsigned int eleOrder = m_oda.getElementOrder();
  const unsigned int nPe = m_oda.getNumNodesPerElement();
  ot::MatCompactRows matRowChunks(nPe, m_uiDof);

  // Loop over all elements, adding row chunks from elemental matrices.
  // Get the node indices on an element using MatvecBaseIn<dim, unsigned int, false>.

  if (m_oda.isActive())
  {
    using CoordT = typename ot::DA<dim>::C;
    using ot::RankI;
    using ScalarT = typename ot::MatCompactRows::ScalarT;
    using IndexT = typename ot::MatCompactRows::IndexT;

    const size_t ghostedNodalSz = m_oda.getTotalNodalSz();
    const ot::TreeNode<CoordT, dim> *odaCoords = m_oda.getTNCoords();
    const std::vector<RankI> &ghostedGlobalNodeId = m_oda.getNodeLocalToGlobalMap();

    std::vector<ot::MatRecord> elemRecords;
    std::vector<IndexT> rowIdxBuffer;
    std::vector<IndexT> colIdxBuffer;
    std::vector<ScalarT> colValBuffer;

    const bool visitEmpty = false;
    const unsigned int padLevel = 0;
    ot::MatvecBaseIn<dim, RankI, false> treeLoopIn(ghostedNodalSz,
                                                   m_uiDof,
                                                   eleOrder,
                                                   visitEmpty,
                                                   padLevel,
                                                   odaCoords,
                                                   &(*ghostedGlobalNodeId.cbegin()),
                                                   *m_oda.getTreePartFront(),
                                                   *m_oda.getTreePartBack());

    // Iterate over all leafs of the local part of the tree.
    while (!treeLoopIn.isFinished())
    {
      const ot::TreeNode<CoordT, dim> subtree = treeLoopIn.getCurrentSubtree();
      const auto subtreeInfo = treeLoopIn.subtreeInfo();

      if (treeLoopIn.isPre() && subtreeInfo.isLeaf())
      {
        const double * nodeCoordsFlat = subtreeInfo.getNodeCoords();
        const RankI * nodeIdsFlat = subtreeInfo.readNodeValsIn();

        // Get elemental matrix for the current leaf element.
        elemRecords.clear();
        this->getElementalMatrix(elemRecords, nodeCoordsFlat, nodeIdsFlat);
        std::sort(elemRecords.begin(), elemRecords.end());

#ifdef __DEBUG__
        if (!elemRecords.size())
          fprintf(stderr, "getElementalMatrix() did not return any rows! (%s:%lu)\n", __FILE__, __LINE__);
#endif// __DEBUG__

        rowIdxBuffer.clear();
        colIdxBuffer.clear();
        colValBuffer.clear();

        // Copy elemental matrix to sorted order.
        size_t countEntries = 0;
        for (const ot::MatRecord &rec : elemRecords)
        {
          const IndexT rowIdx = rec.getRowID() * m_uiDof + rec.getRowDim();
          if (rowIdxBuffer.size() == 0 || rowIdx != rowIdxBuffer.back())
          {
#ifdef __DEBUG__
            if (countEntries != 0 && countEntries != nPe * m_uiDof)
              fprintf(stderr, "Unexpected #entries in row of elemental matrix, "
                              "RowId==%lu RowDim==%lu. Expected %u, got %u.\n",
                              rec.getRowID(), rec.getRowDim(), nPe*m_uiDof, countEntries);
#endif// __DEBUG__
            countEntries = 0;
            rowIdxBuffer.push_back(rowIdx);
          }
          colIdxBuffer.push_back(rec.getColID() * m_uiDof + rec.getColDim());
          colValBuffer.push_back(rec.getMatVal());
        }

        // TODO p2c and c2p matrix multiplications if element has hanging nodes.
#warning TODO use p2c and c2p to support hanging nodes in feMatrix::collectMatrixEntries().

        // Collect the rows of the elemental matrix into matRowChunks.
        for (unsigned int r = 0; r < rowIdxBuffer.size(); r++)
        {
          matRowChunks.appendChunk(rowIdxBuffer[r],
                                   &colIdxBuffer[r * nPe * m_uiDof],
                                   &colValBuffer[r * nPe * m_uiDof]);
        }
      }
      treeLoopIn.step();
    }
  }

  return matRowChunks;
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
void feMatrix<LeafT, dim>::setDiag(Vec& out, double scale)
{
  PetscScalar * outArry=NULL;
  VecGetArray(out,&outArry);

  setDiag(outArry, scale);

  VecRestoreArray(out,&outArry);
}


template <typename LeafT, unsigned int dim>
bool feMatrix<LeafT,dim>::getAssembledMatrix(Mat *J, MatType mtype)
{
  assert(!"Not implemented!");

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
