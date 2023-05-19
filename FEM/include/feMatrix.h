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
         // future: remove the octlist parameter
        feMatrix(const ot::DA<dim>* da, const std::vector<ot::TreeNode<unsigned int, dim>> *, unsigned int dof=1);

        feMatrix(feMatrix &&other);

        ~feMatrix();

        unsigned ndofs() const { return m_uiDof; }

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
        virtual void elementalMatVec(const VECType *in, VECType *out, unsigned int ndofs, const double *coords, double scale, bool isElementBoundary ) = 0;

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
        template <typename AssembleElemental>
        void collectMatrixEntries(AssembleElemental assemble_e);


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

        //TODO Output channel of preMatVec may be unexpected, should clarify use case.
        //     In particular, need to know what input is needed for general boundary conditions,
        //     which would tell us whether we can allow aliasing (in==out)
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

        //TODO Input channel of postMatVec may be unexpected, should clarify use case.
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
         * @note You should set the row/col ids using the LOCAL elemental lexicographic ordering.
         * */
        void getElementalMatrix(std::vector<ot::MatRecord> &records, const double *coords, bool isElementBoundary)
        {
          // If this IS asLeaf().getElementalMatrix(), i.e. there is not an override, don't recurse.
          static bool entered = false;
          if (!entered)
          {
            entered = true;
            asLeaf().getElementalMatrix(records, coords, isElementBoundary);
            entered = false;
          }
          else
            throw std::logic_error("Application didn't override feMatrix::getElementalMatrix().");
        }


};

template <typename LeafT, unsigned int dim>
feMatrix<LeafT,dim>::feMatrix(const ot::DA<dim>* da, const std::vector<ot::TreeNode<unsigned int, dim>> *, unsigned int dof)
  : feMat<dim>(da)
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
  const ot::DA<dim> * m_oda = this->da();

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
  preMatVec(in, inGhostedPtr + this->ndofs() * m_oda->getLocalNodeBegin(), scale);
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
  std::function<void(const VECType *, VECType *, unsigned int, const double *, double, bool)> eleOp =
      std::bind(&feMatrix<LeafT,dim>::elementalMatVec, this, _1, _2, _3, _4, _5, _6);

#ifdef DENDRO_KT_MATVEC_BENCH_H
  bench::t_matvec.start();
#endif
  fem::matvec(inGhostedPtr, outGhostedPtr, m_uiDof, tnCoords, m_oda->getTotalNodalSz(),
      &(*this->octList()->cbegin()), this->octList()->size(),
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
  postMatVec(outGhostedPtr + this->ndofs() * m_oda->getLocalNodeBegin(), out, scale);
  // TODO what is the return value supposed to represent?
}


template <typename LeafT, unsigned int dim>
void feMatrix<LeafT,dim>::setDiag(VECType *out, double scale)
{
  using namespace std::placeholders;   // Convenience for std::bind().

  // Shorter way to refer to our member DA.
  const ot::DA<dim> * m_oda = this->da();

  // Static buffers for ghosting. Check/increase size.
  static std::vector<VECType> outGhosted;
  /// static std::vector<char> outDirty;
  m_oda->template createVector<VECType>(outGhosted, false, true, m_uiDof);
  /// m_oda->template createVector<char>(outDirty, false, true, 1);
  std::fill(outGhosted.begin(), outGhosted.end(), 0);
  VECType *outGhostedPtr = outGhosted.data();

  // Local setDiag().
  const auto * tnCoords = m_oda->getTNCoords();
  std::function<void(VECType *, unsigned int, const double *, double)> eleSet =
      std::bind(&feMatrix<LeafT,dim>::elementalSetDiag, this, _1, _2, _3, _4);

#ifdef DENDRO_KT_GMG_BENCH_H
  bench::t_matvec.start();
#endif
  fem::locSetDiag(outGhostedPtr, m_uiDof, tnCoords, m_oda->getTotalNodalSz(), this->octList(),
      *m_oda->getTreePartFront(), *m_oda->getTreePartBack(),
      eleSet, scale, m_oda->getReferenceElement());
      /// outDirty.data());

#ifdef DENRO_KT_GMG_BENCH_H
  bench::t_matvec.stop();
#endif


#ifdef DENRO_KT_GMG_BENCH_H
  bench::t_ghostexchange.start();
#endif

  // Downstream->Upstream ghost exchange.
  m_oda->template writeToGhostsBegin<VECType>(outGhostedPtr, m_uiDof);
  m_oda->template writeToGhostsEnd<VECType>(outGhostedPtr, m_uiDof);
  /// m_oda->template writeToGhostsBegin<VECType>(outGhostedPtr, m_uiDof, outDirty.data());
  /// m_oda->template writeToGhostsEnd<VECType>(outGhostedPtr, m_uiDof, false, outDirty.data());

#ifdef DENRO_KT_GMG_BENCH_H
  bench::t_ghostexchange.stop();
#endif

  // 5. Copy output data from ghosted buffer.
  m_oda->template ghostedNodalToNodalVec<VECType>(outGhostedPtr, out, true, m_uiDof);
}


template <typename LeafT, unsigned int dim>
template <typename AssembleElemental>
void feMatrix<LeafT, dim>::collectMatrixEntries(AssembleElemental assemble_e)
{
  const ot::DA<dim> &m_oda = *this->da();
  const unsigned int eleOrder = m_oda.getElementOrder();
  const unsigned int nPe = m_oda.getNumNodesPerElement();
  const unsigned int ndofs = m_uiDof;
  const unsigned int n = nPe * ndofs;
  const unsigned int n_squared = n * n;

  // Loop over all elements, adding row chunks from elemental matrices.
  // Get the node indices on an element using MatvecBaseIn<dim, unsigned int, false>.

  if (m_oda.isActive())
  {
    using CoordT = typename ot::DA<dim>::C;
    using ot::RankI;

#ifdef BUILD_WITH_PETSC
    using ScalarT = PetscScalar;
    using IndexT = PetscInt;
#else
    using ScalarT = DendroScalar;
    using IndexT = long long unsigned;
#endif

    const size_t ghostedNodalSz = m_oda.getTotalNodalSz();
    const ot::TreeNode<CoordT, dim> *odaCoords = m_oda.getTNCoords();
    const std::vector<RankI> &ghostedGlobalNodeId = m_oda.getNodeLocalToGlobalMap();

    std::vector<ot::MatRecord> elemRecords;
    std::vector<PetscInt> rowIdxBuffer(n);
    std::vector<ScalarT> colValBuffer(n_squared);

    InterpMatrices<dim, ScalarT> interp_matrices(eleOrder);
    std::vector<ScalarT> wksp_col(n);
    std::vector<ScalarT> wksp_mat(n_squared);

    std::vector<ScalarT> nonhanging(n);

    const bool visitEmpty = false;
    const unsigned int padLevel = 0;
    ot::MatvecBaseIn<dim, RankI, false> treeLoopIn(ghostedNodalSz,
                                                   1,                // node id is scalar
                                                   eleOrder,
                                                   visitEmpty,
                                                   padLevel,
                                                   odaCoords,
                                                   &(*ghostedGlobalNodeId.cbegin()),
                                                   &(*this->octList()->cbegin()),
                                                   this->octList()->size(),
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
        this->getElementalMatrix(elemRecords, nodeCoordsFlat, subtreeInfo.isElementBoundary());

#ifdef __DEBUG__
        if (!elemRecords.size())
          fprintf(stderr, "getElementalMatrix() did not return any rows! (%s:%lu)\n", __FILE__, __LINE__);
#endif// __DEBUG__

        if (elemRecords.size() > 0)
        {
          for (int i = 0; i < nPe; ++i)
            for (int id = 0; id < ndofs; ++id)
              rowIdxBuffer[i * ndofs + id] = nodeIdsFlat[i] * ndofs + id;
          for (int ij = 0; ij < n_squared; ++ij)
          {
            const ot::MatRecord rec = elemRecords[ij];
            const int i = rec.getRowID(), j = rec.getColID();
            const int id = rec.getRowDim(), jd = rec.getColDim();
            colValBuffer[((i * ndofs + id) * nPe + j) * ndofs + jd] = rec.getMatVal();
          }

          // Multiply p2c and c2p.
          if (subtreeInfo.getNumNonhangingNodes() != nPe)
          {
            const unsigned char child_m = subtree.getMortonIndex();

            const std::vector<bool> &nodeNonhangingIn = subtreeInfo.readNodeNonhangingIn();
            const ot::TreeNode<CoordT, dim> * nodeCoordsIn = subtreeInfo.readNodeCoordsIn();

            // Initialize diags nonhanging and hanging.  [ I        ]
            // Pre- and post- Qt to mult by block.       [          ] = nh + h Qt h
            //                                           [   Qt^h_h ]
            for (int nIdx = 0, flat = 0; nIdx < nPe; ++nIdx)
            {
              // Block factorization needs entire faces to be hanging or not.
              const bool is_nonhanging = nodeNonhangingIn[nIdx] and
                nodeCoordsIn[nIdx].getLevel() == subtree.getLevel();
              const ScalarT nh = (is_nonhanging ? 1.0f : 0.0f);
              for (const int dofEnd = flat + ndofs; flat < dofEnd; ++flat)
                nonhanging[flat] = nh;
            }

            // Transpose
            const auto transpose = [&](ScalarT *e_mat) {
              for (int i = 0; i < n; ++i)
                for (int j = i; j < n; ++j)
                  std::swap(e_mat[i*n+j], e_mat[j*n+i]);
            };

            // Right-multiply the hanging block of e_mat by Q.
            const auto Q_right = [&](ScalarT *e_mat) {
              for (int i = 0; i < n; ++i)
              {
                ScalarT *row = &e_mat[i*n];
                ScalarT *wksp = wksp_col.data();
                for (int j = 0; j < n; ++j)
                {
                  const ScalarT entry = row[j];
                  wksp[j] = (1.0 - nonhanging[j]) * entry;  // Diag H
                  row[j] = nonhanging[j] * entry;           // Diag NH
                }

                // Treat each row of e_mat as a column vec and left-mult by Q^t.
                constexpr bool C2P = InterpMatrices<dim, ScalarT>::C2P;
                interp_matrices.template IKD_ParentChildInterpolation<C2P>(
                    wksp, wksp, ndofs, child_m);

                for (int j = 0; j < n; ++j)
                  row[j] += (1.0 - nonhanging[j]) * wksp[j];  // Diag H
              }
            };

            ScalarT *e_mat = colValBuffer.data();
            Q_right(e_mat);
            transpose(e_mat);
            Q_right(e_mat);
            transpose(e_mat);
          }//end mult p2c c2p

          assemble_e(rowIdxBuffer, colValBuffer);
        }
      }
      treeLoopIn.step();
    }
  }
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


/**
 * @brief Collect elemental matrices and feed them to Petsc MatSetValue().
 * @note The user is responsible to call MatAssemblyBegin()/MatAssemblyEnd()
 *       if needed. Not called at the end of getAssembledMatrix(),
 *       in case getAssembledMatrix() needs to be called multiple times
 *       before the final Petsc assembly.
 */
template <typename LeafT, unsigned int dim>
bool feMatrix<LeafT,dim>::getAssembledMatrix(Mat *J, MatType mtype)
{DOLLAR("getAssembledMatrix()")
  using ScalarT = PetscScalar;
  using IndexT = PetscInt;
  const int n = this->da()->getNumNodesPerElement() * this->ndofs();

  preMat();
  collectMatrixEntries(
      [&]( const std::vector<PetscInt>& rowIdxBuffer,
           const std::vector<ScalarT> & colValBuffer )
      {
        MatSetValues(*J, n, rowIdxBuffer.data(), n, rowIdxBuffer.data(), colValBuffer.data(), ADD_VALUES);
      });
  postMat();

  return true;
}

#endif



#endif //DENDRO_KT_FEMATRIX_H
