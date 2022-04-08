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
#include "da_matvec.h"
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
         unsigned int m_uiDof = 1;

         /**@brief element nodal vec in */
         VECType * m_uiEleVecIn = nullptr;

         /***@brief element nodal vecOut */
         VECType * m_uiEleVecOut = nullptr;

         /** elemental coordinates */
         double * m_uiEleCoords = nullptr;

    protected:
         feMatrix() {}

    public:
        /**
         * @brief constructs an FEM stiffness matrix class.
         * @param[in] da: octree DA
         * */
        feMatrix(const ot::DA<dim>* da, const std::vector<ot::TreeNode<unsigned int, dim>> *octList, unsigned int dof=1);

        feMatrix(feMatrix &&other);
        feMatrix & operator=(feMatrix &&other);

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
        void collectMatrixEntries(
            const std::vector<ot::TreeNode<unsigned, dim>> &octList,
            const std::vector<ot::TreeNode<unsigned, dim>> &nodes,
            const std::vector<ot::RankI> &ghostedGlobalNodeId,
            AssembleElemental assemble_e);

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

#ifdef BUILD_WITH_AMAT
        template<typename AMATType>
        bool getAssembledAMat_template(AMATType* J);

        virtual bool getAssembledAMat(
            par::aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> *J
            ) override
        {
          return getAssembledAMat_template(J);
        }

        virtual bool getAssembledAMat(
            par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> *J
            ) override
        {
          return getAssembledAMat_template(J);
        }



        template<typename AMATType>
        bool setDiagonalAMat(const VECType * diag, AMATType* J) const;
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
feMatrix<LeafT,dim>::feMatrix(const ot::DA<dim>* da, const std::vector<ot::TreeNode<unsigned int, dim>> *octList, unsigned int dof)
  : feMat<dim>(da, octList)
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
feMatrix<LeafT, dim> & feMatrix<LeafT, dim>::operator=(feMatrix &&other)
{
  feMat<dim>::operator=(std::forward<feMatrix&&>(other));
  std::swap(this->m_uiDof, other.m_uiDof);
  std::swap(this->m_uiEleVecIn, other.m_uiEleVecIn);
  std::swap(this->m_uiEleVecOut, other.m_uiEleVecOut);
  std::swap(this->m_uiEleCoords, other.m_uiEleCoords);
  return *this;
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
  fem::da_matvec(m_oda, this->m_octList, inGhostedPtr, outGhostedPtr, m_uiDof, eleOp, scale);

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


// SliceIter
class SliceIter
{
  protected:
    const std::vector<bool> &m_slice_ref;
    size_t m_slice_idx;
    size_t m_idx;
  public:
    SliceIter(const std::vector<bool> &slice_ref)
      : m_slice_ref(slice_ref), m_slice_idx(0), m_idx(0)
    {
      while (m_idx < m_slice_ref.size() && !m_slice_ref[m_idx])
        ++m_idx;
    }

    static SliceIter end(const std::vector<bool> &slice_ref)
    {
      SliceIter iter(slice_ref);
      while (iter.get_idx() < slice_ref.size())
        ++iter;
      return iter;
    }

    const SliceIter & operator*() const
    {
      return *this;
    }

    size_t get_idx()    const { return m_idx; }
    size_t get_slice_idx() const { return m_slice_idx; }

    SliceIter & operator++()
    {
      while (m_idx < m_slice_ref.size() && !m_slice_ref[++m_idx]);
      ++m_slice_idx;
      return *this;
    }

    bool operator==(const SliceIter &other) const
    {
      return (&m_slice_ref == &other.m_slice_ref) && (m_slice_idx == other.m_slice_idx);
    }

    bool operator!=(const SliceIter &other) const
    {
      return !operator==(other);
    }
};

// SliceIterRange
//
// Usage:
//     for (const SliceIter &slice_iter : slice_iter_range)
//     {
//       slice_iter.get_idx();
//       slice_iter.get_slice_idx();
//     }
struct SliceIterRange
{
  const std::vector<bool> &m_slice_ref;
  SliceIter begin() const { return SliceIter(m_slice_ref); }
  SliceIter end()   const { return SliceIter::end(m_slice_ref); }
};

// SubMatView
template <typename T>
class SubMatView
{
  protected:
    T *m_base_ptr;
    size_t m_slice_r_sz;
    size_t m_slice_c_sz;
    bool m_row_major;

    std::vector<bool> m_slice_r;
    std::vector<bool> m_slice_c;

  public:
    constexpr static bool ROW_MAJOR = true;
    constexpr static bool COL_MAJOR = false;

    SubMatView() = delete;

    SubMatView(T *base_ptr, const std::vector<bool> &slice_r, const std::vector<bool> &slice_c, bool row_major = true)
      : m_base_ptr(base_ptr),
        m_slice_r(slice_r),
        m_slice_c(slice_c),
        m_slice_r_sz(std::count(slice_r.begin(), slice_r.end(), true)),
        m_slice_c_sz(std::count(slice_c.begin(), slice_c.end(), true)),
        m_row_major(row_major)
    {}

    SubMatView(T *base_ptr, size_t slice_r_sz, size_t slice_c_sz, bool row_major = true)
      : m_base_ptr(base_ptr),
        m_slice_r(std::vector<bool>(slice_r_sz, true)),
        m_slice_c(std::vector<bool>(slice_c_sz, true)),
        m_slice_r_sz(slice_r_sz),
        m_slice_c_sz(slice_c_sz),
        m_row_major(row_major)
    {}

    SubMatView(const SubMatView &other)
      : SubMatView(other.m_base_ptr, other.m_slice_r_sz, other.m_slice_c_sz, other.m_row_major)
    {}

    bool is_row_major() const { return m_row_major; }
    bool is_col_major() const { return !m_row_major; }

    SliceIterRange slice_r_range() const { return SliceIterRange{m_slice_r}; }
    SliceIterRange slice_c_range() const { return SliceIterRange{m_slice_c}; }

    T & operator()(const SliceIter &si, const SliceIter &sj)
    {
      const size_t nr = m_slice_r.size();
      const size_t nc = m_slice_c.size();

      return m_base_ptr[si.get_idx()*(m_row_major ? nr : 1) +
                        sj.get_idx()*(!m_row_major ? nc : 1)];
    }

    SubMatView transpose_view()
    {
      return SubMatView(m_base_ptr, m_slice_c, m_slice_r, !m_row_major);
    }

    void swap(SubMatView &src)
    {
      if (!(src.m_slice_r_sz == m_slice_r_sz && src.m_slice_c_sz == m_slice_c_sz))
        throw std::invalid_argument("Source row and column sizes do not match destination.");
      const SliceIterRange src_slice_r_range = src.slice_r_range();
      const SliceIterRange src_slice_c_range = src.slice_c_range();
      const SliceIterRange dst_slice_r_range = this->slice_r_range();
      const SliceIterRange dst_slice_c_range = this->slice_c_range();
      for (SliceIter src_si = src_slice_r_range.begin(), dst_si = dst_slice_r_range.begin();
           src_si != src_slice_r_range.end() && dst_si != dst_slice_r_range.end();
           (++src_si, ++dst_si))
        for (SliceIter src_sj = src_slice_c_range.begin(), dst_sj = dst_slice_c_range.begin();
             src_sj != src_slice_c_range.end() && dst_sj != dst_slice_c_range.end();
             (++src_sj, ++dst_sj))
        {
          T tmp = src(src_si, src_sj);
          src(src_si, src_sj) = (*this)(dst_si, dst_sj);
          (*this)(dst_si, dst_sj) = tmp;
        }
    }

    void copy_from(const SubMatView &src)
    {
      if (!(src.m_slice_r_sz == m_slice_r_sz && src.m_slice_c_sz == m_slice_c_sz))
        throw std::invalid_argument("Source row and column sizes do not match destination.");
      const SliceIterRange src_slice_r_range = src.slice_r_range();
      const SliceIterRange src_slice_c_range = src.slice_c_range();
      const SliceIterRange dst_slice_r_range = this->slice_r_range();
      const SliceIterRange dst_slice_c_range = this->slice_c_range();
      for (SliceIter src_si = src_slice_r_range.begin(), dst_si = dst_slice_r_range.begin();
           src_si != src_slice_r_range.end() && dst_si != dst_slice_r_range.end();
           (++src_si, ++dst_si))
        for (SliceIter src_sj = src_slice_c_range.begin(), dst_sj = dst_slice_c_range.begin();
             src_sj != src_slice_c_range.end() && dst_sj != dst_slice_c_range.end();
             (++src_sj, ++dst_sj))
          (*this)(dst_si, dst_sj) = src(src_si, src_sj);
    }

};


// collectMatrixEntries()
template <typename LeafT, unsigned int dim>
template <typename AssembleElemental>
void feMatrix<LeafT, dim>::collectMatrixEntries(AssembleElemental assemble_e)
{
  return collectMatrixEntries(*this->m_octList,
                              this->da()->nodes(),
                              this->da()->getNodeLocalToGlobalMap(),
                              assemble_e);
}

// collectMatrixEntries()
template <typename LeafT, unsigned int dim>
template <typename AssembleElemental>
void feMatrix<LeafT, dim>::collectMatrixEntries(
    const std::vector<ot::TreeNode<unsigned, dim>> &octList,
    const std::vector<ot::TreeNode<unsigned, dim>> &nodes,
    const std::vector<ot::RankI> &ghostedGlobalNodeId,
    AssembleElemental assemble_e)
{
  const unsigned int eleOrder = this->da()->getElementOrder();
  const unsigned int nPe = this->da()->getNumNodesPerElement();
  const unsigned int ndofs = m_uiDof;
  const unsigned int n = nPe * ndofs;
  const unsigned int n_squared = n * n;

  // Loop over all elements, adding row chunks from elemental matrices.
  // Get the node indices on an element using MatvecBaseIn<dim, unsigned int, false>.

  if (this->da()->isActive())
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

    const size_t ghostedNodalSz = nodes.size();
    const ot::TreeNode<CoordT, dim> *odaCoords = nodes.data();

    std::vector<ot::MatRecord> elemRecords;
    std::vector<PetscInt> rowIdxBuffer(n);
    std::vector<ScalarT> colValBuffer(n_squared);

    InterpMatrices<dim, ScalarT> interp_matrices(eleOrder);
    std::vector<ScalarT> wksp_col(n);
    std::vector<ScalarT> wksp_mat(n_squared);

    std::vector<ScalarT> nonhanging(n);

    size_t elemIdx = 0;
    const bool visitEmpty = false;
    const unsigned int padLevel = 0;
    ot::MatvecBaseIn<dim, RankI, false> treeLoopIn(ghostedNodalSz,
                                                   1,                // node id is scalar
                                                   eleOrder,
                                                   visitEmpty,
                                                   padLevel,
                                                   odaCoords,
                                                   &(*ghostedGlobalNodeId.cbegin()),
                                                   &(*octList.cbegin()),
                                                   octList.size());

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
        elemIdx++;
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


#ifdef BUILD_WITH_AMAT
    template <typename LeafT, unsigned int dim>
    template<typename AMATType>
    bool feMatrix<LeafT, dim>::getAssembledAMat_template(AMATType* J)
    {
      using DT = typename AMATType::DTType;
      using GI = typename AMATType::GIType;
      using LI = typename AMATType::LIType;

#ifdef BUILD_WITH_PETSC
    using ScalarT = PetscScalar;
    using IndexT = PetscInt;
#else
    using ScalarT = DendroScalar;
    using IndexT = long long unsigned;
#endif

      if(this->m_uiOctDA->isActive())
      {
        const size_t numLocElem = this->m_uiOctDA->getLocalElementSz();
        const unsigned ndofs = m_uiDof;
        const unsigned nPe = this->da()->getNumNodesPerElement();
        const unsigned n = nPe * ndofs;

        typedef Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic> EigenMat;
        EigenMat* eMat[2] = {nullptr, nullptr};
        eMat[0] = new EigenMat();
        eMat[0]->resize(ndofs*nPe, ndofs*nPe);
        const EigenMat * read_eMat[2] = {eMat[0], eMat[1]};

        preMat();
        int eid = 0;
        this->collectMatrixEntries(
            [&] ( const std::vector<IndexT> &rowIdxBuffer,
                  const std::vector<ScalarT> &colValBuffer ) {
            for (int r = 0; r < n; ++r)
              for (int c = 0; c < n; ++c)
                (*(eMat[0]))(r,c) = colValBuffer[r * n + c];
            LI n_i[1]={0};
            LI n_j[1]={0};
            J->set_element_matrix(eid, n_i, n_j, read_eMat, 1u);
            // note that read_eMat[0] points to the same memory as eMat[0].
            ++eid;
          });
        postMat();
        delete eMat[0];
      }
      PetscFunctionReturn(0);
    }


    /**@brief Create a diagonal matrix from a distributed, non-ghosted dof array, no accumulation.
     * For example, if outside this method you assemble the diagonal of A,
     * and compute the inverse, then with this you can turn it back into a matrix.
     */
    template <typename LeafT, unsigned int dim>
    template<typename AMATType>
    bool feMatrix<LeafT, dim>::setDiagonalAMat(const VECType * diag, AMATType* J) const
    {
      using DT = typename AMATType::DTType;
      using GI = typename AMATType::GIType;
      using LI = typename AMATType::LIType;
      using OwnershipT = DendroIntL;

      if(this->m_uiOctDA->isActive())
      {
        /// const size_t numLocElem = this->m_uiOctDA->getLocalElementSz();
        const size_t numLocalNodes = this->m_uiOctDA->getLocalNodalSz();
        const size_t localNodeBegin = this->m_uiOctDA->getLocalNodeBegin();
        const size_t numTotalNodes = this->m_uiOctDA->getTotalNodalSz();
        const unsigned eleOrder = this->m_uiOctDA->getElementOrder();
        const unsigned int nPe = intPow(eleOrder+1, dim);
        const unsigned ndofs = this->ndofs();

        // Copy input to a ghosted vector for traversal.
        std::vector<VECType> ghostedDiag(ndofs * numTotalNodes, 0.0f);
        std::copy_n(diag, ndofs * numLocalNodes, &ghostedDiag[ndofs * localNodeBegin]);

        // Set up loops.
        ot::MatvecBaseIn<dim, VECType> dataLoop(
            numTotalNodes, ndofs, eleOrder,
            false, 0,
            this->m_uiOctDA->getTNCoords(),
            ghostedDiag.data(),
            this->m_octList->data(),
            this->m_octList->size());

        ot::MatvecBaseIn<dim, OwnershipT> ownerLoop(
            numTotalNodes, 1, eleOrder,
            false, 0,
            this->m_uiOctDA->getTNCoords(),
            this->m_uiOctDA->getNodeOwnerElements(),
            this->m_octList->data(),
            this->m_octList->size());

        // Element ID used to determine node ownership status in traversal.
        unsigned int localElementId = 0;
        OwnershipT globElementId = this->m_uiOctDA->getGlobalElementBegin();

        // Elemental diagonal matrix.
        typedef Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic> EigenMat;
        EigenMat* eMat[2] = {nullptr, nullptr};
        eMat[0] = new EigenMat();
        eMat[0]->resize(ndofs*nPe, ndofs*nPe);
        const EigenMat * read_eMat[2] = {eMat[0], eMat[1]};
        // Clear elemental matrix.
        for(unsigned int r = 0; r < (nPe*ndofs); ++r)
          for(unsigned int c = 0; c < (nPe*ndofs); ++c)
            (*(eMat[0]))(r,c) = 0;

        // Loop over nodes and elements.
        while (!dataLoop.isFinished())
        {
          if (dataLoop.isPre() && dataLoop.subtreeInfo().isLeaf())
          {
            const VECType * diagCopy = dataLoop.subtreeInfo().readNodeValsIn();
            const OwnershipT * nodeOwners = ownerLoop.subtreeInfo().readNodeValsIn();
            for (size_t nIdx = 0; nIdx < nPe; ++nIdx)
            {
              // Only transfer nonhanging nodes, and from a unique element.
              if (dataLoop.subtreeInfo().readNodeNonhangingIn()[nIdx]
                  && nodeOwners[nIdx] == globElementId)
              {
                // Overwrite elemental matrix diagonal.
                for (int dof = 0; dof < ndofs; ++dof)
                  (*(eMat[0]))(ndofs * nIdx + dof, ndofs * nIdx + dof) = diagCopy[ndofs * nIdx + dof];
              }
              else
              {
                for (int dof = 0; dof < ndofs; ++dof)
                  (*(eMat[0]))(ndofs * nIdx + dof, ndofs * nIdx + dof) = 0;
              }
            }

            // Send elemental diagonal matrix to aMat.
            LI n_i[1]={0};
            LI n_j[1]={0};
            J->set_element_matrix(localElementId, n_i, n_j, read_eMat, 1u);

            dataLoop.next();
            ownerLoop.next();
            globElementId++;
            localElementId++;
          }
          else
          {
            dataLoop.step();
            ownerLoop.step();
          }
        }
        delete eMat[0];
      }

      PetscFunctionReturn(0);
    }


#endif


#endif //DENDRO_KT_FEMATRIX_H
