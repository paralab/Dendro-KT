
#include "hybridPoissonMat.h"

#include "FEM/include/refel.h"

namespace PoissonEq
{

  namespace detail
  {
    class ElementalMatrix
    {
      public:
        ElementalMatrix() = default;
        ElementalMatrix(size_t n_nodes, int ndofs);
        ElementalMatrix(size_t n_nodes, int ndofs, Eigen::MatrixXd entries);
        ElementalMatrix & operator+=(const ElementalMatrix &E);

      public:
        Eigen::MatrixXd entries = {};
        size_t n_nodes = {};
        int ndofs = {};
    };

    template <int dim>
    ElementalMatrix RAP(
        const ot::TreeNode<uint32_t, dim> &fine_oct,
        const ElementalMatrix &fine_emat,
        const std::vector<bool> &fine_nonhanging,
        const InterpMatrices<dim, double> *interp,
        const ot::TreeNode<uint32_t, dim> &coarse_oct,
        const std::vector<bool> &coarse_nonhanging);
  }


  // Implementation

  template <int dim>
  HybridPoissonMat<dim>::HybridPoissonMat(HybridPoissonMat &&) = default;

  template <int dim>
  HybridPoissonMat<dim> &
    HybridPoissonMat<dim>::operator=(HybridPoissonMat &&) = default;


  // HybridPoissonMat()
  template <int dim>
  HybridPoissonMat<dim>::HybridPoissonMat(const ot::DA<dim> *da, int dof)
    : Base(da, nullptr, dof), m_matfree(da, nullptr, dof), m_emats(da->getLocalElementSz()),
      m_elemental_nonhanging(da->getLocalElementSz(), da->getNumNodesPerElement(), true),
      m_interp(da->getElementOrder())
  {
    // Partition table  (center coordinate -> element id).
    m_partition_table.reserve(da->getLocalElementSz()); //don't invalidate iters
    m_inv_partition_table.resize(da->getLocalElementSz());
    const auto &octlist = *this->octList();
    for (size_t i = 0; i < octlist.size(); ++i)
    {
      const ot::TreeNode<uint32_t, dim> tn = octlist[i];
      std::array<double, dim> coords;
      double size;
      ot::treeNode2Physical(tn, coords.data(), size);
      const Point<dim> corner(coords);
      const Point<dim> center = corner + Point<dim>(size) / 2.0;
      const detail::ElemCenter<dim> key(center);

      // Change the partition id after construction using store_evaluated().
      using detail::HybridPartition;
      HybridPartition partition_id = HybridPartition::opaque;

      using detail::ElemIndex;
      const ElemIndex value = ElemIndex{i, partition_id};

      assert(m_partition_table.find(key) == m_partition_table.end());//unique.

      m_inv_partition_table[i] = m_partition_table.emplace(key, value).first;
    }

    const size_t n_nodes = this->da()->getNumNodesPerElement();
    std::vector<bool> nonhanging(n_nodes, false);
    const ot::TreeNode<uint32_t, dim> *tn_coords = nullptr;
    size_t local_element_id = 0;

    // Store hanging node status of all elements.
    ot::MatvecBaseCoords<dim> loop(
        this->da()->getTotalNodalSz(), this->da()->getElementOrder(),
        false, 0,
        this->da()->getTNCoords(),
        this->octList()->data(),
        this->octList()->size(),
        *this->da()->getTreePartFront(),
        *this->da()->getTreePartBack());
    while (not loop.isFinished())
    {
      if (loop.isPre() and loop.subtreeInfo().isLeaf())
      {
        const int level = loop.getCurrentSubtree().getLevel();
        nonhanging = loop.subtreeInfo().readNodeNonhangingIn();
        tn_coords = loop.subtreeInfo().readNodeCoordsIn();

        // Block factorization needs entire closed faces to be hanging or not.
        for (size_t i = 0; i < n_nodes; ++i)
          if (nonhanging[i] and tn_coords[i].getLevel() != level)
            nonhanging[i] = false;

        m_elemental_nonhanging.set_row(local_element_id, nonhanging);

        loop.next();
        ++local_element_id;
      }
      else
      {
        loop.step();
      }
    }
  }


  template <int dim>
  void HybridPoissonMat<dim>::store_evaluated(size_t local_eid)
  {
    // Those Eigen matrices that are not resized will just be null pointers.

    this->is_evaluated(local_eid, true);
    this->elemental_matrix(local_eid, this->evaluate(local_eid));
  }


  template <int dim>
  detail::ElementalMatrix
    HybridPoissonMat<dim>::evaluate(size_t local_eid) const
  {
    const size_t n_nodes = this->da()->getNumNodesPerElement();
    const int ndofs = this->ndofs();
    const size_t n = n_nodes * ndofs;
    detail::ElementalMatrix emat(n_nodes, ndofs);

    const ot::TreeNode<uint32_t, dim> octant = (*this->octList())[local_eid];
    const bool is_boundary = octant.getIsOnTreeBdry();

    static std::vector<double> node_coords_flat;
    const std::vector<ot::TreeNode<uint32_t, dim>> *no_tn_coords = nullptr;
    ot::fillAccessNodeCoordsFlat(
        false, *no_tn_coords,
        octant, this->da()->getElementOrder(), node_coords_flat);

    static std::vector<ot::MatRecord> elem_records;
    m_matfree.getElementalMatrix(elem_records, node_coords_flat.data(), is_boundary);//beware: not entire element set
    for (const ot::MatRecord &entry : elem_records)
      emat.entries(entry.getRowID() * ndofs + entry.getRowDim(),
                   entry.getColID() * ndofs + entry.getColDim())
          = entry.getMatVal();

    // Multiply on the left since Eigen entries are column-major.

    // Hᵀ A H = Hᵀ (Hᵀ Aᵀ)ᵀ

    emat.entries.transposeInPlace();

    for (size_t col = 0; col < n; ++col)
      this->elemental_c2p( local_eid,
          emat.entries.col(col).data(),
          emat.entries.col(col).data());

    emat.entries.transposeInPlace();

    for (size_t col = 0; col < n; ++col)
      this->elemental_c2p( local_eid,
          emat.entries.col(col).data(),
          emat.entries.col(col).data());

    elem_records.clear();
    node_coords_flat.clear();

    return emat;
  }


  template <int dim>
  void HybridPoissonMat<dim>::elemental_p2c(
      size_t local_eid, const double *in, double *out) const
  {
    HybridPoissonMat<dim>::transform_hanging<false>(
        &m_interp,
        (*this->octList())[local_eid].getMortonIndex(),
        m_elemental_nonhanging[local_eid],
        this->da()->getNumNodesPerElement(), this->ndofs(),
        in, out);
  }

  template <int dim>
  void HybridPoissonMat<dim>::elemental_c2p(
      size_t local_eid, const double *in, double *out) const
  {
    HybridPoissonMat<dim>::transform_hanging<true>(
        &m_interp,
        (*this->octList())[local_eid].getMortonIndex(),
        m_elemental_nonhanging[local_eid],
        this->da()->getNumNodesPerElement(), this->ndofs(),
        in, out);
  }


  template <int dim>
  template <bool use_c2p>
  void HybridPoissonMat<dim>::transform_hanging(
      const InterpMatrices<dim, double> *interp,
      int child_number,
      detail::ConstArrayBool1DView nonhanging,
      size_t n_nodes, int ndofs,
      const double *in, double *out)
  {
    static std::vector<double> on_faces;
    on_faces.resize(n_nodes * ndofs, 0.0);

    if (out != in)
    {
      std::copy_n(in, n_nodes * ndofs, out);
    }

    bool transform = false;

    for (size_t i = 0; i < n_nodes; ++i)
    {
      if (not nonhanging[i])   // Note: ensure interior hanging => exterior also
      {
        transform = true;
        for (int dof = 0; dof < ndofs; ++dof)
          on_faces[i * ndofs + dof] = in[i * ndofs + dof];
      }
    }

    if (transform)
    {
      interp->template IKD_ParentChildInterpolation<use_c2p>(
          on_faces.data(), on_faces.data(), ndofs, child_number);

      for (size_t i = 0; i < n_nodes; ++i)
        if (not nonhanging[i])
          for (int dof = 0; dof < ndofs; ++dof)
            out[i * ndofs + dof] = on_faces[i * ndofs + dof];
    }

    on_faces.clear();
  }


  // elementalMatVec()
  template <int dim>
  void HybridPoissonMat<dim>::elementalMatVec(
      const double* in,
      double* out,
      unsigned int ndofs,
      const double* coords,
      double scale,
      bool isElementBoundary)
  {
    const int nodes = this->da()->getNumNodesPerElement();
    const Point<dim> first_pt = Point<dim>(coords);
    const Point<dim> last_pt = Point<dim>(coords + (nodes - 1) * dim);
    const Point<dim> center = (first_pt + last_pt) / 2.0;
    const detail::ElemCenter<dim> key(center);

    using detail::ElemIndex;
    const ElemIndex index = m_partition_table[key];

    assert(m_partition_table.find(key) != m_partition_table.end());

    using detail::HybridPartition;
    switch (index.partition_id)
    {
      case HybridPartition::opaque:

        this->elemental_p2c(index.local_element_id, in, out);

        this->m_matfree.elementalMatVec(
            out, out, ndofs, coords, scale, isElementBoundary);

        this->elemental_c2p(index.local_element_id, out, out);

        break;

      case HybridPartition::evaluated:
        this->emat_mult(in, out, index.local_element_id);
        break;

      default:
        assert(false);
    }

  }

  // HybridPoissonMat::emat_mult()
  template <int dim>
  void HybridPoissonMat<dim>::emat_mult(const double *in, double *out, size_t local_eid)
  {
    assert(local_eid < this->da()->getLocalElementSz());
    assert(this->is_evaluated(local_eid));
    const size_t n = this->da()->getNumNodesPerElement() * this->ndofs();
    const auto &mat = this->m_emats[local_eid];
    Eigen::Map<const Eigen::VectorXd> vec_in(in, n);
    Eigen::Map<Eigen::VectorXd> vec_out(out, n);
    vec_out = mat * vec_in;
  }


  // HybridPoissonMat::matVec()
  template <int dim>
  void HybridPoissonMat<dim>::matVec(
      const VECType* in,VECType* out, double scale)
  {
    // Copy-pasted from feMatrix.h:matVec(),
    // just changed matvec() -> matvec_no_interpolation()

    using namespace std::placeholders;   // Convenience for std::bind().

    // Shorter way to refer to our member DA.
    const ot::DA<dim> * m_oda = this->da();
    const int m_uiDof = this->ndofs();

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
    this->preMatVec(in, inGhostedPtr + this->ndofs() * m_oda->getLocalNodeBegin(), scale);

    // 2. Upstream->downstream ghost exchange.
    m_oda->template readFromGhostBegin<VECType>(inGhostedPtr, m_uiDof);
    m_oda->template readFromGhostEnd<VECType>(inGhostedPtr, m_uiDof);

    // 3. Local matvec().
    const auto * tnCoords = m_oda->getTNCoords();
    std::function<void(const VECType *, VECType *, unsigned int, const double *, double, bool)> eleOp =
        std::bind(&HybridPoissonMat::elementalMatVec, this, _1, _2, _3, _4, _5, _6);

    assert(m_oda->getElementOrder() == m_oda->getReferenceElement()->getOrder());

    fem::matvec_no_interpolation(
        inGhostedPtr, outGhostedPtr, m_uiDof, tnCoords, m_oda->getTotalNodalSz(),
        &(*this->octList()->cbegin()), this->octList()->size(),
        *m_oda->getTreePartFront(), *m_oda->getTreePartBack(),
        eleOp, scale, m_oda->getElementOrder());

    // 4. Downstream->Upstream ghost exchange.
    m_oda->template writeToGhostsBegin<VECType>(outGhostedPtr, m_uiDof);
    m_oda->template writeToGhostsEnd<VECType>(outGhostedPtr, m_uiDof);

    // 5. Copy output data from ghosted buffer.
    m_oda->template ghostedNodalToNodalVec<VECType>(outGhostedPtr, out, true, m_uiDof);

    // 5.a. Override output data with post-matvec re-initialization.
    this->postMatVec(outGhostedPtr + this->ndofs() * m_oda->getLocalNodeBegin(), out, scale);
  }


  // HybridPoissonMat::setDiag()
  template <int dim>
  void HybridPoissonMat<dim>::setDiag(VECType *out, double scale)
  {
    // Copy-pasted from feMatrix.h:setDiag() and setDiag.h,
    // but set interpolation(false).

    // Shorter way to refer to our member DA.
    const ot::DA<dim> * m_oda = this->da();
    const int ndofs = this->ndofs();

    // Static buffers for ghosting. Check/increase size.
    static std::vector<VECType> outGhosted;
    /// static std::vector<char> outDirty;
    m_oda->template createVector<VECType>(outGhosted, false, true, ndofs);
    /// m_oda->template createVector<char>(outDirty, false, true, 1);
    std::fill(outGhosted.begin(), outGhosted.end(), 0);
    VECType *outGhostedPtr = outGhosted.data();


    // --------------------------
    // Local setDiag().
    // --------------------------

    // Initialize output vector to 0.
    const size_t sz = this->da()->getTotalNodalSz();
    std::fill(outGhostedPtr, outGhostedPtr + ndofs*sz, 0);

    const unsigned int eleOrder = this->da()->getElementOrder();
    const unsigned int npe = this->da()->getNumNodesPerElement();
    const size_t n = npe * ndofs;

    std::vector<double> leafResult(ndofs*npe, 0.0);

    constexpr bool noVisitEmpty = false;
    const ot::TreeNode<uint32_t, dim> *coords = this->da()->getTNCoords();
    const ot::TreeNode<uint32_t, dim> *treePartPtr = &(*this->octList()->cbegin());
    const size_t treePartSz = this->octList()->size();
    const ot::TreeNode<uint32_t, dim> &partFront = *this->da()->getTreePartFront();
    const ot::TreeNode<uint32_t, dim> &partBack = *this->da()->getTreePartBack();
    /// ot::MatvecBaseOut<dim, double, false> treeLoopOut(sz, ndofs, eleOrder, noVisitEmpty, 0, coords, treePartPtr, treePartSz, partFront, partBack);
    ot::MatvecBaseOut<dim, double, true> treeLoopOut(sz, ndofs, eleOrder, noVisitEmpty, 0, coords, treePartPtr, treePartSz, partFront, partBack);

    treeLoopOut.interpolation(false);

    size_t local_element_id = 0;
    while (!treeLoopOut.isFinished())
    {
      if (treeLoopOut.isPre() && treeLoopOut.subtreeInfo().isLeaf())
      {
        const double * nodeCoordsFlat = treeLoopOut.subtreeInfo().getNodeCoords();

        const auto & emat = this->elemental_matrix(local_element_id);
        const auto & diag = emat.entries.diagonal();
        for (size_t i = 0; i < n; ++i)
          leafResult[i] = diag[i];

        treeLoopOut.subtreeInfo().overwriteNodeValsOut(&(*leafResult.begin()));

        treeLoopOut.next();
        ++local_element_id;
      }
      else
        treeLoopOut.step();
    }

    /// size_t writtenSz = treeLoopOut.finalize(outGhostedPtr, dirtyOut);
    size_t writtenSz = treeLoopOut.finalize(outGhostedPtr);

    if (sz > 0 && writtenSz == 0)
      std::cerr << "Warning: locSetDiag() did not write any data! Loop misconfigured?\n";

    // ----------------------
    // Ghost exchange
    // ----------------------

    // Downstream->Upstream ghost exchange.
    m_oda->template writeToGhostsBegin<VECType>(outGhostedPtr, ndofs);
    m_oda->template writeToGhostsEnd<VECType>(outGhostedPtr, ndofs);
    /// m_oda->template writeToGhostsBegin<VECType>(outGhostedPtr, ndofs, outDirty.data());
    /// m_oda->template writeToGhostsEnd<VECType>(outGhostedPtr, ndofs, false, outDirty.data());

    // 5. Copy output data from ghosted buffer.
    m_oda->template ghostedNodalToNodalVec<VECType>(outGhostedPtr, out, true, ndofs);
  }


  // HybridPoissonMat::getAssembledMatrix()
  template <int dim>
  bool HybridPoissonMat<dim>::getAssembledMatrix(Mat *J, MatType mtype)
  {
    // Copy-pasted from feMatrix.h:collectMatrixEntries()
    // and getAssembledMatrix(), then deleted hanging interpolation
    // (it is already built into elemental_matrix()).

    this->preMat();

    const unsigned int eleOrder = this->da()->getElementOrder();
    const unsigned int nPe = this->da()->getNumNodesPerElement();
    const int ndofs = this->ndofs();
    const int n = this->da()->getNumNodesPerElement() * this->ndofs();
    const unsigned int n_squared = n * n;

    // Loop over all elements, adding row chunks from elemental matrices.
    // Get the node indices on an element using MatvecBaseIn<dim, unsigned int, false>.

    if (this->da()->isActive())
    {
      using ot::RankI;

      const size_t ghostedNodalSz = this->da()->getTotalNodalSz();
      const ot::TreeNode<uint32_t, dim> *odaCoords = this->da()->getTNCoords();
      const std::vector<RankI> &ghostedGlobalNodeId = this->da()->getNodeLocalToGlobalMap();

      std::vector<PetscInt> rowIdxBuffer(n);

      using Eigen::Dynamic;
      using Eigen::RowMajor;
      Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> row_major_entries(n, n);

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
                                                     *this->da()->getTreePartFront(),
                                                     *this->da()->getTreePartBack());

      size_t local_element_id = 0;

      // Iterate over all leafs of the local part of the tree.
      while (!treeLoopIn.isFinished())
      {
        const ot::TreeNode<uint32_t, dim> subtree = treeLoopIn.getCurrentSubtree();
        const auto subtreeInfo = treeLoopIn.subtreeInfo();

        if (treeLoopIn.isPre() && subtreeInfo.isLeaf())
        {
          const double * nodeCoordsFlat = subtreeInfo.getNodeCoords();
          const RankI * nodeIdsFlat = subtreeInfo.readNodeValsIn();
          for (int i = 0; i < nPe; ++i)
            for (int id = 0; id < ndofs; ++id)
              rowIdxBuffer[i * ndofs + id] = nodeIdsFlat[i] * ndofs + id;

          // Get elemental matrix for the current leaf element.
          row_major_entries = this->elemental_matrix(local_element_id).entries;

          // Send the values to petsc.
          MatSetValues(
              *J, n, rowIdxBuffer.data(), n, rowIdxBuffer.data(),
              row_major_entries.data(), ADD_VALUES);

          ++local_element_id;
        }
        treeLoopIn.step();
      }
    }

    this->postMat();

    return true;
  }


  // HybridPoissonMat::coarsen()
  template <int dim>
  HybridPoissonMat<dim> HybridPoissonMat<dim>::coarsen(const ot::DA<dim> *da) const
  {
    HybridPoissonMat coarse_mat(da, this->ndofs());
    const HybridPoissonMat &fine_mat = *this;

#warning "In general, need surrogate here to line up process partitions."

    //
    // Derive the coarse evaluated cell set (parents of evaluated fine cells).
    //
    const size_t n_coarse_cells = coarse_mat.octList()->size();
    const size_t n_fine_cells = fine_mat.octList()->size();
    const ot::TreeNode<uint32_t, dim> *coarse_cells = coarse_mat.octList()->data();
    const ot::TreeNode<uint32_t, dim> *fine_cells = fine_mat.octList()->data();
    size_t j = 0;
    for (size_t i = 0; i < n_coarse_cells; ++i)
    {
      const ot::TreeNode<uint32_t, dim> coarse_cell = coarse_cells[i];
      bool evaluated = false;
      assert(j < n_fine_cells);
      if (fine_cells[j].getLevel() == coarse_cell.getLevel())
      {
        evaluated = true;
        ++j;
      }
      else
      {
        assert(fine_cells[j].getParent() == coarse_cell);
        for (; j < n_fine_cells and fine_cells[j].getParent() == coarse_cell; ++j)
        {
          if (fine_mat.is_evaluated(j))
            evaluated = true;
        }
      }
      coarse_mat.is_evaluated(i, true);
    }

#warning "TODO skip tree traversal loops if there are no elements"

    //
    // Evaluate the coarse elemental matrices: Galerkin coarsening operator.
    //
    ot::MatvecBaseCoords<dim> coarse_loop(
        coarse_mat.da()->getTotalNodalSz(), coarse_mat.da()->getElementOrder(),
        false, 0,
        coarse_mat.da()->getTNCoords(),
        coarse_mat.octList()->data(),
        coarse_mat.octList()->size(),
        *coarse_mat.da()->getTreePartFront(),
        *coarse_mat.da()->getTreePartBack());

    ot::MatvecBaseCoords<dim> fine_loop(
        fine_mat.da()->getTotalNodalSz(), fine_mat.da()->getElementOrder(),
        false, 0,
        fine_mat.da()->getTNCoords(),
        fine_mat.octList()->data(),
        fine_mat.octList()->size(),
        *fine_mat.da()->getTreePartFront(),
        *fine_mat.da()->getTreePartBack());

    //  \      /
    //   \____/
    //    ++++

    size_t coarse_local_element_id = 0;
    size_t fine_local_element_id = 0;

    detail::ElementalMatrix fine_emat;

    while (not coarse_loop.isFinished())
    {
      assert(coarse_loop.getCurrentSubtree() == fine_loop.getCurrentSubtree());
      // Descending.
      if (coarse_loop.isPre() and not coarse_loop.subtreeInfo().isLeaf())
      {
        coarse_loop.step();
        fine_loop.step();
      }
      // Ascending.
      else if (not coarse_loop.isPre())
      {
        coarse_loop.next();
        fine_loop.next();
      }
      // Coarse leaf.
      else
      {
        if (coarse_mat.is_evaluated(coarse_local_element_id))
        {
          detail::ElementalMatrix coarse_emat = coarse_mat.proto_emat();

          // Fine grid is at same level or has children as this cell.
          const bool same_level = fine_loop.subtreeInfo().isLeaf();
          if (same_level)
          {
            fine_emat = fine_mat.elemental_matrix(fine_local_element_id);

            coarse_emat += detail::RAP<dim>(
                fine_loop.getCurrentSubtree(),
                fine_emat,
                fine_loop.subtreeInfo().readNodeNonhangingIn(),
                &m_interp,
                coarse_loop.getCurrentSubtree(),
                coarse_loop.subtreeInfo().readNodeNonhangingIn());

            ++fine_local_element_id;
          }
          else
          {
            fine_loop.step();
            while (fine_loop.subtreeInfo().isLeaf())
            {
              const detail::ElementalMatrix &fine_emat =
                  fine_mat.elemental_matrix(fine_local_element_id);

              coarse_emat += detail::RAP<dim>(
                  fine_loop.getCurrentSubtree(),
                  fine_emat,
                  fine_loop.subtreeInfo().readNodeNonhangingIn(),
                  &m_interp,
                  coarse_loop.getCurrentSubtree(),
                  coarse_loop.subtreeInfo().readNodeNonhangingIn());

              ++fine_local_element_id;
              fine_loop.next();
            }
          }
          coarse_mat.elemental_matrix(
              coarse_local_element_id, std::move(coarse_emat));
        }
        ++coarse_local_element_id;
        coarse_loop.next();
        fine_loop.next();
      }
    }

    return coarse_mat;
  }


  template <int dim>
  bool HybridPoissonMat<dim>::is_evaluated(size_t local_eid) const
  {
    return m_emats[local_eid].size() > 0;
  }

  template <int dim>
  void HybridPoissonMat<dim>::is_evaluated(size_t local_eid, bool evaluated)
  {
    using detail::HybridPartition;
    const size_t n = this->da()->getNumNodesPerElement() * this->ndofs();
    if (evaluated)
    {
      m_emats[local_eid].resize(n, n);
      m_inv_partition_table[local_eid]->second
          .partition_id = HybridPartition::evaluated;
    }
    else
    {
      m_emats[local_eid].resize(0, 0);
      m_inv_partition_table[local_eid]->second
          .partition_id = HybridPartition::opaque;
    }
  }

  template <int dim>
  detail::ElementalMatrix HybridPoissonMat<dim>::proto_emat() const
  {
    return detail::ElementalMatrix(
        this->da()->getNumNodesPerElement(),
        this->ndofs());
  }

  template <int dim>
  detail::ElementalMatrix
    HybridPoissonMat<dim>::elemental_matrix(size_t local_eid) const
  {
    if (not this->is_evaluated(local_eid))
    {
      return this->evaluate(local_eid);
    }
    else
    {
      return detail::ElementalMatrix(
          this->da()->getNumNodesPerElement(),
          this->ndofs(),
          m_emats[local_eid]);
    }
  }

  template <int dim>
  void HybridPoissonMat<dim>::elemental_matrix(
      size_t local_eid, detail::ElementalMatrix emat)
  {
    assert(this->is_evaluated(local_eid)); // call is_evaluated(local_eid, true)
    m_emats[local_eid] = std::move(emat.entries);
  }


  // Implementation of detail
  namespace detail
  {
    // ArrayBool2D::ArrayBool2D()
    ArrayBool2D::ArrayBool2D(size_t outer, size_t inner, bool value)
      : m_outer(outer), m_inner(inner), m_vec_bool(outer * inner, value)
    { }

    // ArrayBool2D::set_row()
    void ArrayBool2D::set_row(
        size_t outer_idx, const std::vector<bool> &new_row)
    {
      size_t offset = outer_idx * m_inner;
      for (bool bit : new_row)
        m_vec_bool[offset++] = bit;
      assert(offset == (outer_idx + 1) * m_inner);
    }

    // ArrayBool2D::operator[]()
    ConstArrayBool1DView ArrayBool2D::operator[](size_t outer_idx) const
    {
      return { this, outer_idx * m_inner };
    }


    // ElementalMatrix::ElementalMatrix()
    ElementalMatrix::ElementalMatrix(size_t n_nodes, int ndofs)
      : n_nodes(n_nodes),
        ndofs(ndofs),
        entries{ Eigen::MatrixXd::Zero(n_nodes * ndofs, n_nodes * ndofs) }
    { }

    // ElementalMatrix::ElementalMatrix()
    ElementalMatrix::ElementalMatrix(size_t n_nodes, int ndofs, Eigen::MatrixXd entries)
      : n_nodes(n_nodes),
        ndofs(ndofs),
        entries(std::move(entries))
    {
      const size_t n = n_nodes * ndofs;
      assert(this->entries.rows() == n);
      assert(this->entries.cols() == n);
    }

    // ElementalMatrix::operator+=
    ElementalMatrix & ElementalMatrix::operator+=(const ElementalMatrix &E)
    {
      this->entries += E.entries;
      return *this;
    }

    // RAP()
    template <int dim>
    ElementalMatrix RAP(
        const ot::TreeNode<uint32_t, dim> &fine_oct,
        const ElementalMatrix &fine_emat,
        const std::vector<bool> &fine_nonhanging,
        const InterpMatrices<dim, double> *interp,
        const ot::TreeNode<uint32_t, dim> &coarse_oct,
        const std::vector<bool> &coarse_nonhanging)
    {
      // Assume we only coarsen by one level at a time.
      assert(fine_oct.getLevel() <= coarse_oct.getLevel() + 1);
      assert(coarse_oct.isAncestorInclusive(fine_oct));

      // Assume that nodes marked 'hanging' will actually point to parent nodes.
      // Remember that (hyper)faces are hanging by an entire face at a time.

      // Slow but simple: elemental R and P as dense (n x n) matrices.

      // Find rows of P by analyzing the action upon stored coarse-grid vectors.
      //
      // The stored vector (s) is a union of self nodes and parent nodes.
      // Define the virtual vector (H s) by interpolating hanging nodes
      // from parent. U represents the prolongation for a uniform grid.

      // If the coarse and fine octants are equal:
      //   parent fine := parent coarse              f.s|_par  <-   c.s
      //     self fine := virt. coarse               f.s|_self <- H c.s
      //
      // If the coarse and fine octants differ:
      //   parent fine := virtual coarse             f.s|_par  <-   H c.s
      //     self fine := interp.(virt. coarse)      f.s|_self <- U H c.s
      //
      //     ^ Note: Due to 2:1-balancing and single-level coarsening,
      //     a coarse-grid face containing a fine-grid self node (nonhanging)
      //     cannot itself be hanging, so it may seem that (U H) is unneeded.
      //     HOWEVER, fine-grid self nodes can be *internal* to a coarse-grid
      //     cell, and for these cases chained interpolations are necessary.
      //
      //                                   Note that H is a disjoint union
      //   fine/coarse:  =      ≠          of rows from I and rows from W.
      //  ---------------------------      W interpolates from coarse parent.
      //   P|_par   <-    I  or    H       Rows from I are for nonhanging nodes;
      //   P|_self  <-  H I  or  U H       Rows from W are for hanging nodes.

      const size_t n_nodes = fine_emat.n_nodes;
      const int ndofs = fine_emat.ndofs;
      const size_t n = n_nodes * ndofs;

      const int coarse_chn = coarse_oct.getMortonIndex();
      const int fine_chn = fine_oct.getMortonIndex();

      Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n, n);
      Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);

      // M = (M^T I)^T

      if (fine_oct.getLevel() == coarse_oct.getLevel())
      {
        // coarse.interp if fine nonhanging and coarse hanging.
        for (size_t i = 0; i < n_nodes; ++i)
          if (fine_nonhanging[i] and not coarse_nonhanging[i])
            for (int dof = 0; dof < ndofs; ++dof)
            {
              const size_t row = i * ndofs + dof;
              interp->template IKD_ParentChildInterpolation<(interp->C2P)>(
                  P.col(row).data(), P.col(row).data(), ndofs, coarse_chn);
            }
        P.transposeInPlace();
      }
      else  // (fine_oct.getLevel() > coarse_oct.getLevel())
      {
        // coarse.interp if coarse hanging.
        for (size_t i = 0; i < n_nodes; ++i)
          if (not coarse_nonhanging[i])
            for (int dof = 0; dof < ndofs; ++dof)
            {
              const size_t row = i * ndofs + dof;
              interp->template IKD_ParentChildInterpolation<(interp->C2P)>(
                  P.col(row).data(), P.col(row).data(), ndofs, coarse_chn);
            }
        P.transposeInPlace();

        // fine.interp if fine nonhanging.
        for (size_t i = 0; i < n_nodes; ++i)
          if (fine_nonhanging[i])
            for (int dof = 0; dof < ndofs; ++dof)
            {
              const size_t row = i * ndofs + dof;
              interp->template IKD_ParentChildInterpolation<(interp->C2P)>(
                  H.col(row).data(), H.col(row).data(), ndofs, fine_chn);
            }
        H.transposeInPlace();

        P = H * P;
      }

      //future: Other tensor library might simplify Kroenecker product. xtensor?

      Eigen::MatrixXd RAP = P.transpose() * fine_emat.entries * P;

      return ElementalMatrix(n_nodes, ndofs, std::move(RAP));

      // Check:
      //
      // The stored-to-virtual transformation (V) is a block matrix,
      // where blocks correspond to (hyper)faces in the cell decomposition.
      //
      // Blocks on the diagonal correspond to face interiors.
      // Off-diagonal blocks occur where the interior of a face is
      // interpolated in part from values on the face boundary.
    }
  }





  // Template instantiation.

  template class HybridPoissonMat<2u>;
  template class HybridPoissonMat<3u>;
  template class HybridPoissonMat<4u>;

}//namespace PoissonEq
