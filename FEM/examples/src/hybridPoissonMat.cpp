
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
    : Base(da, nullptr, dof), m_matfree(da, nullptr, dof), m_emats(da->getLocalElementSz())
  {
    // Partition table  (center coordinate -> element id).
    m_partition_table.reserve(da->getLocalElementSz());
    const auto &octlist = da->dist_tree()->getTreePartFiltered(da->stratum());
    for (size_t i = 0; i < octlist.size(); ++i)
    {
      const ot::TreeNode<uint32_t, dim> tn = octlist[i];
      std::array<double, dim> coords;
      double size;
      ot::treeNode2Physical(tn, coords.data(), size);
      const Point<dim> corner(coords);
      const Point<dim> center = corner + Point<dim>(size) / 2.0;
      const detail::ElemCenter<dim> key(center);

      //future: some opaque, some evaluated
      HybridPartition partition_id = evaluated;

      const ElemIndex value = ElemIndex{i, partition_id};

      assert(m_partition_table.find(key) == m_partition_table.end());//unique.

      m_partition_table.emplace(key, value);
    }


    //future: separate element-by-element for subset, not in constructor.

    // Those Eigen matrices that are not resized will just be null pointers.

    // Initialize m_emats (elemental matrices): evaluate getElementalMatrix()
    const size_t n = da->getNumNodesPerElement() * dof;
    for (Eigen::MatrixXd &mat: m_emats)
      mat.resize(n, n);

    size_t local_element_id = 0;
    m_matfree.collectMatrixEntries([&, this](
          const std::vector<PetscInt>& ,//ids
          const std::vector<PetscScalar>& entries )
    {
      auto &mat = this->m_emats[local_element_id];

      for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
          mat(r, c) = entries[r * n + c];

      ++local_element_id;
    });

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

    const ElemIndex index = m_partition_table[key];

    assert(m_partition_table.find(key) != m_partition_table.end());

    switch (index.partition_id)
    {
      case opaque:
        this->m_matfree.elementalMatVec(
            in, out, ndofs, coords, scale, isElementBoundary);
        break;

      case evaluated:
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
    const size_t n = this->da()->getNumNodesPerElement() * this->ndofs();
    const auto &mat = this->m_emats[local_eid];
    Eigen::Map<const Eigen::VectorXd> vec_in(in, n);
    Eigen::Map<Eigen::VectorXd> vec_out(out, n);
    vec_out = mat * vec_in;
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

    const InterpMatrices<dim, double> interpolation(this->da()->getElementOrder());

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
                &interpolation,
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
                  &interpolation,
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
    const size_t n = this->da()->getNumNodesPerElement() * this->ndofs();
    if (evaluated)
      m_emats[local_eid].resize(n, n);
    else
      m_emats[local_eid].resize(0, 0);
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
      assert(false); //TODO elemental assembly
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
      // Define the virtual vector (V s) by interpolating hanging nodes
      // from parent. P' represents the prolongation for a uniform grid.

      // If the coarse and fine octants are equal:
      //   parent fine := parent coarse              f.s|_par  <-   c.s
      //     self fine := virt. coarse               f.s|_self <- V c.s
      //
      // If the coarse and fine octants differ:
      //   parent fine := virtual coarse             f.s|_par  <-    V c.s
      //     self fine := interp.(virt. coarse)      f.s|_self <- P' V c.s
      //
      //     ^ Modify: Due to 2:1-balancing and single-level coarsening,
      //     a fine-grid self node (nonhanging) lives on a face
      //     that can be larger but *NEVER hanging in the coarse grid*.
      //     Therefore the coarse virtual nodes equal coarse self nodes:
      //
      //     self fine := interp.(coarse)            f.s|_self <- P' c.s
      //
      // Therefore, all rows of P are      Note that V is a disjoint union
      // either rows of V or of P':        of rows from I and rows from P''.
      //   fine/coarse:  =      ≠          P'' interpolates from coarse parent.
      //  --------------------------       Rows from I are for nonhanging nodes;
      //   P|_par   <-   I  or  V          Rows from P'' are for hanging nodes.
      //   P|_self  <-   V  or  P'

      const size_t n_nodes = fine_emat.n_nodes;
      const int ndofs = fine_emat.ndofs;
      const size_t n = n_nodes * ndofs;

      const int coarse_chn = coarse_oct.getMortonIndex();
      const int fine_chn = fine_oct.getMortonIndex();

      Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n, n);

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
      }
      else  // (fine_oct.getLevel() > coarse_oct.getLevel())
      {
        // coarse.interp if fine hanging and coarse hanging.
        //   fine.interp if fine nonhanging.
        for (size_t i = 0; i < n_nodes; ++i)
          if (fine_nonhanging[i] or not coarse_nonhanging[i])
          {
            const int chn = fine_nonhanging[i] ? fine_chn : coarse_chn;
            for (int dof = 0; dof < ndofs; ++dof)
            {
              const size_t row = i * ndofs + dof;
              interp->template IKD_ParentChildInterpolation<(interp->C2P)>(
                  P.col(row).data(), P.col(row).data(), ndofs, chn);
            }
          }
      }
      P.transposeInPlace();  // The "columns" defined above are rows as desired.

      //future: Other tensor library might simplify Kroenecker product. xtensor?

      Eigen::MatrixXd RAP = P.adjoint() * fine_emat.entries * P;

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
