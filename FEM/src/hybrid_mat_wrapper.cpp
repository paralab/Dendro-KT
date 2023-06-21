#include "hybrid_mat_wrapper.hpp"

namespace fem
{
  // ---------------------------------------------------------------------------
  // Implementation of detail
  // ---------------------------------------------------------------------------
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
    template ElementalMatrix RAP<2>(
        const ot::TreeNode<uint32_t, 2> &, const ElementalMatrix &,
        const std::vector<bool> &, const InterpMatrices<2, double> *,
        const ot::TreeNode<uint32_t, 2> &, const std::vector<bool> &);

    template ElementalMatrix RAP<3>(
        const ot::TreeNode<uint32_t, 3> &, const ElementalMatrix &,
        const std::vector<bool> &, const InterpMatrices<3, double> *,
        const ot::TreeNode<uint32_t, 3> &, const std::vector<bool> &);

    template ElementalMatrix RAP<4>(
        const ot::TreeNode<uint32_t, 4> &, const ElementalMatrix &,
        const std::vector<bool> &, const InterpMatrices<4, double> *,
        const ot::TreeNode<uint32_t, 4> &, const std::vector<bool> &);

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
}
