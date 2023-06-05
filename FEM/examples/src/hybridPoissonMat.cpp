
#include "hybridPoissonMat.h"

namespace PoissonEq
{
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
    const int n = da->getNumNodesPerElement() * dof;
    for (Eigen::MatrixXd &mat: m_emats)
      mat.resize(n, n);

    // Initialize m_emats by evaluating getElementalMatrix()

    // future: separate element-by-element for subset

    // Elemental matrices.
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
    const int n = this->da()->getNumNodesPerElement() * this->ndofs();
    const auto &mat = this->m_emats[local_eid];
    Eigen::Map<const Eigen::VectorXd> vec_in(in, n);
    Eigen::Map<Eigen::VectorXd> vec_out(out, n);
    vec_out = mat * vec_in;
  }



  // Template instantiation.

  template class HybridPoissonMat<2u>;
  template class HybridPoissonMat<3u>;
  template class HybridPoissonMat<4u>;

}//namespace PoissonEq
