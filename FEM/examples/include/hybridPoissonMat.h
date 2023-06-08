
#ifndef DENDRO_KT_HYBRID_POISSON_MAT_H
#define DENDRO_KT_HYBRID_POISSON_MAT_H

#include "feMatrix.h"
#include "poissonMat.h"

#include <unordered_map>

#include <external/aMat/Eigen/Dense>

namespace PoissonEq
{

  namespace detail
  {
    template <int dim>
    struct ElemCenter
    {
      ElemCenter() = default;
      ElemCenter(const Point<dim> &center) : center(center)
      { }

      bool operator==(const ElemCenter &other) const
      {
        return this->center == other.center;
      }

      Point<dim> center;
    };


    class ElementalMatrix;
  }


  enum HybridPartition { opaque, evaluated };



  // HybridPoissonMat: Extension that stores elemental matrices of subset.
  template <int dim>
  class HybridPoissonMat : public feMatrix<HybridPoissonMat<dim>, dim>
  {
    public:
      using Base = feMatrix<HybridPoissonMat<dim>, dim>;

      HybridPoissonMat(const ot::DA<dim> *da, int dof);
      HybridPoissonMat(HybridPoissonMat &&other);
      HybridPoissonMat & operator=(HybridPoissonMat &&other);


      // -----------------------------------------------------------------------
      // Elemental operations.
      // -----------------------------------------------------------------------

      void elementalMatVec(
          const VECType* in,
          VECType* out,
          unsigned int ndofs,
          const double*coords,
          double scale,
          bool isElementBoundary);

      /// void elementalSetDiag(
      ///     VECType *out,
      ///     unsigned int ndofs,
      ///     const double *coords,
      ///     double scale = 1.0);
      //
      /// void getElementalMatrix(
      ///     std::vector<ot::MatRecord> &records,
      ///     const double *coords,
      ///     bool isElementBoundary);


      // -----------------------------------------------------------------------
      // Produce hybrid coarsened operator. (Galerkin coarsening where explicit)
      // -----------------------------------------------------------------------

      HybridPoissonMat coarsen(const ot::DA<dim> *da) const;


      // -----------------------------------------------------------------------
      // Vector operations.
      // -----------------------------------------------------------------------

      // Override the outer matVec() so as to use custom loops.
      virtual void matVec(const VECType* in,VECType* out, double scale=1.0);

      // zero_boundary()
      bool zero_boundary() const { return m_matfree.zero_boundary(); }

      // zero_boundary()
      void zero_boundary(bool enable) { m_matfree.zero_boundary(enable); }

      // preMatVec()
      bool preMatVec(const VECType* in, VECType* out, double scale=1.0)
      {
        return m_matfree.preMatVec(in, out, scale);
      }

      // postMatVec()
      bool postMatVec(const VECType* in, VECType* out, double scale=1.0)
      {
        return m_matfree.postMatVec(in, out, scale);
      }


    private:
      struct ElemIndex
      {
        size_t local_element_id;
        HybridPartition partition_id;
      };


    private:
      void emat_mult(const double *in, double *out, size_t local_eid);

      bool is_evaluated(size_t local_eid) const;
      void is_evaluated(size_t local_eid, bool evaluated);

      detail::ElementalMatrix proto_emat() const;
      detail::ElementalMatrix elemental_matrix(size_t local_eid) const;  //future: return a view, avoid alloc
      void elemental_matrix(size_t local_eid, detail::ElementalMatrix emat);

    private:
      PoissonMat<dim> m_matfree;
      std::vector<Eigen::MatrixXd> m_emats;

      std::unordered_map<detail::ElemCenter<dim>, ElemIndex> m_partition_table;
  };

}//namespace PoissonEq


// Lazy implementation of hash().. future: hash by tuple or string_view (c++17)

#include <string>

template <int dim>
struct std::hash<PoissonEq::detail::ElemCenter<dim>>
{
  size_t operator()(const PoissonEq::detail::ElemCenter<dim> &key) const noexcept
  {
    std::string str((const char *) &key, sizeof(key));
    return std::hash<std::string>{}(str);
  }
};

#endif//DENDRO_KT_HYBRID_POISSON_MAT_H