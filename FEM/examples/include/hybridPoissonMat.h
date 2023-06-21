
#ifndef DENDRO_KT_HYBRID_POISSON_MAT_H
#define DENDRO_KT_HYBRID_POISSON_MAT_H

#include "feMatrix.h"
#include "poissonMat.h"

#include <unordered_map>

#include "Eigen/Dense"

namespace PoissonEq
{

  namespace detail
  {
    class ElementalMatrix;

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


    enum HybridPartition { opaque, evaluated };

    struct ElemIndex
    {
      size_t local_element_id;
      HybridPartition partition_id;
    };

    class ConstArrayBool1DView;

    class ArrayBool2D  //future: array of byte, not vector<bool>
    {
      friend ConstArrayBool1DView;
      public:
        ArrayBool2D(size_t outer, size_t inner, bool value);
        void set_row(size_t outer_idx, const std::vector<bool> &new_row);
        ConstArrayBool1DView operator[](size_t outer_idx) const;
      private:
        std::vector<bool> m_vec_bool;
        size_t m_outer = 0;
        size_t m_inner = 0;

    };

    class ConstArrayBool1DView  //future: array of byte, not vector<bool>
    {
      public:
        auto operator[](size_t inner_idx) const
        {
          return m_array_ptr->m_vec_bool[offset + inner_idx];
        }
      public:
        const ArrayBool2D *m_array_ptr;
        size_t offset = 0;
    };
  }



  // HybridPoissonMat: Extension that stores elemental matrices of subset.
  template <int dim>
  class HybridPoissonMat : public feMatrix<HybridPoissonMat<dim>, dim>
  {
    public:
      using Base = feMatrix<HybridPoissonMat<dim>, dim>;

      HybridPoissonMat(const ot::DA<dim> *da, int dof);
      HybridPoissonMat(HybridPoissonMat &&other);
      HybridPoissonMat & operator=(HybridPoissonMat &&other);



      // A_global = Fᵀ Hᵀ A H F
      //
      //            F is the global to elemental mapping (puts parent @hanging)
      //            (E is the same but for the coarse grid)
      //            F? is owner injection: Ignore non-owners.  F F? F = F
      //
      //                H (block-diagonal) interpolates parent->hanging virtual
      //                (K does the same but for the coarse grid)
      //
      //                    A is the elemental action on virtual nodes

      // custom operators
      // - elementalMatVec()                .......             A x
      // - elementalSetDiag()               .......        diag(A)
      // - getElementalMatrix()             .......             A

      // feMatrix
      // - matVec()                         .......       Fᵀ Hᵀ * H F x
      // - preMatVec()
      // - postMatVec()
      // - setDiag()
      // - collectMatrixEntries()           .......       Fᵀ Hᵀ * H F
      // - getAssembledMatrix(Petsc Mat)

      // hybridMatrix
      // - matVec()                         .......       Fᵀ    *   F x
      // - getAssembledMatrix()             .......       Fᵀ    *   F
      // - setDiag()                        .......       Fᵀ diag(*) F
      //
      // - evaluate()                       .......          Hᵀ * H
      //   elemental_matrix()
      //
      // - elementalMatVec()                .......          Hᵀ * H x

      // multigrid
      // - vcycle()
      //   - (operator-dependent)
      //     - smoother()
      //     - coarsen() //(R.A.P)  <-+-+       Kᵀ Pᵀ           *        P K
      //     - coarse_solver()         \ \
      //   - (not operator-dependent)  /  |
      //     - restriction()         -+  /   Eᵀ Kᵀ Pᵀ F?ᵀ x
      //     - prolongation()      -----+                             F? P K E x


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

      // aliasing is allowed
      void elemental_p2c(size_t local_eid, const double *in, double *out) const;
      void elemental_c2p(size_t local_eid, const double *in, double *out) const;

      bool is_evaluated(size_t local_eid) const;

      void store_evaluated(size_t local_eid);


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


      // -----------------------------------------------------------------------
      // Assembly.
      // -----------------------------------------------------------------------

      // Override the outer setDiag() so as to use custom loops.
      virtual void setDiag(VECType *out, double scale = 1.0);

      // Override the outer getAssembledMatrix() so as to use custom loops.
      virtual bool getAssembledMatrix(Mat *J, MatType mtype);


    private:
      template <bool use_c2p>
      static void transform_hanging(
          const InterpMatrices<dim, double> *interp,
          int child_number,
          detail::ConstArrayBool1DView nonhanging,
          size_t n_nodes, int ndofs,
          const double *in, double *out);


    private:
      // Evaluates from geometric definition.
      detail::ElementalMatrix evaluate(size_t local_eid) const;

      void is_evaluated(size_t local_eid, bool evaluated);

      detail::ElementalMatrix proto_emat() const;
      detail::ElementalMatrix elemental_matrix(size_t local_eid) const;  //future: return a view, avoid alloc
      void elemental_matrix(size_t local_eid, detail::ElementalMatrix emat);

      void emat_mult(const double *in, double *out, size_t local_eid);

    private:
      mutable PoissonMat<dim> m_matfree;   // Internal getElementalMatrix() is not const.. but should be.
      std::vector<Eigen::MatrixXd> m_emats;

      std::unordered_map<detail::ElemCenter<dim>, detail::ElemIndex> m_partition_table;
      std::vector<typename decltype(m_partition_table)::iterator> m_inv_partition_table;

      detail::ArrayBool2D m_elemental_nonhanging;

      InterpMatrices<dim, double> m_interp;
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
