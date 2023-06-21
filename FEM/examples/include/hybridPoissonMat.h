
#ifndef DENDRO_KT_HYBRID_POISSON_MAT_H
#define DENDRO_KT_HYBRID_POISSON_MAT_H

#include "hybrid_mat_wrapper.hpp"
#include "poissonMat.h"

namespace PoissonEq
{
  template <int dim>
  using HybridPoissonMat = fem::HybridMatWrapper<dim, PoissonMat<dim>>;
}

namespace fem
{
  // Explicit instantiation declarations (definitions in hybridPoissonMat.cpp).
  extern template class HybridMatWrapper<2, PoissonEq::PoissonMat<2>>;
  extern template class HybridMatWrapper<3, PoissonEq::PoissonMat<3>>;
  extern template class HybridMatWrapper<4, PoissonEq::PoissonMat<4>>;
}

#endif//DENDRO_KT_HYBRID_POISSON_MAT_H
