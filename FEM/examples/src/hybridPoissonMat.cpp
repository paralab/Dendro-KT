
#include "hybridPoissonMat.h"

namespace fem
{
  // Explicit instantiation definitions.
  template class HybridMatWrapper<2, PoissonEq::PoissonMat<2>>;
  template class HybridMatWrapper<3, PoissonEq::PoissonMat<3>>;
  template class HybridMatWrapper<4, PoissonEq::PoissonMat<4>>;
}
