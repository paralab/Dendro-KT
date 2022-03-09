
#include "pcoord.h"

namespace periodic
{
  template class PCoord<unsigned, 2>;
  template class PCoord<unsigned, 3>;
  template class PCoord<unsigned, 4>;

  template <> std::array<unsigned, 2> PCoord<unsigned, 2>::m_masks = {-1u, -1u};
  template <> std::array<unsigned, 3> PCoord<unsigned, 3>::m_masks = {-1u, -1u, -1u};
  template <> std::array<unsigned, 4> PCoord<unsigned, 4>::m_masks = {-1u, -1u, -1u, -1u};
}
