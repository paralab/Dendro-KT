#include "TreeNode.h"

#include <assert.h>

unsigned int m_uiMaxDepth = 30;   // @masado Does this go here? Should it be const?

namespace ot {

template <typename T, unsigned int dim>
TreeNode<T,dim>::TreeNode() {
  m_uiLevel = 0;
  m_uiCoords.fill(0);
  m_uiLevel = 0;
  //m_uiWeight = 1;
  //==========
#ifdef __DEBUG_TN__
  //@masado Allow arbitrary dimensions but warn about unexpected cases.
  if (dim < 1 || dim > 4) {
    std::cout << "Warning: Value for dim: " << dim << std::endl;
  }
#endif
} //end function

template<typename T, unsigned int dim>
TreeNode<T,dim>::TreeNode (const std::array<T,dim> coords, unsigned int level) {
  m_uiLevel = level;

  #pragma unroll(dim)
  for (int d = 0; d < dim; d++) { m_uiCoords[d] = coords[d]; }

  //m_uiDim = dim;
  //m_uiWeight = 1;

#ifdef __DEBUG_TN__
  //@masado Allow arbitrary dimensions but warn about unexpected cases.
  if (dim < 1 || dim > 4) {
    std::cout << "Warning: Value for dim: " << dim << std::endl;
  }
  for ( T x : m_uiCoords )
  {
    assert( x < ((unsigned int)(1u << m_uiMaxDepth)));
    assert((x % ((unsigned int)(1u << (m_uiMaxDepth - level)))) == 0);
  }
#endif
} //end function


template<typename T, unsigned int dim>
TreeNode<T,dim>::TreeNode (const TreeNode & other) {
  m_uiLevel = other.m_uiLevel;

  #pragma unroll(dim)
  for (int d = 0; d < dim; d++) { m_uiCoords[d] = other.m_uiCoords[d]; }

  //m_uiDim = other.m_uiDim;
  //m_uiWeight = other.m_uiWeight;
} //end function



template<typename T, unsigned int dim>
TreeNode<T,dim>::TreeNode (const int dummy, const std::array<T,dim> coords, unsigned int level)
{
  m_uiLevel = level;

  #pragma unroll(dim)
  for (int d = 0; d < dim; d++) { m_uiCoords[d] = coords[d]; }

  //m_uiDim = dim;
  //m_uiWeight = 1;
}


} //end namespace ot
