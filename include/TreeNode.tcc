#include <array>
#if __DEBUG_TN__
#include <assert.h>
#endif // __DEBUG_TN__

namespace ot {


//========== Overloaded Operators =========== //

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::operator==(TreeNode<T,dim> const &other) const {
  // std::array::operator== compares element by element.
  //@masado Check if there is a function call overhead. If so, resort to TMP.
  if ((m_uiCoords == other.m_uiCoords) &&
      ((m_uiLevel & MAX_LEVEL) == (other.m_uiLevel & MAX_LEVEL))) {
    return true;
  } else {
    return false;
  }
} //end fn.

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::operator!=(TreeNode<T,dim> const &other) const {
  return (!((*this) == other));
} //end fn.


//
// operator<<()
//
template<typename T, unsigned int dim>
std::ostream& operator<<(std::ostream& os, TreeNode<T,dim> const& other) {
  std::for_each(other.m_uiCoords.begin(), other.m_uiCoords.end(),
      [&os] (T x) { os << x << " "; });
  return (os << other.getLevel());
} //end fn.





template <typename T, unsigned int dim>
inline unsigned int TreeNode<T,dim>::getLevel() const {
  return (m_uiLevel & MAX_LEVEL);
}


/**
  @brief Get min (inclusive lower bound) for a single dimension.
 */
template <typename T, unsigned int dim>
inline T TreeNode<T,dim>::minX(int d) const {
#if __DEBUG_TN__
  assert(0 <= d && d < dim);
#endif
  return m_uiCoords[d];
}

/**
  @brief Get max (exclusive upper bound) for a single dimension.
 */
template <typename T, unsigned int dim>
inline T TreeNode<T,dim>::maxX(int d) const {
#if __DEBUG_TN__
  assert(0 <= d && d < dim);
#endif
  unsigned int len = (1u << (m_uiMaxDepth - getLevel()));
  return (minX(d) + len);
}

/**
  @brief Get min (inclusive lower bound) for all dimensions.
 */
template <typename T, unsigned int dim>
inline std::array<T,dim> TreeNode<T,dim>::minX() const {
  return m_uiCoords;
}

/**
  @brief Get max (exclusive upper bound) for all dimensions.
 */
template <typename T, unsigned int dim>
inline std::array<T,dim> TreeNode<T,dim>::maxX() const {
  unsigned int len = (1u << (m_uiMaxDepth - getLevel()));
  std::array<T,dim> maxes = minX();
#pragma unroll(dim)
  for (int d = 0; d < dim; d++) { maxes[d] += len; }
  return maxes;
} //end function


} //end namespace ot
