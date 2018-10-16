#include <array>
#if __DEBUG_TN__
#include <assert.h>
#endif // __DEBUG_TN__

/// namespace detail {
/// 
/// /**
///   @author Masado Ishii
///   @brief Recursively defined static inline expressions for arbitrary number of terms.
///   @tparam dim Array size, also size of the expression.
///   @remarks I have verified the efficiency of the assembly code for \
///            this struct using a toy lambda example.
/// */
/// // Recursive case.
/// template <unsigned int dim>
/// struct StaticUtils
/// {
///   /**
///     @param acc Function object that accesses the i'th boolean value.
///     @param start Starting index.
///   */
///   template <typename AccType>  // bool (*acc)(unsigned int idx)
///   static bool reduce_and(AccType &&acc, unsigned int start = 0)
///   { return acc(start) && StaticUtils<dim-1>::reduce_and(acc, start+1); }
/// };
/// 
/// // Base case.
/// template <> struct StaticUtils<0u>
/// {
///   template <typename AccType>
///   static bool reduce_and(AccType &&acc, unsigned int start = 0)
///   { return true; }
/// };
/// 
/// } // namespace detail

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



// =============== Getters and Setters ============ //

template <typename T, unsigned int dim>
inline unsigned int TreeNode<T,dim>::getDim() const {
    return dim;
}

template <typename T, unsigned int dim>
inline unsigned int TreeNode<T,dim>::getMaxDepth() const {
    return m_uiMaxDepth;
}

template <typename T, unsigned int dim>
inline unsigned int TreeNode<T,dim>::getLevel() const {
  return (m_uiLevel & MAX_LEVEL);
}

template <typename T, unsigned int dim>
inline unsigned int TreeNode<T,dim>::getFlag() const {
    return m_uiLevel;
}

template <typename T, unsigned int dim>
inline T TreeNode<T,dim>::getX(int d) const {
#if __DEBUG_TN__
  assert(0 <= d && d < dim);
#endif
    return m_uiCoords[d];
}

template <typename T, unsigned int dim>
inline int TreeNode<T,dim>::getAnchor(std::array<T,dim> &xyz) const {
    xyz = m_uiCoords;
    return 1;
}

template <typename T, unsigned int dim>
inline int TreeNode<T,dim>::setFlag(unsigned int w) {
    m_uiLevel = w;
    return 1;
}

template <typename T, unsigned int dim>
inline int TreeNode<T,dim>::orFlag(unsigned int w) {
    m_uiLevel = (m_uiLevel | w);
    return 1;
}


// ================= End Getters and Setters =================== //

// ================= Begin Pseudo-getters ====================== //

template <typename T, unsigned int dim>
inline T TreeNode<T,dim>::getParentX(int d) const {
#if __DEBUG_TN__
  assert(0 <= d && d < dim);
#endif
    return getParent().getX(d);
}


template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getParent() const {
    //For any node at level l, the last (maxD-l) bits are 0.
    //By convention, root's parent is also root.
    std::array<T,dim> parCoords;
    unsigned int parLev = (((m_uiLevel & MAX_LEVEL) > 0)
                           ? ((m_uiLevel & MAX_LEVEL) - 1) : 0);
#pragma unroll(dim)
    for (int d = 0; d < dim; d++)
        parCoords[d] = ((m_uiCoords[d] >> (m_uiMaxDepth - parLev)) << (m_uiMaxDepth - parLev));
    return TreeNode(1, parCoords, parLev);
} //end function

template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getAncestor(unsigned int ancLev) const {
    std::array<T,dim> ancCoords;
#pragma unroll(dim)
    for (int d = 0; d < dim; d++)
        ancCoords[d] = ((m_uiCoords[d] >> (m_uiMaxDepth - ancLev)) << (m_uiMaxDepth - ancLev));
    return TreeNode(1, ancCoords, ancLev);
} //end function


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



template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getDFD() const {
  TreeNode<T,dim> dfd(1, m_uiCoords, m_uiMaxDepth);
  return dfd;
} //end function


template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getDLD() const {
  std::array<T,dim> maxes = maxX();
  std::for_each(maxes.begin(), maxes.end(), [](T &v) { v--; });
  TreeNode<T,dim> dld(1, maxes, m_uiMaxDepth);
  return dld;
} //end function


// Helper routine for getNeighbour() methods.
// If addr_in plus offset does not overflow boundary,  sets addr_out and returns true.
// Otherwise, returns false without setting addr_out.
template <typename T>
inline bool getNeighbour1d(T addr_in,
    unsigned int level, signed char offset, T &addr_out)
{
  ////const unsigned int inv_level = m_uiMaxDepth - level;
  ////addr_in = (addr_in >> inv_level) << inv_level;  // Clear any bits deeper than level.

  if (offset == 0)
  {
    addr_out = addr_in;
    return true;
  }

  unsigned int len = (1u << (m_uiMaxDepth - level));
  if ((offset > 0 && addr_in >= (1u << m_uiMaxDepth) - len) ||
      (offset < 0 && addr_in < len))
  {
    return false;
  }

  addr_out = (offset > 0 ? addr_in + len : addr_in - len);
  return true;
}

template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getNeighbour(unsigned int d, signed char offset) const
{
  T n_addr;
  bool is_valid_neighbour = getNeighbour1d(m_uiCoords[d], getLevel(), offset, n_addr);
  if (is_valid_neighbour)
  {
    std::array<T,dim> n_coords = m_uiCoords;
    n_coords[d] = n_addr;
    return TreeNode<T,dim>(1, n_coords, getLevel());
  }
  else
  {
    return TreeNode<T,dim>(); // Root octant.
  }
}

template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getNeighbour(std::array<signed char,dim> offsets) const
{
  std::array<T,dim> n_coords = m_uiCoords;
  const unsigned int level = getLevel();
  
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
  {
    bool is_valid_neighbour = getNeighbour1d(n_coords[d], level, offsets[d], n_coords[d]);
    if (!is_valid_neighbour)
    {
      return TreeNode<T,dim>();  // Root octant.
    }
  }

  return TreeNode<T,dim>(1, n_coords, level);
} //end function


//// // Named neighbours for 3D.
//// // Assume that 3D dimensions are 0=X 1=Y 2=Z.
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getLeft() const
//// {
////   return getNeighbour(0,-1);
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getRight() const
//// {
////   return getNeighbour(0,+1);
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getFront() const
//// {
////   return getNeighbour(1,-1);
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBack() const 
//// {
////   return getNeighbour(1,+1);
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTop() const
//// {
////   return getNeighbour(2,-1);
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottom() const
//// {
////   return getNeighbour(2,+1);
//// }
//// 
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopLeft() const
//// {
////   return getNeighbour({-1,0,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopRight() const
//// {
////   return getNeighbour({+1,0,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomLeft() const
//// {
////   return getNeighbour({-1,0,-1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomRight() const
//// {
////   return getNeighbour({+1,0,-1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getLeftFront() const
//// {
////   return getNeighbour({-1,0,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getRightFront() const
//// {
////   return getNeighbour({+1,-1,0});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopFront() const
//// {
////   return getNeighbour({0,-1,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomFront() const
//// {
////   return getNeighbour({0,-1,-1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopLeftFront() const
//// {
////   return getNeighbour({-1,-1,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopRightFront() const
//// {
////   return getNeighbour({+1,-1,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomLeftFront() const
//// {
////   return getNeighbour({-1,-1,-1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomRightFront() const
//// {
////   return getNeighbour({+1,-1,-1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getLeftBack() const
//// {
////   return getNeighbour({-1,+1,0});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getRightBack() const
//// {
////   return getNeighbour({+1,+1,0});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopBack() const
//// {
////   return getNeighbour({0,+1,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomBack() const
//// {
////   return getNeighbour({0,+1,-1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopLeftBack() const
//// {
////   return getNeighbour({-1,+1,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getTopRightBack() const
//// {
////   return getNeighbour({+1,+1,+1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomLeftBack() const
//// {
////   return getNeighbour({-1,+1,-1});
//// }
//// template <typename T>
//// inline TreeNode<T,3> TreeNode<T,3>::getBottomRightBack() const
//// {
////   return getNeighbour({+1,+1,-1});
//// }







// ================= End Pseudo-getters ====================== //

// ================= Begin is-tests ============================ //

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::isRoot() const {
  return (this->getLevel() == 0);
}

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::isAncestor(TreeNode<T,dim> const &other) const {
    std::array<T,dim> min1, min2, max1, max2;

    min1 = this->minX();
    min2 = other.minX();
    
    max1 = this->maxX();
    max2 = other.maxX();

    //@masado Again it should be possible to get a single short-circuiting expression \
              using recursive TMP if needed. See StaticUtils at top.
      // Here is the TMP solution if desired.
    ///  bool state1 = ( (this->getLevel() < other.getLevel())  \
                  && detail::StaticUtils<dim>::reduce_and([&min1, &min2] (unsigned int d) { return min2[d] >= min1[d]; }) \
                  && detail::StaticUtils<dim>::reduce_and([&max1, &max2] (unsigned int d) { return max2[d] <= max1[d]; }) );
    bool state1=( (this->getLevel() < other.getLevel()) );
    #pragma unroll(dim)
    for (int d = 0; d < dim; d++)
      state1 = state1 && (min2[d] >= min1[d]) && (max2[d] <= max1[d]);

    return state1;
    // In a previous version there was `state2` involving Hilbert ordering.

} // end function

// ================ End is-tests ========================== //

} //end namespace ot
