#include <array>
#include <iomanip>
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

    // =============== Constructors ================= //

    template <typename T, unsigned int dim>
    TreeNode<T,dim>::TreeNode() {
        m_uiLevel = 0;
        m_uiCoords.fill(0);
        m_uiLevel = 0;

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

        T mask = -(1u << (m_uiMaxDepth - level));

#pragma unroll(dim)
        for (int d = 0; d < dim; d++) { m_uiCoords[d] = (coords[d] & mask); }

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

    } //end function

    template<typename T, unsigned int dim>
    TreeNode<T,dim>::TreeNode (const int dummy, const std::array<T,dim> coords, unsigned int level)
    {
        m_uiLevel = level;

#pragma unroll(dim)
        for (int d = 0; d < dim; d++) { m_uiCoords[d] = coords[d]; }

    }


    template<typename T, unsigned int dim>
    TreeNode<T,dim>& TreeNode<T,dim>  :: operator = (TreeNode<T,dim>   const& other) {
        if (this == (&other)) {return *this;}
        m_uiCoords = other.m_uiCoords;
        m_uiLevel = other.m_uiLevel;

        return *this;
    } //end fn.




//========== Overloaded Operators =========== //

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::operator==(TreeNode<T,dim> const &other) const {
  // Levels must match.
  if (m_uiLevel != other.m_uiLevel)
    return false;

  // Shift so that we only compare the relevant bits in each coordinate.
  unsigned int shiftIrlvnt = m_uiMaxDepth - m_uiLevel;

  // Compare coordinates one dimension at a time.
  for (int d = 0; d < dim; d++)
  {
    if ((m_uiCoords[d] >> shiftIrlvnt) != (other.m_uiCoords[d] >> shiftIrlvnt))
      return false;
  }

  return true;
 
} //end fn.

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::operator!=(TreeNode<T,dim> const &other) const {
  return (!((*this) == other));
} //end fn.


//
// operator<()
//
template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::operator<(TreeNode<T,dim> const &other) const {

  // -- original Morton
  
  // Use the coordinate with the highest level difference (closest to root).
  T maxDiffCoord = 0;
  for (int d = 0; d < dim; d++)
  {
    T diffCoord = m_uiCoords[d] ^ other.m_uiCoords[d];  // Will have 0's where equal.
    maxDiffCoord = (diffCoord > maxDiffCoord ? diffCoord : maxDiffCoord);
  }

  // Find the index of the highest level of difference.
  T levelDiff = 0;
  while (levelDiff < m_uiLevel && levelDiff < other.m_uiLevel
      && !(maxDiffCoord & (1u << (m_uiMaxDepth - levelDiff))))
  {
    levelDiff++;
  }

  // Use that level to compare child numbers.
  // In case of descendantship, ancestor is strictly less than descendant.
  unsigned int myIndex = getMortonIndex(levelDiff);
  unsigned int otherIndex = other.getMortonIndex(levelDiff);
  if (myIndex == otherIndex)
    return m_uiLevel < other.m_uiLevel;
  else
    return myIndex < otherIndex;

  // -- original Morton
}

//
// operator<=()
//
template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::operator<=(TreeNode<T,dim> const &other) const
{
  return operator==(other) || operator<(other);
}

//
// operator<<()
//
template<typename T, unsigned int dim>
std::ostream& operator<<(std::ostream& os, TreeNode<T,dim> const& other) {
  std::for_each(other.m_uiCoords.begin(), other.m_uiCoords.end(),
      [&os] (T x) { os << std::setw(10) << x << " "; });
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
inline unsigned char TreeNode<T,dim>::getMortonIndex(T level) const
{
    unsigned char childNum = 0;

    unsigned int len = (1u << (getMaxDepth() - level));
    unsigned int len_par = (1u << (getMaxDepth() - level + 1u));

#pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
        childNum += ((m_uiCoords[d] % len_par) / len) << d;
    }
    return  childNum;
}

template <typename T, unsigned int dim>
inline unsigned char TreeNode<T,dim>::getMortonIndex() const
{
    return getMortonIndex(m_uiLevel);
}

template <typename T, unsigned int dim>
inline void TreeNode<T,dim>::setMortonIndex(unsigned char child)
{
    const T level = m_uiLevel;  // For now, only set at this level.
    const T selector = 1u << (m_uiMaxDepth - level);
#pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
      const T D = 1u << d;
      T oldCoord = m_uiCoords[d];           // Activate.          // Suppress.
      m_uiCoords[d] = (child & D ? oldCoord | selector : oldCoord & (~selector));
    }
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

template <typename T, unsigned int dim>
inline std::array<char, MAX_LEVEL+1> TreeNode<T,dim>::getBase32Hex() const
{
  // https://en.wikipedia.org/wiki/Base32#base32hex
  const char base32hex[] = "0123456789ABCDEFGHIJKLMNOPQRSTUV";
  std::array<char, MAX_LEVEL+1> str;
  // It is assumed that MAX_LEVEL == m_uiMaxDepth + 1.
  /// for (int ii = 0; ii <= m_uiMaxDepth; ii++)
  for (int ii = 0; ii <= m_uiLevel; ii++)
  {
    str[ii] = base32hex[getMortonIndex(ii)];
  }
  /// str[m_uiMaxDepth+1] = '\0';
  str[m_uiLevel+1] = '\0';
  return str;
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


template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getFirstChildMorton() const {
  const T mask = ~((1u << (m_uiMaxDepth - m_uiLevel)) - 1);
  TreeNode<T,dim> m = *this;
#pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    m.m_uiCoords[d] &= mask;      // Clear anything below parent bit.
  m.m_uiLevel++;
  
  return m;
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
