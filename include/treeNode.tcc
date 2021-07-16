#include <array>
#include <iomanip>
#include "mathUtils.h"
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

    //
    // TreeNode default constructor.
    template <typename T, unsigned int dim>
    TreeNode<T,dim>::TreeNode()
    {
      m_uiLevel = 0;
      m_coords = periodic::PCoord<T, dim>();

      m_isOnTreeBdry = false;

      //==========
#ifdef __DEBUG_TN__
      //@masado Allow arbitrary dimensions but warn about unexpected cases.
      if (dim < 1 || dim > 4)
        std::cout << "Warning: Value for dim: " << dim << std::endl;
#endif
    } //end function


    //
    // TreeNode constructor.
    template<typename T, unsigned int dim>
    TreeNode<T,dim>::TreeNode(const std::array<T,dim> coords, unsigned int level)
    {
      m_uiLevel = level;
      m_isOnTreeBdry = false;
      m_coords = periodic::PCoord<T, dim>(coords);
    } //end function


    //
    // TreeNode copy constructor.
    template<typename T, unsigned int dim>
    TreeNode<T,dim>::TreeNode(const TreeNode & other)
      : m_coords(other.m_coords),
        m_uiLevel(other.m_uiLevel),
        m_isOnTreeBdry(other.m_isOnTreeBdry)
    {
    } //end function


    // TreeNode protected constructor
    template <typename T, unsigned int dim>
    TreeNode<T,dim>::TreeNode(const int dummy, const std::array<T,dim> coords, unsigned int level)
    {
      m_coords = coords;
      m_uiLevel = level;

      m_isOnTreeBdry = false;
    }


    //
    // TreeNode assignment operator
    template<typename T, unsigned int dim>
    TreeNode<T,dim>& TreeNode<T,dim>::operator=(TreeNode<T,dim> const &other)
    {
      m_coords = other.m_coords;
      m_uiLevel = other.m_uiLevel;

      m_isOnTreeBdry = other.m_isOnTreeBdry;

      return *this;
    }




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
    if ((m_coords.coord(d) >> shiftIrlvnt) != (other.m_coords.coord(d) >> shiftIrlvnt))
      return false;
  }

  return true;
 
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
  std::for_each(other.m_coords.coords().begin(), other.m_coords.coords().end(),
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
inline void TreeNode<T,dim>::setLevel(unsigned int lev) {
  m_uiLevel = (lev & MAX_LEVEL);
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
    return m_coords.coord(d);
}

template <typename T, unsigned int dim>
inline const periodic::PRange<T, dim> TreeNode<T, dim>::range() const
{
  return periodic::PRange<T, dim>(m_coords, (1u << (m_uiMaxDepth - getLevel())));
}

template <typename T, unsigned int dim>
inline void TreeNode<T,dim>::setX(int d, T coord) {
#if __DEBUG_TN__
  assert(0 <= d && d < dim);
#endif
  m_coords.coord(d, coord);
}

template <typename T, unsigned int dim>
inline void TreeNode<T,dim>::setX(const std::array<T, dim> &coords)
{
  m_coords = periodic::PCoord<T, dim>(coords);
}



template <typename T, unsigned int dim>
inline bool TreeNode<T, dim>::getIsOnTreeBdry() const
{
  return m_isOnTreeBdry;
}

template <typename T, unsigned int dim>
inline void TreeNode<T, dim>::setIsOnTreeBdry(bool isOnTreeBdry)
{
  m_isOnTreeBdry = isOnTreeBdry;
}




template <typename T, unsigned int dim>
inline unsigned char TreeNode<T,dim>::getMortonIndex(T level) const
{
    unsigned char childNum = 0;

    T len = (1u << (getMaxDepth() - level));
    T len_par = (1u << (getMaxDepth() - level + 1u));

    #pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
        childNum += ((m_coords.coord(d) % len_par) / len) << d;
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
      T oldCoord = m_coords.coord(d);           // Activate.          // Suppress.
      m_coords.coord(d, (child & D ? oldCoord | selector : oldCoord & (~selector)));
    }
}

template <typename T, unsigned int dim>
inline unsigned int TreeNode<T,dim>::getCommonAncestorDepth(const TreeNode &other)
{
  unsigned int depth_rt = m_uiMaxDepth;
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
  {
    unsigned int diff = other.m_coords.coord(d) ^ m_coords.coord(d);
    // Using log2 gives the index of the highest '1'. We want one above that.
    unsigned int depth = m_uiMaxDepth - log2((diff << 1) | 1u);
    depth_rt = (depth < depth_rt ? depth : depth_rt);  // Minimum of depths.
  }
  return depth_rt;
}

template <typename T, unsigned int dim>
inline int TreeNode<T,dim>::getAnchor(std::array<T,dim> &xyz) const {
    xyz = m_coords.coords();
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
inline std::array<char, MAX_LEVEL+1> TreeNode<T,dim>::getBase32Hex(unsigned int lev) const
{
  if (!lev)
    lev = m_uiLevel;

  // https://en.wikipedia.org/wiki/Base32#base32hex
  const char base32hex[] = "0123456789ABCDEFGHIJKLMNOPQRSTUV";
  std::array<char, MAX_LEVEL+1> str;
  // It is assumed that MAX_LEVEL == m_uiMaxDepth + 1.
  /// for (int ii = 0; ii <= m_uiMaxDepth; ii++)
  for (int ii = 0; ii <= lev; ii++)
  {
    str[ii] = base32hex[getMortonIndex(ii)];
  }
  /// str[m_uiMaxDepth+1] = '\0';
  str[lev+1] = '\0';
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
  // parent level 1: (1u << (m_uiMaxDepth-1))  ->  (m_uiMaxDepth-1) 0s
  // parent level 2: (1u << (m_uiMaxDepth-2))  ->  (m_uiMaxDepth-2) 0s
  const unsigned int parLev = (this->getLevel() > 0 ? this->getLevel() - 1 : 0);
  const periodic::PCoord<T, dim> parCoords =
      this->m_coords.truncated(m_uiMaxDepth - parLev);
  return TreeNode(1, parCoords, parLev);
} //end function

template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getAncestor(unsigned int ancLev) const {
  const periodic::PCoord<T, dim> ancCoords =
      this->m_coords.truncated(m_uiMaxDepth - ancLev);
  return TreeNode(1, ancCoords, ancLev);
} //end function


template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getFirstChildMorton() const {
  return getChildMorton(0);
}

template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getChildMorton(unsigned char child) const {
  TreeNode<T,dim> m = *this;
  m.m_uiLevel++;
  m.setMortonIndex(child);
  return m;
}


/**
  @brief Get (inclusive) lower bound for a single dimension.
 */
template <typename T, unsigned int dim>
inline T TreeNode<T,dim>::lowerBound(int d) const {
  return m_coords.coord(d);
}

/**
  @brief Get (exclusive) upper bound for a single dimension.
 */
template <typename T, unsigned int dim>
inline T TreeNode<T,dim>::upperBound(int d) const {
  T len = (1u << (m_uiMaxDepth - getLevel()));
  return lowerBound(d) + len;
}


#if 0
/**
  @brief Get min (inclusive lower bound) for all dimensions.
 */
template <typename T, unsigned int dim>
inline std::array<T,dim> TreeNode<T,dim>::minX() const {
  return m_coords.coords();
}

/**
  @brief Get max (exclusive upper bound) for all dimensions.
 */
template <typename T, unsigned int dim>
inline std::array<T,dim> TreeNode<T,dim>::maxX() const {
  T len = (1u << (m_uiMaxDepth - getLevel()));
  std::array<T,dim> maxes = minX();
#pragma unroll(dim)
  for (int d = 0; d < dim; d++) { maxes[d] += len; }
  return maxes;
} //end function

#endif


template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getDFD() const {
  TreeNode<T,dim> dfd(1, m_coords, m_uiMaxDepth);
  return dfd;
} //end function


template <typename T, unsigned int dim>
inline TreeNode<T,dim> TreeNode<T,dim>::getDLD() const {
  T len = (1u << (m_uiMaxDepth - getLevel()));
  std::array<T,dim> coords = this->m_coords.coords();
  for (int d = 0; d < dim; ++d)
    coords[d] += len - 1;
  return TreeNode<T, dim>(coords, m_uiMaxDepth);
} //end function


#if 0
// Helper routine for getNeighbour() methods.
// If addr_in plus offset does not overflow boundary,  sets addr_out and returns true.
// Otherwise, returns false without setting addr_out.
template <typename T, bool includeDomBdry = false>
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

  T len = (1u << (m_uiMaxDepth - level));
  if ( (!includeDomBdry &&
           ((offset > 0 && addr_in >= (1u << m_uiMaxDepth) - len) ||  // Function is false if >=.
           (offset < 0 && addr_in < len))
       )
       ||
       (includeDomBdry &&
           ((offset > 0 && addr_in > (1u << m_uiMaxDepth) - len) ||   // Function is false if >.
           (offset < 0 && addr_in < len))
       ) )
  {
    return false;
  }

  addr_out = (offset > 0 ? addr_in + len : addr_in - len);
  return true;
}

template <typename T, unsigned int dim>
template <bool includeDomBdry>
inline TreeNode<T,dim> TreeNode<T,dim>::getNeighbour(unsigned int d, signed char offset) const
{
  T n_addr;
  bool is_valid_neighbour = getNeighbour1d<T,includeDomBdry>(m_uiCoords[d], getLevel(), offset, n_addr);
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
template <bool includeDomBdry>
inline TreeNode<T,dim> TreeNode<T,dim>::getNeighbour(std::array<signed char,dim> offsets) const
{
  std::array<T,dim> n_coords = m_uiCoords;
  const unsigned int level = getLevel();

  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
  {
    bool is_valid_neighbour = getNeighbour1d<T,includeDomBdry>(n_coords[d], level, offsets[d], n_coords[d]);
    if (!is_valid_neighbour)
    {
      return TreeNode<T,dim>();  // Root octant.
    }
  }

  return TreeNode<T,dim>(1, n_coords, level);
} //end function
#endif


template <typename T, unsigned int dim>
inline void TreeNode<T,dim>::appendAllNeighbours(std::vector<TreeNode<T,dim>> &nodeList) const
{
  // The set of neighbors is a 3x3x3x... hypercube with the center deleted.
  // We will index the neighbor cube by cycling indices so that the center is moved
  // to the (0,0,0,...) corner, and, skipping the first item,
  // lexicographic order takes it from there.
  //
  // Shift: delta_ref = (delta_physical + 3) % 3
  //        delta_physical = ((delta_ref + 1) % 3) - 1
  //        -1 <--> [2]    +1 <--> [1]    0 <--> [0]

  const unsigned int level = getLevel();
  periodic::PCoord<T, dim> distances;
  for (int d = 0; d < dim; ++d)
    distances.coord(d, (1u << (m_uiMaxDepth - level)));

  const periodic::PCoord<T, dim> &self = this->m_coords;
  const periodic::PCoord<T, dim> plus = self + distances;
  const periodic::PCoord<T, dim> minus = self - distances;

  // Bounds tests on each axis with respect to the unit hypercube.
  std::array<bool, dim> plusValid;
  std::array<bool, dim> minusValid;
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
  {
    plusValid[d] = plus.coord(d) < (1u << m_uiMaxDepth);
    minusValid[d] = minus.coord(d) >= 0 && minus.coord(d) < (1u << m_uiMaxDepth);
  }

  // Precompute strides for sizes of sub-faces.
  std::array<int, dim> stride;
  stride[0] = 1;
  for (int d = 1; d < dim; d++)
    stride[d] = stride[d-1]*3;
  
  //TODO is there a more efficient way than BigO(dx3^d) per call? Such as just BigTheta(3^d)?

  // For each neighbor, compose the address and append to nodeList.
  constexpr int pow3 = intPow(3, dim);
  for (int neighbourIdx = 1; neighbourIdx < pow3; neighbourIdx++)
  {
    bool isNeighbourUsable = true;
    periodic::PCoord<T, dim> n_coords = self;

    int remainder = neighbourIdx;
    for (int d = dim-1; d >= 0; d--)  // Large to small.
    {
      char delta_ref = (remainder < stride[d] ? 0 : remainder < 2*stride[d] ? 1 : 2);
      switch(delta_ref)
      {
        case 1: n_coords.coord(d, plus.coord(d));    isNeighbourUsable &= plusValid[d];
            break;
        case 2: n_coords.coord(d, minus.coord(d));   isNeighbourUsable &= minusValid[d];
            break;
      }
      remainder = remainder - delta_ref * stride[d];
    }

    if (isNeighbourUsable)
      nodeList.push_back(TreeNode<T,dim>(1, n_coords, level));
  }
}  // end function()


// ================= End Pseudo-getters ====================== //

// ================= Begin is-tests ============================ //

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::isRoot() const {
  return (this->getLevel() == 0);
}

template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::isAncestor(TreeNode<T,dim> const &other) const
{
  if (other.getLevel() <= this->getLevel())
    return false;
  return other.m_coords.truncated(m_uiMaxDepth - this->getLevel()) == m_coords;
}


template <typename T, unsigned int dim>
inline bool TreeNode<T,dim>::isAncestorInclusive(TreeNode<T,dim> const &other) const
{
  if (other.getLevel() < this->getLevel())
    return false;
  return other.m_coords.truncated(m_uiMaxDepth - this->getLevel()) == m_coords;
}


// ================ End is-tests ========================== //

} //end namespace ot
