#include "TreeNode.h"

#include <assert.h>

unsigned int m_uiMaxDepth = 30;   // @masado Does this go here? Should it be const?

namespace ot {

// =============== Constructors ================= //

//
// TreeNode()
//
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

//
// TreeNode(coords, level)
//
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

//
// TreeNode(otherTreeNode)
//
template<typename T, unsigned int dim>
TreeNode<T,dim>::TreeNode (const TreeNode & other) {
  m_uiLevel = other.m_uiLevel;

  #pragma unroll(dim)
  for (int d = 0; d < dim; d++) { m_uiCoords[d] = other.m_uiCoords[d]; }

  //m_uiDim = other.m_uiDim;
  //m_uiWeight = other.m_uiWeight;
} //end function

//
// TreeNode(dummy, coords, level)
//
template<typename T, unsigned int dim>
TreeNode<T,dim>::TreeNode (const int dummy, const std::array<T,dim> coords, unsigned int level)
{
  m_uiLevel = level;

  #pragma unroll(dim)
  for (int d = 0; d < dim; d++) { m_uiCoords[d] = coords[d]; }

  //m_uiDim = dim;
  //m_uiWeight = 1;
}


//Assignment operator
template<typename T, unsigned int dim>
TreeNode<T,dim>& TreeNode<T,dim>  :: operator = (TreeNode<T,dim>   const& other) {
  if (this == (&other)) {return *this;}
  m_uiCoords = other.m_uiCoords;
  m_uiLevel = other.m_uiLevel;
  ///m_uiWeight = other.m_uiWeight;

  return *this;
} //end fn.


    //@masado I need to ask how boundary info will be used.
  ///  //
  ///  // isBoundaryOctant(block, type, *flags)
  ///  //
  ///  template <typename T, unsigned int dim>
  ///  bool TreeNode<T,dim>::isBoundaryOctant(const TreeNode<T,dim>& block, int type, TreeNode<T,dim>::Flag2K *flags) const {
  ///  using Flag2K = TreeNode<T,dim>::Flag2K;
  ///  #if __DEBUG_TN__
  ///    if (sizeof(Flag2K)*8 < 2*dim)
  ///    {
  ///      std::cerr << "Error: Type used for flags has "
  ///                << sizeof(Flag2K)*8 << "bits, but "
  ///                << 2*dim << " bits are needed.\n";
  ///      assert(false);
  ///    }
  ///  #endif
  ///    Flag2K _flags = 0;
  ///  
  ///    unsigned int _x = block.getX();
  ///    unsigned int _y = block.getY();
  ///    unsigned int _z = block.getZ();	
  ///    unsigned int _d = block.getLevel();
  ///  
  ///    /*
  ///    // Block has to be an ancestor of the octant or equal to the octant.
  ///    if( (!block.isAncestor(*this)) && (block != *this) ) {
  ///    if (flags) {
  ///     *flags = _flags;
  ///     }
  ///     return false;
  ///     }
  ///     */
  ///  
  ///    if ((type & NEGATIVE) == NEGATIVE) {
  ///      // test if any of the anchor values matches those of the block ...
  ///      if (m_uiX == _x) _flags |= X_NEG_BDY;
  ///      if (m_uiY == _y) _flags |= Y_NEG_BDY;
  ///      if (m_uiZ == _z) _flags |= Z_NEG_BDY;
  ///    }
  ///  
  ///    if ((type & POSITIVE) == POSITIVE) {
  ///      unsigned int len  = (unsigned int)(1u << (m_uiMaxDepth - getLevel()));
  ///      unsigned int blen = ((unsigned int)(1u << (m_uiMaxDepth - _d))) - len;
  ///  
  ///      if (m_uiX == (_x + blen))  _flags |= X_POS_BDY;
  ///      if (m_uiY == (_y + blen))  _flags |= Y_POS_BDY;
  ///      if (m_uiZ == (_z + blen))  _flags |= Z_POS_BDY;
  ///    }
  ///  
  ///    if (flags) {
  ///      *flags = _flags;
  ///    }
  ///    if (_flags) {
  ///      return true;
  ///    }
  ///    return false;
  ///  } //end function
  ///  
  ///  //
  ///  // isBoundary(type, *flags)
  ///  //
  ///  template <typename T, unsigned int dim>
  ///  bool TreeNode<T,dim>::isBoundaryOctant(int type, TreeNode<T,dim>::Flag2K *flags) const {
  ///  using Flag2K = TreeNode<T,dim>::Flag2K;
  ///  #if __DEBUG_TN__
  ///    if (sizeof(Flag2K)*8 < 2*dim)
  ///    {
  ///      std::cerr << "Error: Type used for flags has "
  ///                << sizeof(Flag2K)*8 << "bits, but "
  ///                << 2*dim << " bits are needed.\n";
  ///      assert(false);
  ///    }
  ///  #endif
  ///    Flag2K _flags = 0;
  ///    if ((type & NEGATIVE) == NEGATIVE) {
  ///      // test if any of the anchor values is zero ...  (sufficient ??? )
  ///      if (!m_uiX) _flags |= X_NEG_BDY;
  ///      if (!m_uiY) _flags |=  Y_NEG_BDY;
  ///      if (!m_uiZ) _flags |=   Z_NEG_BDY;
  ///    }
  ///  
  ///    if ((type & POSITIVE) == POSITIVE) {
  ///      unsigned int len  = (unsigned int)(1u << (m_uiMaxDepth - getLevel()));
  ///      unsigned int blen = ((unsigned int)(1u << m_uiMaxDepth)) - len;
  ///  
  ///      if (m_uiX == blen)  _flags |= X_POS_BDY;
  ///      if (m_uiY == blen)  _flags |= Y_POS_BDY;
  ///      if (m_uiZ == blen)  _flags |= Z_POS_BDY;
  ///    }
  ///  
  ///    if (flags) *flags = _flags;
  ///    if (_flags) return true;
  ///  
  ///    return false;
  ///  } //end function






} //end namespace ot
