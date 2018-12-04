/*
 * SFC_Tree.h
 *
 * Masado Ishii  --  UofU SoC, 2018-12-03
 *
 * Based on work by Milinda Fernando and Hari Sundar.
 *   - Algorithms: SC18 "Comparison Free Computations..." TreeSort, TreeConstruction, TreeBalancing
 *   - Code: Dendro4 [sfcSort.h] [construct.cpp]
 *
 * My contribution is to extend the data structures to 4 dimensions (or higher).
 */

#ifndef DENDRO_KT_SFC_TREE_H
#define DENDRO_KT_SFC_TREE_H

#include "TreeNode.h"
#include <vector>

namespace ot
{

template <typename T, unsigned int D>
struct SFC_Tree
{
  
  // Notes:
  //   From sLev to eLev INCLUSIVE
  static void locTreeSort(const TreeNode<T,D> *inp_begin, const TreeNode<T,D> *inp_end,
                          std::vector<TreeNode<T,D>> &out,
                          unsigned int sLev,
                          unsigned int eLev,
                          unsigned int pRot);  // Initial rotation, use 0 if sLev is 1.

};

// Template instantiations.
template struct SFC_Tree<unsigned int, 2>;
template struct SFC_Tree<unsigned int, 3>;
template struct SFC_Tree<unsigned int, 4>;

} // namespace ot

#endif // DENDRO_KT_SFC_TREE_H
