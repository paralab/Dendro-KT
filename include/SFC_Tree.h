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

#include <mpi.h>

#include <vector>

namespace ot
{

//
// BucketInfo{}
//
// Buckets to temporarily represent (interior) nodes in the hyperoctree
// while we carry out breadth-first traversal. See distTreeSort().
template <typename T>
struct BucketInfo          // Adapted from Dendro4 sfcSort.h:132.
{
  int rot_id;
  unsigned int lev;
  T begin;
  T end;
};

template struct BucketInfo<unsigned int>;


template <typename T, unsigned int D>
struct SFC_Tree
{
  
  // Notes:
  //   - This method operates in-place and returns buckets at level eLev.
  //     The members of each BucketInfo object can be used to further
  //     refine the bucket.
  //   - From sLev to eLev INCLUSIVE
  //   - outBuckets will be appended with all the leaf buckets
  //     between sLev and eLev. Recommended to init to empty vector.
  static void locTreeSort(TreeNode<T,D> *points,
                          unsigned int begin, unsigned int end,
                          unsigned int sLev,
                          unsigned int eLev,
                          int pRot,            // Initial rotation, use 0 if sLev is 1.
                          std::vector<BucketInfo<unsigned int>> &outBuckets);

  // Use this to initialize the last argument to locTreeSort.
  // Convenient if you don't care about the buckets and you would rather
  // write `auto' instead of the full parameter type.
  static std::vector<BucketInfo<unsigned int>> getEmptyBucketVector()
  {
    return std::vector<BucketInfo<unsigned int>>();
  }

  // Notes:
  //   - outSplitters contains both the start and end of children at level `lev'
  static void SFC_bucketing(TreeNode<T,D> *points,
                          unsigned int begin, unsigned int end,
                          unsigned int lev,
                          int pRot,
                          std::array<unsigned int, TreeNode<T,D>::numChildren+1> &outSplitters);


  static void distTreeSort(std::vector<TreeNode<T,D>> inp,
                           std::vector<TreeNode<T,D>> &out,
                           double loadFlexibility,
                           MPI_Comm comm);

};

// Template instantiations.
template struct SFC_Tree<unsigned int, 2>;
template struct SFC_Tree<unsigned int, 3>;
template struct SFC_Tree<unsigned int, 4>;

} // namespace ot

#endif // DENDRO_KT_SFC_TREE_H
