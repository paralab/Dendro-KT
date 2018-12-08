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


// Wrapper around std::vector to act like a queue, plus it has a
// single barrier, which can be moved to the end of the queue at any time.
// TODO This should go in some other utilities header file.
template <typename T>
struct BarrierQueue
{
  // Usage:
  //
  // BarrierQueue q;
  // for (int i = 0; i < 5; i++) { q.enqueue(i); }
  // q.reset_barrier();
  // for (int i = 5; i < 10; i++) { q.enqueue(i); }
  //
  // int x;
  // while (q.dequeue(x)) { std::cout << x << ' '; }  // 0 1 2 3 4
  // q.reset_barrier();
  // while (q.dequeue(x)) { std::cout << x << ' '; }  // 5 6 7 8 9

  struct Range
  {
    typename std::vector<T>::iterator m_begin, m_end;
    typename std::vector<T>::iterator begin() { return m_begin; }
    typename std::vector<T>::iterator end() { return m_end; }
  };

  typename std::vector<T>::size_type b;  // An out-of-band barrier.
  std::vector<T> q;             // If you modify this, call reset_barrier() afterward.

  BarrierQueue() : q(), b(0) {};
  BarrierQueue(typename std::vector<T>::size_type s) : q(s), b(0) {};
  BarrierQueue(typename std::vector<T>::size_type s, T val) : q(s, val), b(0) {};
  void clear() { q.clear(); b = 0; }
  void reset_barrier() { b = q.size(); }
  void resize_back(typename std::vector<T>::size_type count) { q.resize(count + b); }
  typename std::vector<T>::size_type get_barrier() { return b; }
  typename std::vector<T>::size_type size() { return q.size(); }
  T front() { return *q.begin(); }
  T back() { return *q.end(); }
  Range leading() { return {q.begin(), q.begin() + b}; }
  Range trailing() { return {q.begin() + b, q.end()}; }
  void enqueue(T val) { q.push_back(val); }
  typename std::vector<T>::size_type dequeue(T &val)
  { if (b > 0) { val = q[0]; q.erase(q.begin()); } return (b > 0 ? b-- : 0); }
};


template <typename T, unsigned int D>
struct SFC_Tree
{
  
  // Notes:
  //   - This method operates in-place and returns buckets at level eLev.
  //     The members of each BucketInfo object can be used to further
  //     refine the bucket.
  //   - From sLev to eLev INCLUSIVE
  //   - outBuckets will be appended with all the ending-level buckets at eLev.
  //     These buckets also represent all the chunks that could be internally unsorted.
  //     (Any TreeNodes at or above eLev will have been sorted.)
  //     Recommended to init to empty vector, i.e. getEmptyBucketVector().
  static void locTreeSort(TreeNode<T,D> *points,
                          unsigned int begin, unsigned int end,
                          unsigned int sLev,
                          unsigned int eLev,
                          int pRot,            // Initial rotation, use 0 if sLev is 1.
                          std::vector<BucketInfo<unsigned int>> &outBuckets,
                          bool makeBuckets = true);

  //TODO remove all bucket-collecting from locTreeSort() (`outBuckets').
  //It is not needed, and besides, the bucketing I implemented in locTreeSort()
  //does not indicate what point in the SFC ordering each bucket belongs to,
  //so it is not really useful anyway.
  //Proper bucketing will be implemented in distTreeSort.
  //Both locTreeSort and distTreeSort will make use of SFC_bucketing(),
  //and the splitters returned by SFC_bucketing are useful in both cases
  //to know where to set bounds for deeper refinement.

  // Use this to initialize the last argument to locTreeSort.
  // Convenient if you don't care about the buckets and you would rather
  // write `auto' instead of the full parameter type.
  static std::vector<BucketInfo<unsigned int>> getEmptyBucketVector()
  {
    return std::vector<BucketInfo<unsigned int>>();
  }

  // Notes:
  //   - outSplitters contains both the start and end of children at level `lev'
  //     This is to be consistent with the Dendro4 SFC_bucketing().
  //
  //     One difference is that here the buckets are ordered by the SFC
  //     (like the returned data is ordered) and so `outSplitters' should be
  //     monotonically increasing; whereas in Dendro4 SFC_bucketing(), the splitters
  //     are in permuted order.
  static void SFC_bucketing(TreeNode<T,D> *points,
                          unsigned int begin, unsigned int end,
                          unsigned int lev,
                          int pRot,
                          std::array<unsigned int, TreeNode<T,D>::numChildren+1> &outSplitters);


  // Notes:
  //   - points will be replaced/resized with globally sorted data.
  static void distTreeSort(std::vector<TreeNode<T,D>> &points,
                           double loadFlexibility,
                           MPI_Comm comm);

  //
  // treeBFTNextLevel()
  //   Takes the queue of BucketInfo in a breadth-first traversal, and finishes
  //   processing the current level. Each dequeued bucket is subdivided,
  //   and the sub-buckets in the corresponding range of `points` are sorted.
  //   Then the sub-buckets are initialized and enqueued to the back.
  //
  static void treeBFTNextLevel(TreeNode<T,D> *points,
      std::vector<BucketInfo<unsigned int>> &bftQueue);


};

// Template instantiations.
template struct SFC_Tree<unsigned int, 2>;
template struct SFC_Tree<unsigned int, 3>;
template struct SFC_Tree<unsigned int, 4>;

} // namespace ot

#endif // DENDRO_KT_SFC_TREE_H
