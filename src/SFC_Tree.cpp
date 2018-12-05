/*
 * SFC_Tree.cpp
 *
 * Masado Ishii  --  UofU SoC, 2018-12-03
 *
 * Based on work by Milinda Fernando and Hari Sundar.
 *   - Algorithms: SC18 "Comparison Free Computations..." TreeSort, TreeConstruction, TreeBalancing
 *   - Code: Dendro4 [sfcSort.h] [construct.cpp]
 *
 * My contribution is to extend the data structures to 4 dimensions (or higher).
 */

#include "SFC_Tree.h"
#include "hcurvedata.h"

namespace ot
{


//
// locTreeSort()
//
template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: locTreeSort(TreeNode<T,D> *points,
                          unsigned int begin, unsigned int end,
                          unsigned int sLev,
                          unsigned int eLev,
                          int pRot,
                          std::vector<BucketInfo<unsigned int>> &outBuckets,
                          bool makeBuckets)
{
  //// Recursive Depth-first, similar to Most Significant Digit First. ////

  if (end <= begin) { return; }

  constexpr char numChildren = TreeNode<T,D>::numChildren;
  constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

  // Reorder the buckets on sLev (current level).
  std::array<unsigned int, TreeNode<T,D>::numChildren+1> tempSplitters;
  SFC_bucketing(points, begin, end, sLev, pRot, tempSplitters);
  // The array `tempSplitters' has numChildren+1 slots, which includes the
  // beginning, middles, and end of the range of children.

  // Lookup tables to apply rotations.
  const char * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
  const int * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    const unsigned int continueThresh = makeBuckets ? 0 : 1;

    // Recurse.
    // Use the splitters to specify ranges for the next level of recursion.
    // Use the results of the recursion to build the list of leaf buckets.
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      // Columns of HILBERT_TABLE are indexed by the Morton rank.
      // According to Dendro4 TreeNode.tcc:199 they are.
      // (There are possibly inconsistencies in the old code...?
      // Don't worry, we can regenerate the table later.)
      char child = rot_perm[child_sfc] - '0';     // Decode from human-readable ASCII.
      int cRot = orientLookup[child];

      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc] <= continueThresh)
        continue;
      // We don't skip a singleton, since a singleton contributes a bucket.
      // We need recursion to calculate the rotation at the leaf level.

      locTreeSort(points,
          tempSplitters[child_sfc], tempSplitters[child_sfc+1],
          sLev+1, eLev,
          cRot,
          outBuckets,
          makeBuckets);
    }
  }
  else if (makeBuckets)
  {
    // This is the leaf level. Use the splitters to build the list of leaf buckets.
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      char child = rot_perm[child_sfc] - '0';     // Decode from human-readable ASCII.
      int cRot = orientLookup[child];

      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc] == 0)
        continue;

      outBuckets.push_back(
          {cRot, sLev+1,
          tempSplitters[child_sfc],
          tempSplitters[child_sfc+1]});
      // These are the parameters that could be used to further refine the bucket.
    }
  }

}// end function()


//
// SFC_bucketing()
//
//   Based on Dendro4 sfcSort.h SFC_bucketing().
//
template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: SFC_bucketing(TreeNode<T,D> *points,
                          unsigned int begin, unsigned int end,
                          unsigned int lev,
                          int pRot,
                          std::array<unsigned int, TreeNode<T,D>::numChildren+1> &outSplitters)
{
  // ==
  // Reorder the points by child number at level `lev', in the order
  // of the SFC, and yield the positions of the splitters.
  // ==

  using TreeNode = TreeNode<T,D>;
  constexpr char numChildren = TreeNode::numChildren;
  constexpr char rotOffset = 2*numChildren;  // num columns in rotations[].

  //
  // Count the number of points in each bucket,
  // indexed by (Morton) child number.
  std::array<int, numChildren> counts;
  counts.fill(0);
  int countAncestors = 0;   // Special bucket to ensure ancestors precede descendants.
  /// for (const TreeNode &tn : inp)
  for (const TreeNode *tn = points + begin; tn < points + end; tn++)
  {
    if (tn->getLevel() < lev)
      countAncestors++;
    else
      counts[tn->getMortonIndex(lev)]++;
  }

  //
  // Compute offsets of buckets in permuted SFC order.
  // Conceptually:
  //   1. Permute counts;  2. offsets=scan(counts);  3. Un-permute offsets.
  //
  // The `outSplitters' array is indexed in SFC order (to match final output),
  // while the `offsets' and `bucketEnds` arrays are indexed in Morton order
  // (for easy lookup using TreeNode.getMortonIndex()).
  //
  std::array<unsigned int, numChildren+1> offsets, bucketEnds;  // Last idx represents ancestors.
  offsets[numChildren] = begin;
  bucketEnds[numChildren] = begin + countAncestors;
  unsigned int accum = begin + countAncestors;                  // Ancestors belong in front.

  std::array<TreeNode, numChildren+1> unsortedBuffer;
  int bufferSize = 0;

  // Logically permute: Scan the bucket-counts in the order of the SFC.
  // Since we want to map [SFC_rank]-->Morton_rank,
  // use the "left" columns of rotations[], aka `rot_perm'.
  const char *rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
  char child_sfc = 0;
  for ( ; child_sfc < numChildren; child_sfc++)
  {
    char child = rot_perm[child_sfc] - '0';  // Decode from human-readable ASCII.
    outSplitters[child_sfc] = accum;
    offsets[child] = accum;           // Start of bucket. Moving marker.
    accum += counts[child];
    bucketEnds[child] = accum;        // End of bucket. Fixed marker.
  }
  outSplitters[child_sfc] = accum;  // Should be the end.

  // Prepare for the in-place movement phase by copying each offsets[] pointee
  // to the rotation buffer. This frees up the slots to be valid destinations.
  // Includes ancestors.
  for (char bucketId = 0; bucketId <= numChildren; bucketId++)
  {
    if (offsets[bucketId] < bucketEnds[bucketId])
      unsortedBuffer[bufferSize++] = points[offsets[bucketId]];  // Copy TreeNode.
  }

  //
  // Finish the movement phase.
  //
  // Invariant: Any offsets[] pointee has been copied into `unsortedBuffer'.
  while (bufferSize > 0)
  {
    TreeNode *bufferTop = &unsortedBuffer[bufferSize-1];
    unsigned char destBucket
      = (bufferTop->getLevel() < lev) ? numChildren : bufferTop->getMortonIndex(lev);

    points[offsets[destBucket]++] = *bufferTop;  // Set down the TreeNode.

    // Follow the cycle by picking up the next element in destBucket...
    // unless we completed a cycle: in that case we made progress with unsortedBuffer.
    if (offsets[destBucket] < bucketEnds[destBucket])
      *bufferTop = points[offsets[destBucket]];    // Copy TreeNode.
    else
      bufferSize--;
  }
}



template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: distTreeSort(std::vector<TreeNode<T,D>> &points,            // The input needs to be rearranged to compute buckets...
                          double loadFlexibility,
                          MPI_Comm comm)
{

  // -- Don't worry about K buckets for now, we'll add that later. --

  // The goal of this function, as explained in Fernando and Sundar's paper,
  // is to refine the list of points into finer sorted buckets until
  // the balancing criterion has been met. Therefore the hyperoctree is
  // traversed in breadth-first order.
  //
  // I've considered two ways to do a breadth first traversal:
  // 1. Repeated depth-first traversal with a stack to hold rotations
  //    (requires storage linear in height of the tree, but more computation)
  // 2. Single breadth-first traversal with a queue to hold rotations
  //    (requires storage linear in the breadth of the tree, but done in a
  //    single pass. Also can take advantage of sparsity and filtering).
  // The second approach is used in Dendro4 par::sfcTreeSort(), so
  // I'm going to assume that linear aux storage is not too much to ask.

  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const int initNumBuckets = nProc;
  const int initNumLevels = ceil(log2(initNumBuckets)); /*TODO Dendro has fastLog2 */
  //TODO this might be an off-by-one error. Might supposed to use one fewer levels.
  //     Or we could change locTreeSort so that [sLev, eLev] is not inclusive of eLev.

  // Initial local sort to get sufficient number of partitions (splitters).
  std::vector<BucketInfo<unsigned int>> initBuckets;
  locTreeSort(&(*points.begin()), 0, points.size(), 0, initNumLevels, 0, initBuckets);

  // TODO Now that we are here at the communication stage, it becomes clear
  // why we need locTreeSort to return buckets/splitters/counts for all leaf octants,
  // both empty and nonempty: We need the buffers to align with those of all
  // other processes, which may have different sets of empty/nonempty buckets.

  bool gotNewBuckets = true;
  while (gotNewBuckets)
  {


    /*if (...)*/
      gotNewBuckets = true;
    /*else*/
      gotNewBuckets = false;
  }

  //New buckets

  // While(There are new splitters)

    // Extend counts_local to complete leaf level, then initialize counts_local
    // by running through the nonempty buckets.

    // Allreduce(counts_local, counts_global)  // TODO I think we should only communicate the new buckets

    // Scan counts_global to find global bucket boundaries.

    // For each processor, locate the position of its ideal partition boundary
    // in relation to the global bucket boundaries. If it is too far off,
    // note the bucket where it lands. Insert this bucket into the queue for
    // further refinement (unless already inserted this round).

    // Refine each bucket that needs it.

  // All to all on list of TreeNodes, according to the final splitters.
  //TODO figure out the 'staged' part with k-parameter.

  // Finish with a local TreeSort to ensure all points are in order.
}




} // namspace ot
