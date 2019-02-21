/**
 * @file:tsort.cpp
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2018-12-03
 * @brief: Based on work by Milinda Fernando and Hari Sundar.
 * - Algorithms: SC18 "Comparison Free Computations..." TreeSort, TreeConstruction, TreeBalancing
 * - Code: Dendro4 [sfcSort.h] [construct.cpp]
 *
 * My contribution is to extend the data structures to 4 dimensions (or higher).
 */

#include "tsort.h"
#include "octUtils.h"


namespace ot
{

//
// locTreeSort()
//
template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: locTreeSort(TreeNode<T,D> *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          RotI pRot)
{
  //// Recursive Depth-first, similar to Most Significant Digit First. ////

  if (end <= begin) { return; }

  constexpr char numChildren = TreeNode<T,D>::numChildren;
  constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

  // Reorder the buckets on sLev (current level).
  std::array<RankI, numChildren+1> tempSplitters;
  RankI ancStart, ancEnd;
  SFC_bucketing(points, begin, end, sLev, pRot, tempSplitters, ancStart, ancEnd);
  // The array `tempSplitters' has numChildren+1 slots, which includes the
  // beginning, middles, and end of the range of children.
  // Ancestor splitters are is ancStart and ancEnd, not tempSplitters.

  // Lookup tables to apply rotations.
  const ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
  const RotI * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    // Recurse.
    // Use the splitters to specify ranges for the next level of recursion.
    // While at first it might seem wasteful that we keep sorting the
    // ancestors and they don't move, in reality there are few ancestors,
    // so it (probably) doesn't matter that much.
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      // Columns of HILBERT_TABLE are indexed by the Morton rank.
      // According to Dendro4 TreeNode.tcc:199 they are.
      // (There are possibly inconsistencies in the old code...?
      // Don't worry, we can regenerate the table later.)
      ChildI child = rot_perm[child_sfc];
      RotI cRot = orientLookup[child];

      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] <= 1)
        continue;

      locTreeSort(points,
          tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
          sLev+1, eLev,
          cRot);
    }
  }
}// end function()


template <typename T, unsigned int D>
struct KeyFunIdentity
{
  const TreeNode<T,D> &operator()(const TreeNode<T,D> &tn) { return tn; }
};


//
// SFC_bucketing()
//
//   Based on Dendro4 sfcSort.h SFC_bucketing().
//
template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: SFC_bucketing(TreeNode<T,D> *points,
                          RankI begin, RankI end,
                          LevI lev,
                          RotI pRot,
                          std::array<RankI, 1+TreeNode<T,D>::numChildren> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd)
{

  //TODO transfer the comments to the fully configurable templated version of this method.

  SFC_bucketing_impl<KeyFunIdentity<T,D>, TreeNode<T,D>, TreeNode<T,D>>(
      points, begin, end, lev, pRot,
      KeyFunIdentity<T,D>(), false /*true*/,
      outSplitters,
      outAncStart, outAncEnd);

///   // ==
///   // Reorder the points by child number at level `lev', in the order
///   // of the SFC, and yield the positions of the splitters.
///   // ==
/// 
///   using TreeNode = TreeNode<T,D>;
///   constexpr char numChildren = TreeNode::numChildren;
///   constexpr char rotOffset = 2*numChildren;  // num columns in rotations[].
/// 
///   //
///   // Count the number of points in each bucket,
///   // indexed by (Morton) child number.
///   std::array<int, numChildren> counts;
///   counts.fill(0);
///   int countAncestors = 0;   // Special bucket to ensure ancestors precede descendants.
///   /// for (const TreeNode &tn : inp)
///   for (const TreeNode *tn = points + begin; tn < points + end; tn++)
///   {
///     if (tn->getLevel() < lev)
///       countAncestors++;
///     else
///       counts[tn->getMortonIndex(lev)]++;
///   }
/// 
///   //
///   // Compute offsets of buckets in permuted SFC order.
///   // Conceptually:
///   //   1. Permute counts;  2. offsets=scan(counts);  3. Un-permute offsets.
///   //
///   // The `outSplitters' array is indexed in SFC order (to match final output),
///   // while the `offsets' and `bucketEnds` arrays are indexed in Morton order
///   // (for easy lookup using TreeNode.getMortonIndex()).
///   //
///   // Note that outSplitters indexing is additionally offset by 1 so that
///   // the ancestor bucket is first, between [0th and 1st) markers.
///   //
///   std::array<RankI, numChildren+1> offsets, bucketEnds;  // Last idx represents ancestors.
///   offsets[numChildren] = begin;
///   bucketEnds[numChildren] = begin + countAncestors;
///   RankI accum = begin + countAncestors;                  // Ancestors belong in front.
/// 
///   std::array<TreeNode, numChildren+1> unsortedBuffer;
///   int bufferSize = 0;
/// 
///   // Logically permute: Scan the bucket-counts in the order of the SFC.
///   // Since we want to map [SFC_rank]-->Morton_rank,
///   // use the "left" columns of rotations[], aka `rot_perm'.
///   const ChildI *rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
///   ChildI child_sfc = 0;
///   for ( ; child_sfc < numChildren; child_sfc++)
///   {
///     ChildI child = rot_perm[child_sfc];
///     outSplitters[child_sfc+1] = accum;
///     offsets[child] = accum;           // Start of bucket. Moving marker.
///     accum += counts[child];
///     bucketEnds[child] = accum;        // End of bucket. Fixed marker.
///   }
///   outSplitters[child_sfc+1] = accum;  // Should be the end.
///   outSplitters[0] = begin;          // Bucket for 0th child (SFC order) is at index 1, this index 0 contains only ancestors.
/// 
///   // Prepare for the in-place movement phase by copying each offsets[] pointee
///   // to the rotation buffer. This frees up the slots to be valid destinations.
///   // Includes ancestors: Recall, we have used index `numChildren' for ancestors.
///   for (char bucketId = 0; bucketId <= numChildren; bucketId++)
///   {
///     if (offsets[bucketId] < bucketEnds[bucketId])
///       unsortedBuffer[bufferSize++] = points[offsets[bucketId]];  // Copy TreeNode.
///   }
/// 
///   //
///   // Finish the movement phase.
///   //
///   // Invariant: Any offsets[] pointee has been copied into `unsortedBuffer'.
///   while (bufferSize > 0)
///   {
///     TreeNode *bufferTop = &unsortedBuffer[bufferSize-1];
///     unsigned char destBucket
///       = (bufferTop->getLevel() < lev) ? numChildren : bufferTop->getMortonIndex(lev);
/// 
///     points[offsets[destBucket]++] = *bufferTop;  // Set down the TreeNode.
/// 
///     // Follow the cycle by picking up the next element in destBucket...
///     // unless we completed a cycle: in that case we made progress with unsortedBuffer.
///     if (offsets[destBucket] < bucketEnds[destBucket])
///       *bufferTop = points[offsets[destBucket]];    // Copy TreeNode.
///     else
///       bufferSize--;
///   }
/// 
///   // Note the first thing that happens: Due to offsets[numChildren] being the
///   // top of the unsortedBuffer stack, the previously sorted ancestors (grandparents)
///   // are swapped in and out of the buffer, in order, without actually changing positions.
///   // Therefore a sequence of direct descendants will be stable with
///   // ancestors preceeding descendants.
}

template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: SFC_locateBuckets(const TreeNode<T,D> *points,
                          RankI begin, RankI end,
                          LevI lev,
                          RotI pRot,
                          std::array<RankI, 2+TreeNode<T,D>::numChildren> &outSplitters)
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
  // The `outSplitters' array is indexed in SFC order (to match final output).
  //
  // Note that outSplitters indexing is additionally offset by 1 so that
  // the ancestor bucket is first, between [0th and 1st) markers.
  //
  RankI accum = begin + countAncestors;                  // Ancestors belong in front.

  // Logically permute: Scan the bucket-counts in the order of the SFC.
  // Since we want to map [SFC_rank]-->Morton_rank,
  // use the "left" columns of rotations[], aka `rot_perm'.
  const ChildI *rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
  ChildI child_sfc = 0;
  for ( ; child_sfc < numChildren; child_sfc++)
  {
    ChildI child = rot_perm[child_sfc];
    outSplitters[child_sfc+1] = accum;
    accum += counts[child];
  }
  outSplitters[child_sfc+1] = accum;  // Should be the end.
  outSplitters[0] = begin;          // Bucket for 0th child (SFC order) is at index 1, this index 0 contains only ancestors.
}



template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: distTreeSort(std::vector<TreeNode<T,D>> &points,
                          double loadFlexibility,
                          MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  // The heavy lifting to globally sort/partition.
  distTreePartition(points, loadFlexibility, comm);

  // Finish with a local TreeSort to ensure all points are in order.
  locTreeSort(&(*points.begin()), 0, points.size(), 0, m_uiMaxDepth, 0);

  /// // DEBUG: print out all the points.  // This debugging section will break.
  /// { std::vector<char> spaces(m_uiMaxDepth*rProc+1, ' ');
  /// spaces.back() = '\0';
  /// for (const TreeNode tn : points)
  ///   std::cout << spaces.data() << tn.getBase32Hex().data() << "\n";
  /// std::cout << spaces.data() << "------------------------------------\n";
  /// }
}


template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: distTreePartition(std::vector<TreeNode<T,D>> &points,
                          double loadFlexibility,
                          MPI_Comm comm)
{

  // -- Don't worry about K splitters for now, we'll add that later. --

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
  //    single pass of the tree. Also can take advantage of sparsity and filtering).
  // The second approach is used in Dendro4 par::sfcTreeSort(), so
  // I'm going to assume that linear aux storage is not too much to ask.

  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  if (nProc == 1)
  {
    locTreeSort(&(*points.begin()), 0, points.size(), 0, m_uiMaxDepth, 0);
    return;
  }

  using TreeNode = TreeNode<T,D>;
  constexpr char numChildren = TreeNode::numChildren;
  constexpr char rotOffset = 2*numChildren;  // num columns in rotations[].

  // The outcome of the BFT will be a list of splitters, i.e. refined buckets.
  // As long as there are `pending splitters', we will refine their corresponding buckets.
  // Initially all splitters are pending.
  std::vector<RankI> splitters(nProc, 0);
  BarrierQueue<RankI> pendingSplitterIdx(nProc);
  for (RankI sIdx = 0; sIdx < nProc; sIdx++)
    pendingSplitterIdx.q[sIdx] = sIdx;
  pendingSplitterIdx.reset_barrier();

  //
  // Phase 1: move down the levels until we have enough buckets
  //   to test our load-balancing criterion.
  BarrierQueue<BucketInfo<RankI>> bftQueue;
  const int initNumBuckets = nProc;
  const BucketInfo<RankI> rootBucket = {0, 0, 0, (RankI) points.size()};
  bftQueue.q.push_back(rootBucket);
        // No-runaway, in case we run out of points.
        // It is `<' because refining m_uiMaxDepth would make (m_uiMaxDepth+1).
  while (bftQueue.q.size() < initNumBuckets && bftQueue.q[0].lev < m_uiMaxDepth)
  {
    treeBFTNextLevel(&(*points.begin()), bftQueue.q);
  }
  // Remark: Due to the no-runaway clause, we are not guaranteed
  // that bftQueue actually holds `initNumBuckets' buckets.

  //
  // Phase 2: Count bucket sizes, communicate bucket sizes,
  //   test load balance, select buckets and refine, repeat.

  RankI sizeG, sizeL = points.size();
  /// sizeG = sizeL;   // Proxy for summing, to test with one proc.
  par::Mpi_Allreduce<RankI>(&sizeL, &sizeG, 1, MPI_SUM, comm);


  /// //TEST  print all ideal splitters pictorally
  /// RankI oldLoc = 0;
  /// for (int rr = 0; rr < nProc; rr++)
  /// {
  ///   RankI idealSplitter = (rr+1) * sizeG / nProc;
  ///   for ( ; oldLoc < idealSplitter; oldLoc++) { std::cout << ' '; }
  ///   std::cout << 'I';     oldLoc++;
  /// }
  /// std::cout << '\n';

  std::vector<unsigned int> DBG_splitterIteration(nProc, 0);  //DEBUG

  // To avoid communicating buckets that did not get updated, we'll cycle:
  //   Select buckets from old queue, -> enqueue in a "new" queue, -> ...
  //   Use the class BarrierQueue to separate old from new.
  // I also group buckets into "blocks" of contiguous siblings before scanning.
  std::vector<RankI> bktCountsL, bktCountsG;  // As single-use containers per level.
  BarrierQueue<RankI> blkBeginG;              // As queue with barrier, to bridge levels.
  blkBeginG.enqueue(0);
  RankI blkNumBkt = bftQueue.size();  // The first block is special. It contains all buckets.
  while (pendingSplitterIdx.size() > 0)
  {
    /// // TEST Print all new buckets pictorally.
    /// { RankI DBG_oldLoc = 0;
    ///   for (BucketInfo<RankI> b : bftQueue.q) { while(DBG_oldLoc < b.end) { std::cout << ' '; DBG_oldLoc++; } if (DBG_oldLoc == b.end) { std::cout << '|'; DBG_oldLoc++; } }
    ///   std::cout << '\n';
    /// }

    bftQueue.reset_barrier();
    blkBeginG.reset_barrier();
    pendingSplitterIdx.reset_barrier();

    // Count buckets locally and globally.
    bktCountsL.clear();
    for (BucketInfo<RankI> b : bftQueue.leading())
      bktCountsL.push_back(b.end - b.begin);
    bktCountsG.resize(bktCountsL.size());
    /// bktCountsG = bktCountsL;              // Proxy for summing, to test with one proc.
    par::Mpi_Allreduce<RankI>(&(*bktCountsL.begin()), &(*bktCountsG.begin()), (int) bktCountsL.size(), MPI_SUM, comm);

    // Compute ranks, test load balance, and collect buckets for refinement.
    // Process each bucket in sequence, one block of buckets at a time.
    RankI bktBeginG;
    while (blkBeginG.dequeue(bktBeginG))
    {
      for (RankI ii = 0; ii < blkNumBkt; ii++)
      {
        const RankI bktCountG = bktCountsG[0];
        const RankI bktEndG = bktBeginG + bktCountG;
        bktCountsG.erase(bktCountsG.begin());

        BucketInfo<RankI> refBkt;
        bftQueue.dequeue(refBkt);
        bool selectBucket = false;
        const bool bktCanBeRefined = (refBkt.lev < m_uiMaxDepth);
        
        // Test the splitter indices that may fall into the current bucket.
        RankI idealSplitterG;
        while (pendingSplitterIdx.get_barrier()
            && (idealSplitterG = (pendingSplitterIdx.front()+1) * sizeG / nProc) <= bktEndG)
        {
          RankI r;
          pendingSplitterIdx.dequeue(r);
          RankI absTolerance = ((r+1) * sizeG / nProc - r * sizeG / nProc) * loadFlexibility;
          if (bktCanBeRefined && (bktEndG - idealSplitterG) > absTolerance)
          {
            // Too far. Mark bucket for refinment. Send splitter back to queue.
            selectBucket = true;
            pendingSplitterIdx.enqueue(r);
            splitters[r] = refBkt.end;      // Will be overwritten. Uncomment if want to see progress.
          }
          else
          {
            // Good enough. Accept the bucket by recording the local splitter.
            splitters[r] = refBkt.end;
          }
          DBG_splitterIteration[r]++;   //DEBUG
        }

        if (selectBucket)
        {
          bftQueue.enqueue(refBkt);      // For refinment.
          blkBeginG.enqueue(bktBeginG);  // Bucket-begin becomes block-begin on next iteration.
        }

        bktBeginG = bktEndG;
      }
    }

    /// // TEST Print all splitters.
    /// { RankI DBG_oldLoc = 0;
    ///   for (RankI s : splitters) { while(DBG_oldLoc < s) { std::cout << '_'; DBG_oldLoc++; }  if (DBG_oldLoc == s) { std::cout << 'x'; DBG_oldLoc++; } }
    ///   std::cout << '\n';
    /// }
    
    // Refine the buckets that we have set aside.
    treeBFTNextLevel(&(*points.begin()), bftQueue.q);

    blkNumBkt = numChildren;  // After the first level, blocks result from refining a single bucket.
  }

  /// // DEBUG: print out all the points.
  /// { std::vector<char> spaces(m_uiMaxDepth*rProc+1, ' ');
  /// spaces.back() = '\0';
  /// for (const TreeNode tn : points)
  ///   std::cout << spaces.data() << tn.getBase32Hex().data() << "\n";
  /// std::cout << spaces.data() << "------------------------------------\n";
  /// }

  //
  // All to all exchange of the points arrays.

  std::vector<unsigned int> sendCnt, sendDspl;
  std::vector<unsigned int> recvCnt(splitters.size()), recvDspl;
  sendCnt.reserve(splitters.size());
  sendDspl.reserve(splitters.size());
  recvDspl.reserve(splitters.size());
  RankI sPrev = 0;
  for (RankI s : splitters)     // Sequential counting and displacement.
  {
    sendDspl.push_back(sPrev);
    sendCnt.push_back(s - sPrev);
    sPrev = s;
  }
  par::Mpi_Alltoall<RankI>(&(*sendCnt.begin()), &(*recvCnt.begin()), 1, comm);
  sPrev = 0;
  for (RankI c : recvCnt)       // Sequential scan.
  {
    recvDspl.push_back(sPrev);
    sPrev += c;
  }
  RankI sizeNew = sPrev;

  std::vector<TreeNode> origPoints = points;   // Sendbuffer is a copy.

  if (sizeNew > sizeL)
    points.resize(sizeNew);

  par::Mpi_Alltoallv<TreeNode>(
      &(*origPoints.begin()), (int*) &(*sendCnt.begin()), (int*) &(*sendDspl.begin()),
      &(*points.begin()), (int*) &(*recvCnt.begin()), (int*) &(*recvDspl.begin()),
      comm);

  points.resize(sizeNew);

  /// // DEBUG: print out all the points.
  /// { std::vector<char> spaces(m_uiMaxDepth*rProc+1, ' ');
  /// spaces.back() = '\0';
  /// for (const TreeNode tn : points)
  ///   std::cout << spaces.data() << tn.getBase32Hex().data() << "\n";
  /// std::cout << spaces.data() << "------------------------------------\n";
  /// }

  //TODO figure out the 'staged' part with k-parameter.

  // After this process, distTreeSort or distTreeConstruction
  // picks up with a local sorting or construction operation.
  // TODO Need to have the global buckets for that to work.
}


template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: treeBFTNextLevel(TreeNode<T,D> *points,
      std::vector<BucketInfo<RankI>> &bftQueue)
{
  if (bftQueue.size() == 0)
    return;

  const LevI startLev = bftQueue[0].lev;

  using TreeNode = TreeNode<T,D>;
  constexpr char numChildren = TreeNode::numChildren;
  constexpr char rotOffset = 2*numChildren;  // num columns in rotations[].

  while (bftQueue[0].lev == startLev)
  {
    BucketInfo<RankI> front = bftQueue[0];
    bftQueue.erase(bftQueue.begin());

    // Refine the current orthant/bucket by sorting the sub-buckets.
    // Get splitters for sub-buckets.
    std::array<RankI, numChildren+1> childSplitters;
    RankI ancStart, ancEnd;
    if (front.begin < front.end)
      SFC_bucketing(points, front.begin, front.end, front.lev, front.rot_id, childSplitters, ancStart, ancEnd);
    else
      childSplitters.fill(front.begin);  // Don't need to sort an empty selection, it's just empty.

    // Put 'ancestor' points one of the closest sibling bucket.
    const bool ancestorsFirst = true;
    if (ancestorsFirst)
      childSplitters[0] = ancStart;
    else
      childSplitters[numChildren] = ancEnd;

    // Enqueue our children in the next level.
    const ChildI * const rot_perm = &rotations[front.rot_id*rotOffset + 0*numChildren];
    const RotI * const orientLookup = &HILBERT_TABLE[front.rot_id*numChildren];
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      ChildI child = rot_perm[child_sfc];
      RotI cRot = orientLookup[child];
      BucketInfo<RankI> childBucket =
          {cRot, front.lev+1, childSplitters[child_sfc+0], childSplitters[child_sfc+1]};

      bftQueue.push_back(childBucket);
    }
  }
}


//
// locTreeConstruction()
//
template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: locTreeConstruction(TreeNode<T,D> *points,
                                  std::vector<TreeNode<T,D>> &tree,
                                  RankI maxPtsPerRegion,
                                  RankI begin, RankI end,
                                  LevI sLev,
                                  LevI eLev,
                                  RotI pRot,
                                  TreeNode<T,D> pNode)
{
  // Most of this code is copied from locTreeSort().

  if (end <= begin) { return; }

  constexpr char numChildren = TreeNode<T,D>::numChildren;
  constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

  using TreeNode = TreeNode<T,D>;

  // Reorder the buckets on sLev (current level).
  std::array<RankI, numChildren+1> tempSplitters;
  RankI ancStart, ancEnd;
  SFC_bucketing(points, begin, end, sLev, pRot, tempSplitters, ancStart, ancEnd);
  // The array `tempSplitters' has numChildren+2 slots, which includes the
  // beginning, middles, and end of the range of children, and ancestors are in front.

  // Lookup tables to apply rotations.
  const ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
  const RotI * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

  TreeNode cNode = pNode.getFirstChildMorton();

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    // We satisfy the completeness property because we iterate over
    // all possible children here. For each child, append either
    // a leaf orthant or a non-empty complete subtree.
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      // Columns of HILBERT_TABLE are indexed by the Morton rank.
      // According to Dendro4 TreeNode.tcc:199 they are.
      // (There are possibly inconsistencies in the old code...?
      // Don't worry, we can regenerate the table later.)
      ChildI child = rot_perm[child_sfc];
      RotI cRot = orientLookup[child];
      cNode.setMortonIndex(child);

      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] > maxPtsPerRegion)
      {
        // Recursively build a complete sub-tree out of this bucket's points.
        // Use the splitters to specify ranges for the next level of recursion.
        locTreeConstruction(
            points, tree, maxPtsPerRegion,
            tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
            sLev+1, eLev,
            cRot,
            cNode);
      }
      else
      {
        // Append a leaf orthant.
        tree.push_back(cNode);
      }
    }
  }
  else   // We have reached eLev. Violate `maxPtsPerRegion' to satisfy completeness.
  {
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      ChildI child = rot_perm[child_sfc];
      cNode.setMortonIndex(child);
      tree.push_back(cNode);
    }
  }

}  // end function


template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: distTreeConstruction(std::vector<TreeNode<T,D>> &points,
                                   std::vector<TreeNode<T,D>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  tree.clear();

  // The heavy lifting to globally sort/partition.
  distTreePartition(points, loadFlexibility, comm);

  // Instead of locally sorting, locally complete the tree.
  // Since we don't have info about the global buckets, construct from the top.
  const LevI leafLevel = m_uiMaxDepth;
  locTreeConstruction(&(*points.begin()), tree, maxPtsPerRegion,
                      0, (RankI) points.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      0, TreeNode<T,D>());
  // When (sLev,eLev)==(0,m_uiMaxDepth), nodes with level m_uiMaxDepth+1 are created.
  // This must be leading to incorrect ancestry tests because duplicates do
  // not always get removed properly in that case.

  // We have now introduced duplicate sections of subtrees at the
  // edges of the partition.

  // For now:
  // Rather than do a complicated elimination of duplicates,
  // perform another global sort, removing duplicates locally, and then
  // eliminate at most one duplicate from the end of each processor's partition.

  distTreeSort(tree, loadFlexibility, comm);
  locRemoveDuplicates(tree);

  // Some processors could end up being empty, so exclude them from communicator.
  MPI_Comm nonemptys;
  MPI_Comm_split(comm, (tree.size() > 0 ? 1 : MPI_UNDEFINED), rProc, &nonemptys);

  if (tree.size() > 0)
  {
    int nNE, rNE;
    MPI_Comm_rank(nonemptys, &rNE);
    MPI_Comm_size(nonemptys, &nNE);

    // At this point, the end of our portion of the tree is possibly a duplicate of,
    // or an ancestor of, the beginning of the next processors portion of the tree.

    // Exchange to test if our end is a duplicate.
    TreeNode<T,D> nextBegin;
    MPI_Request request;
    MPI_Status status;
    if (rNE > 0)
      par::Mpi_Isend<TreeNode<T,D>>(&(*tree.begin()), 1, rNE-1, 0, nonemptys, &request);
    if (rNE < nNE-1)
      par::Mpi_Recv<TreeNode<T,D>>(&nextBegin, 1, rNE+1, 0, nonemptys, &status);

    // If so, delete our end.
    if (rNE > 0)
      MPI_Wait(&request, &status);
    if (rNE < nNE-1 && (tree.back() == nextBegin || tree.back().isAncestor(nextBegin)))
      tree.pop_back();
  }
}


template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: locRemoveDuplicates(std::vector<TreeNode<T,D>> &tnodes)
{
  const TreeNode<T,D> *tEnd = &(*tnodes.end());
  TreeNode<T,D> *tnCur = &(*tnodes.begin());
  size_t numUnique = 0;

  while (tnCur < tEnd)
  {
    // Find next leaf.
    TreeNode<T,D> *tnNext;
    while ((tnNext = tnCur + 1) < tEnd &&
        (*tnCur == *tnNext || tnCur->isAncestor(*tnNext)))
      tnCur++;

    // Move the leaf.
    if (&tnodes[numUnique] < tnCur)
      tnodes[numUnique] = *tnCur;
    numUnique++;

    tnCur++;
  }

  tnodes.resize(numUnique);
}


template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: locRemoveDuplicatesStrict(std::vector<TreeNode<T,D>> &tnodes)
{
  const TreeNode<T,D> *tEnd = &(*tnodes.end());
  TreeNode<T,D> *tnCur = &(*tnodes.begin());
  size_t numUnique = 0;

  while (tnCur < tEnd)
  {
    // Find next leaf.
    TreeNode<T,D> *tnNext;
    while ((tnNext = tnCur + 1) < tEnd &&
        (*tnCur == *tnNext))  // Strict equality only; ancestors retained.
      tnCur++;

    // Move the leaf.
    if (&tnodes[numUnique] < tnCur)
      tnodes[numUnique] = *tnCur;
    numUnique++;

    tnCur++;
  }

  tnodes.resize(numUnique);
}


//
// propagateNeighbours()
//
template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: propagateNeighbours(std::vector<TreeNode<T,D>> &srcNodes)
{
  std::vector<std::vector<TreeNode<T,D>>> treeLevels = stratifyTree(srcNodes);
  srcNodes.clear();

  ///std::cout << "Starting at        level " << m_uiMaxDepth << ", level size \t " << treeLevels[m_uiMaxDepth].size() << "\n";  //DEBUG

  // Bottom-up traversal using stratified levels.
  for (unsigned int l = m_uiMaxDepth; l > 0; l--)
  {
    const unsigned int lp = l-1;  // Parent level.

    const size_t oldLevelSize = treeLevels[lp].size();

    for (const TreeNode<T,D> &tn : treeLevels[l])
    {
      // Append neighbors of parent.
      TreeNode<T,D> tnParent = tn.getParent();
      treeLevels[lp].push_back(tnParent);
      tnParent.appendAllNeighbours(treeLevels[lp]);

      /* Might need to intermittently remove duplicates... */
    }

    // TODO Consider more efficient algorithms for removing duplicates from lp level.
    locTreeSort(&(*treeLevels[lp].begin()), 0, treeLevels[lp].size(), 1, lp, 0);
    locRemoveDuplicates(treeLevels[lp]);

    ///const size_t newLevelSize = treeLevels[lp].size();
    ///std::cout << "Finished adding to level " << lp << ", level size \t " << oldLevelSize << "\t -> " << newLevelSize << "\n";  // DEBUG
  }

  // Reserve space before concatenating all the levels.
  size_t newSize = 0;
  for (const std::vector<TreeNode<T,D>> &trLev : treeLevels)
    newSize += trLev.size();
  srcNodes.reserve(newSize);

  // Concatenate all the levels.
  for (const std::vector<TreeNode<T,D>> &trLev : treeLevels)
    srcNodes.insert(srcNodes.end(), trLev.begin(), trLev.end());
}


//
// locTreeBalancing()
//
template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: locTreeBalancing(std::vector<TreeNode<T,D>> &points,
                                 std::vector<TreeNode<T,D>> &tree,
                                 RankI maxPtsPerRegion)
{
  const LevI leafLevel = m_uiMaxDepth;

  locTreeConstruction(&(*points.begin()), tree, maxPtsPerRegion,
                      0, (RankI) points.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      0, TreeNode<T,D>());

  /// //DEBUG
  /// std::cout << "Original tree:\n";
  /// for (const TreeNode<T,D> &tn : tree)
  ///   std::cout << "\t" << tn.getBase32Hex().data() << "\n";
  /// std::cout << "\n";

  propagateNeighbours(tree);

  std::vector<TreeNode<T,D>> newTree;
  locTreeConstruction(&(*tree.begin()), newTree, 1,
                      0, (RankI) tree.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      0, TreeNode<T,D>());

  tree = newTree;
}


//
// distTreeBalancing()
//
template <typename T, unsigned int D>
void
SFC_Tree<T,D>:: distTreeBalancing(std::vector<TreeNode<T,D>> &points,
                                   std::vector<TreeNode<T,D>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  distTreeConstruction(points, tree, maxPtsPerRegion, loadFlexibility, comm);
  propagateNeighbours(tree);
  std::vector<TreeNode<T,D>> newTree;
  distTreeConstruction(tree, newTree, 1, loadFlexibility, comm);

  tree = newTree;
}




} // namspace ot
