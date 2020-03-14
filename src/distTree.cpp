
#include "distTree.h"
#include "meshLoop.h"
#include "tsort.h"
#include "nsort.h"

namespace ot
{

  // A class to manage bucket sort.
  // Allows sending items to more than one bucket.
  // Useful if the cost of traversing the original, or computing bucket ids,
  // is expensive or inconvenient.
  template <typename IdxT = size_t, typename BdxT = int>
  class BucketMultiplexer
  {
      struct SrcDestPair
      {
        IdxT idx;
        BdxT bucket;
      };

    private:
      std::vector<IdxT> m_bucketCounts;
      std::vector<SrcDestPair> m_srcDestList;

    public:
      BucketMultiplexer() = delete;
      BucketMultiplexer(BdxT numBuckets, IdxT listReserveSz = 0)
        : m_bucketCounts(numBuckets, 0)
      {
        m_srcDestList.reserve(listReserveSz);
      }

      /// // Declarations
      /// inline void addToBucket(IdxT index, BdxT bucket);
      /// inline IdxT getTotalItems() const;
      /// std::vector<IdxT> getBucketCounts() const;
      /// std::vector<IdxT> getBucketOffsets() const;
      /// template <typename X, typename Y>
      /// inline void transferCopies(Y * dest, const X * src) const;


      // Definitions

      // addToBucket()
      inline void addToBucket(IdxT index, BdxT bucket)
      {
        SrcDestPair srcDestPair;
        srcDestPair.idx = index;
        srcDestPair.bucket = bucket;

        m_srcDestList.emplace_back(srcDestPair);
        m_bucketCounts[bucket]++;
      }

      // getTotalItems()
      inline IdxT getTotalItems() const
      {
        return (IdxT) m_srcDestList.size();
      }

      // getBucketCounts()
      std::vector<IdxT> getBucketCounts() const
      {
        return m_bucketCounts;
      }

      // getBucketOffsets()
      std::vector<IdxT> getBucketOffsets() const
      {
        std::vector<IdxT> bucketOffsets(1, 0);
        for (IdxT c : m_bucketCounts)
          bucketOffsets.push_back(bucketOffsets.back() + c);
        bucketOffsets.pop_back();

        return bucketOffsets;
      }

      // transferCopies()
      template <typename X, typename Y>
      inline void transferCopies(Y * dest, const X * src) const
      {
        std::vector<IdxT> bucketOffsets = getBucketOffsets();

        for (const SrcDestPair &sd : m_srcDestList)
          dest[bucketOffsets[sd.bucket]++] = src[sd.idx];
      }
  };

  //
  // generateGridHierarchy()
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::generateGridHierarchy(bool isFixedNumStrata, unsigned int lev, MPI_Comm comm)
  {
    /**
     * @author Masado Ishii
     * @date 2020-02-21
     */

    // /////////////////////////////////////////////////////////////////////
    //
    // Assume the fine grid is 2:1 balanced.
    //
    // Coarsen each family of siblings opportunistically from the fine grid
    // into the coarse grid, but do not coarsen if it would violate 2:1-balancing.
    //
    // Potential coarse elements are represented by `candidates'.
    // Illegal coarse elements are prevented using `disqualifiers'.
    //
    // Define
    //     candidates := {parent of x : x is in the fine grid}
    //     disqualifiers := closure of {parents of candidates} U {parents of neighbors of disqualifiers}
    //     coarse grid := {for all candidates c: if c is not disqualified, then c; else select(fine grid, descendants of c)}
    //
    // Note that we do not want to disqualify neighbors of parents of disqualified elements.
    // We really do want to propagate only parents of neighbors of disqualified elements.
    //
    // Define `relevant disqualifiers' as disqualifiers which are also candidates.
    //
    //
    // The strategy for generating the coarse grid is as follows:
    //
    //     1. Generate candidates as parents of all local fine grid elements.
    //
    //     2. Initialize disqualifiers as parents of candidates.
    //
    //     3. Propagate disqualifiers level by level:
    //        3.1 Stratify disqualifiers by level.
    //        3.2 Starting from the deepest level, add all parents of neighbors.
    //            3.2.1 Simultaneously traverse the list of candidates,
    //                    so as to eliminate disqualifiers that are not relevant.
    //            3.2.2 Need to check the before and after address of the
    //                    disqualifier against the local partition splitters.
    //            3.2.3 Any disqualifier that belongs to another partition
    //                    can be added to a sendcount.
    //
    //     4. Collect and send/recv candidates and disqualifiers that belong
    //          to another partition.
    //
    //     5. Simultaneously traverse candidates, disqualifiers, and the fine grid
    //        5.1 If there exists a candidate, and it's not disqualified,
    //              add it as a cell in the coarse grid.
    //        5.2 Otherwise, add the subtree from the fine grid.
    //
    //     6. A candidate survives iff it is part of the coarse grid,
    //        however it's possible that some fine grid elements
    //        did not see their surviving parent candidate.
    //        Need to check ends / globally to remove extraneous fine grid.
    //        (Keep candidates).
    //         
    //
    // /////////////////////////////////////////////////////////////////////
    //
    // So far I have proved that
    //     Claim1:
    //     relevant disqualifiers == closure of {parents of candidates}
    //                               U {parents of neighbors of relevant disqualifiers}
    //
    // which may become helpful in determining a bound on the number of times
    // needed to propagate the disqualifiers nonredundantly, as a function of
    // the dimension of the domain.
    //
    // [Proof of Claim1]
    //
    //     Lemma1: A disqualifier at level L intersects with fine grid elements
    //             of level (L+1) or deeper.
    //     [Pf of Lemma1]
    //         By induction on the level L.
    //
    //         Let q be a disqualifier at level L.
    //         Case1: q is a parent of a candidate, which is a parent of a
    //                fine grid element. Therefore q intersects with a fine
    //                grid element of level (L+2), and intersects with fine
    //                grid elements no coarser than level (L+1).
    //
    //         Case2: If L is a coarser level than the deepest disqualifier,
    //                q can be a parent of a neighbor of a disqualifier n
    //                of level (L+1). By the inductive hypothesis, n intersects
    //                with fine grid elements of level (L+2) or finer.
    //                Due to 2:1 balancing, q intersects with fine grid elements
    //                of level (L+1) or finer.
    //                                                             [Lemma1 QED]
    //     
    //     Proof of Claim1:
    //         Let q be a disqualified candidate of level L.
    //         Case1: q is not a parent of a neighbor of a disqualifier.
    //                Since q is a disqualifier, the only other possibility
    //                is that q is a parent of a candidate.
    //
    //         Case2: q is a parent of a neighbor of a disqualifier,
    //                and q is a parent of a candidate.
    //
    //         Case3: q is a parent of a neighbor of a disqualifier
    //                n of level (L+1), and q is not a parent of a candidate.
    //                Since q intersects fine grid elements of level (L+1) or
    //                finer but is not the parent of a candidate, it must be
    //                that q intersects only fine grid elements of level (L+1).
    //                Now, if n were _not_ a candidate, then n would intersect
    //                only fine grid elements of level (L+3) or finer, but
    //                this would violate 2:1 balancing of the fine grid.
    //                Therefore n is a candidate, which implies that
    //                q is a parent of a disqualified candidate.
    //
    //                                                             [Claim1 QED]
    //
    // /////////////////////////////////////////////////////////////////////



      /// // Multilevel grids, finest first. Must initialize with at least one level.
      /// std::vector<std::vector<TreeNode<T, dim>>> m_gridStrata;
      /// std::vector<TreeNode<T, dim>> m_tpFrontStrata;
      /// std::vector<TreeNode<T, dim>> m_tpBackStrata;

    int nProc, rProc;
    MPI_Comm_size(comm, &nProc);
    MPI_Comm_rank(comm, &rProc);

    MPI_Comm activeComm = comm;
    int nProcActive = nProc;
    int rProcActive = rProc;

    // Determine the number of grids in the grid hierarchy.
    if (isFixedNumStrata)
      m_numStrata = lev;
    else
    {
      LevI observedMaxDepth_loc = 0, observedMaxDepth_glob = 0;
      for (const TreeNode<T, dim> & tn : m_gridStrata[0])
        if (observedMaxDepth_loc < tn.getLevel())
          observedMaxDepth_loc = tn.getLevel();

      par::Mpi_Allreduce<LevI>(&observedMaxDepth_loc, &observedMaxDepth_glob, 1, MPI_MAX, comm);

      m_numStrata = 1 + (observedMaxDepth_glob - lev);
    }

    //
    // Create successively coarser grids using an opportunistic policy.
    //
    // Any set of siblings is coarsened to create their parent, unless
    // doing so would violate the 2:1 balancing constraint in the coarse grid.
    //
    for (int coarseStratum = 1; coarseStratum < m_numStrata; coarseStratum++)
    {
      // Identify fineGrid and coarseGrid.
      std::vector<TreeNode<T, dim>> &fineGrid = m_gridStrata[coarseStratum-1];
      std::vector<TreeNode<T, dim>> &coarseGrid = m_gridStrata[coarseStratum];


      // Coarsen from the fine grid to the coarse grid.

      // 1. Generate candidates as parents of all local fine grid elements.
      std::vector<TreeNode<T, dim>> candidates;

      using LoopPostSkipEmptyT = MeshLoopPostSkipEmpty<T, dim>;
      LevI prevLev = 0;
      for (const MeshLoopFrame<T, dim> &subtree : LoopPostSkipEmptyT(fineGrid))
      {
        if (!subtree.isLeaf() && subtree.getLev() == prevLev - 1)
          candidates.emplace_back(fineGrid[subtree.getBeginIdx()].getParent());
        prevLev = subtree.getLev();
      }
      // The candidates are all distinct on any rank. Dups possible globally.


      //     2. Initialize disqualifiers as parents of candidates.
      std::vector<TreeNode<T, dim>> disqualifiers;
      prevLev = 0;
      for (const MeshLoopFrame<T, dim> &subtree : LoopPostSkipEmptyT(candidates))
      {
        if (!subtree.isLeaf() && subtree.getLev() == prevLev - 1)
          disqualifiers.emplace_back(candidates[subtree.getBeginIdx()].getParent());
        prevLev = subtree.getLev();
      }
      // Disqualifiers initially distinct.

      // 3. Propagate disqualifiers level by level:
      //    3.1 Stratify disqualifiers by level.
      //    3.2 Starting from the deepest level, add all parents of neighbors.

      std::vector<std::vector<TreeNode<T, dim>>>
          stratDisq = stratifyTree(disqualifiers);

      // Bottom-up propagation using stratified levels.
      size_t num_disq = stratDisq[0].size();
      std::vector<TreeNode<T, dim>> nbrBuffer;
      for (LevI l = m_uiMaxDepth; l > 0; l--)
      {
        const LevI lp = l-1;  // Parent level.

        nbrBuffer.clear();
        for (const TreeNode<T, dim> &tn : stratDisq[l])
          tn.appendAllNeighbours(nbrBuffer);
        for (const TreeNode<T, dim> &nbr : nbrBuffer)
          stratDisq[lp].push_back(nbr.getParent());

        SFC_Tree<T, dim>::locTreeSort(&(*stratDisq[lp].begin()), 0, stratDisq[lp].size(), 1, lp, 0);
        SFC_Tree<T, dim>::locRemoveDuplicates(stratDisq[lp]);

        num_disq += stratDisq[lp].size();
      }
      // After adding new disqualifiers to a level, that level is made distinct.

      disqualifiers.clear();
      disqualifiers.reserve(num_disq);
      for (const std::vector<TreeNode<T, dim>> &disqLev : stratDisq)
        disqualifiers.insert(disqualifiers.end(), disqLev.begin(), disqLev.end());
      stratDisq.clear();
      stratDisq.shrink_to_fit();

      // 4. Use the fine grid splitters to partition the candidates and disqualifiers.
      //    (Could use other splitters, but the fine grid splitters are handy.)

      //TODO what if not everybody has something to broadcast?
      // We need to add a bool at each level saying are we active.
      // Assume the finest level has contents and thus splitters on every rank.
      // Use the fine splitters to finalize the coarse contents.
      // Redistribute the coarse contents, and evaluate the coarse active/inactive,
      // create a new comm that includes only the nonempty ranks.
      // We can terminate early if the comm has a single rank.
      std::vector<TreeNode<T, dim>> fsplitters =
          SFC_Tree<T, dim>::dist_bcastSplitters(&m_tpFrontStrata[coarseStratum-1], activeComm);

      //TODO this whole section could be factored out as a general
      // method that sends octants to their owners


      BucketMultiplexer<size_t, int> candidate_bmpx(nProcActive, candidates.size());
      BucketMultiplexer<size_t, int> disqualif_bmpx(nProcActive, disqualifiers.size());

      // Synchronize traversal of three lists of TreeNodes.
      // Descend into empty subtrees and control ascension manually.
      MeshLoopInterface<T, dim, true, true, false> lpSplitters(fsplitters);
      MeshLoopInterface<T, dim, true, true, false> lpCandidates(candidates);
      MeshLoopInterface<T, dim, true, true, false> lpDisqualifiers(disqualifiers);
      int splitterCount = 0;
      /// std::vector<size_t> scountCandidates(nProc, 0);
      /// std::vector<size_t> scountDisqualifiers(nProc, 0);
      while (!lpSplitters.isFinished())
      {
        const MeshLoopFrame<T, dim> &subtreeSplitters = lpSplitters.getTopConst();
        const MeshLoopFrame<T, dim> &subtreeCandidates = lpCandidates.getTopConst();
        const MeshLoopFrame<T, dim> &subtreeDisqualifiers = lpDisqualifiers.getTopConst();

        int splittersInSubtree = subtreeSplitters.getTotalCount();

        // Case 1: There are no splitters in the subtree.
        //     --> add all items to current bucket.
        // Case 2: The splitter subtree is a leaf.
        //     --> No items can be deeper than the current subtree.
        //         Advance the bucket and add the items to it.
        // Case 2a: The splitter subtree is a nonempty nonleaf, and the item subtree is a leaf.
        //     --> add the current item to all buckets split by the splitters.
        // Case 2b: The splitter subtree is a nonempty nonleaf, and the item subtree is not a leaf.
        //     --> descend.

        if (subtreeSplitters.isEmpty() || subtreeSplitters.isLeaf() ||
            (subtreeCandidates.isEmpty() && subtreeDisqualifiers.isEmpty()))
        {
          if (subtreeSplitters.isLeaf())
            ++splitterCount;

          /// scountCandidates[splitterCount - 1] += subtreeCandidates.getTotalCount();
          /// scountDisqualifiers[splitterCount - 1] += subtreeDisqualifiers.getTotalCount();

          for (size_t cIdx = subtreeCandidates.getBeginIdx(); cIdx < subtreeCandidates.getEndIdx(); ++cIdx)
            candidate_bmpx.addToBucket(cIdx, splitterCount);
          for (size_t dIdx = subtreeDisqualifiers.getBeginIdx(); dIdx < subtreeDisqualifiers.getEndIdx(); ++dIdx)
            disqualif_bmpx.addToBucket(dIdx, splitterCount);

          lpSplitters.next();
          lpCandidates.next();
          lpDisqualifiers.next();
        }
        else
        {
          // A candidate that overlaps multiple partitions should be duplicated.
          if (!subtreeCandidates.isEmpty() && subtreeCandidates.isLeaf() && splittersInSubtree > 0)
          {
            size_t candidateIdx = subtreeCandidates.getBeginIdx();
            int bucketIdx;
            for (bucketIdx = splitterCount; bucketIdx < splitterCount + splittersInSubtree; bucketIdx++)
              candidate_bmpx.addToBucket(candidateIdx, bucketIdx);
            candidate_bmpx.addToBucket(candidateIdx, bucketIdx);
          }
  
          // A disqualifier that overlaps multiple partitions should be duplicated.
          if (!subtreeCandidates.isEmpty() && subtreeDisqualifiers.isLeaf() && splittersInSubtree > 0)
          {
            size_t disqualifIdx = subtreeDisqualifiers.getBeginIdx();
            int bucketIdx;
            for (bucketIdx = splitterCount; bucketIdx < splitterCount + splittersInSubtree; bucketIdx++)
              disqualif_bmpx.addToBucket(disqualifIdx, bucketIdx);
            disqualif_bmpx.addToBucket(disqualifIdx, bucketIdx);
          }

          /// scountCandidates[splitterCount - 1] += subtreeCandidates.getAncCount();
          /// scountDisqualifiers[splitterCount - 1] += subtreeDisqualifiers.getAncCount();

          lpSplitters.step();
          lpCandidates.step();
          lpDisqualifiers.step();
        }
      }

      /// std::vector<size_t> soffsetsCandidates(nProc, 0);
      /// { int offset = candidates.size();
      ///   for (int i = nProc-1; i >= 0; i--)
      ///     soffsetsCandidates[i] = (offset -= scountCandidates[i]);
      /// }

      /// std::vector<size_t> soffsetsDisqualifiers(nProc, 0);
      /// { int offset = candidates.size();
      ///   for (int i = nProc-1; i >= 0; i--)
      ///     soffsetsDisqualifiers[i] = (offset -= scountDisqualifiers[i]);
      /// }

      // Send counts, offsets, buffer for candidates.
      std::vector<size_t> scountCandidates = candidate_bmpx.getBucketCounts();
      std::vector<size_t> soffsetsCandidates = candidate_bmpx.getBucketOffsets();
      std::vector<TreeNode<T, dim>> sendbufCandidates(candidate_bmpx.getTotalItems());
      candidate_bmpx.transferCopies(sendbufCandidates.data(), candidates.data());

      // Send counts, offsets, buffer for disqualifiers.
      std::vector<size_t> scountDisqualifiers = disqualif_bmpx.getBucketCounts();
      std::vector<size_t> soffsetsDisqualifiers = disqualif_bmpx.getBucketOffsets();
      std::vector<TreeNode<T, dim>> sendbufDisqualifiers(disqualif_bmpx.getTotalItems());
      disqualif_bmpx.transferCopies(sendbufDisqualifiers.data(), disqualifiers.data());



      //TODO exchange scounts rcounts

      std::vector<size_t> rcountCandidates(nProc, 0);
      std::vector<size_t> rcountDisqualifiers(nProc, 0);

      // Send/recv candidates and disqualifiers

      std::vector<TreeNode<T, dim>> recvCandidates, recvDisqualifiers;



      //TODO


      //            3.2.1 Simultaneously traverse the list of candidates,
      //                    so as to eliminate disqualifiers that are not relevant.
      //            3.2.2 Need to check the before and after address of the
      //                    disqualifier against the local partition splitters.
      //            3.2.3 Any disqualifier that belongs to another partition
      //                    can be added to a sendcount.





      // 4. Collect and send/recv candidates and disqualifiers that belong
      //      to another partition.

      // 5. Simultaneously traverse candidates, disqualifiers, and the fine grid
      //    5.1 If there exists a candidate, and it's not disqualified,
      //          add it as a cell in the coarse grid.
      //    5.2 Otherwise, add the subtree from the fine grid.


      // After finalize the candidates, repartition and get partFront/Back.

      //TODO test for empty, replace activeComm

    }


  }



  // Explicit instantiations
  template class DistTree<unsigned int, 2u>;
  template class DistTree<unsigned int, 3u>;
  template class DistTree<unsigned int, 4u>;

}
