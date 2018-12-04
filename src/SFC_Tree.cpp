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
//   `out' will be resized to inp_end - inp_begin.
//
template<typename T, unsigned int D>
void
SFC_Tree<T,D>:: locTreeSort(const TreeNode<T,D> *inp_begin, const TreeNode<T,D> *inp_end,
                   std::vector<TreeNode<T,D>> &out,
                   unsigned int sLev,
                   unsigned int eLev,
                   unsigned int pRot)
{
  //// Recursive Depth-first Most Significant Digit First. ////

  // There is a global stack declared in hcurvedata.h:
  //   - std::vector<unsigned int> RotationID_Stack;
  //   - unsigned int rotationStackPointer;
  // TODO It seems that in later versions of Dendro this was replaced
  //      by a local std::vector stack.
  //      Maybe in the recursive locTreeSort we can just get away
  //      with storing the rotation stack in the call stack (pRot).

  using TreeNode = TreeNode<T,D>;
  constexpr unsigned int numChildren = TreeNode::numChildren;
  constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

  //
  // Reorder the points according to the SFC at level sLev.

  //DEBUG
  printf("Level==%u,  Input Size==%lu\n", sLev, (inp_end - inp_begin));

  out.resize(inp_end - inp_begin);

  // Count the number of points in each bucket,
  // indexed by (Morton) child number.
  std::array<int, numChildren> counts;
  counts.fill(0);
  /// for (const TreeNode &tn : inp)
  for (const TreeNode *tn = inp_begin; tn != inp_end; tn++)
  {
    counts[tn->getMortonIndex(sLev)]++;
  }

  // Compute offsets of buckets in permuted SFC order.
  // Conceptually:
  //   1. Permute counts;  2. offsets=scan(counts);  3. Un-permute offsets.
  //
  // - Since this function is recursive, we are inside a single parent octant
  //   for the duration of the body. This means we apply a single permutation
  //   to the buckets for all points.
  // - We need to "Un-permute offsets" so that we can index using Morton index.
  int accum = 0;
  std::array<int, numChildren> offsets;
  // Logically permute: Scan the bucket-counts in the order of the SFC.
  // Since we want to map [SFC_rank]-->Morton_rank,
  // use the "left" columns of rotations[], aka `rot_perm'.
  const char *rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
  for (int child_sfc = 0; child_sfc < numChildren; child_sfc++)
  {
    char child = rot_perm[child_sfc] - '0';  // Decode from human-readable ASCII.
    offsets[child] = accum;
    accum += counts[child];
  }

  // Move points from `inp' to `out' according to `offsets'
  /// for (const TreeNode &tn : inp)
  for (const TreeNode *tn = inp_begin; tn != inp_end; tn++)
  {
    unsigned char child = tn->getMortonIndex(sLev);
    out[offsets[child]++] = *tn;
  }

  //
  // Recurse.
  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    std::vector<TreeNode> reorderedChild;
    const char *orientLookup = &HILBERT_TABLE[pRot*numChildren];
    for (int child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      char child = rot_perm[child_sfc] - '0';  // Decode from human-readable ASCII.

      if (counts[child] <= 1)
        continue;

      // Columns of HILBERT_TABLE are indexed by the Morton rank.
      // According to Dendro4 TreeNode.tcc:199 they are.
      // (There are possibly inconsistencies in the old code...
      // Don't worry, we'll be regenerating the table later.)
      unsigned int cRot = orientLookup[child];

      // Recall that offsets[child] attaned the last+1 index during MOVE.
      TreeNode * const out_begin = &out[offsets[child] - counts[child]];
      TreeNode * const out_end = &out[offsets[child]];
      locTreeSort(out_begin, out_end, reorderedChild, sLev + 1, eLev, cRot);

      // Now that `reorderedChild' contains the reorderedChild points,
      // copy them into `out'.
      for (int ii = 0; ii < counts[child]; ii++)
      {
        out_begin[ii] = reorderedChild[ii];
      }
      /// for (TreeNode *outPtr = out_begin, int ii = 0; outPtr != out_end; outPtr++, ii++)
      /// {
      ///   *outPtr = reorderedChild[ii];
      /// }
    }
  }//end if()
}// end function()
//TODO: Consider making this in-place: So `inp' will no longer be const, but we can
//      avoid allocating (and joining) lots of std::vector. To do it, we just need an
//      extra buffer of std::array<TreeNode, numChildren> during the
//      MOVE phase; then repeat: {skip belonging points},
//      {move points to buffers}, {accept points from buffers}.
//      Downside: Perhaps not cache-friendly.

} // namspace ot
