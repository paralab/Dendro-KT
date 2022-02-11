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
#include "tnUtils.h"

#include "filterFunction.h"

#include "meshLoop.h"

#include <vector>
#include <set>


namespace ot
{


//
// SFC_bucketing()
//
//   Based on Dendro4 sfcSort.h SFC_bucketing().
//
template<typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: SFC_bucketing(TreeNode<T,dim> *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd)
{

  //TODO transfer the comments to the fully configurable templated version of this method.

  SFC_bucketing_impl<KeyFunIdentity_TN<T,dim>, TreeNode<T,dim>, TreeNode<T,dim>>(
      points, begin, end, lev, sfc,
      KeyFunIdentity_TN<T,dim>(), true, true,
      outSplitters,
      outAncStart, outAncEnd);

///   // ==
///   // Reorder the points by child number at level `lev', in the order
///   // of the SFC, and yield the positions of the splitters.
///   // ==
/// 
///   using TreeNode = TreeNode<T,dim>;
///   constexpr char numChildren = nchild(dim);
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


template<typename T, unsigned int dim>
void
SFC_Tree<T,dim>::SFC_locateBuckets(const TreeNode<T,dim> *points,
                                 RankI begin, RankI end,
                                 LevI lev,
                                 SFC_State<dim> sfc,
                                 std::array<RankI, 1+nchild(dim)> &outSplitters,
                                 RankI &outAncStart,
                                 RankI &outAncEnd)
{
  SFC_locateBuckets_impl<KeyFunIdentity_TN<T,dim>, TreeNode<T,dim>, TreeNode<T,dim>>(
      points, begin, end, lev, sfc,
      KeyFunIdentity_TN<T,dim>(), true, true,
      outSplitters,
      outAncStart, outAncEnd);
}


//
// tsearch_lower_bound()
//
template <typename T, unsigned int dim>
size_t SFC_Tree<T, dim>::tsearch_lower_bound(
    const std::vector<TreeNode<T, dim>> &sortedOcts,
    const TreeNode<T, dim> &key)
{
  return tsearch_equal_range(
      &(*sortedOcts.begin()),
      key,
      0, sortedOcts.size(),
      1, SFC_State<dim>::root()).first;
}


//
// tsearch_upper_bound()
//
template <typename T, unsigned int dim>
size_t SFC_Tree<T, dim>::tsearch_upper_bound(
    const std::vector<TreeNode<T, dim>> &sortedOcts,
    const TreeNode<T, dim> &key)
{
  return tsearch_equal_range(
      &(*sortedOcts.begin()),
      key,
      0, sortedOcts.size(),
      1, SFC_State<dim>::root()).second;
}


//
// tsearch_equal_range()
//
template <typename T, unsigned int dim>
std::pair<size_t, size_t> SFC_Tree<T, dim>::tsearch_equal_range(
      const std::vector<TreeNode<T, dim>> &sortedOcts,
      const TreeNode<T, dim> &key)
{
  return tsearch_equal_range(
      &(*sortedOcts.begin()),
      key,
      0, sortedOcts.size(),
      1, SFC_State<dim>::root());
}


//
// tsearch_equal_range()  (implementation for all 3 bounds)
//
template <typename T, unsigned int dim>
std::pair<size_t, size_t> SFC_Tree<T, dim>::tsearch_equal_range(
      const TreeNode<T, dim> *sortedOcts,
      const TreeNode<T, dim> &key,
      size_t begin, size_t end,
      LevI sLev,
      SFC_State<dim> sfc)
{
  // Keep track of both first greater_or_equal and first greater.
  // If greater is found before equal, then it is also greater_or_equal.
  // The search is over when both have been found, so filled()==true.
  struct ERange {
    bool filled() const { return m_found_geq && m_found_greater; }
    std::pair<size_t, size_t> const pair() { return {m_geq_g[0], m_geq_g[1]}; }

    void equal(const size_t idx_eq) {
      if (m_found_geq) return;
      m_geq_g[0] = idx_eq;
      m_found_geq = true;
    }

    void greater(const size_t idx_greater) {
      if (m_found_greater) return;
      m_geq_g[m_found_geq] = idx_greater;
      m_found_geq = true;
      m_geq_g[1] = idx_greater;
      m_found_greater = true;
    }

    size_t m_geq_g[2];  bool m_found_geq;  bool m_found_greater;

  } range = {{end, end}, false, false};

  using Oct = TreeNode<T, dim>;
  const auto level = [](const Oct &oct) -> LevI { return oct.getLevel(); };
  const auto cnum = [](const Oct &oct, LevI l) -> sfc::ChildNum
      { return sfc::ChildNum(oct.getMortonIndex(l)); };
  const auto coarserLevel = [](const Oct &a, const Oct &b) -> LevI
      { return fminf(a.getLevel(), b.getLevel()); };
  const auto shareSubtree = [](LevI l, const Oct &a, const Oct &b) -> bool
      { return l <= a.getCommonAncestorDepth(b); };

  const Oct * &x = sortedOcts;  // alias for brevity
  LevI lp = sLev - 1;
  size_t i = begin;

  // Pre:  forall(j < i), x[j] < key;
  //       lp <= level(key);
  //       lp == 0  or  subtree{lp}(x[i-1]) == subtree{lp}(key)
  while (i < end && !range.filled())
  {
    if (level(x[i]) < lp or !shareSubtree(lp, x[i], key))
      range.greater(i);
    else
    {
      // Pre: lp <= level(x[i]), level(key);
      //      subtree{lp}(x[i]) == subtree{lp}(key)
      const LevI lBoth = coarserLevel(x[i], key);
      while (lp < lBoth and shareSubtree(lp + 1, x[i], key))
      {
        sfc = sfc.child_curve(cnum(key, lp + 1));
        ++lp;
      }

      if (lp == lBoth)
        if (level(x[i]) == level(key))
          range.equal(i),  ++i;
        else if (level(x[i]) > level(key))
          range.greater(i);
        else
          ++i;
      else
        if (sfc.child_rank(cnum(x[i], lp + 1))
            > sfc.child_rank(cnum(key, lp + 1)))
          range.greater(i);
        else
          ++i;
    }
  }

  return range.pair();
}



template <typename X>
struct Segment
{
  X *ptr = nullptr;
  size_t begin = 0;
  size_t end = 0;

  Segment(X *ptr_, size_t begin_, size_t end_)
    : ptr(ptr_), begin(begin_), end(end_) {}

  bool nonempty() const       { return begin < end; }
  bool empty() const          { return !nonempty(); }
  const X & operator*() const { return ptr[begin]; }
  X & operator*()             { return ptr[begin]; }
  Segment & operator++()      { ++begin; return *this; }
};

template <typename X>
Segment<X> segment_all(std::vector<X> &vec)
{
  return Segment<X>{vec.data(), 0, vec.size()};
}

template <typename X>
Segment<const X> segment_all(const std::vector<X> &vec)
{
  return Segment<const X>{vec.data(), 0, vec.size()};
}


template <typename X>
struct Keeper
{
  X *ptr = nullptr;
  size_t begin = 0;
  size_t end = 0;
  size_t out = 0;
  bool kept = false;

  Keeper(X *ptr_, size_t begin_, size_t end_)
    : ptr(ptr_), begin(begin_), end(end_), out(begin_) {}

  bool nonempty() const       { return begin < end; }
  bool empty() const          { return !nonempty(); }
  const X & operator*() const { return ptr[begin]; }
  X & operator*()             { return ptr[begin]; }
  Keeper & operator++()       { ++begin; kept = false; return *this; }
  void keep()                 { if (!kept) ptr[out++] = ptr[begin]; kept = true; }
  void unkeep()               { --out; kept = false; }
  bool can_store() const      { return out < begin; }
  bool store(const X &x)      { return can_store() && (ptr[out++] = x, true); }
  bool adv_store(const X &x)  { operator++();  return store(x); }
};





/* Appends info for all keys in subtree and advances both sequences past subtree. */
template <typename T, unsigned int dim>
void overlaps_lower_bound_rec(
    Segment<const TreeNode<T, dim>> &sortedOcts,
    Segment<const TreeNode<T, dim>> &sortedKeys,
    std::vector<size_t> &lineage,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<size_t> &beginOverlapsBounds,
    std::vector<size_t> &overlapsBounds);

/* Appends lower bound for all keys in subtree and advances both sequences past subtree. */
template <typename T, unsigned int dim>
void lower_bound_rec(
    Segment<const TreeNode<T, dim>> &sortedOcts,
    Segment<const TreeNode<T, dim>> &sortedKeys,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<size_t> &lowerBounds);




template <typename T, int dim>
Overlaps<T, dim>::Overlaps(
    const std::vector<TreeNode<T, dim>> &sortedOcts,
    const std::vector<TreeNode<T, dim>> &sortedKeys)
  :
    m_sortedOcts(sortedOcts),
    m_sortedKeys(sortedKeys)
{
  /** For each key, returns two things from sortedOcts:
   *    - The index of the first element not less than the key, and
   *    - a list of zero or more indices of ancestor octants (appended).
   *  From these, the subset of sortedOcts overlapping each key can be recovered.
   *  Strict ancestor overlaps may be dispersed in the input,
   *  while inclusive descendant overlaps must occur in a contiguous sequence.
   *
   * @param sortedOcts [in] A sorted list of octants, not necessarily unique.
   * @param sortedKeys [in] A sorted list of octants, not necessarily unique.
   * @param beginOverlaps [out] Begining of segment in overlaps for each key.
   * @param overlaps [out] Concatenated lists of ancestor overlaps and lower bounds.
   */
  using Oct = TreeNode<T, dim>;
  Segment<const Oct> segSortedOcts(&(*sortedOcts.cbegin()), 0, sortedOcts.size());
  Segment<const Oct> segSortedKeys(&(*sortedKeys.cbegin()), 0, sortedKeys.size());
  std::vector<size_t> lineage;
  m_beginOverlaps.clear();
  m_overlaps.clear();

  overlaps_lower_bound_rec<T, dim>(
      segSortedOcts,
      segSortedKeys,
      lineage,
      Oct(),
      SFC_State<dim>::root(),
      m_beginOverlaps,
      m_overlaps);
}

template <typename T, int dim>
void Overlaps<T, dim>::keyOverlaps(
    const size_t keyIdx, std::vector<size_t> &overlapIdxs)
{
  // Index 0 of the search result indicates possible descendant overlaps.
  // Index 1 and above of the search result indicate ancestor overlaps.
  // Put ancestors first to maintain sorted order.
  overlapIdxs.clear();
  keyOverlapsAncestors(keyIdx, overlapIdxs);
  size_t descendant = m_overlaps[m_beginOverlaps[keyIdx] + 0];
  while (descendant < m_sortedOcts.size() &&
         m_sortedKeys[keyIdx].isAncestorInclusive(m_sortedOcts[descendant]))
    overlapIdxs.push_back(descendant++);
}

template <typename T, int dim>
void Overlaps<T, dim>::keyOverlapsAncestors(
    const size_t keyIdx, std::vector<size_t> &overlapIdxs)
{
  // Index 1 and above of the search result indicate ancestor overlaps.
  const size_t listSize =
      (keyIdx < m_beginOverlaps.size() - 1 ?
          m_beginOverlaps[keyIdx + 1] - m_beginOverlaps[keyIdx] :
          m_overlaps.size() - m_beginOverlaps[keyIdx]);

  overlapIdxs.clear();
  for (size_t j = 1; j < listSize; ++j)
    overlapIdxs.push_back(m_overlaps[m_beginOverlaps[keyIdx] + j]);
}

template class Overlaps<unsigned, 2>;
template class Overlaps<unsigned, 3>;
template class Overlaps<unsigned, 4>;




//
// lower_bound()
//
template <typename T, unsigned int dim>
std::vector<size_t> SFC_Tree<T, dim>::lower_bound(
    const std::vector<TreeNode<T, dim>> &sortedOcts,
    const std::vector<TreeNode<T, dim>> &sortedKeys)
{
  assert(isLocallySorted(sortedOcts));
  assert(isLocallySorted(sortedKeys));

  using Oct = TreeNode<T, dim>;
  Segment<const Oct> segSortedOcts(&(*sortedOcts.cbegin()), 0, sortedOcts.size());
  Segment<const Oct> segSortedKeys(&(*sortedKeys.cbegin()), 0, sortedKeys.size());
  std::vector<size_t> lowerBounds;

  lower_bound_rec<T, dim>(
      segSortedOcts,
      segSortedKeys,
      Oct(),
      SFC_State<dim>::root(),
      lowerBounds);

  return lowerBounds;
}

// overlaps_lower_bound_rec()
template <typename T, unsigned int dim>
void overlaps_lower_bound_rec(
    Segment<const TreeNode<T, dim>> &sortedOcts,
    Segment<const TreeNode<T, dim>> &sortedKeys,
    std::vector<size_t> &lineage,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<size_t> &beginOverlapsBounds,
    std::vector<size_t> &overlapsBounds)
{
  // As leaf(s)
  while (sortedKeys.nonempty() && subtree == *sortedKeys)
  {
    beginOverlapsBounds.push_back(overlapsBounds.size());
    overlapsBounds.push_back(sortedOcts.begin);
    overlapsBounds.insert(overlapsBounds.end(), lineage.begin(), lineage.end());
    ++sortedKeys;
  }
  size_t pushed = 0;
  while (sortedOcts.nonempty() && subtree == *sortedOcts)
  {
    lineage.push_back(sortedOcts.begin);
    ++pushed;
    ++sortedOcts;
  }

  // As subtree
  if (sortedKeys.nonempty() && subtree.isAncestor(*sortedKeys))
  {
    // Traverse child subtrees.
    for (sfc::SubIndex c(0); c < nchild(dim); ++c)
      overlaps_lower_bound_rec<T, dim>(
          sortedOcts, sortedKeys, lineage,
          subtree.getChildMorton(sfc.child_num(c)),
          sfc.subcurve(c),
          beginOverlapsBounds,
          overlapsBounds);
  }
  else
  {
    // Skip subtree in sortedOcts.
    while (sortedOcts.nonempty() && subtree.isAncestor(*sortedOcts))
      ++sortedOcts;
  }
  while (pushed-- > 0)
    lineage.pop_back();
}

// lower_bound_rec()
template <typename T, unsigned int dim>
void lower_bound_rec(
    Segment<const TreeNode<T, dim>> &sortedOcts,
    Segment<const TreeNode<T, dim>> &sortedKeys,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<size_t> &lowerBounds)
{
  // As leaf(s)
  while (sortedKeys.nonempty() && subtree == *sortedKeys)
  {
    lowerBounds.push_back(sortedOcts.begin);
    ++sortedKeys;
  }
  while (sortedOcts.nonempty() && subtree == *sortedOcts)
  {
    ++sortedOcts;
  }

  // As subtree
  if (sortedKeys.nonempty() && subtree.isAncestor(*sortedKeys))
  {
    // Traverse child subtrees.
    for (sfc::SubIndex c(0); c < nchild(dim); ++c)
      lower_bound_rec<T, dim>(
          sortedOcts, sortedKeys,
          subtree.getChildMorton(sfc.child_num(c)),
          sfc.subcurve(c),
          lowerBounds);
  }
  else
  {
    // Skip subtree in sortedOcts.
    while (sortedOcts.nonempty() && subtree.isAncestor(*sortedOcts))
      ++sortedOcts;
  }
}


//
// removeDescendants()
//
template <typename T, unsigned dim>
void SFC_Tree<T, dim>::removeDescendants(
      std::vector<TreeNode<T, dim>> &sortedOcts,
      const std::vector<TreeNode<T, dim>> &sortedKeys)
{
  const std::vector<size_t> lowerBounds = lower_bound(sortedOcts, sortedKeys);
  size_t kept = 0;
  size_t i = 0;
  for (size_t keyIdx = 0; keyIdx < sortedKeys.size(); ++keyIdx)
  {
    while (i < lowerBounds[keyIdx])
      sortedOcts[kept++] = sortedOcts[i++];
    while (i < sortedOcts.size() &&
           sortedKeys[keyIdx].isAncestorInclusive(sortedOcts[i]))
      ++i;
  }
  while (i < sortedOcts.size())
    sortedOcts[kept++] = sortedOcts[i++];
  sortedOcts.resize(kept);
}

//
// retainDescendants()
//
template <typename T, unsigned dim>
void SFC_Tree<T, dim>::retainDescendants(
      std::vector<TreeNode<T, dim>> &sortedOcts,
      const std::vector<TreeNode<T, dim>> &sortedKeys)
{
  const std::vector<size_t> lowerBounds = lower_bound(sortedOcts, sortedKeys);
  size_t kept = 0;
  size_t i = 0;
  for (size_t keyIdx = 0; keyIdx < sortedKeys.size(); ++keyIdx)
  {
    while (i < lowerBounds[keyIdx])
      ++i;
    while (i < sortedOcts.size() &&
           sortedKeys[keyIdx].isAncestorInclusive(sortedOcts[i]))
      sortedOcts[kept++] = sortedOcts[i++];
  }
  sortedOcts.resize(kept);
}

//
// removeEqual()
//
template <typename T, unsigned dim>
void SFC_Tree<T, dim>::removeEqual(
      std::vector<TreeNode<T, dim>> &sortedOcts,
      const std::vector<TreeNode<T, dim>> &sortedKeys)
{
  const std::vector<size_t> lowerBounds = lower_bound(sortedOcts, sortedKeys);
  size_t kept = 0;
  size_t i = 0;
  for (size_t keyIdx = 0; keyIdx < sortedKeys.size(); ++keyIdx)
  {
    while (i < lowerBounds[keyIdx])
      sortedOcts[kept++] = sortedOcts[i++];
    while (i < sortedOcts.size() && sortedOcts[i] == sortedKeys[keyIdx])
      ++i;
  }
  while (i < sortedOcts.size())
    sortedOcts[kept++] = sortedOcts[i++];
  sortedOcts.resize(kept);
}




template<typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: distTreeSort(std::vector<TreeNode<T,dim>> &points,
                          double loadFlexibility,
                          MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  // The heavy lifting to globally sort/partition.
  distTreePartition(points, 0, loadFlexibility, comm);

  // Finish with a local TreeSort to ensure all points are in order.
  locTreeSort(&(*points.begin()), 0, points.size(), 0, m_uiMaxDepth, SFC_State<dim>::root());

  /// // DEBUG: print out all the points.  // This debugging section will break.
  /// { std::vector<char> spaces(m_uiMaxDepth*rProc+1, ' ');
  /// spaces.back() = '\0';
  /// for (const TreeNode tn : points)
  ///   std::cout << spaces.data() << tn.getBase32Hex().data() << "\n";
  /// std::cout << spaces.data() << "------------------------------------\n";
  /// }
}



struct P2PPartners
{
  std::vector<int> m_dest;
  std::vector<int> m_src;
  MPI_Comm m_comm;
  int m_comm_size;
  int m_comm_rank;

  void reserve(size_t ndest, size_t nsrc)
  {
    assert(ndest >= 0);
    assert(nsrc >= 0);
    m_dest.reserve(ndest);
    m_src.reserve(nsrc);
  }

  P2PPartners() = default;

  P2PPartners(const std::vector<int> &dest,
              const std::vector<int> &src,
              MPI_Comm comm)
    : m_dest(dest), m_src(src), m_comm(comm)
  {}

  void reset(size_t ndest, size_t nsrc, MPI_Comm comm) {
    nDest(ndest);  nSrc(nsrc);  m_comm = comm;
    MPI_Comm_size(comm, &m_comm_size);
    MPI_Comm_rank(comm, &m_comm_rank);
  }

  MPI_Comm comm() const { return m_comm; }
  int comm_size() const { return m_comm_size; }
  int comm_rank() const { return m_comm_rank; }

  size_t nDest() const { return m_dest.size(); }
  size_t nSrc()  const { return m_src.size(); }

  void nDest(size_t size) { m_dest.resize(size); }
  void nSrc(size_t size)  { m_src.resize(size); }

  int dest(size_t i) const { assert(i < m_dest.size());  return m_dest[i]; }
  int src(size_t i)  const { assert(i < m_src.size());   return m_src[i]; }

  void dest(size_t i, int d) { assert(i < m_dest.size());  m_dest[i] = d; }
  void src(size_t i, int s)  { assert(i < m_src.size());   m_src[i] = s; }
};


template <typename ScalarT = int>
struct P2PScalar
{
  const P2PPartners *m_partners = nullptr;
  MPI_Comm comm() const    { assert(m_partners != nullptr);  return m_partners->comm(); }
  int dest(size_t i) const { assert(m_partners != nullptr);  return m_partners->dest(i); }
  int src(size_t i)  const { assert(m_partners != nullptr);  return m_partners->src(i); }

  static constexpr int LEN = 1;
  std::vector<ScalarT> m_sendScalar;
  std::vector<ScalarT> m_recvScalar;
  std::vector<MPI_Request> m_requests;

  void reserve(int ndest, int nsrc)
  {
    assert(ndest >= 0);
    assert(nsrc >= 0);
    m_sendScalar.reserve(LEN * ndest);
    m_recvScalar.reserve(LEN * nsrc);
    m_requests.reserve(ndest);
  }

  P2PScalar() = default;
  P2PScalar(const P2PPartners *partners) { reset(partners); }

  void reset(const P2PPartners *partners)
  {
    assert(partners != nullptr);
    m_partners = partners;
    m_sendScalar.resize(LEN * partners->nDest());
    m_recvScalar.resize(LEN * partners->nSrc());
    m_requests.resize(partners->nDest());
  }

  void send(int destIdx, ScalarT scalar) {
    assert(destIdx < m_partners->nDest());
    m_sendScalar[destIdx] = scalar;  // future: LEN > 1
    par::Mpi_Isend(&(m_sendScalar[destIdx]), LEN, dest(destIdx), 0, comm(), &m_requests[destIdx]);
  }

  ScalarT recv(int srcIdx) {
    assert(srcIdx < m_partners->nSrc());
    MPI_Status status;
    par::Mpi_Recv(&(m_recvScalar[srcIdx]), LEN, src(srcIdx), 0, comm(), &status);
    return m_recvScalar[srcIdx];
  }

  void recv(int srcIdx, ScalarT &scalar) {
    scalar = this->recv(srcIdx);
  }

  void wait_all() {
    for (MPI_Request &request : m_requests)
      MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
};


template <typename X>
struct P2PVector
{
  const std::vector<int> &m_dest;
  const std::vector<int> &m_src;
  MPI_Comm m_comm;

  std::vector<MPI_Request> m_requests;

  P2PVector(const P2PPartners *partners)
    : m_dest(partners->m_dest), m_src(partners->m_src), m_comm(partners->m_comm),
      m_requests(partners->nDest())
  { }

  void send(int destIdx, const std::vector<X> &vector) {
    par::Mpi_Isend(&(*vector.cbegin()), vector.size(), m_dest[destIdx], 0, m_comm, &m_requests[destIdx]);
  }

  void recv(int srcIdx, std::vector<X> &vector) {
    MPI_Status status;
    par::Mpi_Recv(&(*vector.begin()), vector.size(), m_src[srcIdx], 0, m_comm, &status);
  }

  void wait_all() {
    for (MPI_Request &request : m_requests)
      MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
};


// P2PMeta
struct P2PMeta
{
  const P2PPartners *m_partners = nullptr;

  std::vector<int> m_send_meta;
  std::vector<int> m_recv_meta;
  std::vector<MPI_Request> m_requests;
  int m_recv_total = 0;

  long long unsigned m_bytes_sent = 0;
  long long unsigned m_bytes_rcvd = 0;

  MPI_Comm comm() const    { assert(m_partners != nullptr);  return m_partners->comm(); }
  int dest(size_t i) const { assert(m_partners != nullptr);  return m_partners->dest(i); }
  int src(size_t i)  const { assert(m_partners != nullptr);  return m_partners->src(i); }

  int * send_sizes()   { assert(m_send_meta.size());  return &m_send_meta[0]; }
  int * send_offsets() { assert(m_send_meta.size());  return &m_send_meta[m_partners->nDest()]; }
  int * recv_sizes()   { assert(m_recv_meta.size());  return &m_recv_meta[0]; }
  int * recv_offsets() { assert(m_recv_meta.size());  return &m_recv_meta[m_partners->nSrc()]; }

  const int recv_total() const { return m_recv_total; }

  long long unsigned bytes_sent() const { return m_bytes_sent; }
  long long unsigned bytes_rcvd() const { return m_bytes_rcvd; }

  void reserve(int ndest, int nsrc)
  {
    m_send_meta.reserve(2 * ndest);
    m_recv_meta.reserve(2 * nsrc);
    m_requests.reserve(ndest);
  }

  P2PMeta() = default;
  P2PMeta(const P2PPartners *partners) { reset(partners); }

  void reset(const P2PPartners *partners)
  {
    m_send_meta.clear();
    m_recv_meta.clear();
    m_requests.clear();

    m_partners = partners;
    m_send_meta.resize(2 * partners->nDest(), 0);
    m_recv_meta.resize(2 * partners->nSrc(), 0);
    m_requests.resize(partners->nDest());
  }

  // Usage: schedule_send(), recv_size() ... tally_recvs(), send(), recv()

  void schedule_send(int destIdx, int size, int offset) {
    send_sizes()[destIdx] = size;
    send_offsets()[destIdx] = offset;
  }

  void recv_size(int srcIdx, int size) {
    recv_sizes()[srcIdx] = size;
  }

  void tally_recvs() {
    int sum = 0;
    for (int i = 0; i < m_partners->nSrc(); ++i) {
      recv_offsets()[i] = sum;
      sum += recv_sizes()[i];
    }
    m_recv_total = sum;
  }

  template <typename X>
  void send(const X *send_buffer) {
    for (int i = 0; i < m_partners->nDest(); ++i)
    {
      par::Mpi_Isend(&send_buffer[send_offsets()[i]],
                     send_sizes()[i],
                     dest(i), 0, comm(), &m_requests[i]);
      m_bytes_sent += send_sizes()[i] * sizeof(X);
    }
  }

  template <typename X>
  void recv(X *recv_buffer) {
    MPI_Status status;
    for (int i = 0; i < m_partners->nSrc(); ++i)
    {
      par::Mpi_Recv(&recv_buffer[recv_offsets()[i]],
                    recv_sizes()[i],
                    src(i), 0, comm(), &status);
      m_bytes_rcvd += recv_sizes()[i] * sizeof(X);
    }
  }

  template <typename X>
  void send(const std::vector<X> &send_buffer) { send(&(*send_buffer.begin())); }

  template <typename X>
  void recv(const std::vector<X> &recv_buffer) {
    recv_buffer.resize(recv_total());
    recv(&(*recv_buffer.begin()));
  }

  void wait_all() {
    for (MPI_Request &request : m_requests)
      MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
};







template <int nbuckets>
using Buckets = std::array<size_t, nbuckets + 1>;


// imported from restart:test/restart/restart.cpp
// future: move implementation to where it belongs.
template <typename T, unsigned int dim>
inline Buckets<nchild(dim)+1> bucket_sfc(
    TreeNode<T, dim> *xs, size_t begin, size_t end, int child_level, const SFC_State<int(dim)> sfc)
{
  using X = TreeNode<T, dim>;
  constexpr int nbuckets = nchild(dim) + 1;
  Buckets<nbuckets> sfc_buckets = {};            // Zeros
  {
    Buckets<nbuckets> buckets = {};
    const auto pre_bucket = [=](const X &oct) -> int {
        return (oct.getLevel() >= child_level) + oct.getMortonIndex(child_level);
    };  // Easy to compute bucket, will permute counts later.

    for (size_t i = begin; i < end; ++i)         // Count.
      ++buckets[pre_bucket(xs[i])];

    // Permuted prefix sum.
    buckets[1 + sfc.child_num(sfc::SubIndex(0))] += buckets[0];
    for (sfc::SubIndex s(1); s < nchild(dim); ++s)
      buckets[1 + sfc.child_num(s)] += buckets[1 + sfc.child_num(s.minus(1))];

    static std::vector<X> copies;
    copies.resize(end - begin);

    for (size_t i = end; i-- > begin; ) // backward
      copies[--buckets[pre_bucket(xs[i])]] = xs[i];
    for (size_t i = begin; i < end; ++i)
      xs[i] = copies[i - begin];

    for (sfc::SubIndex s(0); s < nchild(dim); ++s)    // Permute.
      sfc_buckets[1 + s] = begin + buckets[1 + sfc.child_num(s)];
    sfc_buckets[0] = begin;
    sfc_buckets[nbuckets] = end;
  }
  return sfc_buckets;
}


class DistPartPlot
{
  private:
    MPI_Comm m_comm;
    int m_comm_size;
    int m_comm_rank;

    long long unsigned m_Ng;
    int m_fine_level;
    int m_row = 0;
    int m_obj = 0;

    std::ofstream m_file;
    std::string m_root_name;

    static const char * color(int r) {
        const int NCOLORS = 8;
        const char * const palette[NCOLORS] =
            {"E41A1C",
             "377EB8",
             "4DAF4A",
             "984EA3",
             "FF7F00",
             "FFFF33",
             "A65628",
             "F781BF"};
        return palette[r % NCOLORS];
    }

    static const char * white() { return "FFFFFF"; }
    static const char * black() { return "000000"; }
    static const char * dark_grey() { return "333333"; }
    static const char * light_grey() { return "AAAAAA"; }

    struct X { double min; double max;  explicit X(double m, double M) : min(m), max(M) {} };
    struct Y { double min; double max;  explicit Y(double m, double M) : min(m), max(M) {} };

    void rectangle_obj(X x, Y y, const char *color, int object)
    {
      const double pad = 0;
      m_file << "set object " << 1 + object << " rect from "
             << x.min + pad << "," << y.min + pad << " to "
             << x.max - pad << "," << y.max - pad
             << " back"
             << " fillcolor rgb \"#" << color << "\""
             << " linewidth 1"
             << "\n";
    }

    void rectangle_all(X x, Y y, const char *color)
    {
      rectangle_obj(x, y, color, m_obj + m_comm_rank);
      m_obj += m_comm_size;
    }

    void rectangle_root(X x, Y y, const char *color)
    {
      if (m_comm_rank == 0)
        rectangle_obj(x, y, color, m_obj);
      m_obj++;
    }

  public:
    // DistPartPlot()
    DistPartPlot(long long unsigned Ng, int nblocks, int fine_level, const std::string &fileprefix, MPI_Comm comm)
      : m_comm(comm),
        m_Ng(Ng),
        m_fine_level(fine_level),
        m_root_name(fileprefix + "_root.txt")
    {
      MPI_Comm_size(comm, &m_comm_size);
      MPI_Comm_rank(comm, &m_comm_rank);

      if (m_comm_rank == 0)
      {
        std::ofstream rootFile(m_root_name);
        rootFile << "set title \"" << fileprefix << "\"\n";
        for (int r = 0; r < m_comm_size; ++r)
          rootFile << "load \"" << fileprefix << "_" << r << ".txt\"\n";
        /// rootFile << "set size square\n";
        rootFile << "set key off\n";
        rootFile << "set xrange [0:" << m_Ng << "]\n";
        rootFile << "set yrange [" << -1 - fine_level << ":" << 1 + fine_level << "]\n";
        rootFile << "plot 0\n";
        rootFile << "pause mouse keypress\n";
        rootFile.close();
      }

      m_file.open(fileprefix + "_" + std::to_string(m_comm_rank) + ".txt");

      // Splitters.
      const auto ideal = [=](int blk) { return blk * Ng / nblocks; };
      for (int blk = 0; blk < nblocks; ++blk)
      {
        rectangle_root(X(ideal(blk), ideal(blk+1)), Y(0,1), white());
        rectangle_root(X(ideal(blk), ideal(blk+1)), Y(-1,0), white());
      }
    }

    // close()
    void close()
    {
      m_file.close();

      if (m_comm_rank == 0)
        fprintf(stderr, "Run `gnuplot %s`\n", m_root_name.c_str());
    }

    // ~DistPartPlot()
    ~DistPartPlot()  { close(); }

    using LLU = long long unsigned;

    void row(const std::vector<LLU> &global_begin,
             const std::vector<LLU> &global_end,
             const std::vector<LLU> &local_size,
             const std::vector<LLU> &process_offset_end)
    {
      const size_t size = global_begin.size();
      for (size_t i = 0; i < size; ++i)
      {
        rectangle_root(X(global_begin[i], global_end[i]), Y(1+m_row, 1+m_row+1), light_grey());
        rectangle_all(X(global_begin[i] + process_offset_end[i] - local_size[i],
                        global_begin[i] + process_offset_end[i]),
                      Y(-1-m_row-1, -1-m_row),
                      color(m_comm_rank));
      }
      ++m_row;
    }
};




#define DEBUG_BUCKET_ARRAY 0

template <typename T, int dim>
struct BucketRef
{
  using LLU = long long unsigned;
  LLU              & local_begin;
  LLU              & local_end;
  LLU              & global_begin;
  LLU              & global_end;
  TreeNode<T, unsigned(dim)> & octant;
  SFC_State<dim>   & sfc;
  char             & split;
#if DEBUG_BUCKET_ARRAY
  LLU & local_size;
  LLU & process_offset_end;  // MPI_Scan over local_size
#endif

  void mark_split()         { split = true; }
  bool marked_split() const { return split; }
};

template <typename T, int dim>
struct BucketArray
{
  using LLU = long long unsigned;
  std::vector<LLU>              m_local_begin;
  std::vector<LLU>              m_local_end;
  std::vector<LLU>              m_global_begin;
  std::vector<LLU>              m_global_end;
  std::vector<TreeNode<T, unsigned(dim)>> m_octant;
  std::vector<SFC_State<dim>>   m_sfc;
  std::vector<char>             m_split;
#if DEBUG_BUCKET_ARRAY
  std::vector<LLU>              m_local_size;
  std::vector<LLU>              m_process_offset_end;
#endif
  static LLU s_allreduce_sz;
  static LLU s_allreduce_ct;

  BucketArray() = default;

  static void reset_log() { s_allreduce_sz = 0;  s_allreduce_ct = 0; }
  static LLU allreduce_sz() { return s_allreduce_sz; }
  static LLU allreduce_ct() { return s_allreduce_ct; }

  BucketRef<T, dim> ref(size_t i)
  {
    return BucketRef<T, dim>{ m_local_begin[i],
                              m_local_end[i],
                              m_global_begin[i],
                              m_global_end[i],
                              m_octant[i],
                              m_sfc[i],
                              m_split[i],
#if DEBUG_BUCKET_ARRAY
                              m_local_size[i],
                              m_process_offset_end[i],
#endif
    };
  }

  void reserve(size_t capacity)
  {
    m_local_begin.reserve(capacity);
    m_local_end.reserve(capacity);
    m_global_begin.reserve(capacity);
    m_global_end.reserve(capacity);
    m_octant.reserve(capacity);
    m_sfc.reserve(capacity);
    m_split.reserve(capacity);
#if DEBUG_BUCKET_ARRAY
    m_local_size.reserve(capacity);
    m_process_offset_end.reserve(capacity);
#endif
  }

  void reset()
  {
    m_local_begin.clear();
    m_local_end.clear();
    m_global_begin.clear();
    m_global_end.clear();
    m_octant.clear();
    m_sfc.clear();
    m_split.clear();
#if DEBUG_BUCKET_ARRAY
    m_local_size.clear();
    m_process_offset_end.clear();
#endif
  }

  void reset(size_t size, TreeNode<T, unsigned(dim)> octant, SFC_State<dim> sfc)
  {
    reset();
    m_local_begin.push_back(0);
    m_local_end.push_back(size);
    m_global_begin.push_back(0);
    m_global_end.push_back(0);
    m_octant.push_back(octant);
    m_sfc.push_back(sfc);
    m_split.push_back(false);
#if DEBUG_BUCKET_ARRAY
    m_local_size.push_back(size);
    m_process_offset_end.push_back(0);
#endif
  }

  void push_children(const BucketRef<T, dim> parent, Buckets<1+nchild(dim)> &children)
  {
    const TreeNode<T, unsigned(dim)> oct = parent.octant;
    const SFC_State<dim> sfc = parent.sfc;
    for (sfc::SubIndex s(0); s < nchild(dim); ++s)
    {
      m_local_begin.push_back(children[1+s]);  // assume ancestors front
      m_local_end.push_back(children[1+s+1]);
      m_global_begin.push_back(0);
      m_global_end.push_back(0);
      m_octant.push_back(oct.getChildMorton(sfc.child_num(s)));
      m_sfc.push_back(sfc.subcurve(s));
      m_split.push_back(false);
#if DEBUG_BUCKET_ARRAY
      m_local_size.push_back(children[1+s+1] - children[1+s]);
      m_process_offset_end.push_back(0);
#endif
    }
  }

  void all_reduce(MPI_Comm comm)
  {
    par::Mpi_Allreduce(&(*m_local_begin.begin()), &(*m_global_begin.begin()), size(), MPI_SUM, comm);
    par::Mpi_Allreduce(&(*m_local_end.begin()), &(*m_global_end.begin()), size(), MPI_SUM, comm);
#if DEBUG_BUCKET_ARRAY
    par::Mpi_Scan(&(*m_local_size.begin()), &(*m_process_offset_end.begin()), size(), MPI_SUM, comm);
#endif
    s_allreduce_sz += size();
    s_allreduce_ct += 2;
  }

  void plot(DistPartPlot &plot)
  {
#if DEBUG_BUCKET_ARRAY
    plot.row(m_global_begin, m_global_end, m_local_size, m_process_offset_end);
#endif
  }

  size_t size() const { return m_local_begin.size(); }

  struct Iterator
  {
    BucketArray & m_bucket_array;
    size_t m_i;

    Iterator & operator++() { ++m_i; return *this; }
    BucketRef<T, dim> operator*() { return m_bucket_array.ref(m_i); }
    bool operator!=(const Iterator &that) const { return m_i != that.m_i; }
  };

  Iterator begin() { return Iterator{*this, 0u}; }
  Iterator end()   { return Iterator{*this, size()}; }
};
template <typename T, int dim>  long long unsigned BucketArray<T, dim>::s_allreduce_sz = 0;
template <typename T, int dim>  long long unsigned BucketArray<T, dim>::s_allreduce_ct = 0;

#undef DEBUG_BUCKET_ARRAY


template void distTreePartition_kway<unsigned, 2u>( MPI_Comm comm,
    std::vector<TreeNode<unsigned, 2u>> &octants,
    const double sfc_tol, const int kway,
    const TreeNode<unsigned, 2u> root,
    const SFC_State<2> sfc);
template void distTreePartition_kway<unsigned, 3u>( MPI_Comm comm,
    std::vector<TreeNode<unsigned, 3u>> &octants,
    const double sfc_tol, const int kway,
    const TreeNode<unsigned, 3u> root,
    const SFC_State<3> sfc);
template void distTreePartition_kway<unsigned, 4u>( MPI_Comm comm,
    std::vector<TreeNode<unsigned, 4u>> &octants,
    const double sfc_tol, const int kway,
    const TreeNode<unsigned, 4u> root,
    const SFC_State<4> sfc);

template <typename...Args>
void tfprintf(Args...args) { const bool use = false; if (use) fprintf(args...); }

template <typename T, unsigned int dim>
void distTreePartition_kway(
    MPI_Comm comm,
    std::vector<TreeNode<T, dim>> &octants,
    const double sfc_tol,
    const int kway,
    TreeNode<T, dim> root,
    SFC_State<int(dim)> sfc)
{
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);
  int nblocks = std::min(comm_size, kway);

  const MPI_Comm comm_in = comm;
  const int comm_size_in = comm_size;
  const int comm_rank_in = comm_rank;

  using LLU = long long unsigned;
  const size_t kway_roundup = binOp::next_power_of_pow_2_dim<dim>(kway);
  assert(kway_roundup > 0);

  BucketArray<T, int(dim)> parent_buckets,  child_buckets;
  parent_buckets.reserve(kway_roundup * nchild(dim));
  child_buckets.reserve(kway_roundup * nchild(dim));

  P2PPartners p2p_partners;  p2p_partners.reserve(kway, 2 * kway);
  P2PScalar<> p2p_sizes;     p2p_sizes.reserve(kway, 2 * kway);
  P2PMeta p2p_meta;          p2p_meta.reserve(kway, 2 * kway);

  std::vector<MPI_Comm> to_be_freed;

  const auto bucket_split = [](std::vector<TreeNode<T, dim>> &v, BucketRef<T, int(dim)> b)
  {
    return bucket_sfc(
        &(*v.begin()),
        b.local_begin,
        b.local_end,
        b.octant.getLevel() + 1,
        b.sfc);
    //future: use locate instead of bucketing if octants is already sorted.
  };

  // Splitters.
  std::vector<size_t> local_block;
  local_block.reserve(kway + 1);

  std::vector<TreeNode<T, dim>> received_octants;

  std::string plot_prefix = "partition";

  while (nblocks > 1)
  {
    LLU Ng;
    LLU const Nl = octants.size();
    par::Mpi_Allreduce(&Nl, &Ng, 1, MPI_SUM, comm);

    received_octants.reserve((Ng + comm_size_in - 1) / comm_size_in * (1 + sfc_tol * 2));

    parent_buckets.reset(octants.size(), root, sfc);
    child_buckets.reset();


    // Initial buckets.
    int depth = 0;
    for (; depth + root.getLevel() < m_uiMaxDepth and (1 << (depth*dim)) < nblocks; ++depth)
    {
      child_buckets.reset();
      for (BucketRef<T, int(dim)> b : parent_buckets)
      {
        const size_t split_sz = nchild(dim) + 1;
        Buckets<split_sz> split = bucket_split(octants, b);
        child_buckets.push_children(b, split);
      }
      std::swap(child_buckets, parent_buckets);
    }
    const int initial_depth = depth;

    /// DistPartPlot plot(Ng, nblocks, m_uiMaxDepth - initial_depth, plot_prefix, comm);

    // Allreduce parents to get global begins of parents.
    parent_buckets.all_reduce(comm);
    /// parent_buckets.plot(plot);

    // Mapping of splitters within array, induces mapping within buckets.
    const auto ideal =    [=](int blk) { assert(blk <= nblocks); return blk * Ng / nblocks; };
    const auto ideal_sz = [=](int blk) { assert(blk <= nblocks); return ideal(blk + 1) - ideal(blk); };
    const auto next_blk = [=](LLU item) { assert(item <= Ng);    return int((item * nblocks + Ng - 1) / Ng); };
    const LLU min_tol =  Ng                / nblocks * sfc_tol;
    const LLU max_tol = (Ng + nblocks - 1) / nblocks * sfc_tol;

    const auto too_wide = [=](LLU begin, LLU end) -> bool {
      const bool wider_than_margins = end - begin > 2 * min_tol + 1;
      const LLU margin_left = begin + min_tol < end ? begin + min_tol + 1 : end;
      const LLU margin_right = end >= min_tol ? end - min_tol : 0;
      const int far_blk_begin = next_blk(margin_left);
      const int far_blk_end = next_blk(margin_right);
      return wider_than_margins and far_blk_begin < far_blk_end;
    };

    const auto local_block_sz = [&](int blk) {
      assert(blk < nblocks);
      return local_block[blk+1] - local_block[blk];
    };

    // Mapping of tasks within blocks.
    const auto block_to_task =  [=](int blk) { return blk * comm_size / nblocks; };
    const auto block_tasks =    [=](int blk) { return block_to_task(blk+1) - block_to_task(blk); };
    const auto blk_id_to_task = [=](int blk, int blk_id) { return block_to_task(blk) + blk_id; };

    const auto task_to_block_ =  [=](int task) { return ((task + 1) * nblocks - 1) / comm_size; };
    const auto task_to_block = [=](int task) {
      int blk = task_to_block_(task);
      assert(block_to_task(blk) <= task and task < block_to_task(blk+1));
      return blk;
    };

    const int self_blk = task_to_block(comm_rank);
    const int self_blk_id = comm_rank - block_to_task(self_blk);

    const auto dest_blk_id =  [=](int blk) { return self_blk_id % block_tasks(blk); };//future: more balanced
    const auto src_blk_id =   [=](int blk, int src) { return src == 0 ? self_blk_id : block_tasks(self_blk); };
    const auto srcs_per_blk = [=](int blk) {
      if (self_blk_id == block_tasks(blk)) return 0;
      if (self_blk_id == 0 and block_tasks(blk) > block_tasks(self_blk)) return 2;
      else return 1;
    };

    // Splitters
    local_block.clear();
    local_block.resize(nblocks + 1, -1);
    local_block[nblocks] = octants.size();

    // commit()
    const auto commit = [&](int blk, const BucketRef<T, int(dim)> b) {
        assert(blk < nblocks);
        assert(b.global_begin <= ideal(blk) and ideal(blk) <= b.global_end);
        const LLU dist_begin = ideal(blk) - b.global_begin;
        const LLU dist_end = b.global_end - ideal(blk);
        local_block[blk] = (dist_begin <= dist_end ? b.local_begin : b.local_end);
    };

    // Keep splitting buckets to tolerance or until max depth reached.
    for (; parent_buckets.size() > 0 and depth + root.getLevel() < m_uiMaxDepth; ++depth)
    {
      child_buckets.reset();

      // If all splitters acceptable or there are none, commit. Else, split.
      for (BucketRef<T, int(dim)> b : parent_buckets)
      {
        if (not too_wide(b.global_begin, b.global_end))
        {
          const int begin = next_blk(b.global_begin);
          const int end = next_blk(b.global_end);
          for (int blk = begin; blk < end; ++blk)
            commit(blk, b);
        }
        else
        {
          b.mark_split();
          const size_t split_sz = nchild(dim) + 1;
          Buckets<split_sz> split = bucket_split(octants, b);
          child_buckets.push_children(b, split);
        }
      }

      // Allreduce children to get global begins of children.
      child_buckets.all_reduce(comm);
      /// child_buckets.plot(plot);

      // Commit any splitters that are not inheritted by children.
      size_t cb = 0;
      for (BucketRef<T, int(dim)> b : parent_buckets)
        if (b.marked_split())
        {
          const int parent_begin = next_blk(b.global_begin);
          const int child_begin = next_blk(child_buckets.ref(cb).global_begin);
          for (int blk = parent_begin; blk < child_begin; ++blk)
            commit(blk, b);
          cb += nchild(dim);
        }

      std::swap(parent_buckets, child_buckets);
    }

    // If ran out of levels, commit any remaining splitters.
    for (const BucketRef<T, int(dim)> b : parent_buckets)
    {
      const int begin = next_blk(b.global_begin);
      const int end = next_blk(b.global_end);
      for (int blk = begin; blk < end; ++blk)
        commit(blk, b);
    }

    // Validate splitters.
    assert((std::find(local_block.begin(), local_block.end(), -1) == local_block.end()));
    assert(std::is_sorted(local_block.begin(), local_block.end()));


    // Exchange data to match block boundaries.

    int total_srcs = 0;
    for (int blk = 0; blk < nblocks; ++blk)
      if (blk != self_blk)
        total_srcs += srcs_per_blk(blk);

    assert(par::mpi_sum(total_srcs, comm) == comm_size * (nblocks - 1));

    p2p_partners.reset(nblocks - 1, total_srcs, comm);
    p2p_sizes.reset(&p2p_partners);
    p2p_meta.reset(&p2p_partners);

    for (int blk = 0, dst_idx = 0; blk < nblocks; ++blk)
      if (blk != self_blk)
      {
        p2p_partners.dest(dst_idx, blk_id_to_task(blk, dest_blk_id(blk)));
        p2p_sizes.send(dst_idx, local_block_sz(blk));
        p2p_meta.schedule_send(dst_idx, local_block_sz(blk), local_block[blk]);
        dst_idx++;
      }

    for (int blk = 0, src_idx = 0; blk < nblocks; ++blk)
      if (blk != self_blk)
        for (int s = 0; s < srcs_per_blk(blk); ++s)
        {
          p2p_partners.src(src_idx, blk_id_to_task(blk, src_blk_id(blk, s)));
          p2p_meta.recv_size(src_idx,  p2p_sizes.recv(src_idx));
          src_idx++;
        }

    p2p_meta.tally_recvs();

    p2p_meta.send(octants);

    received_octants.clear();
    received_octants.resize(p2p_meta.recv_total());

    // Copy local segment to end.
    received_octants.insert(received_octants.end(),
        &octants[local_block[self_blk]],
        &octants[local_block[self_blk+1]]);

    assert(par::mpi_sum(LLU(received_octants.size()), comm) == Ng);

    // Receive remote segements into beginning.
    p2p_meta.recv(&received_octants[0]);  // ptr: Don't resize vector
    p2p_sizes.wait_all();
    p2p_meta.wait_all();

    plot_prefix += "_" + std::to_string(self_blk);

    std::swap(octants, received_octants);

    if (comm_size > kway)
    {
      MPI_Comm new_comm;
      MPI_Comm_split(comm, self_blk, self_blk_id, &new_comm);
      to_be_freed.push_back(new_comm);

      comm = new_comm;
      MPI_Comm_size(comm, &comm_size);
      MPI_Comm_rank(comm, &comm_rank);
      nblocks = std::min(comm_size, kway);
    }
    else
      break;  // Avoid splitting comm at the very end.
  }

  /// // For per-call stats, use BucketArray<T, dim>::reset_log() before bucketing.
  /// par::MinMeanMax<LLU>
  ///     allreduce_ct = par::Mpi_ReduceMinMeanMax(BucketArray<T, dim>::allreduce_ct(), comm_in),
  ///     allreduce_sz = par::Mpi_ReduceMinMeanMax(BucketArray<T, dim>::allreduce_sz(), comm_in);
  /// if (comm_rank_in == 0)
  ///   printf("NP=%-5d \t ct=%llu/%llu/%llu \t sz=%llu/%llu/%llu\n",
  ///       comm_size_in,
  ///       allreduce_ct.m_glob_min, LLU(allreduce_ct.m_glob_mean), allreduce_ct.m_glob_max,
  ///       allreduce_sz.m_glob_min, LLU(allreduce_sz.m_glob_mean), allreduce_sz.m_glob_max);

  /// par::MinMeanMax<LLU> exchange_sz = par::Mpi_ReduceMinMeanMax(p2p_meta.bytes_rcvd(), comm_in);
  /// if (comm_rank_in == 0)
  ///   printf("kway=%-3d \t NP=%-5d \t sz=%llu/%llu/%llu\n",
  ///       kway, comm_size_in, exchange_sz.m_glob_min, LLU(exchange_sz.m_glob_mean), exchange_sz.m_glob_max);

  for (MPI_Comm new_comm : to_be_freed)
    MPI_Comm_free(&new_comm);
}



template<typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: distTreePartition(std::vector<TreeNode<T,dim>> &points,
                          unsigned int noSplitThresh,
                          double loadFlexibility,
                          MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  if (nProc == 1)
    return;

  par::SendRecvSchedule sched = distTreePartitionSchedule(points, noSplitThresh, loadFlexibility, comm);

  std::vector<TreeNode<T,dim>> origPoints = points;   // Sendbuffer is a copy.

  size_t sizeNew = sched.rdispls.back() + sched.rcounts.back();
  points.resize(sizeNew);

  par::Mpi_Alltoallv<TreeNode<T,dim>>(
      &origPoints[0], &sched.scounts[0], &sched.sdispls[0],
      &points[0],     &sched.rcounts[0], &sched.rdispls[0],
      comm);

  //Future: Use Mpi_Alltoallv_Kway()

  // After this process, distTreeSort or distTreeConstruction
  // picks up with a local sorting or construction operation.
  // TODO Need to have the global buckets for that to work.
}

template<typename T, unsigned int dim>
par::SendRecvSchedule
SFC_Tree<T,dim>:: distTreePartitionSchedule(std::vector<TreeNode<T,dim>> &points,
                          unsigned int noSplitThresh,
                          double loadFlexibility,
                          MPI_Comm comm)
{

  // -- Don't worry about K splitters for now, we'll add that later. --

  // The goal of this function, as explained in Fernando and Sundar's paper,
  // is to refine the list of points into finer sorted buckets until
  // the load-balancing criterion has been met. Therefore the hyperoctree is
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
    // Note: distTreePartition() is only responsible for partitioning the points,
    // which is a no-op with nProc==1.

    /// locTreeSort(&(*points.begin()), 0, points.size(), 0, m_uiMaxDepth, SFC_State<dim>::root());
    return par::SendRecvSchedule{};
  }

  using TreeNode = TreeNode<T,dim>;
  constexpr char numChildren = nchild(dim);

  // The outcome of the BFT will be a list of splitters, i.e. refined buckets.
  std::vector<unsigned int> splitters(nProc, 0);

  std::vector<BucketInfo<RankI>> newBuckets, allBuckets, mergedBuckets;
  std::vector<RankI> newCountsL, newCountsG, allCountsG, allCountsGScan, mergedCountsG;
  std::vector<RankI> splitBucketIndex;

  // TODO a more efficient way to do splitterBucketGrid when nProc is large.
  
  // Phase 1: move down the levels until we have roughly enough buckets
  //   to test our load-balancing criterion.
  const int initNumBuckets = nProc;
  newBuckets.push_back(BucketInfo<RankI>::root(points.size()));
        // No-runaway, in case we run out of points.
        // It is `<' because refining m_uiMaxDepth would make (m_uiMaxDepth+1).

  // @masado: There is a condition that you need to handle, what is initNumBuckets cannot be reached my the m_uiMaxDepth. 
  // May be you need to perform, communicator split, or something. This can cause problems for larger nProc. - Milinda.  
  while (newBuckets.size() < initNumBuckets && newBuckets[0].lev < m_uiMaxDepth)
  {
    // Prep for phase 2.
    splitBucketIndex.resize(newBuckets.size());
    std::iota(splitBucketIndex.begin(), splitBucketIndex.end(), 0);
    allBuckets = newBuckets;

    // Actual work of phase 1.
    treeBFTNextLevel(&(*points.begin()), newBuckets);
  }
  // Remark: Due to the no-runaway clause, we are not guaranteed
  // that bftQueue actually holds `initNumBuckets' buckets. Is this important?
  // @masado : You need to handdle this case, - Milinda. 
  // TODO what are the repercussions of having too few buckets?

  // Phase 2: Count bucket sizes, communicate bucket sizes,
  //   test load balance, select buckets and refine, repeat.
  RankI sizeG, sizeL = points.size();
  par::Mpi_Allreduce<RankI>(&sizeL, &sizeG, 1, MPI_SUM, comm);

  while (newBuckets.size() > 0)
  {
    // Count sizes of new buckets locally and globally.
    newCountsL.resize(newBuckets.size());
    newCountsG.resize(newBuckets.size());
    for (unsigned int ii = 0; ii < newBuckets.size(); ii++)
      newCountsL[ii] = newBuckets[ii].end - newBuckets[ii].begin;
    par::Mpi_Allreduce<RankI>(&(*newCountsL.begin()), &(*newCountsG.begin()),
        (int) newCountsL.size(), MPI_SUM, comm);

    // Merge new buckets with the rest of the buckets.
    unsigned int splitIndex=0;
    unsigned int bIndex=0;
    for (unsigned int ii = 0; ii < allBuckets.size(); ii++)
    {
      if( (splitIndex < splitBucketIndex.size()) && (ii==splitBucketIndex[splitIndex]))
      {
        if(allBuckets[ii].lev<m_uiMaxDepth)
        { 
          // is actually splitted. 
          assert( (bIndex + numChildren) <= newBuckets.size() );
          for(unsigned int w=bIndex; w < (bIndex + numChildren); w++)
          {
            mergedBuckets.push_back(newBuckets[w]);
            mergedCountsG.push_back(newCountsG[w]);
          }
          
          bIndex+=numChildren;
        }
        else
        {
          mergedBuckets.push_back(allBuckets[ii]); 
          mergedCountsG.push_back(allCountsG[ii]);
        }
        splitIndex++;
      }
      else
      {
        mergedBuckets.push_back(allBuckets[ii]);
        mergedCountsG.push_back(allCountsG[ii]);
      }
    }
    std::swap(mergedBuckets, allBuckets);
    std::swap(mergedCountsG, allCountsG);
    mergedBuckets.clear();
    mergedCountsG.clear();

    // Inclusive scan of bucket counts before comparing w/ ideal splitters.
    allCountsGScan.resize(allCountsG.size());
    allCountsGScan[0] = allCountsG[0];
    for(unsigned int k=1;k<allCountsG.size();k++)
      allCountsGScan[k] = allCountsGScan[k-1] + allCountsG[k];

    newBuckets.clear();
    splitBucketIndex.clear();

    // Index the buckets that need to be refined.
    RankI idealLoadBalance=0;
    for(int i=0;i<nProc-1;i++)
    {
      idealLoadBalance+=((i+1)*sizeG/nProc -i*sizeG/nProc);
      DendroIntL toleranceLoadBalance = ((i+1)*sizeG/nProc -i*sizeG/nProc) * loadFlexibility;
      unsigned int loc=(std::lower_bound(allCountsGScan.begin(), allCountsGScan.end(), idealLoadBalance) - allCountsGScan.begin());

      if((abs(allCountsGScan[loc]-idealLoadBalance) > toleranceLoadBalance) && (allBuckets[loc].lev < m_uiMaxDepth))
      {
        if(!splitBucketIndex.size()  || splitBucketIndex.back()!=loc)
          splitBucketIndex.push_back(loc);
      }
      else
      {
        if ((loc + 1) < allBuckets.size())
          splitters[i] = allBuckets[loc + 1].begin;
        else
          splitters[i] = allBuckets[loc].begin;
        //TODO probably easier to just use allBuckets[loc].end
      }
    }
    splitters[nProc-1] = points.size();

    // Filter the buckets that need to be refined, using (a subset of) splitBucketIndex[].
    for (int k = 0; k < splitBucketIndex.size(); k++)
    {
      const BucketInfo<RankI> b = allBuckets[splitBucketIndex[k]];
      if (b.lev < m_uiMaxDepth)
        newBuckets.push_back(b);
    }

    // Refining step.
    treeBFTNextLevel(&(*points.begin()), newBuckets);
  }

  /// //
  /// // Adjust splitters to respect noSplitThresh.
  /// //
  /// // Look for a level such that 0 < globSz[lev] <= noSplitThresh and globSz[lev-1] > noSplitThresh.
  /// // If such a level exists, then set final level to lev-1.
  /// for (int r = 0; r < nProc; r++)
  /// {
  ///   LevI lLev = finalSplitterLevels[r];
  ///   LevI pLev = (lLev > 0 ? lLev - 1 : lLev);

  ///   if (0 < splitterCountsGrid[lLev * nProc + r])
  ///   {
  ///     while (pLev > 0 && splitterCountsGrid[pLev * nProc + r] <= noSplitThresh)
  ///       pLev--;

  ///     if (splitterCountsGrid[lLev * nProc + r] <= noSplitThresh &&
  ///         splitterCountsGrid[pLev * nProc + r] > noSplitThresh)
  ///       finalSplitterLevels[r] = pLev;
  ///   }
  /// }

  /// // The output of the bucketing is a list of splitters marking ends of partition.
  /// for (int r = 0; r < nProc; r++)
  ///   splitters[r] = splitterBucketGrid[finalSplitterLevels[r] * nProc + r];

  //
  // All to all exchange of the points arrays.
    
  std::vector<int> sendCnt, sendDspl;
  std::vector<int> recvCnt(splitters.size()), recvDspl;
  sendCnt.reserve(splitters.size());
  sendDspl.reserve(splitters.size());
  recvDspl.reserve(splitters.size());
  unsigned int sPrev = 0;

  /*for(unsigned int i=1;i<splitters.size();i++)
  {
    if(splitters[i-1]>splitters[i])
      std::cout<<"rank: "<<rProc<<" spliter["<<(i-1)<<"] : "<<splitters[i-1]<<" < splitter["<<i<<"]: "<<splitters[i]<<std::endl;
  }*/

  for (RankI s : splitters)     // Sequential counting and displacement.
  {
    //DEBUG
    if ((long) s - (long) sPrev < 0)
      fprintf(stderr, "[%d] Negative count: %lld - %lld\n", rProc, (long long) s, (long long) sPrev);

    sendDspl.push_back(sPrev);
    assert((s - sPrev) >=0);
    sendCnt.push_back(s - sPrev);
    sPrev = s;
  }
  par::Mpi_Alltoall<int>(&(*sendCnt.begin()), &(*recvCnt.begin()), 1, comm);
  sPrev = 0;
  for (RankI c : recvCnt)       // Sequential scan.
  {
    recvDspl.push_back(sPrev);
    sPrev += c;
  }

  par::SendRecvSchedule sched;
  sched.scounts = sendCnt;
  sched.sdispls = sendDspl;
  sched.rcounts = recvCnt;
  sched.rdispls = recvDspl;

  return sched;
}


template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: treeBFTNextLevel(TreeNode<T,dim> *points,
      std::vector<BucketInfo<RankI>> &bftQueue)
{
  if (bftQueue.size() == 0)
    return;

  const LevI startLev = bftQueue[0].lev;

  using TreeNode = TreeNode<T,dim>;
  constexpr char numChildren = nchild(dim);
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
    {
      SFC_bucketing(points, front.begin, front.end, front.lev+1, SFC_State<dim>(front.rot), childSplitters, ancStart, ancEnd);

      // Put 'ancestor' points one of the closest sibling bucket.
      const bool ancestorsFirst = true;
      if (ancestorsFirst)
        childSplitters[0] = ancStart;
      else
        childSplitters[numChildren] = ancEnd;
    }
    else
    {
      childSplitters.fill(front.begin);  // Don't need to sort an empty selection, it's just empty.
    }

    // Enqueue our children in the next level.
    const SFC_State<dim> sfc(front.rot);
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      BucketInfo<RankI> childBucket = {
          sfc.subcurve(child_sfc).state(),
          front.lev+1,
          childSplitters[child_sfc+0],
          childSplitters[child_sfc+1] };

      bftQueue.push_back(childBucket);
    }
  }
}


//
// locTreeConstruction()
//
template <typename T, unsigned int dim>
void
locTreeConstruction_rec(TreeNode<T,dim> *points,
                                  std::vector<TreeNode<T,dim>> &tree,
                                  RankI maxPtsPerRegion,
                                  RankI begin, RankI end,
                                  LevI sLev,
                                  LevI eLev,
                                  SFC_State<dim> sfc,
                                  TreeNode<T,dim> pNode);

template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: locTreeConstruction(TreeNode<T,dim> *points,
                                  std::vector<TreeNode<T,dim>> &tree,
                                  RankI maxPtsPerRegion,
                                  RankI begin, RankI end,
                                  LevI sLev,
                                  LevI eLev,
                                  SFC_State<dim> sfc,
                                  TreeNode<T,dim> pNode)
{
  locTreeConstruction_rec<T, dim>(points, tree, maxPtsPerRegion, begin, end, sLev, eLev, sfc, pNode);
}

template <typename T, unsigned int dim>
void
locTreeConstruction_rec(TreeNode<T,dim> *points,
                                  std::vector<TreeNode<T,dim>> &tree,
                                  RankI maxPtsPerRegion,
                                  RankI begin, RankI end,
                                  LevI sLev,
                                  LevI eLev,
                                  SFC_State<dim> sfc,
                                  TreeNode<T,dim> pNode)
{
  // Most of this code is copied from locTreeSort().

  if (end <= begin) { return; }

  constexpr char numChildren = nchild(dim);

  // Reorder the buckets on sLev (current level).
  std::array<RankI, numChildren+1> tempSplitters;
  RankI ancStart, ancEnd;
  SFC_Tree<T, dim>::SFC_bucketing(points, begin, end, sLev, sfc, tempSplitters, ancStart, ancEnd);
  // The array `tempSplitters' has numChildren+2 slots, which includes the
  // beginning, middles, and end of the range of children, and ancestors are in front.

  TreeNode<T,dim> cNode = pNode.getFirstChildMorton();

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    // We satisfy the completeness property because we iterate over
    // all possible children here. For each child, append either
    // a leaf orthant or a non-empty complete subtree.
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      cNode.setMortonIndex(sfc.child_num(child_sfc));

      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] > maxPtsPerRegion)
      {
        // Recursively build a complete sub-tree out of this bucket's points.
        // Use the splitters to specify ranges for the next level of recursion.
        locTreeConstruction_rec<T, dim>(
            points, tree, maxPtsPerRegion,
            tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
            sLev+1, eLev,
            sfc.subcurve(child_sfc),
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
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      cNode.setMortonIndex(sfc.child_num(child_sfc));
      tree.push_back(cNode);
    }
  }

}  // end function


template <typename T, unsigned int dim>
void SFC_Tree<T, dim>::locCompleteResolved(
      const TreeNode<T, dim> *octants,
      std::vector<TreeNode<T, dim>> &tree,
      RankI begin, RankI end,
      SFC_State<dim> sfc,
      TreeNode<T,dim> pNode)
{
  if (end <= begin)  // No required resolution.
  {
    tree.push_back(pNode);
    return;
  }

  // Find buckets on sLev (current level).
  std::array<RankI, nchild(dim)+1> tempSplitters;
  RankI ancStart, ancEnd;
  SFC_locateBuckets(octants, begin, end, pNode.getLevel()+1, sfc, tempSplitters, ancStart, ancEnd);

  if (ancStart < ancEnd)        // Reached required resolution.
    tree.push_back(pNode);
  else                          // Keep subdividing.
    for (sfc::SubIndex child_sfc(0); child_sfc < nchild(dim); ++child_sfc)
      locCompleteResolved(
          octants, tree,
          tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
          sfc.subcurve(child_sfc),
          pNode.getChildMorton(sfc.child_num(child_sfc)));
}






//
// locTreeConstructionWithFilter(points)
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: locTreeConstructionWithFilter(
                                  const ibm::DomainDecider &decider,
                                  TreeNode<T,dim> *points,
                                  std::vector<TreeNode<T,dim>> &tree,
                                  RankI maxPtsPerRegion,
                                  RankI begin, RankI end,
                                  LevI sLev,
                                  LevI eLev,
                                  SFC_State<dim> sfc,
                                  TreeNode<T,dim> pNode)
{
  if (end <= begin) { return; }

  constexpr char numChildren = nchild(dim);

  std::array<RankI, numChildren+1> tempSplitters;
  RankI ancStart, ancEnd;
  SFC_bucketing(points, begin, end, sLev, sfc, tempSplitters, ancStart, ancEnd);

  TreeNode<T,dim> cNode = pNode.getFirstChildMorton();

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      cNode.setMortonIndex(sfc.child_num(child_sfc));

      double physCoords[dim];
      double physSize;
      treeNode2Physical(cNode, physCoords, physSize);

      const ibm::Partition childRegion = decider(physCoords, physSize);

      if (childRegion != ibm::IN)
      {
        if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] > maxPtsPerRegion)
        {
          locTreeConstructionWithFilter(
              decider,
              points, tree, maxPtsPerRegion,
              tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
              sLev+1, eLev,
              sfc.subcurve(child_sfc),
              cNode);
        }
        else
        {
          // Append a leaf orthant.
          tree.push_back(cNode);
        }
      }
    }
  }
  else   // We have reached eLev. Violate `maxPtsPerRegion' to satisfy completeness.
  {
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      cNode.setMortonIndex(sfc.child_num(child_sfc));

      double physCoords[dim];
      double physSize;
      treeNode2Physical(cNode, physCoords, physSize);

      const ibm::Partition childRegion = decider(physCoords, physSize);

      if (childRegion != ibm::IN)
        tree.push_back(cNode);
    }
  }

}  // end function


//
// locTreeConstructionWithFilter()
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: locTreeConstructionWithFilter( const ibm::DomainDecider &decider,
                                               bool refineAll,
                                               std::vector<TreeNode<T,dim>> &tree,
                                               LevI sLev,
                                               LevI eLev,
                                               SFC_State<dim> sfc,
                                               TreeNode<T,dim> pNode)
{
  constexpr char numChildren = nchild(dim);

  TreeNode<T,dim> cNode = pNode.getFirstChildMorton();

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      cNode.setMortonIndex(sfc.child_num(child_sfc));

      double physCoords[dim];
      double physSize;
      treeNode2Physical(cNode, physCoords, physSize);

      const ibm::Partition childRegion = decider(physCoords, physSize);

      if (childRegion != ibm::IN)
      {
        if (childRegion == ibm::INTERCEPTED ||
            childRegion == ibm::OUT && refineAll)
        {
          locTreeConstructionWithFilter( decider, refineAll, tree, sLev + 1, eLev, sfc.subcurve(child_sfc), cNode);
        }
        else
        {
          // Append a leaf orthant.
          tree.push_back(cNode);
        }
      }
    }
  }
  else   // sLev == eLev. Append all children.
  {
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      cNode.setMortonIndex(sfc.child_num(child_sfc));

      double physCoords[dim];
      double physSize;
      treeNode2Physical(cNode, physCoords, physSize);

      const ibm::Partition childRegion = decider(physCoords, physSize);

      if (childRegion != ibm::IN)
        tree.push_back(cNode);
    }
  }

}  // end function


template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: distTreeConstruction(std::vector<TreeNode<T,dim>> &points,
                                   std::vector<TreeNode<T,dim>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  tree.clear();

  // The heavy lifting to globally sort/partition.
  distTreePartition(points, maxPtsPerRegion, loadFlexibility, comm);

  // Instead of locally sorting, locally complete the tree.
  // Since we don't have info about the global buckets, construct from the top.
  const LevI leafLevel = m_uiMaxDepth;
  locTreeConstruction(&(*points.begin()), tree, maxPtsPerRegion,
                      0, (RankI) points.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      SFC_State<dim>::root(),
                      TreeNode<T,dim>());
  // When (sLev,eLev)==(0,m_uiMaxDepth), nodes with level m_uiMaxDepth+1 are created.
  // This must be leading to incorrect ancestry tests because duplicates do
  // not always get removed properly in that case.

  // We have now introduced duplicate sections of subtrees at the
  // edges of the partition.

  distRemoveDuplicates(tree, loadFlexibility, false, comm);
}


template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: distTreeConstructionWithFilter(
    const ibm::DomainDecider &decider,
    std::vector<TreeNode<T,dim>> &points,
    std::vector<TreeNode<T,dim>> &tree,
    RankI maxPtsPerRegion,
    double loadFlexibility,
    MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  tree.clear();

  // The heavy lifting to globally sort/partition.
  distTreePartition(points, maxPtsPerRegion, loadFlexibility, comm);

  // Instead of locally sorting, locally complete the tree.
  // Since we don't have info about the global buckets, construct from the top.
  const LevI leafLevel = m_uiMaxDepth;
  locTreeConstructionWithFilter(decider,
                      &(*points.begin()), tree, maxPtsPerRegion,
                      0, (RankI) points.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      SFC_State<dim>::root(),
                      TreeNode<T,dim>());

  distRemoveDuplicates(tree, loadFlexibility, false, comm);
}


template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>::distTreeConstructionWithFilter( const ibm::DomainDecider &decider,
                                               bool refineAll,
                                               std::vector<TreeNode<T,dim>> &tree,
                                               LevI eLev,
                                               double loadFlexibility,
                                               MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  constexpr char numChildren = nchild(dim);
  constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

  std::vector<TreeNode<T,dim>> leafs;

  struct SubtreeList
  {
    std::vector<TreeNode<T,dim>> oct;
    std::vector<sfc::RotIndex::Type> rot;

    void swap(SubtreeList &other)
    {
      std::swap(this->oct, other.oct);
      std::swap(this->rot, other.rot);
    }

    void clear()
    {
      this->oct.clear();
      this->rot.clear();
    }
  };

  SubtreeList uncommittedSubtrees;
  SubtreeList tmpUncommittedSubtrees;
  SubtreeList refined;

  enum Phase : int {SINGLE_RANK = 0, ALL_RANKS = 1};
  int phase = SINGLE_RANK;

  if (rProc == 0)
  {
    uncommittedSubtrees.oct.push_back(TreeNode<T,dim>());
    uncommittedSubtrees.rot.push_back(0);
  }

  for (int nextLevel = 1; nextLevel <= eLev; ++nextLevel)
  {
    for (size_t ii = 0; ii < uncommittedSubtrees.oct.size(); ++ii)
    {
      const SFC_State<dim> sfc(sfc::RotIndex(uncommittedSubtrees.rot[ii]));
      TreeNode<T,dim> cNode = uncommittedSubtrees.oct[ii].getFirstChildMorton();

      for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
      {
        cNode.setMortonIndex(sfc.child_num(child_sfc));

        double physCoords[dim];
        double physSize;
        treeNode2Physical(cNode, physCoords, physSize);

        const ibm::Partition childRegion = decider(physCoords, physSize);

        if (childRegion != ibm::IN)
        {
          if (childRegion == ibm::INTERCEPTED ||
              childRegion == ibm::OUT && refineAll)
          {
            refined.oct.push_back(cNode);
            refined.rot.push_back(sfc.subcurve(child_sfc).state());
          }
          else
            leafs.push_back(cNode);
        }
      }
    }

    uncommittedSubtrees.clear();
    uncommittedSubtrees.swap(refined);

    if (phase == SINGLE_RANK)
    {
      if (rProc == 0 && uncommittedSubtrees.oct.size() >= nProc)
        phase = ALL_RANKS;

      par::Mpi_Bcast(&phase, 1, 0, comm);
    }

    if (phase == ALL_RANKS
        && par::loadImbalance(uncommittedSubtrees.oct.size(), comm) > loadFlexibility)
    {
      long long int locSz = uncommittedSubtrees.oct.size();
      long long int globSz = 0;
      par::Mpi_Allreduce(&locSz, &globSz, 1, MPI_SUM, comm);
      const size_t newLocSz = globSz / nProc + (rProc < globSz % nProc);
      par::scatterValues(uncommittedSubtrees.oct, tmpUncommittedSubtrees.oct, newLocSz, comm);
      par::scatterValues(uncommittedSubtrees.rot, tmpUncommittedSubtrees.rot, newLocSz, comm);
      uncommittedSubtrees.clear();
      uncommittedSubtrees.swap(tmpUncommittedSubtrees);
    }
  }
  leafs.insert(leafs.end(), uncommittedSubtrees.oct.begin(), uncommittedSubtrees.oct.end());
  distTreeSort(leafs, loadFlexibility, comm);
  tree = leafs;
}



template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: distRemoveDuplicates(std::vector<TreeNode<T,dim>> &tree, double loadFlexibility, bool strict, MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  // For now:
  // Rather than do a complicated elimination of duplicates,
  // perform another global sort, removing duplicates locally, and then
  // eliminate at most one duplicate from the end of each processor's partition.

  distTreeSort(tree, loadFlexibility, comm);
  if (!strict)
    locRemoveDuplicates(tree);
  else
    locRemoveDuplicatesStrict(tree);

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
    TreeNode<T,dim> nextBegin;
    MPI_Request request;
    MPI_Status status;
    if (rNE > 0)
      par::Mpi_Isend<TreeNode<T,dim>>(&(*tree.begin()), 1, rNE-1, 0, nonemptys, &request);
    if (rNE < nNE-1)
      par::Mpi_Recv<TreeNode<T,dim>>(&nextBegin, 1, rNE+1, 0, nonemptys, &status);

    // If so, delete our end.
    if (rNE > 0)
      MPI_Wait(&request, &status);
    if (rNE < nNE-1 && (tree.back() == nextBegin || !strict && tree.back().isAncestor(nextBegin)))
      tree.pop_back();
  }
  if (nonemptys != MPI_COMM_NULL)
    MPI_Comm_free(&nonemptys);
}


template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: locRemoveDuplicates(std::vector<TreeNode<T,dim>> &tnodes)
{
  const TreeNode<T,dim> *tEnd = &(*tnodes.end());
  TreeNode<T,dim> *tnCur = &(*tnodes.begin());
  size_t numUnique = 0;

  while (tnCur < tEnd)
  {
    // Find next leaf.
    TreeNode<T,dim> *tnNext;
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


template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: locRemoveDuplicatesStrict(std::vector<TreeNode<T,dim>> &tnodes)
{
  const TreeNode<T,dim> *tEnd = &(*tnodes.end());
  TreeNode<T,dim> *tnCur = &(*tnodes.begin());
  size_t numUnique = 0;

  while (tnCur < tEnd)
  {
    // Find next leaf.
    TreeNode<T,dim> *tnNext;
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
// distCoalesceSiblings()
//
template <typename T, unsigned int dim>
void SFC_Tree<T, dim>::distCoalesceSiblings( std::vector<TreeNode<T, dim>> &tree,
                                           MPI_Comm comm_ )
{
  MPI_Comm comm = comm_;

  int nProc, rProc;
  MPI_Comm_size(comm, &nProc);
  MPI_Comm_rank(comm, &rProc);

  int locDone = false;
  int globDone = false;

  while (!globDone)
  {
    // Exclude self if don't contain any TreeNodes.
    bool isActive = (tree.size() > 0);
    MPI_Comm activeComm;
    MPI_Comm_split(comm, (isActive ? 1 : MPI_UNDEFINED), rProc, &activeComm);
    if (comm != comm_ && comm != MPI_COMM_NULL)
      MPI_Comm_free(&comm);
    comm = activeComm;

    if (!isActive)
      break;

    MPI_Comm_size(comm, &nProc);
    MPI_Comm_rank(comm, &rProc);


    // Assess breakage on front and back.
    TreeNode<T, dim> locFrontParent = tree.front().getParent();
    TreeNode<T, dim> locBackParent = tree.back().getParent();
    bool locFrontIsBroken = false;
    bool locBackIsBroken = false;

    int sendLeft = 0, recvRight = 0;

    int idx = 0;
    while (idx < tree.size()
        && !locFrontIsBroken
        && locFrontParent.isAncestorInclusive(tree[idx]))
    {
      if (tree[idx].getParent() != locFrontParent)
        locFrontIsBroken = true;
      else
        sendLeft++;

      idx++;
    }

    idx = tree.size()-1;
    while (idx >= 0
        && !locBackIsBroken
        && locBackParent.isAncestorInclusive(tree[idx]))
    {
      if (tree[idx].getParent() != locBackParent)
        locBackIsBroken = true;

      idx--;
    }


    // Check with left and right ranks to see if an exchange is needed.
    TreeNode<T, dim> leftParent, rightParent;
    bool leftIsBroken, rightIsBroken;

    bool exchangeLeft = false;
    bool exchangeRight = false;

    constexpr int tagToLeft1 = 72;
    constexpr int tagToRight1 = 73;
    constexpr int tagToLeft2 = 74;
    constexpr int tagToRight2 = 75;
    constexpr int tagToLeft3 = 76;
    constexpr int tagToLeft4 = 77;
    MPI_Status status;

    int leftRank = (rProc > 0 ? rProc - 1 : MPI_PROC_NULL);
    int rightRank = (rProc < nProc-1 ? rProc+1 : MPI_PROC_NULL);

    par::Mpi_Sendrecv(&locFrontParent, 1, leftRank, tagToLeft1,
                      &rightParent, 1, rightRank, tagToLeft1, comm, &status);
    par::Mpi_Sendrecv(&locBackParent, 1, rightRank, tagToRight1,
                      &leftParent, 1, leftRank, tagToRight1, comm, &status);

    par::Mpi_Sendrecv(&locFrontIsBroken, 1, leftRank, tagToLeft2,
                      &rightIsBroken, 1, rightRank, tagToLeft2, comm, &status);
    par::Mpi_Sendrecv(&locBackIsBroken, 1, rightRank, tagToRight2,
                      &leftIsBroken, 1, leftRank, tagToRight2, comm, &status);

    if (rProc > 0)
      exchangeLeft = (!locFrontIsBroken && !leftIsBroken && locFrontParent == leftParent);

    if (rProc < nProc-1)
      exchangeRight = (!locBackIsBroken && !rightIsBroken && locBackParent == rightParent);


    // Do the exchanges. Send to left, recv from right.
    if (!exchangeLeft)
      leftRank = MPI_PROC_NULL;
    if (!exchangeRight)
      rightRank = MPI_PROC_NULL;

    par::Mpi_Sendrecv(&sendLeft, 1, leftRank, tagToLeft3,
                      &recvRight, 1, rightRank, tagToLeft3, comm, &status);

    if (exchangeRight)
    {
      tree.resize(tree.size() + recvRight);
    }

    par::Mpi_Sendrecv(&(*tree.begin()), sendLeft, leftRank, tagToLeft4,
                      &(*tree.end()) - recvRight, recvRight, rightRank, tagToLeft4,
                      comm, &status);

    if (exchangeLeft)
    {
      tree.erase(tree.begin(), tree.begin() + sendLeft);
    }


    // Global reduction to find out if the tree partition has converged.
    locDone = (!exchangeLeft && !exchangeRight);
    par::Mpi_Allreduce(&locDone, &globDone, 1, MPI_LAND, comm);
  }
  if (comm != comm_ && comm != MPI_COMM_NULL)
    MPI_Comm_free(&comm);

}


template <typename T, unsigned int dim>
std::vector<TreeNode<T, dim>>
SFC_Tree<T, dim>::locRemesh( const std::vector<TreeNode<T, dim>> &inTree,
                           const std::vector<OCT_FLAGS::Refine> &refnFlags )
{
  // TODO need to finally make a seperate minimal balancing tree routine
  // that remembers the level of the seeds.
  // For now, this hack should work because we remove duplicates.
  // With a proper level-respecting treeBalancing() routine, don't need to
  // make all siblings of all treeNodes for OCT_NO_CHANGE and OCT_COARSEN.
  constexpr ChildI NumChildren = 1u << dim;
  std::vector<TreeNode<T, dim>> outTree;
  std::vector<TreeNode<T, dim>> seed;
  for (size_t i = 0; i < inTree.size(); ++i)
  {
    switch(refnFlags[i])
    {
      case OCT_FLAGS::OCT_NO_CHANGE:
        for (ChildI child_m = 0; child_m < NumChildren; ++child_m)
          seed.push_back(inTree[i].getParent().getChildMorton(child_m));
        break;

      case OCT_FLAGS::OCT_COARSEN:
        for (ChildI child_m = 0; child_m < NumChildren; ++child_m)
          seed.push_back(inTree[i].getParent().getParent().getChildMorton(child_m));
        break;

      case OCT_FLAGS::OCT_REFINE:
        for (ChildI child_m = 0; child_m < NumChildren; ++child_m)
          seed.push_back(inTree[i].getChildMorton(child_m));
        break;

      default:
        throw std::invalid_argument("Unknown OCT_FLAGS::Refine flag.");
    }
  }

  SFC_Tree<T, dim>::locTreeSort(seed);
  SFC_Tree<T, dim>::locRemoveDuplicates(seed);
  SFC_Tree<T, dim>::locTreeBalancing(seed, outTree, 1);

  return outTree;
}

template <typename T, unsigned int dim>
void
SFC_Tree<T, dim>::distRemeshWholeDomain( const std::vector<TreeNode<T, dim>> &inTree,
                                       const std::vector<OCT_FLAGS::Refine> &refnFlags,
                                       std::vector<TreeNode<T, dim>> &outTree,
                                       double loadFlexibility,
                                       MPI_Comm comm )
{
  constexpr ChildI NumChildren = 1u << dim;

  outTree.clear();

  // TODO need to finally make a seperate minimal balancing tree routine
  // that remembers the level of the seeds.
  // For now, this hack should work because we remove duplicates.
  // With a proper level-respecting treeBalancing() routine, don't need to
  // make all siblings of all treeNodes for OCT_NO_CHANGE and OCT_COARSEN.
  std::vector<TreeNode<T, dim>> seed;
  for (size_t i = 0; i < inTree.size(); ++i)
  {
    switch(refnFlags[i])
    {
      case OCT_FLAGS::OCT_NO_CHANGE:
        for (ChildI child_m = 0; child_m < NumChildren; ++child_m)
          seed.push_back(inTree[i].getParent().getChildMorton(child_m));
        break;

      case OCT_FLAGS::OCT_COARSEN:
        for (ChildI child_m = 0; child_m < NumChildren; ++child_m)
          seed.push_back(inTree[i].getParent().getParent().getChildMorton(child_m));
        break;

      case OCT_FLAGS::OCT_REFINE:
        for (ChildI child_m = 0; child_m < NumChildren; ++child_m)
          seed.push_back(inTree[i].getChildMorton(child_m));
        break;

      default:
        throw std::invalid_argument("Unknown OCT_FLAGS::Refine flag.");
    }
  }

  SFC_Tree<T, dim>::distTreeSort(seed, loadFlexibility, comm);
  SFC_Tree<T, dim>::distRemoveDuplicates(seed, loadFlexibility, RM_DUPS_AND_ANC, comm);
  SFC_Tree<T, dim>::distTreeBalancing(seed, outTree, 1, loadFlexibility, comm);
  SFC_Tree<T, dim>::distCoalesceSiblings(outTree, comm);
}


template <typename T, unsigned int dim>
void
SFC_Tree<T, dim>::distRemeshSubdomain( const std::vector<TreeNode<T, dim>> &inTree,
                                       const std::vector<OCT_FLAGS::Refine> &refnFlags,
                                       std::vector<TreeNode<T, dim>> &outTree,
                                       double loadFlexibility,
                                       MPI_Comm comm )
{
  constexpr ChildI NumChildren = 1u << dim;

  /// SFC_Tree<T, dim>::distCoalesceSiblings(inTree, comm);

  std::vector<TreeNode<T, dim>> res;
  for (size_t i = 0; i < inTree.size(); ++i)
  {
    switch(refnFlags[i])
    {
      case OCT_FLAGS::OCT_NO_CHANGE:
        res.push_back(inTree[i]);
        break;

      case OCT_FLAGS::OCT_COARSEN:
        res.push_back(inTree[i].getParent());
        break;

      case OCT_FLAGS::OCT_REFINE:
        for (ChildI child_m = 0; child_m < NumChildren; ++child_m)
          res.push_back(inTree[i].getChildMorton(child_m));
        break;

      default:
        throw std::invalid_argument("Unknown OCT_FLAGS::Refine flag.");
    }
  }

  outTree = inTree;
  locTreeSort(res);
  SFC_Tree<T, dim>::locMatchResolution(outTree, res);
  SFC_Tree<T, dim>::distTreeSort(outTree, loadFlexibility, comm);
  SFC_Tree<T, dim>::distRemoveDuplicates(outTree, loadFlexibility, RM_DUPS_AND_ANC, comm);
  SFC_Tree<T, dim>::distMinimalBalanced(outTree, loadFlexibility, comm);
  SFC_Tree<T, dim>::distCoalesceSiblings(outTree, comm);
}

template <typename T, unsigned int dim>
std::vector<TreeNode<T, dim>> SFC_Tree<T, dim>::getSurrogateGrid(
    RemeshPartition remeshPartition,
    const std::vector<TreeNode<T, dim>> &oldTree,
    const std::vector<TreeNode<T, dim>> &newTree,
    MPI_Comm comm)
{
  std::vector<TreeNode<T, dim>> surrogateTree;

  if (remeshPartition == SurrogateInByOut)  // old default
  {
    // Create a surrogate tree, which is identical to the oldTree,
    // but partitioned to match the newTree.
    surrogateTree = SFC_Tree<T, dim>::getSurrogateGrid(oldTree, newTree, comm);
  }
  else
  {
    // Create a surrogate tree, which is identical to the newTree,
    // but partitioned to match the oldTree.
    surrogateTree = SFC_Tree<T, dim>::getSurrogateGrid(newTree, oldTree, comm);
  }

  return surrogateTree;
}


template <typename T, unsigned int dim>
std::vector<int> getSendcounts(const std::vector<TreeNode<T, dim>> &items,
                               const std::vector<TreeNode<T, dim>> &frontSplitters)
{
  int numSplittersSeen = 0;
  int ancCarry = 0;
  std::vector<int> scounts(frontSplitters.size(), 0);

  MeshLoopInterface_Sorted<T, dim, true, true, false> itemLoop(items);
  MeshLoopInterface_Sorted<T, dim, true, true, false> splitterLoop(frontSplitters);
  while (!itemLoop.isFinished())
  {
    const MeshLoopFrame<T, dim> &itemSubtree = itemLoop.getTopConst();
    const MeshLoopFrame<T, dim> &splitterSubtree = splitterLoop.getTopConst();

    if (splitterSubtree.isEmpty())
    {
      scounts[numSplittersSeen-1] += itemSubtree.getTotalCount();
      scounts[numSplittersSeen-1] += ancCarry;
      ancCarry = 0;

      itemLoop.next();
      splitterLoop.next();
    }
    else if (itemSubtree.isEmpty() && ancCarry == 0)
    {
      numSplittersSeen += splitterSubtree.getTotalCount();

      itemLoop.next();
      splitterLoop.next();
    }
    else
    {
      ancCarry += itemSubtree.getAncCount();

      if (splitterSubtree.isLeaf())
      {
        numSplittersSeen++;

        scounts[numSplittersSeen-1] += ancCarry;
        ancCarry = 0;
      }

      itemLoop.step();
      splitterLoop.step();
    }
  }

  return scounts;
}


template <typename T, unsigned int dim>
std::vector<TreeNode<T, dim>>
SFC_Tree<T, dim>::getSurrogateGrid( const std::vector<TreeNode<T, dim>> &replicateGrid,
                                    const std::vector<TreeNode<T, dim>> &splittersFromGrid,
                                    MPI_Comm comm )
{
  std::vector<TreeNode<T, dim>> surrogateGrid;

  int nProc, rProc;
  MPI_Comm_size(comm, &nProc);
  MPI_Comm_rank(comm, &rProc);

  // Temporary activeComm, in case splittersFromGrid has holes, this
  // make it more convenient to construct surrogate grid.
  const bool isSplitterGridActive = splittersFromGrid.size() > 0;
  MPI_Comm sgActiveComm;
  {
  MPI_Comm_split(comm, (isSplitterGridActive ? 1 : MPI_UNDEFINED), rProc, &sgActiveComm);
  }

  std::vector<int> sgActiveList;
  std::vector<TreeNode<T, dim>> splitters;
  {
  splitters = SFC_Tree<T, dim>::dist_bcastSplitters(
      &splittersFromGrid.front(),
      comm,
      sgActiveComm,
      isSplitterGridActive,
      sgActiveList);
  }

  std::vector<int> surrogateSendCountsCompact = getSendcounts<T, dim>(replicateGrid, splitters);
  std::vector<int> surrogateSendCounts(nProc, 0);
  for (int i = 0; i < sgActiveList.size(); ++i)
    surrogateSendCounts[sgActiveList[i]] = surrogateSendCountsCompact[i];

  std::vector<int> surrogateRecvCounts(nProc, 0);

  {
  par::Mpi_Alltoall(surrogateSendCounts.data(), surrogateRecvCounts.data(), 1, comm);
  }

  std::vector<int> surrogateSendDispls(1, 0);
  surrogateSendDispls.reserve(nProc + 1);
  for (int c : surrogateSendCounts)
    surrogateSendDispls.push_back(surrogateSendDispls.back() + c);
  surrogateSendDispls.pop_back();

  std::vector<int> surrogateRecvDispls(1, 0);
  surrogateRecvDispls.reserve(nProc + 1);
  for (int c : surrogateRecvCounts)
    surrogateRecvDispls.push_back(surrogateRecvDispls.back() + c);

  surrogateGrid.resize(surrogateRecvDispls.back());

  // Copy replicateGrid grid to surrogate grid.
  {
  par::Mpi_Alltoallv_sparse(replicateGrid.data(),
                            surrogateSendCounts.data(),
                            surrogateSendDispls.data(),
                            surrogateGrid.data(),
                            surrogateRecvCounts.data(),
                            surrogateRecvDispls.data(),
                            comm);
  }

  if (sgActiveComm != MPI_COMM_NULL)
    MPI_Comm_free(&sgActiveComm);
  return surrogateGrid;
}



//
// propagateNeighbours()
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: propagateNeighbours(std::vector<TreeNode<T,dim>> &srcNodes)
{
  std::vector<std::vector<TreeNode<T,dim>>> treeLevels = stratifyTree(srcNodes);
  srcNodes.clear();

  ///std::cout << "Starting at        level " << m_uiMaxDepth << ", level size \t " << treeLevels[m_uiMaxDepth].size() << "\n";  //DEBUG

  // Bottom-up traversal using stratified levels.
  for (unsigned int l = m_uiMaxDepth; l > 0; l--)
  {
    const unsigned int lp = l-1;  // Parent level.

    const size_t oldLevelSize = treeLevels[lp].size();

    const std::vector<TreeNode<T, dim>> &childList = treeLevels[l];
    std::vector<TreeNode<T, dim>> &parentList = treeLevels[lp];
    for (size_t i = 0; i < childList.size(); ++i)
      if (i == 0 || childList[i-1].getParent() != childList[i].getParent())
        childList[i].getParent().appendAllNeighbours(parentList);

    // TODO Consider more efficient algorithms for removing duplicates from lp level.
    locTreeSort(&(*treeLevels[lp].begin()), 0, treeLevels[lp].size(), 1, lp, SFC_State<dim>::root());
    locRemoveDuplicates(treeLevels[lp]);

    ///const size_t newLevelSize = treeLevels[lp].size();
    ///std::cout << "Finished adding to level " << lp << ", level size \t " << oldLevelSize << "\t -> " << newLevelSize << "\n";  // DEBUG
  }

  // Reserve space before concatenating all the levels.
  size_t newSize = 0;
  for (const std::vector<TreeNode<T,dim>> &trLev : treeLevels)
    newSize += trLev.size();
  srcNodes.reserve(newSize);

  // Concatenate all the levels.
  for (const std::vector<TreeNode<T,dim>> &trLev : treeLevels)
    srcNodes.insert(srcNodes.end(), trLev.begin(), trLev.end());
}


//
// locTreeBalancing()
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: locTreeBalancing(std::vector<TreeNode<T,dim>> &points,
                                 std::vector<TreeNode<T,dim>> &tree,
                                 RankI maxPtsPerRegion)
{
  const LevI leafLevel = m_uiMaxDepth;

  locTreeConstruction(&(*points.begin()), tree, maxPtsPerRegion,
                      0, (RankI) points.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      SFC_State<dim>::root(),
                      TreeNode<T,dim>());

  propagateNeighbours(tree);

  std::vector<TreeNode<T,dim>> newTree;
  locTreeConstruction(&(*tree.begin()), newTree, 1,
                      0, (RankI) tree.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      SFC_State<dim>::root(),
                      TreeNode<T,dim>());

  tree = newTree;
}

//
// locTreeBalancingWithFilter()
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: locTreeBalancingWithFilter(
                                 const ibm::DomainDecider &decider,
                                 std::vector<TreeNode<T,dim>> &points,
                                 std::vector<TreeNode<T,dim>> &tree,
                                 RankI maxPtsPerRegion)
{
  const LevI leafLevel = m_uiMaxDepth;

  locTreeConstructionWithFilter(
                      decider,
                      &(*points.begin()), tree, maxPtsPerRegion,
                      0, (RankI) points.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      SFC_State<dim>::root(),
                      TreeNode<T,dim>());

  propagateNeighbours(tree);

  std::vector<TreeNode<T,dim>> newTree;
  locTreeConstructionWithFilter(
                      decider,
                      &(*tree.begin()), newTree, 1,
                      0, (RankI) tree.size(),
                      1, leafLevel,         //TODO is sLev 0 or 1?
                      SFC_State<dim>::root(),
                      TreeNode<T,dim>());

  tree = newTree;
}


//
// distTreeBalancing()
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: distTreeBalancing(std::vector<TreeNode<T,dim>> &points,
                                   std::vector<TreeNode<T,dim>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  distTreeConstruction(points, tree, maxPtsPerRegion, loadFlexibility, comm);
  propagateNeighbours(tree);
  distRemoveDuplicates(tree, loadFlexibility, true, comm);   // Duplicate neighbours could cause over-refinement.
  std::vector<TreeNode<T,dim>> newTree;
  distTreeConstruction(tree, newTree, 1, loadFlexibility, comm);  // Still want only leaves.

  tree = newTree;
}

//
// distTreeBalancingWithFilter()
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: distTreeBalancingWithFilter(
                                   const ibm::DomainDecider &decider,
                                   std::vector<TreeNode<T,dim>> &points,
                                   std::vector<TreeNode<T,dim>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  distTreeConstructionWithFilter(decider, points, tree, maxPtsPerRegion, loadFlexibility, comm);
  propagateNeighbours(tree);
  distRemoveDuplicates(tree, loadFlexibility, true, comm);   // Duplicate neighbours could cause over-refinement.
  std::vector<TreeNode<T,dim>> newTree;
  distTreeConstructionWithFilter(decider, tree, newTree, 1, loadFlexibility, comm);  // Still want only leaves.

  tree = newTree;
}



template <typename X>
using VecVec = std::vector<std::vector<X>>;

struct RangeUnion
{
  using Range = std::pair<size_t, size_t>;
  std::vector<std::pair<size_t, size_t>> m_ranges;

  RangeUnion() = default;
  RangeUnion(const size_t n_ranges) : m_ranges(n_ranges, Range{0, 0})  {}

  template <typename X>
  RangeUnion(const std::vector<X> &xs) : m_ranges{ Range{0, xs.size()} }  {}

  size_t n_ranges()      const { return m_ranges.size(); }
  bool has(size_t i)     const { return i < n_ranges(); }
  size_t begin(size_t i) const { return m_ranges[i].first; }
  size_t end(size_t i)   const { return m_ranges[i].second; }
  size_t & begin(size_t i)     { return m_ranges[i].first; }
  size_t & end(size_t i)       { return m_ranges[i].second; }

  struct It
  {
    public:
      const RangeUnion *ru;
      size_t outer;
      size_t inner;
    private:
      void begin_range(size_t i) {
        outer = i;
        inner = ru->has(i) ? ru->begin(i) : -1;
      }

      void begin_valid() {
        while (outer < ru->m_ranges.size() and inner == ru->end(outer))
          begin_range(++outer);
      }

    public:
      It(const RangeUnion *ru_) : ru(ru_)  { begin_range(0); begin_valid(); }
      bool empty()    const { return outer == ru->m_ranges.size(); }
      bool nonempty() const { return not empty(); }
      It & operator++() { ++inner; begin_valid(); return *this; }

      template <typename X>
        const X & into(const VecVec<X> &xs) { return xs[outer][inner]; }
      template <typename X>
        X & into(VecVec<X> &xs)             { return xs[outer][inner]; }
  };

  It iterator() const { return It(this); }
};


template <int nbuckets>
using Buckets = std::array<size_t, nbuckets + 1>;

template <int nbuckets>
struct BucketUnion
{
  static Buckets<nbuckets> empty_buckets() { return Buckets<nbuckets>{}; }

  std::vector<Buckets<nbuckets>> m_buckets;

  BucketUnion() = default;
  BucketUnion(const size_t n_levels) : m_buckets(n_levels, empty_buckets())  {}

  RangeUnion bucket(size_t bucket) const {
    RangeUnion ru;
    using Range = RangeUnion::Range;
    for (const Buckets<nbuckets> &split : m_buckets)
      ru.m_ranges.push_back(Range{split[bucket], split[bucket + 1]});
    return ru;
  }
};


template <int nbuckets, typename X, typename KeyMap>
Buckets<nbuckets> bucket(X *xs, size_t begin, size_t end, KeyMap keymap)
{
  static std::vector<X> copies;
  copies.resize(end - begin);
  Buckets<nbuckets> buckets = {};
  for (size_t i = begin; i < end; ++i)
    ++buckets[keymap(xs[i])];
  for (int b = 1; b < buckets.size(); ++b)
    buckets[b] += buckets[b-1];
  for (size_t i = end; i-- > begin; ) // backward
    copies[--buckets[keymap(xs[i])]] = xs[i];
  for (size_t i = begin; i < end; ++i)
    xs[i] = copies[i - begin];
  for (int b = 0; b < buckets.size(); ++b)
    buckets[b] += begin;
  return buckets;
}

template <int nbuckets, typename X, typename KeyMap>
Buckets<nbuckets> bucket(std::vector<X> &xs, size_t begin, size_t end, KeyMap keymap)
{
  return bucket<nbuckets>(xs.data(), begin, end, keymap);
}

template <int nbuckets, typename X, typename KeyMap>
Buckets<nbuckets> bucket(Segment<X> &xs, KeyMap keymap)
{
  return bucket<nbuckets>(xs.ptr, xs.begin, xs.end, keymap);
}

template <int nbuckets, typename X, typename KeyMap>
BucketUnion<nbuckets> bucket(VecVec<X> &xs, const RangeUnion &ru, KeyMap keymap)
{
  BucketUnion<nbuckets> buckets;
  for (size_t level = 0; level < ru.n_ranges(); ++level)
    buckets.m_buckets.push_back(bucket<nbuckets>(xs[level], ru.begin(level), ru.end(level), keymap));
  return buckets;
}

template <int nbuckets>
class Bucketing
{
  size_t m_begin = 0;
  mutable Buckets<nbuckets> m_buckets = {};
  mutable bool m_done_counting = false;
  mutable bool m_done_indexing = false;
  size_t m_running_total = 0;

  public:
    Bucketing() = default;
    Bucketing(size_t begin) : m_begin(begin)  {}

    // Must iterate over elements in the same order on both passes.
    void count(int bucket) {
      assert(!m_done_counting);
      ++m_buckets[bucket];
      ++m_running_total;
    }

    size_t index(int bucket) {
      finish_counting();
      return m_buckets[bucket]++;
    }

    size_t total() const {
      finish_counting();
      return m_buckets[nbuckets] - m_buckets[0];
    }

    size_t running_total() const {
      return m_running_total;
    }

    const Buckets<nbuckets> & buckets() const {
      finish_indexing();
      return m_buckets;
    }

  private:
    void finish_counting() const
    {
      if (!m_done_counting)
      {
        for (int b = 1; b < m_buckets.size(); ++b)
          m_buckets[b] += m_buckets[b-1];
        for (int b = 0; b < m_buckets.size(); ++b)
          m_buckets[b] += m_begin;
        for (int b = nbuckets; b > 0; --b)
          m_buckets[b] = m_buckets[b-1];
        m_buckets[0] = m_begin;
        m_done_counting = true;
      }
    }

    void finish_indexing() const
    {
      if (!m_done_indexing)
      {
        assert(m_buckets[nbuckets-1] == m_buckets[nbuckets]);
        for (int b = nbuckets; b > 0; --b)
          m_buckets[b] = m_buckets[b-1];
        m_buckets[0] = m_begin;
        m_done_indexing = true;
      }
    }
};

// Append a set of buckets to output
template <int nbuckets, typename X, typename ItemInBucket>
Buckets<nbuckets> bucket_dup(
    const VecVec<X> &xs,
    const RangeUnion &ru,
    ItemInBucket itemInBucket,
    std::vector<X> &output)
{
  Bucketing<nbuckets> bucketing;
  for (RangeUnion::It it = ru.iterator(); it.nonempty(); ++it)
  {
    const X &x = it.into(xs);
    for (int b = 0; b < nbuckets; ++b)
      if (itemInBucket(x, b))
        bucketing.count(b);
  }

  output.resize(output.size() + bucketing.total());
  X *out = &(*output.end() - bucketing.total());

  // Forward on second pass.
  for (RangeUnion::It it = ru.iterator(); it.nonempty(); ++it)
  {
    const X &x = it.into(xs);
    for (int b = 0; b < nbuckets; ++b)
      if (itemInBucket(x, b))
        out[bucketing.index(b)] = x;
  }

  return bucketing.buckets();
}

// Sorts and duplicates the parents as needed.
template <typename T, unsigned int dim>
void appendNeighbours_rec(
    const int octLevel,
    Segment<TreeNode<T, dim>> interior,
    VecVec<TreeNode<T, dim>> &exterior,
    const RangeUnion &exterior_ranges,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<TreeNode<T, dim>> &neighbours)
{
  using Oct = TreeNode<T, dim>;
  using OctList = std::vector<Oct>;

  if (exterior_ranges.iterator().empty() and interior.empty())
    return;

  //
  // Base case: Subtree is same level as octants.
  //
  if (subtree.getLevel() == octLevel)
  {
    if (exterior_ranges.iterator().nonempty())
      neighbours.push_back(subtree);
    return;
  }


  //
  // Bucket subtree-internal octants.
  //

  // key: child number relative to subtree, permuted by sfc
  const auto descend = [&subtree, &sfc](const Oct &oct) -> sfc::SubIndex {
    const int childLevel = subtree.getLevel() + 1;
    return sfc.child_rank(sfc::ChildNum(oct.getMortonIndex(childLevel)));
  };

  // Assumes no ancestors
  Buckets<nchild(dim)> int_child_buckets =
    bucket<nchild(dim)>(interior, descend);


  //
  // Bucket subtree-exterior octants (single child incidence).
  //

  std::array<TreeNode<T, dim>, nchild(dim)> morton_children;
  for (int c = 0; c < nchild(dim); ++c)
    morton_children[c] = subtree.getChildMorton(c);

  // In the periodic case, need to eliminate non-unique children,
  // since the recursion is unguided.
  unsigned int periodic_axes = 0;
  for (int d = 0; d < dim; ++d)
    if (morton_children[1u << d] == morton_children[0])
      periodic_axes |= (1u << d);

  std::bitset<nchild(dim)> uniq_children;
  for (int c = 0; c < nchild(dim); ++c)
    if ((c & periodic_axes) == 0)
      uniq_children[c] = true;

  assert(uniq_children.all());  // Note: If periodic, remove assertion.

  const auto incident = [&](const Oct &oct0, const Oct &oct1) -> bool {
    int count_edge = 0,  count_overlaps = 0;
    for (int d = 0; d < dim; ++d)
      if (oct0.minX(d) == oct1.maxX(d) or oct1.minX(d) == oct0.maxX(d))
        ++count_edge;
      else if (oct0.minX(d) < oct1.maxX(d) and oct1.minX(d) < oct0.maxX(d))
        ++count_overlaps;
    return count_edge + count_overlaps == dim and count_edge >= 1;
  };

  // key: exclusively incident child number, or nchild(dim)
  const auto incidence = [&](const Oct &oct) -> sfc::SubIndex {
    int numIncident = 0;
    sfc::ChildNum incidentChild(-1);
    for (sfc::ChildNum c(0); c < nchild(dim); ++c)
      if (incident(oct, morton_children[c])) { ++numIncident; incidentChild = c; }
    return numIncident == 1 ? sfc.child_rank(incidentChild)
                            : sfc::SubIndex(nchild(dim));
  };

  BucketUnion<nchild(dim) + 1> ext_child_buckets =
      bucket<nchild(dim) + 1>(exterior, exterior_ranges, incidence);


  //
  // Duplicate subtree-exterior octants (multi-child) and cross-child octants.
  //

  Bucketing<nchild(dim) + 1> dup_bucketing;
    // only nchild(dim) buckets used, but want to match ext_child_buckets.

  const RangeUnion multi_child = ext_child_buckets.bucket(nchild(dim));

  // Count multi-child exterior
  for (RangeUnion::It it = multi_child.iterator(); it.nonempty(); ++it)
  {
    const Oct &oct = it.into(exterior);
    for (sfc::ChildNum c(0); c < nchild(dim); ++c)
      if (uniq_children[c] && incident(morton_children[c], oct))
        dup_bucketing.count(sfc.child_rank(c));
  }
  // Count cross-child interior
  for (size_t i = interior.begin; i < interior.end; ++i)
  {
    const Oct &oct = interior.ptr[i];
    for (sfc::ChildNum c(0); c < nchild(dim); ++c)
      if (uniq_children[c] && incident(morton_children[c], oct))
        dup_bucketing.count(sfc.child_rank(c));
  }

  // Prepare child output vector.
  OctList &children_exterior = exterior[subtree.getLevel() + 1];
  children_exterior.resize(dup_bucketing.total());

  // Duplicate multi-child exterior
  for (RangeUnion::It it = multi_child.iterator(); it.nonempty(); ++it)
  {
    const Oct &oct = it.into(exterior);
    for (sfc::ChildNum c(0); c < nchild(dim); ++c)
      if (uniq_children[c] && incident(morton_children[c], oct))
        children_exterior[dup_bucketing.index(sfc.child_rank(c))] = oct;
  }
  // Duplicate cross-child interior
  for (size_t i = interior.begin; i < interior.end; ++i)
  {
    const Oct &oct = interior.ptr[i];
    for (sfc::ChildNum c(0); c < nchild(dim); ++c)
      if (uniq_children[c] && incident(morton_children[c], oct))
        children_exterior[dup_bucketing.index(sfc.child_rank(c))] = oct;
  }

  // Recurse
  ext_child_buckets.m_buckets.push_back(dup_bucketing.buckets());
  for (sfc::SubIndex i(0); i < nchild(dim); ++i)
  {
    Segment<Oct> child_interior(interior.ptr,
                                int_child_buckets[i],
                                int_child_buckets[i+1]);

    RangeUnion child_exterior = ext_child_buckets.bucket(i);

    appendNeighbours_rec<T, dim>(
        octLevel,
        child_interior,
        exterior,
        child_exterior,
        subtree.getChildMorton(sfc.child_num(i)),
        sfc.subcurve(i),
        neighbours);
  }

  ext_child_buckets.m_buckets.pop_back();
  children_exterior.clear();
}


template <typename T, unsigned int dim>
void appendNeighboursOfParents(
    int octLevel,
    const std::vector<TreeNode<T, dim>> &octList,  // assumes no ancestors
    std::vector<TreeNode<T, dim>> &parentList)
{
  if (octLevel <= 1)
    return;

  // Parents. Unique within any sorted segment.
  std::vector<TreeNode<T, dim>> parents = octList;
  TreeNode<T, dim> lastKeptParent;
  bool keptAParent = false;
  Keeper<TreeNode<T, dim>> parentStore(&(*parents.begin()), 0, parents.size());
  while (parentStore.nonempty())
  {
    TreeNode<T, dim> nextParent = (*parentStore).getParent();
    if (not keptAParent or nextParent != lastKeptParent)
    {
      assert(parentStore.adv_store(nextParent));
      keptAParent = true;
      lastKeptParent = nextParent;
    }
    else
      ++parentStore;
  }
  parents.erase(parents.begin() + parentStore.out, parents.end());

  VecVec<TreeNode<T, dim>> exterior_aux(m_uiMaxDepth + 1);
  RangeUnion exterior_range(exterior_aux[0]);

  // Add neighbours.
  appendNeighbours_rec<T, dim>(
      octLevel - 1,
      segment_all(parents),
      exterior_aux,
      exterior_range,
      TreeNode<T, dim>(),
      SFC_State<dim>::root(),
      parentList);
}



template <typename T, unsigned int dim>
void mergeSorted_rec(
    Segment<const TreeNode<T, dim>> &first,
    Segment<const TreeNode<T, dim>> &second,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<TreeNode<T, dim>> &out)
{
  using Oct = TreeNode<T, dim>;

  const auto equals_subtree = [&subtree](const Segment<const Oct> &seg) {
    return seg.nonempty() and *seg == subtree;
  };
  const auto in_subtree = [&subtree](const Segment<const Oct> &seg) {
    return seg.nonempty() and subtree.isAncestorInclusive(*seg);
  };
  const auto output = [&out](Segment<const Oct> &seg) {
    out.push_back(*seg);  ++seg;
  };


  if (first.empty() and second.empty())
    return;

  while (equals_subtree(first))
    output(first);
  while (equals_subtree(second))
    output(second);

  if (not in_subtree(first))
    while (in_subtree(second))
      output(second);
  else if (not in_subtree(second))
    while (in_subtree(first))
      output(first);
  else
    for (sfc::SubIndex i(0); i < nchild(dim); ++i)
      mergeSorted_rec<T, dim>(
          first, second,
          subtree.getChildMorton(sfc.child_num(i)),
          sfc.subcurve(i),
          out);
}


template <typename T, unsigned int dim>
void mergeSorted(std::vector<TreeNode<T, dim>> &octList, size_t prefix)
{
  using Oct = TreeNode<T, dim>;

  static std::vector<Oct> output;
  output.clear();
  output.reserve(octList.size());

  Segment<const Oct> first(octList.data(), 0, prefix);
  Segment<const Oct> second(octList.data(), prefix, octList.size());
  mergeSorted_rec<T, dim>(first, second, Oct(), SFC_State<dim>::root(), output);

  std::swap(octList, output);
}


//
// locMinimalBalanced()
//
template <typename T, unsigned int dim>
void SFC_Tree<T, dim>::locMinimalBalanced(std::vector<TreeNode<T, dim>> &tree)
{
  using Oct = TreeNode<T, dim>;
  using OctList = std::vector<Oct>;

  if (tree.size() == 0)
    return;

  OctList resolution;

  // Propagate neighbors by levels
  {
    std::set<int> levels;
    std::map<int, OctList> octLevels;
    for (const Oct &oct : tree)
    {
      levels.insert(oct.getLevel());
      octLevels[oct.getLevel()].push_back(oct);
    }

    const int coarsest = *levels.begin(),  finest = *levels.rbegin();
    for (int level = finest; level > coarsest; --level)
    {
      const OctList &childList = octLevels[level];
      OctList &parentList = octLevels[level-1];

      {DOLLAR("recursive_add_neighbors")
        const size_t prefix = parentList.size();
        appendNeighboursOfParents<T, dim>(level, childList, parentList);
        mergeSorted(parentList, prefix);
        locRemoveDuplicates(parentList);
      }
    }

    size_t sumSizes = 0;
    for (const std::pair<int, OctList> &lev_list : octLevels)
      sumSizes += lev_list.second.size();

    resolution.reserve(sumSizes);
    for (const std::pair<int, OctList> &lev_list : octLevels)
      resolution.insert(resolution.end(), lev_list.second.begin(), lev_list.second.end());
  }
  locTreeSort(resolution);
  locRemoveDuplicates(resolution);
  locResolveTree(tree, std::move(resolution));
}




//
// locResolveTree_rec()
//
template <typename T, unsigned int dim>
void locResolveTree_rec(
    Keeper<TreeNode<T, dim>> &domain,
    Keeper<TreeNode<T, dim>> &res,
    bool complete,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<TreeNode<T, dim>> &extra);


//
// locResolveTree()
//
template <typename T, unsigned int dim>
void SFC_Tree<T, dim>::locResolveTree(
    std::vector<TreeNode<T, dim>> &tree,
    std::vector<TreeNode<T, dim>> &&res)
{
  using Oct = TreeNode<T, dim>;
  Keeper<Oct> keep_domain(&(*tree.begin()), 0, tree.size());
  Keeper<Oct> keep_res(&(*res.begin()), 0, res.size());
  std::vector<Oct> extra;

  locResolveTree_rec<T, dim>(
      keep_domain,
      keep_res,
      false,
      Oct(),
      SFC_State<dim>::root(),
      extra);

  tree.erase(tree.begin() + keep_domain.out, tree.end());
  res.erase(res.begin() + keep_res.out, res.end());

  if (res.size() > tree.size())
    std::swap(tree, res);

  tree.insert(tree.end(), res.begin(), res.end());
  tree.insert(tree.end(), extra.begin(), extra.end());
  res.clear();
  extra.clear();

  SFC_Tree<T, dim>::locTreeSort(tree);
}


//
// locResolveTree_rec()
//
template <typename T, unsigned int dim>
void locResolveTree_rec(
    Keeper<TreeNode<T, dim>> &domain,  // sorted nonoverlapping
    Keeper<TreeNode<T, dim>> &res,     // sorted, maybe overlapping
    bool complete,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<TreeNode<T, dim>> &extra)
{
  using Oct = TreeNode<T, dim>;

  struct SubtreeKeeper {
    Keeper<Oct> &it;  const Oct &root;  const bool complete;
    SubtreeKeeper(Keeper<Oct> &it_, const Oct &root_, bool complete_ = false)
      :  it(it_), root(root_), complete(complete_) {}

    bool has_root() const { return it.nonempty() and *it == root; }
    bool nonempty() const { return complete or it.nonempty() and root.isAncestorInclusive(*it); }
    bool empty() const    { return not nonempty(); }

    // Return true if all subtree octants were kept in keeper, otherwise false.
    bool keep_all() {
      if (complete and (it.empty() or not root.isAncestorInclusive(*it))) return false;
      while (it.nonempty() and root.isAncestorInclusive(*it))
        it.keep(), ++it;
      return true;
    }
  };

  SubtreeKeeper dom_tree(domain, subtree, complete);
  SubtreeKeeper res_tree(res, subtree);

  if (dom_tree.empty())
    // Advance past subtree in res.
    while (res_tree.nonempty())
      ++res_tree.it;
  else
  {
    bool res_should_keep_root = false;
    bool res_kept_root = false;

    if (res_tree.has_root())
      res_tree.it.keep(), res_kept_root = true;
    while (res_tree.has_root())
      ++res_tree.it;

    // Leaf or empty. No more detail in res; conform to original domain.
    if (res_tree.empty())
    {
      // Try to keep in dom, unless have to store root finer than dom.
      // Then try to keep in res, if already there,
      // otherwise insert copy of root into whichever has room,
      // otherwise push into auxiliary vector.
      if (not dom_tree.keep_all())
        if (res_kept_root)
          res_should_keep_root = true;
        else if (not (res_tree.it.store(subtree) or dom_tree.it.store(subtree)))
          extra.push_back(subtree);
    }

    if (res_kept_root and not res_should_keep_root)
      res_tree.it.unkeep();

    // When res has more detail, recurse on subtrees.
    if (res_tree.nonempty())
    {
      complete = complete or dom_tree.has_root();
      while (dom_tree.has_root())
        ++dom_tree.it;

      for (sfc::SubIndex c(0); c < nchild(dim); ++c)
        locResolveTree_rec<T, dim>(
            domain, res, complete,
            subtree.getChildMorton(sfc.child_num(c)),
            sfc.subcurve(c),
            extra);
    }
  }
}



template <typename T, unsigned dim>
void locMatchResolution_rec(
    Segment<const TreeNode<T, dim>> &domain,
    Segment<const TreeNode<T, dim>> &res,
    bool complete,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<TreeNode<T, dim>> &outTree);

//
// locMatchResolution()
//
template <typename T, unsigned dim>
void SFC_Tree<T, dim>::locMatchResolution(
    std::vector<TreeNode<T, dim>> &tree, const std::vector<TreeNode<T, dim>> &res)
{
  assert(isLocallySorted(tree));
  assert(isLocallySorted(res));

  using Oct = TreeNode<T, dim>;
  std::vector<Oct> domain;
  std::swap(domain, tree);
  Segment<const Oct> segDomain(&(*domain.cbegin()), 0, domain.size());
  Segment<const Oct> segRes(&(*res.cbegin()), 0, res.size());
  locMatchResolution_rec<T, dim>(
      segDomain,
      segRes,
      false,
      Oct(),
      SFC_State<dim>::root(),
      tree);
}

template <typename T, unsigned dim>
void locMatchResolution_rec(
    Segment<const TreeNode<T, dim>> &domain,
    Segment<const TreeNode<T, dim>> &res,
    bool complete,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc,
    std::vector<TreeNode<T, dim>> &outTree)
{
  if (complete or (domain.nonempty() and subtree.isAncestorInclusive(*domain)))
  {
    if (domain.nonempty() and *domain == subtree)
      complete = true;

    // Advance past parents
    while (domain.nonempty() and *domain == subtree)
      ++domain;
    while (res.nonempty() and *res == subtree)
      ++res;

    if (res.nonempty() and subtree.isAncestor(*res))
    {
      // Advance past children in domain and res.
      for (sfc::SubIndex c(0); c < nchild(dim); ++c)
        locMatchResolution_rec<T, dim>(
            domain, res, complete,
            subtree.getChildMorton(sfc.child_num(c)),
            sfc.subcurve(c),
            outTree);
    }
    else
    {
      // Append leaf and advance past children in domain.
      outTree.push_back(subtree);
      while (domain.nonempty() and subtree.isAncestor(*domain))
        ++domain;
    }
  }
  else
    // Advance past subtree in res.
    while (res.nonempty() && subtree.isAncestorInclusive(*res))
      ++res;

  assert((domain.empty() or !subtree.isAncestorInclusive(*domain)));
  assert((res.empty() or !subtree.isAncestorInclusive(*res)));
}


//
// unstableOctants()
//
template <typename T, unsigned int dim>
std::vector<TreeNode<T, dim>> SFC_Tree<T, dim>::unstableOctants(
    const std::vector<TreeNode<T, dim>> &tree,
    const bool dangerLeft,
    const bool dangerRight)
{
  using Oct = TreeNode<T, dim>;
  using OctList = std::vector<Oct>;

  if (!dangerLeft && !dangerRight)
    return OctList();
  if (tree.size() == 0)
    return OctList();

  const Oct front = tree.front(),  back = tree.back();

  // Want all oct of (oct, nbr) such that
  //   nbr.isAncestor(front)  or  (nbr < front)  or
  //   nbr.isAncestor(back)   or  (nbr > back)
  // Can test ancestry immediately, but comparisons need sort.

  // Optimization: If an octant has an ancestor residing in the partition
  //               and sharing no sides with the octant, then it is stable.
  const auto paddedAncestor = [](const Oct &oct) -> Oct {
    if (oct.getLevel() == 0)
      return oct;
    const int height = m_uiMaxDepth - oct.getLevel();
    const int rootMask = (1u << oct.getLevel()) - 1;
    int levDeviate = oct.getLevel();
    for (int d = 0; d < dim; ++d)
    {
      T x = oct.getX(d);
      x >>= height;
      const int levDeviateLeft = ((x & rootMask) == 0 ? 1 : (oct.getLevel() - binOp::lowestOnePos(x)));
      x = ~x;
      const int levDeviateRight = ((x & rootMask) == 0 ? 1 : (oct.getLevel() - binOp::lowestOnePos(x)));
      levDeviate = fmin(fmin(levDeviateLeft, levDeviateRight), levDeviate);
    }
    return oct.getAncestor(levDeviate - 1);
  };

  std::vector<char> isUnstable(tree.size(), false);
  OctList octNbr;
  OctList allNbr;
  std::vector<size_t> allSrc;
  for (size_t i = 0; i < tree.size(); ++i)
  {
    bool knownStable = false;
    {
      const Oct pad = paddedAncestor(tree[i]);
      if (! ((dangerLeft && pad.isAncestor(front))
             || (dangerRight && pad.isAncestor(back))) )
        knownStable = true;
    }

    bool knownUnstable = false;
    if (!knownStable)
    {
      octNbr.clear();
      tree[i].appendAllNeighbours(octNbr);
      for (const Oct &nbr : octNbr)
        if ((dangerLeft && nbr.isAncestor(front))
            || (dangerRight && nbr.isAncestor(back)))
          knownUnstable = true;
    }

    if (knownUnstable)
      isUnstable[i] = true;
    else if (!knownStable)//unknown
    {
      allNbr.insert(allNbr.end(), octNbr.begin(), octNbr.end());
      allSrc.insert(allSrc.end(), octNbr.size(), i);
    }
  }
  locTreeSort(allNbr, allSrc);

  if (dangerLeft)
  {
    // Points to first element not less than front.
    // Strict ancestors of front already handled above.
    const size_t lbound = tsearch_lower_bound(allNbr, front);
    for (size_t n = 0; n < lbound; ++n)
      isUnstable[allSrc[n]] = true;
  }
  if (dangerRight)
  {
    // Deepest last descendant in true SFC order.
    Oct backDLD = dld(back);

    // Points to first element greater than back.
    // Strict ancestors of back already handled above.
    /// const size_t ubound = tsearch_upper_bound(allNbr, back);  // after oct range start in SFC
    const size_t ubound = tsearch_upper_bound(allNbr, backDLD);   // after oct range end in SFC
    for (size_t n = ubound; n < allNbr.size(); ++n)
      isUnstable[allSrc[n]] = true;
  }

  OctList unstable;
  for (size_t i = 0; i < tree.size(); ++i)
    if (isUnstable[i])
      unstable.push_back(tree[i]);
  return unstable;
}


std::vector<int> recvFromActive(
    const std::vector<int> &activeList,
    const std::vector<int> &sendToActive,
    MPI_Comm comm)
{
  //future: choose a better collective pattern

  int commSize, commRank;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  std::map<int, int> invActive;
  for (int ar = 0; ar < activeList.size(); ++ar)
    invActive[activeList[ar]] = ar;

  const bool isActive = invActive.find(commRank) != invActive.end();

  std::vector<int> sendToSizes(commSize, 0);
  for (const int arSend : sendToActive)
    sendToSizes[activeList[arSend]] = 1;

  int recvFromSz = 0;
  {
    std::vector<int> recvFromSizes(commSize, 0);
    par::Mpi_Allreduce( &(*sendToSizes.cbegin()),
                        &(*recvFromSizes.begin()),
                        commSize, MPI_SUM, comm );
    recvFromSz = recvFromSizes[commRank];
  }

  if (!isActive)
  {
    assert(sendToActive.size() == 0);
    assert(recvFromSz == 0);
  }

  std::set<int> recvFromGlobal;
  std::vector<MPI_Request> requests(sendToActive.size());
  for (size_t i = 0; i < sendToActive.size(); ++i)
    par::Mpi_Isend((int*)nullptr, 0, activeList[sendToActive[i]], 0, comm, &requests[i]);
  for (size_t j = 0; j < recvFromSz; ++j)
  {
    MPI_Status status;
    par::Mpi_Recv((int*)nullptr, 0, MPI_ANY_SOURCE, 0, comm, &status);
    recvFromGlobal.insert(status.MPI_SOURCE);
  }
  for (MPI_Request &request : requests)
    MPI_Wait(&request, MPI_STATUS_IGNORE);

  MPI_Barrier(comm);

  std::vector<int> recvFromActive;
  for (int r : recvFromGlobal)
    recvFromActive.push_back(invActive[r]);

  return recvFromActive;
}




template <typename T, unsigned dim>
void SFC_Tree<T, dim>::distMinimalBalanced(
      std::vector<TreeNode<T, dim>> &tree, double sfc_tol, MPI_Comm comm)
{
  // Based on the distributed balancing routines in:
  // @article{doi:10.1137/070681727,
  //   author = {Sundar, Hari and Sampath, Rahul S. and Biros, George},
  //   title = {Bottom-Up Construction and 2:1 Balance Refinement of Linear Octrees in Parallel},
  //   journal = {SIAM Journal on Scientific Computing},
  //   year = {2008},
  //   volume = {30}, number = {5}, pages = {2675-2708},
  //   doi = {10.1137/070681727}, URL = { https://doi.org/10.1137/070681727 }
  // }

  using Oct = TreeNode<T, dim>;
  using OctList = std::vector<Oct>;

  int commSize, commRank;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  const bool isActive = tree.size() > 0;

  locMinimalBalanced(tree);

  // The unstable octants (processor boundary octants) may need to be refined.
  const OctList unstableOwned = unstableOctants(tree, commRank > 0, commRank < commSize - 1);

  // Splitters of active ranks.
  std::vector<int> active;
  const PartitionFrontBack<T, dim> partition =
      allgatherSplitters(tree.size() > 0, tree.front(), tree.back(), comm, &active);

  std::map<int, int> invActive;
  for (int ar = 0; ar < active.size(); ++ar)
    invActive[active[ar]] = ar;

  const int activeRank = (isActive ? invActive[commRank] : -1);

  // (Round 1)
  // Find insulation layers of unstable octs.
  // Source indices later used to effectively undo the sort.
  OctList insulationOfOwned;
  std::vector<size_t> beginInsulationOfOwned, endInsulationOfOwned;
  for (const Oct &u : unstableOwned)
  {
    beginInsulationOfOwned.push_back(insulationOfOwned.size());
    u.appendAllNeighbours(insulationOfOwned);
    endInsulationOfOwned.push_back(insulationOfOwned.size());
  }

  // Every insulation octant overlaps a range of active ranks.
  // Expect return indices relative to active list.
  std::vector<IntRange> insulationProcRanges =
      treeNode2PartitionRanks(insulationOfOwned, partition, &active);

  // (Round 1)  Stage queries to be sent.
  std::set<int> queryDestSet;
  std::map<int, std::vector<Oct>> sendQueryInform;
  std::map<int, std::vector<Oct>> sendInformOnly;
  for (size_t ui = 0; ui < unstableOwned.size(); ++ui)
  {
    std::set<int> unstableProcSet;
    for (size_t i = beginInsulationOfOwned[ui];
                i < endInsulationOfOwned[ui]; ++i)
      for (int r = insulationProcRanges[i].min;
               r <= insulationProcRanges[i].max; ++r)
        unstableProcSet.insert(r);
    unstableProcSet.erase(activeRank);  // Don't send to self.

    if (unstableProcSet.size() == 1)  // Insulation of ui completed in 1st round.
      sendInformOnly[*unstableProcSet.begin()].push_back(unstableOwned[ui]);
    else
      for (int r : unstableProcSet)
        sendQueryInform[r].push_back(unstableOwned[ui]);

    queryDestSet.insert(unstableProcSet.cbegin(), unstableProcSet.cend());
  }

  std::vector<int> queryDest(queryDestSet.cbegin(), queryDestSet.cend());
  std::vector<int> querySrc = recvFromActive(active, queryDest, comm);

  std::map<int, std::vector<Oct>> recvQueryInform;
  std::map<int, std::vector<Oct>> recvInformOnly;

  // (Round 1)  Send/recv
  std::vector<int> queryDestGlobal, querySrcGlobal;
  for (int r : queryDest)  queryDestGlobal.push_back(active[r]);
  for (int r : querySrc)   querySrcGlobal.push_back(active[r]);
  {
    P2PPartners partners(queryDestGlobal, querySrcGlobal, comm);

    // Sizes
    P2PScalar<> szInformOnly(&partners), szQueryInform(&partners);
    for (int i = 0; i < queryDest.size(); ++i)
    {
      szInformOnly.send(i, sendInformOnly[queryDest[i]].size());
      szQueryInform.send(i, sendQueryInform[queryDest[i]].size());
    }
    for (int j = 0; j < querySrc.size(); ++j)
    {
      recvInformOnly[querySrc[j]].resize(szInformOnly.recv(j));
      recvQueryInform[querySrc[j]].resize(szQueryInform.recv(j));
    }
    szInformOnly.wait_all();
    szQueryInform.wait_all();

    // Payload
    P2PVector<Oct> vcInformOnly(&partners), vcQueryInform(&partners);
    for (int i = 0; i < queryDest.size(); ++i)
    {
      vcInformOnly.send(i, sendInformOnly[queryDest[i]]);
      vcQueryInform.send(i, sendQueryInform[queryDest[i]]);
    }
    for (int j = 0; j < querySrc.size(); ++j)
    {
      vcInformOnly.recv(j, recvInformOnly[querySrc[j]]);
      vcQueryInform.recv(j, recvQueryInform[querySrc[j]]);
    }
    vcInformOnly.wait_all();
    vcQueryInform.wait_all();
  }

  // (Round 2)  Recreate the insulation layers of remote unstable query octants.
  OctList insulationRemote;
  std::vector<int> insulationRemoteOrigin;
  for (int r : querySrc)
  {
    const size_t oldSz = insulationRemote.size();
    for (const Oct &remoteUnstable : recvQueryInform[r])
      remoteUnstable.appendAllNeighbours(insulationRemote);
    const size_t newSz = insulationRemote.size();
    std::fill_n(std::back_inserter(insulationRemoteOrigin), newSz - oldSz, r);
  }
  locTreeSort(insulationRemote, insulationRemoteOrigin);

  // Need to send owned unstable octants that overlap with remote insulation.
  Overlaps<T, dim> insRemotePerUnstableOwned(
      insulationRemote, unstableOwned);

  std::map<int, std::vector<Oct>> sendQueryAnswer;
  std::vector<size_t> overlapIdxs;
  std::set<int> octAnswerProcs;
  for (size_t ui = 0; ui < unstableOwned.size(); ++ui)
  {
    overlapIdxs.clear();
    insRemotePerUnstableOwned.keyOverlapsAncestors(ui, overlapIdxs);

    octAnswerProcs.clear();
    for (size_t insIdx : overlapIdxs)
      octAnswerProcs.insert(insulationRemoteOrigin[insIdx]);  // future: Use scatter not gather
    for (int r : octAnswerProcs)
      sendQueryAnswer[r].push_back(unstableOwned[ui]);
  }

  // Do not need to send if already sent.
  for (int r : querySrc)
  {
    removeEqual(sendQueryAnswer[r], sendInformOnly[r]);
    removeEqual(sendQueryAnswer[r], sendQueryInform[r]);
  }

  // (Round 2)  Send/recv
  std::map<int, std::vector<Oct>> recvQueryAnswer;
  {
    P2PPartners partners(querySrcGlobal, queryDestGlobal, comm);

    // Sizes
    P2PScalar<> szQueryAnswer(&partners);
    for (int i = 0; i < querySrc.size(); ++i)
      szQueryAnswer.send(i, sendQueryAnswer[querySrc[i]].size());
    for (int j = 0; j < queryDest.size(); ++j)
      recvQueryAnswer[queryDest[j]].resize(szQueryAnswer.recv(j));
    szQueryAnswer.wait_all();

    // Payload
    P2PVector<Oct> vcQueryAnswer(&partners);
    for (int i = 0; i < querySrc.size(); ++i)
      vcQueryAnswer.send(i, sendQueryAnswer[querySrc[i]]);
    for (int j = 0; j < queryDest.size(); ++j)
      vcQueryAnswer.recv(j, recvQueryAnswer[queryDest[j]]);
    vcQueryAnswer.wait_all();
  }

  // Construct received insulation layers of unstable octants.
  const auto appendVec = [](OctList &into, const OctList &from) {
    into.insert(into.end(), from.cbegin(), from.cend());
  };

  OctList unstablePlus;
  {
    size_t total = unstableOwned.size();
    for (int r : querySrc)
    {
      total += recvInformOnly[r].size();
      total += recvQueryInform[r].size();
    }
    for (int r : queryDest)
      total += recvQueryAnswer[r].size();
    unstablePlus.reserve(total);
  }
  appendVec(unstablePlus, unstableOwned);
  for (int r : querySrc)
  {
    appendVec(unstablePlus, recvInformOnly[r]);
    appendVec(unstablePlus, recvQueryInform[r]);
  }
  for (int r : queryDest)
    appendVec(unstablePlus, recvQueryAnswer[r]);

  // Balance unstable octants against received insulation.
  locTreeSort(unstablePlus);
  locMinimalBalanced(unstablePlus);
  retainDescendants(unstablePlus, unstableOwned);

  // Replace unstable with unstable balanced.
  removeEqual(tree, unstableOwned);
  appendVec(tree, unstablePlus);
  locTreeSort(tree);
}







//
// getContainingBlocks() - Used for tagging points on the processor boundary.
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: getContainingBlocks(TreeNode<T,dim> *points,
                                  RankI begin, RankI end,
                                  const TreeNode<T,dim> *splitters,
                                  int numSplitters,
                                  std::vector<int> &outBlocks)
{
  int dummyNumPrevBlocks = 0;
  getContainingBlocks(points,
      begin, end,
      splitters,
      0, numSplitters,
      1, SFC_State<dim>::root(),
      dummyNumPrevBlocks, outBlocks.size(), outBlocks);
}

namespace util {
  void markProcNeighbour(int proc, int startSize, std::vector<int> &neighbourList)
  {
    if (neighbourList.size() == startSize || neighbourList.back() != proc)
      neighbourList.push_back(proc);
  }
}  // namespace ot::util


//
// getContainingBlocks() - recursive implementation (inorder traversal).
//
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>:: getContainingBlocks(TreeNode<T,dim> *points,
                                  RankI begin, RankI end,
                                  const TreeNode<T,dim> *splitters,
                                  RankI sBegin, RankI sEnd,
                                  LevI lev, SFC_State<dim> sfc,
                                  int &numPrevBlocks,
                                  const int startSize,
                                  std::vector<int> &outBlocks)
{
  // Idea:
  // If a bucket contains points but no splitters, the points belong to the block of the most recent splitter.
  // If a bucket contains points and splitters, divide and conquer by refining the bucket and recursing.
  // In an ancestor or leaf bucket, contained splitters and points are equal, so splitter <= point.
  constexpr ChildI numChildren = nchild(dim);

  // Bucket points.
  std::array<RankI, 1+numChildren> pointBuckets;
  RankI ancStart, ancEnd;
  SFC_bucketing(points, begin, end, lev, sfc, pointBuckets, ancStart, ancEnd);

  // Count splitters.
  std::array<RankI, numChildren> numSplittersInBucket;
  numSplittersInBucket.fill(0);
  RankI numAncSplitters = 0;
  for (int s = sBegin; s < sEnd; s++)
  {
    if (splitters[s].getLevel() < lev)
      numAncSplitters++;
    else
      numSplittersInBucket[
          sfc.child_rank(sfc::ChildNum(
              splitters[s]
                  .getMortonIndex(lev)))]++;
  }


  // Mark any splitters in the ancestor bucket. Splitters preceed points.
  numPrevBlocks += numAncSplitters;
  if (numPrevBlocks > 0 && ancEnd > ancStart)
    util::markProcNeighbour(numPrevBlocks - 1, startSize, outBlocks);

  // Mark splitters in child buckets.
  if (lev < m_uiMaxDepth)
  {
    for (sfc::SubIndex child_sfc(0); child_sfc < numChildren; ++child_sfc)
    {
      if (pointBuckets[child_sfc+1] > pointBuckets[child_sfc])
      {
        if (numSplittersInBucket[child_sfc] > 0)
          getContainingBlocks(points, pointBuckets[child_sfc], pointBuckets[child_sfc+1],
              splitters, numPrevBlocks, numPrevBlocks + numSplittersInBucket[child_sfc],
              lev+1, sfc.subcurve(child_sfc),
              numPrevBlocks, startSize, outBlocks);
        else
          util::markProcNeighbour(numPrevBlocks - 1, startSize, outBlocks);
      }
      else
        numPrevBlocks += numSplittersInBucket[child_sfc];
    }
  }
  else
  {
    // In leaf buckets splitters preceed points, just as in ancestor buckets.
    for (ChildI child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      numPrevBlocks += numSplittersInBucket[child_sfc];
      if (numPrevBlocks > 0 && pointBuckets[child_sfc+1] > pointBuckets[child_sfc])
        util::markProcNeighbour(numPrevBlocks - 1, startSize, outBlocks);
    }
  }

  // Emit a block id for blocks with points but not splitters.
}


template <typename T, unsigned int dim>
PartitionFront<T, dim> SFC_Tree<T, dim>::allgatherSplitters(
    bool nonempty_,
    const TreeNode<T, dim> &front_,
    MPI_Comm comm,
    std::vector<int> *activeList)
{
  int commSize, commRank;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  std::vector<TreeNode<T, dim>> splitters(commSize);
  std::vector<char> isNonempty(commSize, false);
  const TreeNode<T, dim> front = (nonempty_ ? front_ : TreeNode<T, dim>());
  const char nonempty = nonempty_;
  par::Mpi_Allgather(&front, &(*splitters.begin()), 1, comm);
  par::Mpi_Allgather(&nonempty, &(*isNonempty.begin()), 1, comm);

  for (int r = commSize - 2; r >= 0; --r)
    if (!isNonempty[r])
      splitters[r] = splitters[r + 1];

  if (activeList != nullptr)
  {
    activeList->clear();
    for (int r = 0; r < commSize; ++r)
      if (isNonempty[r])
        activeList->push_back(r);
  }

  PartitionFront<T, dim> partition;
  partition.m_fronts = splitters;
  return partition;
}

template <typename T, unsigned int dim>
PartitionFrontBack<T, dim> SFC_Tree<T, dim>::allgatherSplitters(
    bool nonempty_,
    const TreeNode<T, dim> &front_,
    const TreeNode<T, dim> &back_,
    MPI_Comm comm,
    std::vector<int> *activeList)
{
  int commSize, commRank;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  std::vector<TreeNode<T, dim>> fronts(commSize);
  std::vector<TreeNode<T, dim>> backs(commSize);
  std::vector<char> isNonempty(commSize, false);
  const TreeNode<T, dim> front = (nonempty_ ? front_ : TreeNode<T, dim>());
  const TreeNode<T, dim> back = (nonempty_ ? back_ : TreeNode<T, dim>());
  const char nonempty = nonempty_;
  par::Mpi_Allgather(&front, &(*fronts.begin()), 1, comm);
  par::Mpi_Allgather(&back, &(*backs.begin()), 1, comm);
  par::Mpi_Allgather(&nonempty, &(*isNonempty.begin()), 1, comm);

  for (int r = commSize - 2; r >= 0; --r)
    if (!isNonempty[r])
    {
      fronts[r] = fronts[r + 1];
      backs[r] = fronts[r + 1];   // To maintain ordering: f[r]<=b[r]<=f[r+1]
    }

  if (activeList != nullptr)
  {
    activeList->clear();
    for (int r = 0; r < commSize; ++r)
      if (isNonempty[r])
        activeList->push_back(r);
  }

  PartitionFrontBack<T, dim> partition;
  partition.m_fronts = fronts;
  partition.m_backs = backs;
  return partition;
}


//
// treeNode2PartitionRank()  -- relative to global splitters, empty ranks allowed.
//
template <typename T, unsigned int dim>
std::vector<int> SFC_Tree<T, dim>::treeNode2PartitionRank(
    const std::vector<TreeNode<T, dim>> &treeNodes,
    const PartitionFront<T, dim> &partitionSplitters)
{
  // Result
  std::vector<int> rankIds(treeNodes.size(), 0);

  std::vector<TreeNode<T, dim>> splitters = partitionSplitters.m_fronts;

  // Root as a valid splitter would be a special case indicating 0 owns all.
  if (splitters[0] == TreeNode<T, dim>())
    return rankIds;

  // Tail splitters of empty ranks should be ignored.
  auto splitTail = splitters.crbegin();
  while (splitTail != splitters.crend() && *splitTail == TreeNode<T, dim>())
    ++splitTail;
  const size_t numSplitters = std::distance(splitTail, splitters.crend());

  if (numSplitters == 1)
    return rankIds;

  // Concatenate [splitters | elements]  (rely on stable sort, splitters come first.)
  std::vector<TreeNode<T, dim>> keys;
  splitters.resize(numSplitters);
  keys.insert(keys.end(), splitters.cbegin(), splitters.cend());
  keys.insert(keys.end(), treeNodes.cbegin(), treeNodes.cend());

  // Indices into result, which we use after sorting. [{-1,...-1} | indices]
  std::vector<size_t> indices(keys.size());
  std::fill(indices.begin(), indices.begin() + numSplitters, -1);
  std::iota(indices.begin() + numSplitters, indices.end(), 0);

  SFC_Tree<T, dim>::locTreeSort(keys, indices);  // Assumed to be stable.

  int rank = -1;
  for (size_t ii = 0; ii < keys.size(); ++ii)
    if (indices[ii] == -1)
      ++rank;
    else
      rankIds[indices[ii]] = rank;

  return rankIds;
}

//
// treeNode2PartitionRanks()  -- relative to active list, empty ranks allowed in partition.
//
template <typename T, unsigned int dim>
std::vector<IntRange> SFC_Tree<T, dim>::treeNode2PartitionRanks(
    const std::vector<TreeNode<T, dim>> &treeNodes_,
    const PartitionFrontBack<T, dim> &partition,
    const std::vector<int> *activePtr)
{
  if (treeNodes_.size() == 0)
    return std::vector<IntRange>();

  const std::vector<int> &active = *activePtr;  // future: recover active

  // Active splitters.
  std::vector<TreeNode<T, dim>> splitters;
  for (int ar = 0; ar < active.size(); ++ar)
  {
    splitters.push_back(partition.m_fronts[active[ar]]);
    splitters.push_back(partition.m_backs[active[ar]]);
  }
  assert(isLocallySorted(splitters));

  // Sort treeNodes if not already sorted.
  const bool sorted = isLocallySorted<T, dim>(treeNodes_);
  std::vector<TreeNode<T, dim>> sortedTreeNodes;
  std::vector<size_t> src;
  if (!sorted)
  {
    sortedTreeNodes = treeNodes_;
    src.resize(sortedTreeNodes.size());
    std::iota(src.begin(), src.end(), 0);
    locTreeSort(sortedTreeNodes, src);
  }
  const std::vector<TreeNode<T, dim>> &treeNodes =
      (sorted ? treeNodes_ : sortedTreeNodes);
  assert(treeNodes.size() == treeNodes_.size());

  // Lower bound: For each octant in treeNodes, first splitter equal or greater.
  std::vector<size_t> lowerBounds = lower_bound(splitters, treeNodes);

  // Overlapped processes for each octant:
  //   - oct.isAncestorInclusive(fronts[r])  OR
  //   - oct.isAncestorInclusive(backs[r])   OR
  //   - lower_bound(fronts[r]) < oct < lowerBound(backs[r])
  std::vector<IntRange> ranges(treeNodes.size());
  for (size_t i = 0; i < treeNodes.size(); ++i)
  {
    size_t j = lowerBounds[i];
    if (j < splitters.size() && j % 2 == 1)  // back splitter
      ranges[i].include(j/2);
    while (j < splitters.size()
        && treeNodes[i].isAncestorInclusive(splitters[j]))
    {
      ranges[i].include(j/2);
      ++j;
    }
  }

  // Undo sort if not originally sorted.
  if (!sorted)
  {
    std::vector<IntRange> unsortedRanges(ranges.size());
    for (size_t i = 0; i < src.size(); ++i)
      unsortedRanges[src[i]] = ranges[i];
    std::swap(ranges, unsortedRanges);
  }

  return ranges;
}



/** @brief Successively computes 0th child in SFC order to given level. */
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>::firstDescendant(TreeNode<T,dim> &parent,
                               RotI &pRot,
                               LevI descendantLev)
{
  constexpr unsigned int NUM_CHILDREN = 1u << dim;
  constexpr unsigned int rotOffset = 2*NUM_CHILDREN;  // num columns in rotations[].

  while (parent.getLevel() < descendantLev)
  {
    const ot::ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*NUM_CHILDREN];
    const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*NUM_CHILDREN];

    ot::ChildI child0 = rot_perm[0];
    parent = parent.getChildMorton(child0);
    pRot = orientLookup[child0];
  }
}


/** @brief Successively computes (n-1)th child in SFC order to given level. */
template <typename T, unsigned int dim>
void
SFC_Tree<T,dim>::lastDescendant(TreeNode<T,dim> &parent,
                              RotI &pRot,
                              LevI descendantLev)
{
  constexpr unsigned int NUM_CHILDREN = 1u << dim;
  constexpr unsigned int rotOffset = 2*NUM_CHILDREN;  // num columns in rotations[].

  while (parent.getLevel() < descendantLev)
  {
    const ot::ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*NUM_CHILDREN];
    const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*NUM_CHILDREN];

    ot::ChildI childLast = rot_perm[NUM_CHILDREN-1];
    parent = parent.getChildMorton(childLast);
    pRot = orientLookup[childLast];
  }
}


// define static data member bucketStableAux
template <typename T, unsigned int dim>
std::vector<char> SFC_Tree<T, dim>::bucketStableAux;


// Template instantiations.
template struct SFC_Tree<unsigned int, 2>;
template struct SFC_Tree<unsigned int, 3>;
template struct SFC_Tree<unsigned int, 4>;


// --------------------------------------------------------------------------

template <typename T, unsigned int dim>
bool is2to1Balanced(const std::vector<TreeNode<T, dim>> &tree_, MPI_Comm comm)
{
  using Oct = TreeNode<T, dim>;
  using OctList = std::vector<Oct>;

  const auto locSortUniq = [](OctList &octList) {
    SFC_Tree<T, dim>::locTreeSort(octList);
    SFC_Tree<T, dim>::locRemoveDuplicates(octList);
  };

  OctList tree = tree_;
  locSortUniq(tree);

  // Create coarse neighbor search keys.
  OctList keys;
  for (const Oct &oct : tree)
    oct.appendCoarseNeighbours(keys);
  locSortUniq(keys);

  // The tree is not balanced if and only if the tree has a strict ancestor
  // of any of the coarse keys.
  // Search for keys by the tree partition.
  // If an ancestor of a key exists in the tree, then the partition
  // of the process owning the ancestor would also cover the key.
  // (The opposite might not be true..
  // we can't force an ancestor onto every rank that begins with a descendant).

  // Distribute keys by the tree partition.
  std::vector<int> keyDest = SFC_Tree<T, dim>::treeNode2PartitionRank(
      keys,
      SFC_Tree<T, dim>::allgatherSplitters(
          tree.size() > 0, tree.front(), comm) );
  keys = par::sendAll(keys, keyDest, comm);
  locSortUniq(keys);

  // Find strict descendants of tree[i] in keys.
  /// std::vector<TreeNode<T, dim>> offenders;
  char locBalanced = true;
  std::vector<size_t> lowerBounds = SFC_Tree<T, dim>::lower_bound(keys, tree);
  for (size_t i = 0; i < tree.size(); ++i)
    for (size_t j = lowerBounds[i]; j < keys.size() && tree[i].isAncestorInclusive(keys[j]); ++j)
      if (tree[i].isAncestor(keys[j]))
      {
        locBalanced = false;
        /// offenders.push_back(tree[i]);
        break;
      }

  char globBalanced = locBalanced;
  par::Mpi_Allreduce(&locBalanced, &globBalanced, 1, MPI_LAND, comm);
  return globBalanced;
}

template bool is2to1Balanced<unsigned, 2u>(const std::vector<TreeNode<unsigned, 2u>> &, MPI_Comm);
template bool is2to1Balanced<unsigned, 3u>(const std::vector<TreeNode<unsigned, 3u>> &, MPI_Comm);
template bool is2to1Balanced<unsigned, 4u>(const std::vector<TreeNode<unsigned, 4u>> &, MPI_Comm);



template <typename T, unsigned int dim>
bool isLocallySorted(const std::vector<TreeNode<T, dim>> &octList)
{
  return octList.size() == lenContainedSorted<T, dim>(
      &(*octList.cbegin()),
      0, octList.size(),
      TreeNode<T, dim>(),
      SFC_State<dim>::root());
}
template bool isLocallySorted<unsigned, 2u>(const std::vector<TreeNode<unsigned, 2u>> &);
template bool isLocallySorted<unsigned, 3u>(const std::vector<TreeNode<unsigned, 3u>> &);
template bool isLocallySorted<unsigned, 4u>(const std::vector<TreeNode<unsigned, 4u>> &);


template <typename T, unsigned int dim>
size_t lenContainedSorted(
    const TreeNode<T, dim> *octList,
    size_t begin, size_t end,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc)
{
  if (begin == end)
    return 0;
  const size_t parBegin = begin;
  while (begin < end && octList[begin] == subtree)
    ++begin;
  if (begin < end && subtree.isAncestor(octList[begin]))
  {
    for (sfc::SubIndex c(0); c < nchild(dim); ++c)
      begin += lenContainedSorted<T, dim>(
          octList, begin, end,
          subtree.getChildMorton(sfc.child_num(c)),
          sfc.subcurve(c));
  }
  return begin - parBegin;
}

template size_t lenContainedSorted<unsigned, 2u>(
    const TreeNode<unsigned, 2u> *, size_t, size_t, TreeNode<unsigned, 2u>, SFC_State<2u>);
template size_t lenContainedSorted<unsigned, 3u>(
    const TreeNode<unsigned, 3u> *, size_t, size_t, TreeNode<unsigned, 3u>, SFC_State<3u>);
template size_t lenContainedSorted<unsigned, 4u>(
    const TreeNode<unsigned, 4u> *, size_t, size_t, TreeNode<unsigned, 4u>, SFC_State<4u>);





} // namspace ot
