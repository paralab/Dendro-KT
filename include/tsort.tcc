/**
 * @file:tsort.tcc
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-01-11
 */

#include <numeric>

namespace ot
{


//
// locTreeSort()
//
template<typename T, unsigned int dim, class PointType>
void
locTreeSort_rec(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc);

template<typename T, unsigned int dim>
template <class PointType>
void
SFC_Tree<T,dim>:: locTreeSort(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc)
{
  DOLLAR("locTreeSort()");
  locTreeSort_rec<T, dim, PointType>(points, begin, end, sLev, eLev, sfc);
}

template<typename T, unsigned int dim, class PointType>
void
locTreeSort_rec(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc)
{
  //// Recursive Depth-first, similar to Most Significant Digit First. ////

  if (end <= begin) { return; }

  // Reorder the buckets on sLev (current level).
  std::array<RankI, nchild(dim)+1> tempSplitters;
  RankI ancStart, ancEnd;
  /// SFC_bucketing(points, begin, end, sLev, pRot, tempSplitters, ancStart, ancEnd);
  SFC_Tree<T, dim>::template SFC_bucketing_impl<KeyFunIdentity_Pt<PointType>, PointType, PointType>(
      points, begin, end, sLev, sfc,
      KeyFunIdentity_Pt<PointType>(), true, true,
      tempSplitters,
      ancStart, ancEnd);

  // The array `tempSplitters' has numChildren+1 slots, which includes the
  // beginning, middles, and end of the range of children.
  // Ancestor splitters are is ancStart and ancEnd, not tempSplitters.

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    // Recurse.
    // Use the splitters to specify ranges for the next level of recursion.
    for (char child_sfc = 0; child_sfc < nchild(dim); child_sfc++)
    {
      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] <= 1)
        continue;

      if (sLev > 0)
      {
        locTreeSort_rec<T, dim, PointType>(points,
            tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
            sLev+1, eLev,
            sfc.subcurve(sfc::SubIndex(child_sfc)));
      }
      else   // Special handling if we have to consider the domain boundary.
      {
        locTreeSort_rec<T, dim, PointType>(points,
            tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
            sLev+1, eLev,
            sfc);
      }
    }
  }
}// end function()


//
// locTreeSort() (with parallel companion array)
//
template<typename T, unsigned int dim, class KeyFun, typename PointType, typename KeyType, bool useCompanions, typename... Companion>
void
locTreeSort_rec(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          Companion* ... companions
                          );

template<typename T, unsigned int dim>
template <class KeyFun, typename PointType, typename KeyType, bool useCompanions, typename... Companion>
void
SFC_Tree<T,dim>:: locTreeSort(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          Companion* ... companions
                          )
{
  locTreeSort_rec<T, dim, KeyFun, PointType, KeyType, useCompanions>(
      points, begin, end, sLev, eLev, sfc, keyfun, companions...);
}

template<typename T, unsigned int dim, class KeyFun, typename PointType, typename KeyType, bool useCompanions, typename... Companion>
void
locTreeSort_rec(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          Companion* ... companions
                          )

{
  //// Recursive Depth-first, similar to Most Significant Digit First. ////

  if (end <= begin) { return; }

  constexpr char numChildren = nchild(dim);
  constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

  // Reorder the buckets on sLev (current level).
  std::array<RankI, numChildren+1> tempSplitters;
  RankI ancStart, ancEnd;
  SFC_Tree<T, dim>::template SFC_bucketing_general<KeyFun, PointType, KeyType, useCompanions, Companion...>(
      points, begin, end, sLev, sfc,
      keyfun, true, true,
      tempSplitters,
      ancStart, ancEnd,
      companions...
      );

  // The array `tempSplitters' has numChildren+1 slots, which includes the
  // beginning, middles, and end of the range of children.
  // Ancestor splitters are is ancStart and ancEnd, not tempSplitters.

  if (sLev < eLev)  // This means eLev is further from the root level than sLev.
  {
    // Recurse.
    // Use the splitters to specify ranges for the next level of recursion.
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] <= 1)
        continue;

      if (sLev > 0)
      {
        locTreeSort_rec<T, dim, KeyFun, PointType, KeyType, useCompanions, Companion...>
            (points,
            tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
            sLev+1, eLev,
            sfc.subcurve(sfc::SubIndex(child_sfc)),  // This branch uses cRot.
            keyfun,
            companions...
            );
      }
      else   // Special handling if we have to consider the domain boundary.
      {
        locTreeSort_rec<T, dim, KeyFun, PointType, KeyType, useCompanions, Companion...>
            (points,
            tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
            sLev+1, eLev,
            sfc,                         // This branch uses pRot.
            keyfun,
            companions...
            );
      }
    }
  }
}// end function()


//
// SFC_bucketing_impl()
//
template <typename T, unsigned int dim>
template <class KeyFun, typename PointType, typename KeyType>
void
SFC_Tree<T,dim>:: SFC_bucketing_impl(PointType *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          bool separateAncestors,
                          bool ancestorsFirst,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd)
{
  // Call the "companion" implementation without giving or using companions.
  SFC_Tree<T,dim>::template SFC_bucketing_general<KeyFun, PointType, KeyType, false, int>(
      points,
      begin, end, lev, sfc,
      keyfun, separateAncestors, ancestorsFirst,
      outSplitters, outAncStart, outAncEnd,
      (int*) nullptr
      );
}




template <typename T, unsigned int dim>
template <class KeyFun, typename PointType, typename KeyType, typename CompanionHead, typename... CompanionTail>
void SFC_Tree<T, dim>::SFC_bucketStable(
    const PointType *points,
    RankI begin,
    RankI end,
    LevI lev,
    KeyFun keyfun,
    bool separateAncestors,
    const std::array<RankI, nchild(dim)+1> &offsets,     // last idx represents ancestors.
    const std::array<RankI, nchild(dim)+1> &bucketEnds,  // last idx represents ancestors.
    CompanionHead * companionHead,
    CompanionTail* ... companionTail)
{
  // Recursively unwrap the pack of companions
  // until call the single-parameter "values" version.
  SFC_bucketStable<KeyFun, PointType, KeyType>(
      points,
      begin,
      end,
      lev,
      keyfun,
      separateAncestors,
      offsets,
      bucketEnds,
      companionHead);   // bucket the head companion array
  SFC_bucketStable<KeyFun, PointType, KeyType>(
      points,
      begin,
      end,
      lev,
      keyfun,
      separateAncestors,
      offsets,
      bucketEnds,
      companionTail...);  // recursive on tail of companion arrays
}


template <typename T, unsigned int dim>
template <class KeyFun, typename PointType, typename KeyType, typename ValueType>
void SFC_Tree<T, dim>::SFC_bucketStable(
    const PointType *points,
    RankI begin,
    RankI end,
    LevI lev,
    KeyFun keyfun,
    bool separateAncestors,
    std::array<RankI, nchild(dim)+1> offsets,            // last idx represents ancestors.
    const std::array<RankI, nchild(dim)+1> &bucketEnds,  // last idx represents ancestors.
    ValueType *values)
{
  SFC_Tree<T, dim>::bucketStableAux.resize(sizeof(ValueType) * (end - begin));
  ValueType *auxv = reinterpret_cast<ValueType*>(SFC_Tree<T, dim>::bucketStableAux.data());

  // Bucket to aux[]
  for (size_t ii = begin; ii < end; ++ii)
  {
    const TreeNode<T,dim> key = keyfun(points[ii]);
    const unsigned char destBucket
        = (separateAncestors && key.getLevel() < lev)
          ? nchild(dim)
          : key.getMortonIndex(lev);
    assert(offsets[destBucket] < bucketEnds[destBucket]);
    auxv[(offsets[destBucket]++) - begin] = values[ii];
  }

  // Restore to values[]
  for (size_t ii = begin; ii < end; ++ii)
    values[ii] = auxv[ii - begin];
}




//
// SFC_bucketing_general()
//
template <typename T, unsigned int dim>
template <class KeyFun, typename PointType, typename KeyType, bool useCompanions, typename... Companion>
void
SFC_Tree<T,dim>:: SFC_bucketing_general(PointType *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          bool separateAncestors,
                          bool ancestorsFirst,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd,
                          Companion* ... companions
                          )
{
  using TreeNode = TreeNode<T,dim>;
  constexpr char numChildren = nchild(dim);
  constexpr char rotOffset = 2*numChildren;  // num columns in rotations[].

  // -- Reconstruct offsets and bucketEnds from returned splitters. -- //
  SFC_locateBuckets_impl<KeyFun,PointType,KeyType>(
      points, begin, end, lev, sfc,
      keyfun, separateAncestors, ancestorsFirst,
      outSplitters, outAncStart, outAncEnd);

  std::array<RankI, numChildren+1> offsets, bucketEnds;  // Last idx represents ancestors.

  for (sfc::SubIndex child_sfc(0); child_sfc < nchild(dim); ++child_sfc)
  {
    sfc::ChildNum child_m = sfc.child_num(child_sfc);
    offsets[child_m] = outSplitters[child_sfc];
    bucketEnds[child_m] = outSplitters[child_sfc+1];
  }
  offsets[numChildren] = outAncStart;
  bucketEnds[numChildren] = outAncEnd;

  // Perform bucketing using computed bucket offsets.
  if (useCompanions)
    SFC_Tree<T, dim>::SFC_bucketStable<KeyFun, PointType, KeyType>(
        points, begin, end, lev, keyfun,
        separateAncestors,
        offsets, bucketEnds,
        companions...);
  // Wait to bucket the keys until all companions have been bucketed.
  SFC_Tree<T, dim>::SFC_bucketStable<KeyFun, PointType, KeyType>(
      points, begin, end, lev, keyfun,
      separateAncestors,
      offsets, bucketEnds,
      points);
}


//
// SFC_locateBuckets_impl()
//
template <typename T, unsigned int dim>
template <class KeyFun, typename PointType, typename KeyType>
void
SFC_Tree<T,dim>:: SFC_locateBuckets_impl(const PointType *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          bool separateAncestors,
                          bool ancestorsFirst,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd)
{
  using TreeNode = TreeNode<T,dim>;
  constexpr char numChildren = nchild(dim);
  constexpr char rotOffset = 2*numChildren;  // num columns in rotations[].

  std::array<int, numChildren> counts;
  counts.fill(0);
  int countAncestors = 0;   // Special bucket to ensure ancestors bucketed properly.
  for (const PointType *pt = points + begin; pt < points + end; pt++)
  {
    const KeyType &tn = keyfun(*pt);
    if (separateAncestors && tn.getLevel() < lev)
      countAncestors++;
    else
      counts[tn.getMortonIndex(lev)]++;
  }

  /// std::array<RankI, numChildren+1> offsets, bucketEnds;  // Last idx represents ancestors.
  RankI accum;
  if (ancestorsFirst)
    accum = begin + countAncestors;                  // Ancestors belong in front.
  else
    accum = begin;                                   // Else first child is front.

  ChildI child_sfc = 0;
  for ( ; child_sfc < numChildren; child_sfc++)
  {
    ChildI child = sfc.child_num(sfc::SubIndex(child_sfc));
    outSplitters[child_sfc] = accum;
    /// offsets[child] = accum;           // Start of bucket. Moving marker.
    accum += counts[child];
    /// bucketEnds[child] = accum;        // End of bucket. Fixed marker.
  }
  outSplitters[child_sfc] = accum;  // Should be the end of siblings..

  if (ancestorsFirst)
  {
    /// offsets[numChildren] = begin;
    /// bucketEnds[numChildren] = begin + countAncestors;
    outAncStart = begin;
    outAncEnd = begin + countAncestors;
  }
  else
  {
    /// offsets[numChildren] = accum;
    /// bucketEnds[numChildren] = accum + countAncestors;
    outAncStart = accum;
    outAncEnd = accum + countAncestors;
  }
}



//
// SFC_Tree::dist_bcastSplitters()
//
template <typename T, unsigned int dim>
std::vector<TreeNode<T,dim>> SFC_Tree<T,dim>::dist_bcastSplitters(const TreeNode<T,dim> *start, MPI_Comm comm)
{
  DOLLAR("dist_bcastSplitters.singleComm");
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  using TreeNode = TreeNode<T,dim>;
  std::vector<TreeNode> splitters(nProc);
  par::Mpi_Allgather(start, splitters.data(), 1, comm);

  return splitters;
}


//
// SFC_Tree::dist_bcastSplitters() (when activeComm != globalComm)
//
template <typename T, unsigned int dim>
std::vector<TreeNode<T, dim>> SFC_Tree<T, dim>::dist_bcastSplitters(
    const TreeNode<T, dim> *start,
    MPI_Comm globalComm,
    MPI_Comm activeComm,
    bool isActive,
    std::vector<int> &activeList)
{
  DOLLAR("dist_bcastSplitters.2Comm");
  int rProc, nProc;
  MPI_Comm_size(globalComm, &nProc);
  MPI_Comm_rank(globalComm, &rProc);

  int activeSize, activeRank;;
  if (isActive)
  {
    MPI_Comm_size(activeComm, &activeSize);
    MPI_Comm_rank(activeComm, &activeRank);
  }

  // Decide on a global root, who must also be an active rank.
  const int voteRoot = (isActive ? rProc : nProc);
  int globalRoot = -1, activeRoot = -1;
  par::Mpi_Allreduce(&voteRoot, &globalRoot, 1, MPI_MIN, globalComm);
  if (rProc == globalRoot)
    activeRoot = activeRank;
  par::Mpi_Bcast(&activeRoot, 1, globalRoot, globalComm); // For active ranks.
  par::Mpi_Bcast(&activeSize, 1, globalRoot, globalComm); // For inactive ranks.

  activeList.clear();
  activeList.resize(activeSize);
  std::vector<TreeNode<T, dim>> activeSplitters(activeSize);

  if (isActive)
  {
    // Collect from active ranks to active root.
    par::Mpi_Gather(&rProc, &activeList[0], 1, activeRoot, activeComm);
    par::Mpi_Gather(start, &activeSplitters[0], 1, activeRoot, activeComm);
  }
  // Share with everyone.
  par::Mpi_Bcast(&activeList[0], activeSize, globalRoot, globalComm);
  par::Mpi_Bcast(&activeSplitters[0], activeSize, globalRoot, globalComm);

  return activeSplitters;
}




//
// treeNode2PartitionRank()  -- relative to active splitters
//
template <typename T, unsigned int dim>
std::vector<int> SFC_Tree<T, dim>::treeNode2PartitionRank(
    const std::vector<TreeNode<T, dim>> &treeNodes,
    const std::vector<TreeNode<T, dim>> &partitionFrontSplitters)
{
  DOLLAR("treeNode2PartitionRank.tcc");
  // Result
  std::vector<int> rankIds(treeNodes.size(), -1);

  // Concatenate [elements | splitters]
  std::vector<TreeNode<T, dim>> keys;
  keys.insert(keys.end(), treeNodes.cbegin(), treeNodes.cend());
  keys.insert(keys.end(), partitionFrontSplitters.cbegin(), partitionFrontSplitters.cend());

  // Indices into result, which we use after sorting. [indices | {-1,...-1}]
  std::vector<size_t> indices(treeNodes.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::fill_n(std::back_inserter(indices), partitionFrontSplitters.size(), -1);

  int rank = -1;

  SFC_Tree<T, dim>::locTreeSort(keys, indices);
  size_t next_ii = 0;
  for (size_t ii = 0; ii < keys.size(); ii = next_ii)
  {
    bool hasFrontSplitter = false;
    next_ii = ii;
    while (next_ii < keys.size() && keys[next_ii].getX() == keys[ii].getX())
    {
      const bool isFrontSplitter = (indices[next_ii] == -1);
      hasFrontSplitter |= isFrontSplitter;

      next_ii++;
    }

    if (hasFrontSplitter)
      rank++;

    for (size_t jj = ii; jj < next_ii; ++jj)
      if (indices[jj] != -1)
        rankIds[indices[jj]] = rank;
  }

  return rankIds;
}


//
// treeNode2PartitionRank()  -- mapped to global rank ids
//
template <typename T, unsigned int dim>
std::vector<int> SFC_Tree<T, dim>::treeNode2PartitionRank(
    const std::vector<TreeNode<T, dim>> &treeNodes,
    const std::vector<TreeNode<T, dim>> &partitionFrontSplitters,
    const std::vector<int> &partitionActiveList)
{
  std::vector<int> rankIds = SFC_Tree<T, dim>::treeNode2PartitionRank(
      treeNodes, partitionFrontSplitters);

  for (int &id : rankIds)
    id = partitionActiveList[id];

  return rankIds;
}




} // end namespace ot
