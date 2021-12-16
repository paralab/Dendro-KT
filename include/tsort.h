/**
 * @file:tsort.h
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2018-12-03
 * @brief: Based on work by Milinda Fernando and Hari Sundar.
 * - Algorithms: SC18 "Comparison Free Computations..." TreeSort, TreeConstruction, TreeBalancing
 * - Code: Dendro4 [sfcSort.h] [construct.cpp]
 *
 * My contribution is to extend the data structures to 4 dimensions (or higher).
 */

#ifndef DENDRO_KT_SFC_TREE_H
#define DENDRO_KT_SFC_TREE_H

#include "treeNode.h"
#include <mpi.h>
#include <vector>
#include "hcurvedata.h"
#include "parUtils.h"
#include "filterFunction.h"
#include <stdio.h>

namespace type
{
  /**
   * StrongIndex
   * Based on pattern by Jonathan Boccara:
   *    https://www.fluentcpp.com/2016/12/08/strong-types-for-strong-interfaces/
   */
  template <typename T, typename Unique>
  struct StrongIndex
  {
    using Type = T;
    T m_i;

    // Construction & conversion
    explicit StrongIndex(const T i) : m_i(i) {}
    operator T() const { return m_i; }
    T get() const { return m_i; }

    // Arithmetic
    StrongIndex & operator++() { ++m_i; return *this; }
    StrongIndex & operator+=(const T d) { m_i += d; return *this; }
    StrongIndex operator+(const T d) const { return StrongIndex(m_i + d); }
        // others as needed
  };
};

namespace ot
{

using LevI   = unsigned int;
using RankI  = DendroIntL;
using RotI   = int;
using ChildI = char;

namespace OCT_FLAGS
{
  enum Refine {OCT_NO_CHANGE = 0, OCT_REFINE = 1, OCT_COARSEN = 2};
}

enum GridAlignment { CoarseByFine, FineByCoarse };
enum RemeshPartition { SurrogateOutByIn, SurrogateInByOut };


namespace sfc
{
  using ChildNum = type::StrongIndex<unsigned char, struct ChildNum_>;
  using SubIndex = type::StrongIndex<unsigned char, struct SubIndex_>;
  using RotIndex = type::StrongIndex<int, struct RotIndex_>;
}

template <int dim>
struct SFC_State
{
  constexpr static int nchild() { return 1 << dim; }
  static SFC_State root() { return SFC_State(); }

  SFC_State() = default;

  SFC_State(const sfc::RotIndex rotation)
    : m_rotation(rotation)
  {}
  inline sfc::RotIndex state() const { return m_rotation; }

  inline sfc::ChildNum child_num(sfc::SubIndex i) const;    // rot_perm
  inline sfc::SubIndex child_rank(sfc::ChildNum cn) const;  // rot_inv
  inline SFC_State subcurve(sfc::SubIndex i) const;         // traversal
  inline SFC_State child_curve(sfc::ChildNum cn) const;     // traversal

  sfc::RotIndex m_rotation{0};
};


template <int dim>
sfc::ChildNum SFC_State<dim>::child_num(sfc::SubIndex i) const
{
  return sfc::ChildNum(rotations[(m_rotation * 2 + 0) * nchild() + i]);
}

template <int dim>
sfc::SubIndex SFC_State<dim>::child_rank(sfc::ChildNum cn) const
{
  return sfc::SubIndex(rotations[(m_rotation * 2 + 1) * nchild() + cn]);
}

template <int dim>
SFC_State<dim> SFC_State<dim>::child_curve(sfc::ChildNum cn) const
{
  return SFC_State{sfc::RotIndex(HILBERT_TABLE[m_rotation * nchild() + cn])};
}

template <int dim>
SFC_State<dim> SFC_State<dim>::subcurve(sfc::SubIndex i) const
{
  return child_curve(child_num(i));
}



//
// BucketInfo{}
//
// Buckets to temporarily represent (interior) nodes in the hyperoctree
// while we carry out breadth-first traversal. See distTreeSort().
template <typename T>
struct BucketInfo          // Adapted from Dendro4 sfcSort.h:132.
{
  sfc::RotIndex rot;
  LevI lev;
  T begin;
  T end;

  static BucketInfo root(T size)
  {
    return BucketInfo{ sfc::RotIndex(0), 0, 0, size};
  }
};

template struct BucketInfo<RankI>;


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

template <typename T, unsigned int dim>
struct KeyFunIdentity_TN
{
  const TreeNode<T,dim> &operator()(const TreeNode<T,dim> &tn) { return tn; }
};

template <typename PointType>
struct KeyFunIdentity_Pt
{
  const PointType &operator()(const PointType &pt) { return pt; }
};

template <typename T, unsigned int dim>
struct KeyFunIdentity_maxDepth
{
  const TreeNode<T,dim> operator()(TreeNode<T,dim> tn) { tn.setLevel(m_uiMaxDepth); return tn; }
};


template <typename T, int dim>
struct PartitionSplitters
{
  std::vector<TreeNode<T, dim>> m_firsts;
};


template <typename T, unsigned int dim>
struct SFC_Tree
{
  template <class PointType>
  static void locTreeSort(std::vector<PointType> &points)
  {
    SFC_Tree<T, dim>::locTreeSort(&(*points.begin()), 0, (RankI) points.size(), 0, m_uiMaxDepth, SFC_State<dim>::root());
  }


  template <class PointType, typename... CompanionT>
  static void locTreeSort(std::vector<PointType> &points, std::vector<CompanionT>& ... companions)
  {
    SFC_Tree<T, dim>::locTreeSort<KeyFunIdentity_Pt<PointType>,
                                PointType,
                                PointType,
                                true,
                                CompanionT...
                                >
         (points.data(),
          0, (RankI) points.size(),
          1, m_uiMaxDepth, SFC_State<dim>::root(),
          KeyFunIdentity_Pt<PointType>(),
          (companions.data())...
          );
  }



  template <class PointType>
  static void locTreeSortMaxDepth(std::vector<PointType> &points)
  {
    SFC_Tree<T, dim>::locTreeSort< KeyFunIdentity_maxDepth<T, dim>,
                                 PointType, TreeNode<T, dim>, false, int>
      (&(*points.begin()), 0, (RankI) points.size(), 0, m_uiMaxDepth, SFC_State<dim>::root(),
       KeyFunIdentity_maxDepth<T, dim>(), (int*) nullptr);
  }

  template <class PointType, typename... CompanionT>
  static void locTreeSortMaxDepth(std::vector<PointType> &points, std::vector<CompanionT>& ... companions)
  {
    SFC_Tree<T, dim>::locTreeSort< KeyFunIdentity_maxDepth<T, dim>,
                                 PointType, TreeNode<T, dim>, true, CompanionT...>
      (&(*points.begin()), 0, (RankI) points.size(), 0, m_uiMaxDepth, SFC_State<dim>::root(),
       KeyFunIdentity_maxDepth<T, dim>(), (companions.data())...);
  }



  // Notes:
  //   - This method operates in-place.
  //   - From sLev to eLev INCLUSIVE
  template <class PointType>   // = TreeNode<T,dim>
  static void locTreeSort(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc);            // Initial rotation, use 0 if sLev is 1.

  // Notes:
  //   - Allows the generality of a ``key function,''
  //        i.e. function to produce TreeNodes-like objects to sort by.
  //   - Otherwise, same as above except shuffles a parallel companion array along with the TreeNodes.
  template <class KeyFun, typename PointType, typename KeyType, bool useCompanions, typename... Companion>
  static void locTreeSort(PointType *points,
                          RankI begin, RankI end,
                          LevI sLev,
                          LevI eLev,
                          SFC_State<dim> sfc,            // Initial rotation, use 0 if sLev is 1.
                          KeyFun keyfun,
                          Companion * ... companions
                          );

  // Notes:
  //   - outSplitters contains both the start and end of children at level `lev'
  //     This is to be consistent with the Dendro4 SFC_bucketing().
  //
  //     One difference is that here the buckets are ordered by the SFC
  //     (like the returned data is ordered) and so `outSplitters' should be
  //     monotonically increasing; whereas in Dendro4 SFC_bucketing(), the splitters
  //     are in permuted order.
  //
  //   - The size of outSplitters is 2+numChildren, which are splitters for
  //     1+numChildren buckets. The leading bucket holds ancestors and the
  //     remaining buckets are for children.
  static void SFC_bucketing(TreeNode<T,dim> *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd);

  /**
   * @tparam KeyFun KeyType KeyFun::operator()(PointType);
   * @tparam KeyType must support the public interface of TreeNode<T,dim>.
   * @tparam PointType passive data type.
   * @param ancestorsFirst If true, ancestor bucket precedes all siblings, else follows all siblings.
   */
  // Notes:
  //   - Buckets points based on TreeNode "keys" generated by applying keyfun(point).
  template <class KeyFun, typename PointType, typename KeyType>
  static void SFC_bucketing_impl(PointType *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          bool separateAncestors,
                          bool ancestorsFirst,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd);


  // Notes:
  //   - Same as above except shuffles a parallel companion array along with the TreeNodes.
  //   - In actuality the above version calls this one.
  template <class KeyFun, typename PointType, typename KeyType, bool useCompanions, typename... Companion>
  static void SFC_bucketing_general(PointType *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          bool separateAncestors,
                          bool ancestorsFirst,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd,
                          Companion * ... companions
                          );

  static std::vector<char> bucketStableAux;

  // Template-recursively buckets one companion array at a time.
  template <class KeyFun, typename PointType, typename KeyType, typename ValueType>
  static void SFC_bucketStable(
      const PointType *points,
      RankI begin,
      RankI end,
      LevI lev,
      KeyFun keyfun,
      bool separateAncestors,
      std::array<RankI, nchild(dim)+1> offsets,            // last idx represents ancestors.
      const std::array<RankI, nchild(dim)+1> &bucketEnds,  // last idx represents ancestors.
      ValueType *values);
  template <class KeyFun, typename PointType, typename KeyType, typename CompanionHead, typename... CompanionTail>
  static void SFC_bucketStable(
      const PointType *points,
      RankI begin,
      RankI end,
      LevI lev,
      KeyFun keyfun,
      bool separateAncestors,
      const std::array<RankI, nchild(dim)+1> &offsets,     // last idx represents ancestors.
      const std::array<RankI, nchild(dim)+1> &bucketEnds,  // last idx represents ancestors.
      CompanionHead * companionHead,
      CompanionTail* ... companionTail);



  /**
   * @tparam KeyFun KeyType KeyFun::operator()(PointType);
   * @tparam KeyType must support the public interface of TreeNode<T,dim>.
   * @tparam PointType passive data type.
   * @param ancestorsFirst If true, ancestor bucket precedes all siblings, else follows all siblings.
   */
  // Notes:
  //   - Buckets points based on TreeNode "keys" generated by applying keyfun(point).
  //   - Same parameters as SFC_bucketing_impl, except does not move points, hence read-only.
  template <class KeyFun, typename PointType, typename KeyType>
  static void SFC_locateBuckets_impl(const PointType *points,
                          RankI begin, RankI end,
                          LevI lev,
                          SFC_State<dim> sfc,
                          KeyFun keyfun,
                          bool separateAncestors,
                          bool ancestorsFirst,
                          std::array<RankI, 1+nchild(dim)> &outSplitters,
                          RankI &outAncStart,
                          RankI &outAncEnd);



  // Notes:
  //   - Same parameters as SFC_bucketing, except does not move points.
  //   - This method is read only.
  static void SFC_locateBuckets(const TreeNode<T,dim> *points,
                                RankI begin, RankI end,
                                LevI lev,
                                SFC_State<dim> sfc,
                                std::array<RankI, 1+nchild(dim)> &outSplitters,
                                RankI &outAncStart,
                                RankI &outAncEnd);


  // -----

  /** Return index to first element _not less than_ key. */
  static size_t tsearch_lower_bound(
      const std::vector<TreeNode<T, dim>> &sortedOcts,
      const TreeNode<T, dim> &key);

  /** Return index to first element _greater than_ key. */
  static size_t tsearch_upper_bound(
      const std::vector<TreeNode<T, dim>> &sortedOcts,
      const TreeNode<T, dim> &key);

  /** Find first geq and first greater. */
  static std::pair<size_t, size_t> tsearch_equal_range(
      const std::vector<TreeNode<T, dim>> &sortedOcts,
      const TreeNode<T, dim> &key);

  /** Find first geq and first greater. */
  // Only handles octants, not boundary points.
  static std::pair<size_t, size_t> tsearch_equal_range(
      const TreeNode<T, dim> *sortedOcts,
      const TreeNode<T, dim> &key,
      size_t begin, size_t end,
      LevI sLev,
      SFC_State<dim> sfc);


  /** For each key, returns two things from sortedOcts:
   *    - The index of the first element not less than the key, and
   *    - a list of zero or more indices of ancestor octants (appended).
   *  From these, the subset of sortedOcts overlapping each key can be recovered.
   *  Strict ancestor overlaps may be dispersed in the input,
   *  while inclusive descendant overlaps must occur in a contiguous sequence.
   *
   * @param sortedOcts [in] A sorted list of octants, not necessarily unique.
   * @param uniqKeys [in] A sorted list of unique, nonoverlapping octants.
   * @param beginOverlapsBounds [out] Begining of segment in overlapsBounds for each key.
   * @param overlapsBounds [out] Concatenated lists of ancestor overlaps and lower bounds.
   */
  static void overlaps_lower_bound(
      const std::vector<TreeNode<T, dim>> &sortedOcts,
      const std::vector<TreeNode<T, dim>> &uniqKeys,
      std::vector<size_t> &beginOverlapsBounds,
      std::vector<size_t> &overlapsBounds);

  // -----




  // Notes:
  //   - points will be replaced/resized with globally sorted data.
  static void distTreeSort(std::vector<TreeNode<T,dim>> &points,
                           double loadFlexibility,
                           MPI_Comm comm);

  // This method does most of the work for distTreeSort and distTreeConstruction.
  // It includes the breadth-first global sorting phase and Alltoallv()
  // but does not sort locally.
  //
  // pFinalOctants is an output parameter of the global refinement structure.
  // If it is NULL then it is unused.
  // If it is not NULL then it is cleared and filled with the output data.
  //
  // @param noSplitThresh takes precedence over loadFlexibility,
  //        such that, for any non-empty bucket of global contents <= noSplitThresh
  //        whose parent has global contents > noSplitThresh,
  //        the parent will not be split across processors, but will
  //        land completely onto a single processor.
  //        To ignore this parameter, set noSplitThresh=0.
  //
  // Notes:
  //   - points will be replaced/resized with globally sorted data.
  static void distTreePartition(std::vector<TreeNode<T,dim>> &points,
                           unsigned int noSplitThresh,
                           double loadFlexibility,
                           MPI_Comm comm);

  static par::SendRecvSchedule
    distTreePartitionSchedule(std::vector<TreeNode<T,dim>> &points,
                           unsigned int noSplitThresh,
                           double loadFlexibility,
                           MPI_Comm comm);

  //
  // treeBFTNextLevel()
  //   Takes the queue of BucketInfo in a breadth-first traversal, and finishes
  //   processing the current level. Each dequeued bucket is subdivided,
  //   and the sub-buckets in the corresponding range of `points` are sorted.
  //   Then the sub-buckets are initialized and enqueued to the back.
  //
  static void treeBFTNextLevel(TreeNode<T,dim> *points,
      std::vector<BucketInfo<RankI>> &bftQueue);


  /**
   * @brief Broadcast the first TreeNode from every processor so we have global access to the splitter list.
   */
  static std::vector<TreeNode<T,dim>> dist_bcastSplitters(const TreeNode<T,dim> *start, MPI_Comm comm);

  static std::vector<TreeNode<T, dim>> dist_bcastSplitters(
      const TreeNode<T, dim> *start,
      MPI_Comm globalComm,
      MPI_Comm activeComm,
      bool isActive,
      std::vector<int> &activeList);

  /**
   * @brief Allgather the first TreeNode from every processor.
   * @description
   *        An empty rank is represented by getting its successor's splitter,
   *        or the root TreeNode if it has no successor.
   *        This method does not call MPI_Comm_split().
   */

  static PartitionSplitters<T, dim> allgatherSplitters(
      bool nonempty,
      const TreeNode<T, dim> &front,
      MPI_Comm comm,
      std::vector<int> *activeList = nullptr);

  /**
   * @brief Map a set of treeNodes in the domain to the partition ranks
   *        that own them. Repeat splitters are allowed.
   *        Empty ranks in the partition are allowed.
   */
  static std::vector<int> treeNode2PartitionRank(
      const std::vector<TreeNode<T, dim>> &treeNodes,
      const PartitionSplitters<T, dim> &partitionSplitters);


  /** @brief Map any collection of treeNodes in the domain
   *         to the partition ranks that own them.
   *         The rank ids are returned
   *         in the range [0 .. partitionFrontSplitters.size()-1].
   */
  static std::vector<int> treeNode2PartitionRank(
      const std::vector<TreeNode<T,dim>> &treeNodes,
      const std::vector<TreeNode<T,dim>> &partitionFrontSplitters);

  /** @brief Map any collection of treeNodes in the domain
   *         to the partition ranks that own them.
   *         partitionFrontSplitters contains front elements from active ranks.
   *         partitionActiveList contains the global rank ids of active ranks.
   *         The rank ids are returned
   *         in the range [0 .. max{partitionActiveList}].
   */
  static std::vector<int> treeNode2PartitionRank(
      const std::vector<TreeNode<T,dim>> &treeNodes,
      const std::vector<TreeNode<T,dim>> &partitionFrontSplitters,
      const std::vector<int> &partitionActiveList);


  // -------------------------------------------------------------

  // Notes:
  //   - (Sub)tree will be built by appending to `tree'.
  static void locTreeConstruction(TreeNode<T,dim> *points,
                                  std::vector<TreeNode<T,dim>> &tree,
                                  RankI maxPtsPerRegion,
                                  RankI begin, RankI end,
                                  LevI sLev,
                                  LevI eLev,
                                  SFC_State<dim> sfc,
                                  TreeNode<T,dim> pNode);

  static void distTreeConstruction(std::vector<TreeNode<T,dim>> &points,
                                   std::vector<TreeNode<T,dim>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm);

  static void locCompleteResolved(
      const TreeNode<T, dim> *octants,
      std::vector<TreeNode<T, dim>> &tree,  // expects empty
      RankI begin, RankI end,
      SFC_State<dim> sfc,
      TreeNode<T,dim> pNode);


  static void locTreeConstructionWithFilter( const ibm::DomainDecider &decider,
                                             TreeNode<T,dim> *points,
                                             std::vector<TreeNode<T,dim>> &tree,
                                             RankI maxPtsPerRegion,
                                             RankI begin, RankI end,
                                             LevI sLev,
                                             LevI eLev,
                                             SFC_State<dim> sfc,
                                             TreeNode<T,dim> pNode);

  static void locTreeConstructionWithFilter( const ibm::DomainDecider &decider,
                                             bool refineAll,
                                             std::vector<TreeNode<T,dim>> &tree,
                                             LevI sLev,
                                             LevI eLev,
                                             SFC_State<dim> sfc,
                                             TreeNode<T,dim> pNode);

  static void distTreeConstructionWithFilter(
                                   const ibm::DomainDecider &decider,
                                   std::vector<TreeNode<T,dim>> &points,
                                   std::vector<TreeNode<T,dim>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm);

  static void distTreeConstructionWithFilter( const ibm::DomainDecider &decider,
                                              bool refineAll,
                                              std::vector<TreeNode<T,dim>> &tree,
                                              LevI eLev,
                                              double loadFlexibility,
                                              MPI_Comm comm);

  static constexpr bool RM_DUPS_AND_ANC = false;
  static constexpr bool RM_DUPS_ONLY = true;

  static void distRemoveDuplicates(std::vector<TreeNode<T,dim>> &tree,
                                   double loadFlexibility,
                                   bool strict,
                                   MPI_Comm comm);

  // Removes duplicate/ancestor TreeNodes from a sorted list of TreeNodes.
  // Notes:
  //   - Removal is done in a single pass in-place. The vector may be shrunk.
  static void locRemoveDuplicates(std::vector<TreeNode<T,dim>> &tnodes);

  // Notes:
  //   - Nodes only removed if strictly equal to other nodes. Ancestors retained.
  static void locRemoveDuplicatesStrict(std::vector<TreeNode<T,dim>> &tnodes);

  /**
   * distCoalesceSiblings()
   *
   * @brief If all siblings are leafs, push them onto the first incident rank.
   *
   * Enforcing this criterion is a prerequisite to intergrid transfer.
   *
   * Simpler than keepSiblingLeafsTogether.
   */
  static void distCoalesceSiblings( std::vector<TreeNode<T, dim>> &tree,
                                    MPI_Comm comm );


  // -------------------------------------------------------------

  static std::vector<TreeNode<T, dim>> locRemesh( const std::vector<TreeNode<T, dim>> &inTree,
                                                const std::vector<OCT_FLAGS::Refine> &refnFlags );

  /**
   * @note Whichever of the input and output grids is controlling partitioning
   *       of the surrogate grid, it is assumed to either
   *       be coarser or have coalesced siblings.
   *       Old default was SurrogateInByOut .
   */
  static void distRemeshWholeDomain( const std::vector<TreeNode<T, dim>> &inTree,
                                     const std::vector<OCT_FLAGS::Refine> &refnFlags,
                                     std::vector<TreeNode<T, dim>> &outTree,
                                     double loadFlexibility,
                                     MPI_Comm comm );

  // When remeshing with the SFC_Tree interface, surrogate grid is optional.
  // Use getSurrogateGrid method after distRemeshWholeDomain() to recover the surrogate.
  static std::vector<TreeNode<T, dim>> getSurrogateGrid(
      RemeshPartition remeshPartition,
      const std::vector<TreeNode<T, dim>> &oldTree,
      const std::vector<TreeNode<T, dim>> &newTree,
      MPI_Comm comm);

  static std::vector<TreeNode<T, dim>> getSurrogateGrid( const std::vector<TreeNode<T, dim>> &replicateGrid,
                                                       const std::vector<TreeNode<T, dim>> &splittersFromGrid,
                                                       MPI_Comm comm );

  // -------------------------------------------------------------

  /**
   * @brief Create auxiliary octants in bottom-up order to close the 2:1-balancing constraint.
   */
  static void propagateNeighbours(std::vector<TreeNode<T,dim>> &tree);

  // Notes:
  //   - Constructs a tree based on distribution of points, then balances and completes.
  //   - Initializes tree with balanced complete tree.
  static void locTreeBalancing(std::vector<TreeNode<T,dim>> &points,
                               std::vector<TreeNode<T,dim>> &tree,
                               RankI maxPtsPerRegion);

  static void distTreeBalancing(std::vector<TreeNode<T,dim>> &points,
                                   std::vector<TreeNode<T,dim>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm);

  static void locTreeBalancingWithFilter(
                               const ibm::DomainDecider &decider,
                               std::vector<TreeNode<T,dim>> &points,
                               std::vector<TreeNode<T,dim>> &tree,
                               RankI maxPtsPerRegion);

  static void distTreeBalancingWithFilter(
                                   const ibm::DomainDecider &decider,
                                   std::vector<TreeNode<T,dim>> &points,
                                   std::vector<TreeNode<T,dim>> &tree,
                                   RankI maxPtsPerRegion,
                                   double loadFlexibility,
                                   MPI_Comm comm);

  /** Maintains effective domain, splitting octants that violate 2:1-balance. */
  static void locMinimalBalanced(std::vector<TreeNode<T, dim>> &tree);

  /** Maintains effective domain, splitting octants larger than res octants. */
  static void locResolveTree(std::vector<TreeNode<T, dim>> &tree,
                             const std::vector<TreeNode<T, dim>> &res);

  /** Assumes sorted balanced tree.
   * Returns octants whose potential neighbors are not strictly contained
   * in the SFC domain spanned by {first..last}. */
  static std::vector<TreeNode<T, dim>> unstableOctants(
      const std::vector<TreeNode<T, dim>> &tree,
      const bool dangerLeft = true,     // mark unstable if go outside first
      const bool dangerRight = true);   // mark unstable if go outside last

  // -------------------------------------------------------------

  /**
   * @brief Given partition splitters and a list of (unordered) points, finds every block that contains at least some of the points.
   * @param splitters an array that holds the leading boundary of each block.
   * @note Assumes that the points are at the deepest level.
   * @note Assumes that the partition splitters are already SFC-sorted.
   */
  // Use this one.
  static void getContainingBlocks(TreeNode<T,dim> *points,
                                  RankI begin, RankI end,
                                  const TreeNode<T,dim> *splitters,
                                  int numSplitters,
                                  std::vector<int> &outBlocks);

  // Recursive implementation.
  static void getContainingBlocks(TreeNode<T,dim> *points,
                                  RankI begin, RankI end,
                                  const TreeNode<T,dim> *splitters,
                                  RankI sBegin, RankI sEnd,
                                  LevI lev, SFC_State<dim> sfc,
                                  int &numPrevBlocks,
                                  const int startSize,
                                  std::vector<int> &outBlocks);

  // -------------------------------------------------------------

  /** @brief Successively computes 0th child in SFC order to given level. */
  static void firstDescendant(TreeNode<T,dim> &parent,
                              RotI &pRot,
                              LevI descendantLev);

  /** @brief Successively computes (n-1)th child in SFC order to given level. */
  static void lastDescendant(TreeNode<T,dim> &parent,
                             RotI &pRot,
                             LevI descendantLev);

};



/** Assumes tree is a distributed tree with no overlaps. */
template <typename T, unsigned int dim>
bool is2to1Balanced(const std::vector<TreeNode<T, dim>> &tree, MPI_Comm comm);

template <typename T, unsigned int dim>
bool isLocallySorted(const std::vector<TreeNode<T, dim>> &octList);

template <typename T, unsigned int dim>
size_t lenContainedSorted(
    const TreeNode<T, dim> *octList,
    size_t begin, size_t end,
    TreeNode<T, dim> subtree,
    SFC_State<dim> sfc);

} // namespace ot

#include "tsort.tcc"

#endif // DENDRO_KT_SFC_TREE_H
