/**
 * @author Masado Ishii
 * @date   2023-04-11
 * @brief  Find local elements adjacent to partition boundaries
 */

#ifndef DENDRO_KT_PARTITION_BORDER_HPP
#define DENDRO_KT_PARTITION_BORDER_HPP

// Function declaration for linkage purposes.
inline void link_partition_border_tests() {};

#include "include/treeNode.h"
#include "include/tsort.h"
#include "leaf_sets.hpp"

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  /// namespace traversal_quantifiers
  /// {
  ///   template <typename Contents>
  ///   struct Container { Contents what; };

  ///   template <typename Contents>
  ///   struct Any : public Container<Contents> {
  ///     using Container<Contents>::Container;
  ///     template <class Up>
  ///     operator Any<Up>() & { return static_cast<Any<Up>&>(static_cast<Container<Up>&>(*this)); }
  ///     //TODO make sure how to static cast references if Contents& -> Up& is valid
  ///   };

  ///   template <typename Contents>
  ///   struct All : public Container<Contents> {
  ///   };

  ///   template <int dim>
  ///   Any<LeafListView<dim>> any(LeafListView<dim> view) { return { view }; }

  ///   template <int dim>
  ///   All<LeafListView<dim>> all(LeafListView<dim> view) { return { view }; }
  /// };


  // LocalAdjacencyList
  struct LocalAdjacencyList
  {
    int local_rank;
    std::vector<int> neighbor_ranks;
  };

  // sfc_partition(): Returns partition indices bordered by local. (No overlap)
  template <unsigned dim>
  LocalAdjacencyList sfc_partition(
      int local_rank,
      bool is_active,
      const std::vector<int> &active_list,
      const PartitionFrontBack<uint32_t, dim> &endpoints);


  // where_border()
  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_border(
      LeafListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set,
      Emit &&emit);

  // where_border_or_overlap()
  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_border_or_overlap(
      LeafListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set,
      Emit &&emit);

  // border_any() (leaf to set)
  template <int dim, typename AnySet>
  bool border_any(
      TreeNode<uint32_t, dim> octant,
      const LeafSet<dim, AnySet> &any_set);

  // border_or_overlap_any() (leaf to set)
  template <int dim, typename AnySet>
  bool border_or_overlap_any(
      TreeNode<uint32_t, dim> octant,
      const LeafSet<dim, AnySet> &any_set);

  // border_any() (set to set)
  template <int dim, typename PSet, typename QSet>
  bool border_any(
      const LeafSet<dim, PSet> &P,
      const LeafSet<dim, QSet> &Q);

  // border_or_overlap_any() (set to set)
  template <int dim, typename PSet, typename QSet>
  bool border_or_overlap_any(
      const LeafSet<dim, PSet> &P,
      const LeafSet<dim, QSet> &Q);

  // adjacent()
  template <unsigned dim>
  bool adjacent(
      const TreeNode<uint32_t, dim> &a,
      const TreeNode<uint32_t, dim> &b);
}


// =============================================================================
// Tests
// =============================================================================
#ifdef DOCTEST_LIBRARY_INCLUDED
#include "include/tnUtils.h"
namespace ot
{

  DOCTEST_TEST_SUITE("Partition Border")
  {
    template <int dim>
    struct LeafVector
    {
      std::vector<TreeNode<uint32_t, dim>> vec;

      static LeafVector sorted(std::vector<TreeNode<uint32_t, dim>> vec) {
        SFC_Tree<uint32_t, dim>::locTreeSort(vec);
        return LeafVector{std::move(vec)};
      }

      void sort() {
        SFC_Tree<uint32_t, dim>::locTreeSort(this->vec);
      }

      // view()
      LeafListView<dim> view() const { return vec_leaf_list_view<dim>(this->vec); }

      // range()
      LeafRange<dim> range() const { return vec_leaf_range<dim>(this->vec); }
    };

    // uniform_refine_morton()
    template <unsigned dim, typename Emit>
    void uniform_refine_morton(TreeNode<uint32_t, dim> region, int depth, Emit &&emit)
    {
      if (depth == 0)
        emit(region);
      else
        for (sfc::ChildNum c(0); c < nchild(dim); ++c)
          uniform_refine_morton(region.getChildMorton(c), depth - 1, emit);
    }

    DOCTEST_TEST_CASE("adjacent")
    {
      CHECK_FALSE( adjacent(morton_lineage<2>({}), morton_lineage<2>({})) );
      CHECK_FALSE( adjacent(morton_lineage<2>({}), morton_lineage<2>({0})) );
      CHECK_FALSE( adjacent(morton_lineage<2>({}), morton_lineage<2>({1})) );
      CHECK_FALSE( adjacent(morton_lineage<2>({}), morton_lineage<2>({2})) );
      CHECK_FALSE( adjacent(morton_lineage<2>({}), morton_lineage<2>({3})) );
      CHECK_FALSE( adjacent(morton_lineage<2>({0}), morton_lineage<2>({0})) );

      CHECK( adjacent(morton_lineage<2>({0}), morton_lineage<2>({1})) );
      CHECK( adjacent(morton_lineage<2>({0}), morton_lineage<2>({2})) );
      CHECK( adjacent(morton_lineage<2>({0}), morton_lineage<2>({3})) );

      CHECK( adjacent(morton_lineage<2>({1}), morton_lineage<2>({0})) );
      CHECK( adjacent(morton_lineage<2>({1}), morton_lineage<2>({2})) );
      CHECK( adjacent(morton_lineage<2>({1}), morton_lineage<2>({3})) );

      CHECK( adjacent(morton_lineage<2>({2}), morton_lineage<2>({0})) );
      CHECK( adjacent(morton_lineage<2>({2}), morton_lineage<2>({1})) );
      CHECK( adjacent(morton_lineage<2>({2}), morton_lineage<2>({3})) );

      CHECK( adjacent(morton_lineage<2>({3}), morton_lineage<2>({0})) );
      CHECK( adjacent(morton_lineage<2>({3}), morton_lineage<2>({1})) );
      CHECK( adjacent(morton_lineage<2>({3}), morton_lineage<2>({2})) );

      CHECK( adjacent(morton_lineage<2>({0, 1}),
                      morton_lineage<2>({1, 0})) );
      CHECK_FALSE( adjacent(morton_lineage<2>({0, 1}),
                            morton_lineage<2>({1, 1})) );
    }

    DOCTEST_TEST_CASE("border_any with leaf")
    {
      constexpr int dim = 2;
      using Octant = TreeNode<uint32_t, dim>;
      _InitializeHcurve(dim);

      //  _______________
      // |       |     ¡_|
      // |       |     !_|
      // |       |  _ _¡_|
      // |_______|_|_|_!_|
      // |       |  ^ ^¡_|
      // |       |     !_|
      // |       |     ¡_|
      // |_______|_____!_|

      const LeafVector<dim> right_side = LeafVector<dim>::sorted({
        morton_lineage<dim>({1, 1, 1}),
        morton_lineage<dim>({1, 1, 3}),
        morton_lineage<dim>({1, 3, 1}),
        morton_lineage<dim>({1, 1, 3}),
        morton_lineage<dim>({3, 1, 1}),
        morton_lineage<dim>({3, 1, 3}),
        morton_lineage<dim>({3, 3, 1}),
        morton_lineage<dim>({3, 1, 3})
      });
      const auto view = right_side.view();

      CHECK( border_any<dim>(morton_lineage<dim>({3, 1, 0}), view) );

      CHECK_FALSE( border_any<dim>(morton_lineage<dim>({3, 0, 1}), view) );

      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("Simple 2D partition boundary")
    {
      // This partitioning should work with both Hilbert and Z curve.
      //  _______________
      // |???    |    ???|     Two partitions defined by space-filling curve.
      // |???    |    ???|     The last cells are in one of the "???" corners.
      // |       |       |
      // |.......!_______|     Higher partition gets the top 6 regions.
      // |       ¡   |   |
      // |       !...!___|     Lower partition gets the bottom 4 regions.
      // |       |   ¡.|.| ___/
      // |_______|___|_|_|

      constexpr int dim = 2;
      using Octant = TreeNode<uint32_t, dim>;
      _InitializeHcurve(dim);

      // Partition definitions.
      const Octant root = {};

      const LeafVector<dim> part_low = LeafVector<dim>::sorted({
        morton_lineage(root, {0}),
        morton_lineage(root, {1, 0}),
        morton_lineage(root, {1, 1, 0}),
        morton_lineage(root, {1, 1, 1})
      });
      const LeafVector<dim> part_high = LeafVector<dim>::sorted({
        morton_lineage(root, {1, 1, 2}),
        morton_lineage(root, {1, 1, 3}),
        morton_lineage(root, {1, 2}),
        morton_lineage(root, {1, 3}),
        morton_lineage(root, {2}),
        morton_lineage(root, {3}),
      });

      // Uniformly refine each region by 3 levels.
      const int refine_depth = 3;
      const size_t expected_interpart_low = 5 * intPow(2, refine_depth);
      const size_t expected_interpart_high = 6 * intPow(2, refine_depth);

      LeafVector<dim> list_low,  list_high;
      for (Octant region : part_low.vec)
        uniform_refine_morton(region, refine_depth, [&list_low](Octant oct) {
            list_low.vec.push_back(oct);
        });
      for (Octant region : part_high.vec)
        uniform_refine_morton(region, refine_depth, [&list_high](Octant oct) {
            list_high.vec.push_back(oct);
        });
      list_low.sort();
      list_high.sort();

      /// using namespace traversal_quantifiers;

      // spoof
      const auto any = [](auto &&x) { return std::forward<decltype(x)>(x); };
      const auto all = [](auto &&x) { return std::forward<decltype(x)>(x); };

      LeafVector<dim> interpart_low,  interpart_high;

      DOCTEST_SUBCASE("With any_set of LeafListView")
      {
        where_border(all(list_low.view()), any(part_high.view()), [&interpart_low](const Octant *oct) {
            interpart_low.vec.push_back(*oct);
        });
        where_border(all(list_high.view()), any(part_low.view()), [&interpart_high](const Octant *oct) {
            interpart_high.vec.push_back(*oct);
        });
      }

      DOCTEST_SUBCASE("With any_set of LeafRange")
      {
        where_border(all(list_low.view()), any(part_high.range()), [&interpart_low](const Octant *oct) {
            interpart_low.vec.push_back(*oct);
        });
        where_border(all(list_high.view()), any(part_low.range()), [&interpart_high](const Octant *oct) {
            interpart_high.vec.push_back(*oct);
        });

        CHECK( border_any(any(part_low.range()), any(part_high.range())) );

        if (refine_depth > 1)
        {
          const size_t tail_length = intPow(2, dim * (refine_depth - 1));
          const LeafRange<dim> tail = LeafRange<dim>::make(
              *(list_high.vec.end() - tail_length), list_high.vec.back());
          CHECK_FALSE( border_any(any(part_low.range()), any(tail)) );
        }
      }

      CHECK( interpart_low.vec.size() == expected_interpart_low );
      CHECK( interpart_high.vec.size() == expected_interpart_high );

      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("sfc_partition")
    {
      constexpr int dim = 2;
      using Octant = TreeNode<uint32_t, dim>;
      _InitializeHcurve(dim);

      // Partition definitions.
      const Octant root = {};
      const LeafVector<dim> parts[2] = {
        LeafVector<dim>::sorted({
          morton_lineage(root, {0, 0}),
          morton_lineage(root, {0, 1}),
          morton_lineage(root, {0, 2}),
          morton_lineage(root, {0, 3}),
          morton_lineage(root, {1, 0}),
          morton_lineage(root, {1, 1}),
          morton_lineage(root, {1, 2}),
          morton_lineage(root, {1, 3})
        }),
        LeafVector<dim>::sorted({
          morton_lineage(root, {2, 0}),
          morton_lineage(root, {2, 1}),
          morton_lineage(root, {2, 2}),
          morton_lineage(root, {2, 3}),
          morton_lineage(root, {3, 0}),
          morton_lineage(root, {3, 1}),
          morton_lineage(root, {3, 2}),
          morton_lineage(root, {3, 3})
        })
      };

      const bool is_active = true;
      const std::vector<int> active_list = {0, 1};
      const PartitionFrontBack<uint32_t, dim> endpoints = {
        {parts[0].vec.front(), parts[1].vec.front()},
        {parts[0].vec.back(), parts[1].vec.back()}
      };

      for (int rank: {0, 1})
      {
        LocalAdjacencyList adjacency_list =
          sfc_partition<dim>(rank, is_active, active_list, endpoints);

        CHECK( adjacency_list.local_rank == rank );
        CHECK( adjacency_list.neighbor_ranks.size() == 1 );
        if (adjacency_list.neighbor_ranks.size() == 1)
          CHECK( adjacency_list.neighbor_ranks[0] != rank );
      }

      _DestroyHcurve();
    }


    DOCTEST_TEST_CASE("sfc_partition do not overlap")
    {
      constexpr int dim = 2;
      using Octant = TreeNode<uint32_t, dim>;
      _InitializeHcurve(dim);

      // Two identical partitions consisting of a single leaf.
      const Octant root = {};
      const LeafVector<dim> part =
        LeafVector<dim>::sorted({
          morton_lineage(root, {0, 3}),
        });
      const PartitionFrontBack<uint32_t, dim> endpoints = {
        {part.vec.front(), part.vec.front()},
        {part.vec.back(), part.vec.back()}
      };

      const Octant single = part.vec[0];
      CHECK_FALSE( border_any<dim>(
            LeafRange<dim>::make(single, single),
            LeafRange<dim>::make(single, single)) );

      const bool is_active = true;
      const std::vector<int> active_list = {0, 1};

      for (int rank: {0, 1})
      {
        LocalAdjacencyList adjacency_list =
          sfc_partition<dim>(rank, is_active, active_list, endpoints);

        CHECK( adjacency_list.local_rank == rank );
        CHECK( adjacency_list.neighbor_ranks.size() == 0 );
      }

      _DestroyHcurve();
    }
  }

}
#endif//DOCTEST_LIBRARY_INCLUDED


// =============================================================================
// Implementation
// =============================================================================
namespace ot
{

  // sfc_partition(): Returns partition indices where local borders or overlaps.
  template <unsigned dim>
  LocalAdjacencyList sfc_partition(
      int local_rank,
      bool is_active,
      const std::vector<int> &active_list,
      const PartitionFrontBack<uint32_t, dim> &endpoints)
  {
    std::vector<int> neighbor_ranks;

    if (not is_active)
    {
      return { local_rank, std::move(neighbor_ranks) };
    }

    // active_view: Interleave partition front and back octants.
    std::vector<TreeNode<uint32_t, dim>> range_list;
    range_list.reserve(endpoints.m_fronts.size() * 2);
    for (int r: active_list)
    {
      range_list.push_back(endpoints.m_fronts[r]);
      range_list.push_back(endpoints.m_backs[r]);
    }
    const LeafRangeListView<dim> active_view =
        vec_leaf_range_list_view<dim>(range_list);

    // Construct local partition range.
    const LeafRange<dim> local_range = LeafRange<dim>::make(
        endpoints.m_fronts[local_rank], endpoints.m_backs[local_rank]);

    // Search: subset of range-on-range.
    where_ranges_border( active_view, local_range,
        [&](const TreeNode<uint32_t, dim> *range_first)
    {
      const size_t active_idx = (range_first - active_view.begin()) / 2;
      const int remote_rank = active_list[active_idx];
      if (remote_rank != local_rank)
        neighbor_ranks.push_back(remote_rank);
    });

    return { local_rank, std::move(neighbor_ranks) };
  }



  // where_border()
  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_border(
      LeafListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set_,
      Emit &&emit)
  {
    const AnySet &any_set = any_set_.cast();

    if (all_list.is_singleton())  //TODO allow duplicates
    {
      if (border_any<dim>(all_list.root(), any_set))
        emit(all_list.begin());
    }
    // Main result is that, by pruning, we can skip many accesses to all_list.
    else if (border_or_overlap_any<dim>(all_list.root(), any_set))
    {
      // future: Subtrees of all_list: jump-start searches in any_set by parent.
      for (sfc::SubIndex s(0); s < nchild(dim); ++s)
      {
        const LeafListView<dim> sublist = all_list.subdivide(s);
        if (sublist.any())
          where_border<dim>(sublist, any_set, std::forward<Emit>(emit));
      }
    }
  }

  // where_border_or_overlap()
  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_border_or_overlap(
      LeafListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set_,
      Emit &&emit)
  {
    const AnySet &any_set = any_set_.cast();

    if (all_list.is_singleton())  //TODO allow duplicates
    {
      if (border_or_overlap_any<dim>(all_list.root(), any_set))
        emit(all_list.begin());
    }
    // Main result is that, by pruning, we can skip many accesses to all_list.
    else if (border_or_overlap_any<dim>(all_list.root(), any_set))
    {
      // future: Subtrees of all_list: jump-start searches in any_set by parent.
      for (sfc::SubIndex s(0); s < nchild(dim); ++s)
      {
        const LeafListView<dim> sublist = all_list.subdivide(s);
        if (sublist.any())
          where_border_or_overlap<dim>(sublist, any_set, std::forward<Emit>(emit));
      }
    }
  }


  // where_ranges_border()
  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_ranges_border(
      LeafRangeListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set_,
      Emit &&emit)
  {
    const AnySet &any_set = any_set_.cast();

    if (all_list.is_single_range())
    {
      if (border_any<dim>(all_list.range(), any_set))
        emit(all_list.begin());
    }
    else if (all_list.is_singleton())
    {
      if (border_any<dim>(all_list.range(), any_set))
        for (const auto *ptr = all_list.begin(); ptr != all_list.end(); ptr += 2)
          emit(ptr);
    }
    // Main result is that, by pruning, we can skip many accesses to all_list.
    else if (border_or_overlap_any<dim>(all_list.root(), any_set))
    {
      for (int i = 0, n = nchild(dim); i < n; ++i)
      {
        // Length-wise segment, not child.
        // This ensures the recursion makes progress.
        LeafRangeListView<dim> sublist = all_list.fair_segment(i, n);
        if (sublist.any())
          where_ranges_border<dim>(sublist, any_set, std::forward<Emit>(emit));
      }
    }
  }

  // where_ranges_border_or_overlap()
  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_ranges_border_or_overlap(
      LeafRangeListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set_,
      Emit &&emit)
  {
    const AnySet &any_set = any_set_.cast();

    if (all_list.is_single_range())
    {
      if (border_or_overlap_any<dim>(all_list.range(), any_set))
        emit(all_list.begin());
    }
    // Main result is that, by pruning, we can skip many accesses to all_list.
    else if (border_or_overlap_any<dim>(all_list.root(), any_set))
    {
      // future: Subtrees of all_list: jump-start searches in any_set by parent.
      const TreeNode<uint32_t, dim> *prev_end = nullptr;
      for (sfc::SubIndex s(0); s < nchild(dim); ++s)
      {
        LeafRangeListView<dim> sublist = all_list.subdivide(s);
        if (prev_end != nullptr and sublist.begin() < prev_end)  //redundant
          sublist = sublist.shrink_begin();
        prev_end = sublist.end();
        if (sublist.any())
          where_ranges_border_or_overlap<dim>(sublist, any_set, std::forward<Emit>(emit));
      }
    }
  }




  namespace detail
  {
    template <unsigned dim>
    bool descendants_adjacent_to_leaf(
        TreeNode<uint32_t, dim> ancestor, TreeNode<uint32_t, dim> leaf)
    {
      return ancestor.isAncestor(leaf) or adjacent(ancestor, leaf);
    }
  }

  template <int dim, bool overlaps_ok, typename AnySet>
  bool border_any_impl(
      TreeNode<uint32_t, dim> octant,
      const LeafSet<dim, AnySet> &any_set_)
  {
    using namespace detail;
    const AnySet &any_set = any_set_.cast();

    // Assume that any_set.any().
    // Assume that any_set.root() has descendants satisfying search criteria.

    if (not overlaps_ok)
    {
      // This would violate precondition.
      assert(not octant.isAncestorInclusive(any_set.root()));
    }

    if (any_set.is_singleton())
    {
      return overlaps_ok or not any_set.root().isAncestor(octant);
    }
    else
    {
      if (overlaps_ok and octant.isAncestorInclusive(any_set.root()))
        return true;
    }

    // Split any_set into descendants and recurse.
    for (sfc::SubIndex s(0); s < nchild(dim); ++s)
    {
      const AnySet segment_subset = any_set.subdivide(s);
      if (segment_subset.any() and
          descendants_adjacent_to_leaf(segment_subset.root(), octant))
      {
        if (border_any_impl<dim, overlaps_ok>(octant, segment_subset))
          return true;
        // future: return where. Save on related searches by skipping prefix.
      }
    }
    return false;
  }

  template <int dim, typename AnySet>
  bool border_any(
      TreeNode<uint32_t, dim> octant,
      const LeafSet<dim, AnySet> &any_set_)
  {
    using namespace detail;
    const AnySet &any_set = any_set_.cast();
    if (any_set.none())
      return false;
    if (descendants_adjacent_to_leaf(any_set.root(), octant))
      return border_any_impl<dim, false>(octant, any_set);
    else
      return false;
  }

  template <int dim, typename AnySet>
  bool border_or_overlap_any(
      TreeNode<uint32_t, dim> octant,
      const LeafSet<dim, AnySet> &any_set_)
  {
    using namespace detail;
    const AnySet &any_set = any_set_.cast();
    if (any_set.none())
      return false;
    if (octant.isAncestorInclusive(any_set.root()) or
        descendants_adjacent_to_leaf(any_set.root(), octant))
      return border_any_impl<dim, true>(octant, any_set);
    else
      return false;
  }

  template <int dim, bool overlaps_ok, typename PSet, typename QSet>
  bool border_any_impl(
      const LeafSet<dim, PSet> &P_,
      const LeafSet<dim, QSet> &Q_)
  {
    using namespace detail;
    const PSet &P = P_.cast();
    const QSet &Q = Q_.cast();
    // Assume neither is none.
    // Assume adjacent or descendants are.

    if (adjacent(P.root(), Q.root()) or overlaps_ok)
      if (P.is_singleton() and Q.is_singleton())
        return true;

    const auto may_border = [](const auto &R, const auto &S) {
      return descendants_adjacent_to_leaf(R, S)
          or descendants_adjacent_to_leaf(S, R);
    };

    // Split the non-singleton(s) and recurse.
    if (P.is_singleton())
      return border_any<dim>(P.root(), Q);
    else if (Q.is_singleton())
      return border_any<dim>(Q.root(), P);
    else
    {
      // future: separate implementations depending on
      //         whether subdivision is costly or not.
      std::bitset<nchild(dim)> p_init, q_init;
      std::array<PSet, nchild(dim)> p_sub;
      std::array<QSet, nchild(dim)> q_sub;
      for (sfc::SubIndex pc(0); pc < nchild(dim); ++pc)
        if (may_border(P.scope().subdivide(pc).m_root, Q.root()))
        {
          p_sub[pc] = P.subdivide(pc);
          p_init[pc] = p_sub[pc].any();
        }
      for (sfc::SubIndex qc(0); qc < nchild(dim); ++qc)
        if (may_border(Q.scope().subdivide(qc).m_root, P.root()))
        {
          q_sub[qc] = Q.subdivide(qc);
          q_init[qc] = q_sub[qc].any();
        }
      // future: prioritize pairs by highest dimensionality of intersection.
      for (sfc::SubIndex pc(0); pc < nchild(dim); ++pc)
        if (p_init[pc])
          for (sfc::SubIndex qc(0); qc < nchild(dim); ++qc)
            if (q_init[qc])
              if (may_border(p_sub[pc].root(), q_sub[qc].root()))
                if (border_any_impl<dim, overlaps_ok>(p_sub[pc], q_sub[qc]))
                  return true;
    }

    return false;
  }


  template <int dim, typename PSet, typename QSet>
  bool border_any(
      const LeafSet<dim, PSet> &P_,
      const LeafSet<dim, QSet> &Q_)
  {
    using namespace detail;
    const PSet &P = P_.cast();
    const QSet &Q = Q_.cast();
    if (P.none() or Q.none())
      return false;
    if (adjacent(P.root(), Q.root())
        or P.root().isAncestorInclusive(Q.root())
        or Q.root().isAncestorInclusive(P.root()))
    {
      return border_any_impl<dim, false>(P, Q);
    }
    else
      return false;
  }

  template <int dim, typename PSet, typename QSet>
  bool border_or_overlap_any(
      const LeafSet<dim, PSet> &P_,
      const LeafSet<dim, QSet> &Q_)
  {
    using namespace detail;
    const PSet &P = P_.cast();
    const QSet &Q = Q_.cast();
    if (P.none() or Q.none())
      return false;
    if (adjacent(P.root(), Q.root())
        or P.root().isAncestorInclusive(Q.root())
        or Q.root().isAncestorInclusive(P.root()))
    {
      return border_any_impl<dim, true>(P, Q);
    }
    else
      return false;
  }


  // adjacent()
  template <unsigned dim>
  bool adjacent(
      const TreeNode<uint32_t, dim> &a,
      const TreeNode<uint32_t, dim> &b)
  {
    const auto a_range = a.range();
    const auto b_range = b.range();
    const periodic::IntersectionMagnitude intersection =
        periodic::intersect_magnitude(a_range, b_range);
    return intersection.nonempty and intersection.dimension < dim;
  }


}

#endif//DENDRO_KT_PARTITION_BORDER_HPP
