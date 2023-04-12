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

  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_border(
      LeafListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set,
      Emit &&emit);

  template <int dim, typename AnySet>
  bool border_any(
      TreeNode<uint32_t, dim> octant,
      const LeafSet<dim, AnySet> &any_set);

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

      // slice()
      LeafListView<dim> slice(size_t begin, size_t end) const {
        return LeafListView<dim>(&vec[begin], &vec[end]);
      }

      // view()
      LeafListView<dim> view() const {
        return this->slice(0, this->vec.size());
      }

      // range()
      LeafRange<dim> range() const {
        return this->vec.empty() ?
          LeafRange<dim>::make_empty()
          :
          LeafRange<dim>::make(this->vec.front(), this->vec.back());
      }
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

    DOCTEST_TEST_CASE("border_any")
    {
      constexpr int dim = 2;
      using Octant = TreeNode<uint32_t, dim>;
      _InitializeHcurve(dim);

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
      // |       |       |
      // |       |       |     Two partitions defined by space-filling curve.
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
      }

      CHECK( interpart_low.vec.size() == expected_interpart_low );
      CHECK( interpart_high.vec.size() == expected_interpart_high );

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
  // where_border()
  template <int dim, typename AnySet, typename Emit>  // void emit(const TreeNode *tn)
  void where_border(
      LeafListView<dim> all_list,
      const LeafSet<dim, AnySet> &any_set_,
      Emit &&emit)
  {
    const AnySet &any_set = any_set_.cast();

    // Main result is that, by pruning, we can skip many accesses to all_list.
    if (border_any<dim>(all_list.root(), any_set))
    {
      if (all_list.is_singleton())
        emit(all_list.begin());
      else
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

  template <int dim, typename AnySet>
  bool border_any_impl(
      TreeNode<uint32_t, dim> octant,
      const LeafSet<dim, AnySet> &any_set_)
  {
    using namespace detail;
    const AnySet &any_set = any_set_.cast();

    // Assume that any_set.root() has descendants adjacent to octant.

    if (adjacent(any_set.root(), octant) and any_set.is_singleton())
      return true;

    // Split any_set into descendants and recurse.
    for (sfc::SubIndex s(0); s < nchild(dim); ++s)
    {
      const AnySet segment_subset = any_set.subdivide(s);
      if (segment_subset.any() and
          descendants_adjacent_to_leaf(segment_subset.root(), octant))
      {
        if (border_any_impl<dim>(octant, segment_subset))
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
    if (descendants_adjacent_to_leaf(any_set.root(), octant))
      return border_any_impl<dim>(octant, any_set);
    else
      return false;
  }


  // adjacent()
  template <unsigned dim>
  bool adjacent(
      const TreeNode<uint32_t, dim> &a,
      const TreeNode<uint32_t, dim> &b)
  {
    //       [----]      [----]  [----]  [----]  [----]  [----]     [----]
    //  [--]          [--]       [--]     [--]     [--]       [--]         [--]

    const auto a_range = a.range();
    const auto b_range = b.range();

    // Closed bounds overlap on each axis.
    bool closed_overlap = true;
    for (int d = 0; d < dim; ++d)
      closed_overlap &=
           (a_range.closedContains(d, b_range.min(d)))
        or (b_range.closedContains(d, a_range.min(d)));

    bool open_overlap = true;
    for (int d = 0; d < dim; ++d)
      open_overlap &=
           (a_range.openContains(d, b_range.min(d)))
        or (b_range.openContains(d, a_range.min(d)));

    return closed_overlap and not open_overlap;
  }


}

#endif//DENDRO_KT_PARTITION_BORDER_HPP
