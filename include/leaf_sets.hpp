/**
 * @author Masado Ishii
 * @date   2023-03-30
 * @brief  Loop over pairs of neighboring leafs.
 */

#ifndef DENDRO_KT_LEAF_SETS_HPP
#define DENDRO_KT_LEAF_SETS_HPP

// Function declaration for linkage purposes.
inline void link_leaf_sets_tests() {};

#include "include/treeNode.h"
#include "include/tsort.h"
#include "include/sfc_search.h"
#include "include/mathUtils.h"

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  template <int dim, typename Derived>
    struct LeafSet;

  // LeafSet implementations
  template <int dim>  struct LeafListView;   // loop over explicit array.
  template <int dim>  struct LeafRange;  // loop over implicit range.
  template <int dim>  struct LeafRangeListView;  // array of implicit ranges

  template <int dim, typename V>
  LeafListView<dim> vec_leaf_list_view(const V &vector);  // STL vector or array

  template <int dim, typename V>
  LeafRange<dim> vec_leaf_range(const V &vector);  // STL vector or array

  template <int dim, typename V>
  LeafRangeListView<dim> vec_leaf_range_list_view(const V &vector);  // STL vector or array


  // DescendantSet
  template <int dim>
  struct DescendantSet;

  // Octant
  template <int dim>  struct Octant;

  /**
   * DescendantSet
   */
  // future: consider combining with SFC_Region
  template <int dim>
  struct DescendantSet
  {
    DescendantSet subdivide(sfc::SubIndex s) const;
    DescendantSet child(sfc::ChildNum c) const;
    DescendantSet select(TreeNode<uint32_t, dim> subtree) const;

    TreeNode<uint32_t, dim> m_root;
    SFC_State<dim> m_sfc;
  };


  /**
   * LeafSet
   * @brief Base class to group kinds of leaf sets (LeafListView, LeafRange).
   */
  template <int dim, typename Derived>
  struct LeafSet
  {
    public:
      // Down-cast.
      Derived & cast() { return static_cast<Derived &>(*this); }
      const Derived & cast() const { return static_cast<const Derived &>(*this); }

      LeafSet(DescendantSet<dim> scope) : m_scope(scope) {}

      LeafSet() = default;
      explicit LeafSet(const LeafSet &other) = default;
      explicit LeafSet(LeafSet &&other) = default;
      LeafSet & operator=(const LeafSet &other) = default;
      LeafSet & operator=(LeafSet &&other) = default;

      TreeNode<uint32_t, dim> root() const
      {
        return m_scope.m_root;
      }

      SFC_State<dim> sfc() const
      {
        return m_scope.m_sfc;
      }

      DescendantSet<dim> scope() const
      {
        return m_scope;
      }

    private:
      LeafSet(const Derived &);  // Protect from partial copies in CRTP.

    public:
      DescendantSet<dim> m_scope;
  };


  /**
   * LeafListView
   *
   * @brief Sublist of explicit sorted array of leaf octants.
   */
  template <int dim>
  struct LeafListView : public LeafSet<dim, LeafListView<dim>>
  {
    using Base = LeafSet<dim, LeafListView<dim>>;
    using Base::Base;
    public:
      LeafListView() = default;
      LeafListView( const TreeNode<uint32_t, dim> *begin,
                const TreeNode<uint32_t, dim> *end,
                DescendantSet<dim> scope = {} );

      bool any() const;
      bool none() const;
      bool is_singleton() const;
      bool is_singleton_of(const TreeNode<uint32_t, dim> &member) const;
      // If this->is_singleton(), then this->is_singleton_of(this->root()).

      bool operator==(const LeafListView &other) const;
      bool operator!=(const LeafListView &other) const;

      LeafListView subdivide(sfc::SubIndex s) const;
      LeafListView child(sfc::ChildNum c) const;

      // Specific to LeafListView
      const TreeNode<uint32_t, dim> *begin() const;
      const TreeNode<uint32_t, dim> *end() const;

    public:
      const TreeNode<uint32_t, dim> *m_begin;
      const TreeNode<uint32_t, dim> *m_end;
  };


  /**
   * LeafRange
   *
   * @brief Implicit range along the space-filling curve.
   */
  template <int dim>
  struct LeafRange : public LeafSet<dim, LeafRange<dim>>
  {
    using Base = LeafSet<dim, LeafRange<dim>>;
    using Base::Base;
    public:
      LeafRange() = default;

      static LeafRange make_empty(DescendantSet<dim> scope = {});

      static LeafRange make(TreeNode<uint32_t, dim> first,
                            TreeNode<uint32_t, dim> last,
                            DescendantSet<dim> scope = {});

      bool any() const;
      bool none() const;
      bool is_singleton() const;
      bool is_singleton_of(const TreeNode<uint32_t, dim> &member) const;
      // If this->is_singleton(), then this->is_singleton_of(this->root()).

      bool operator==(const LeafRange &other) const;
      bool operator!=(const LeafRange &other) const;

      LeafRange subdivide(sfc::SubIndex s) const;
      LeafRange child(sfc::ChildNum c) const;

      // Specific to LeafRange (experimental)
      TreeNode<uint32_t, dim> first() const { return m_first; }
      TreeNode<uint32_t, dim> last() const { return m_last; }

    private:
      LeafRange( TreeNode<uint32_t, dim> first,
                 TreeNode<uint32_t, dim> last,
                 bool nonempty,
                 DescendantSet<dim> scope = {} );

    public:
      TreeNode<uint32_t, dim> m_first;
      TreeNode<uint32_t, dim> m_last;
      bool m_nonempty;
  };


  /**
   * LeafRangeListView
   *
   * @brief Sublist of implicit ranges along the space-filling curve.
   * @note Top ranges must be nonempty. Listed as {first,last, first,last, ...}.
   * @note Ranges may not overlap, except
   *       if the first of one range equals the last of the previous range.
   */
  template <int dim>
  struct LeafRangeListView : public LeafSet<dim, LeafRangeListView<dim>>
  {
    using Base = LeafSet<dim, LeafRangeListView<dim>>;
    using Base::Base;
    public:
      LeafRangeListView() = default;

      // (end - begin) % 2 == 0
      LeafRangeListView( const TreeNode<uint32_t, dim> *begin,
                         const TreeNode<uint32_t, dim> *end,
                         DescendantSet<dim> scope = {} );

      bool any() const;
      bool none() const;
      bool is_singleton() const;
      bool is_singleton_of(const TreeNode<uint32_t, dim> &member) const;
      // If this->is_singleton(), then this->is_singleton_of(this->root()).

      bool operator==(const LeafRangeListView &other) const;
      bool operator!=(const LeafRangeListView &other) const;

      //future: Split a range, expose partial_front() and partial_back(),
      //        let user decide which way to shift the front and back.
      /// LeafRangeListView subdivide(sfc::SubIndex s) const;
      /// LeafRangeListView child(sfc::ChildNum c) const;

      // Specific to LeafRangeListView (experimental)

      // Partitions the list of ranges into `count` approximately equal segments,
      // and returns the `pick`-th segment (starting from 0).
      LeafRangeListView fair_segment(int pick, int count) const;

      // Specific to LeafRangeListView
      const TreeNode<uint32_t, dim> *begin() const;
      const TreeNode<uint32_t, dim> *end() const;
      bool is_single_range() const;
      LeafRange<dim> range() const;
      LeafRangeListView shrink_begin() const;
      LeafRangeListView shrink_end() const;

    private:
      struct AsList { };
      struct AsRange { };

      LeafRangeListView(AsList, LeafListView<dim> sublist);

      LeafRangeListView(
          AsRange, LeafListView<dim> sublist, LeafRange<dim> subrange);

      bool is_subrange() const;

    public:
      LeafListView<dim> m_sublist;
      LeafRange<dim> m_subrange;
  };

}

// =============================================================================
// Tests
// =============================================================================
#ifdef DOCTEST_LIBRARY_INCLUDED

#include "include/tnUtils.h"

namespace ot
{
  DOCTEST_TEST_SUITE("DescendantSet")
  {
    DOCTEST_TEST_CASE("Default is root")
    {
      constexpr int dim = 2;
      CHECK( DescendantSet<dim>().m_root == TreeNode<uint32_t, dim>{} );
      CHECK( DescendantSet<dim>().m_sfc.state() == SFC_State<dim>::root().state() );
    }

    DOCTEST_TEST_CASE("Select 4th generation")
    {
      constexpr int dim = 2;
      _InitializeHcurve(dim);
      const TreeNode<uint32_t, dim> root = {};
      const TreeNode<uint32_t, dim> descendant = morton_lineage(root, {1,1,1,1});
      CHECK( DescendantSet<dim>().select(descendant).m_root.getLevel() == 4 );
      _DestroyHcurve();
    }
  }

  DOCTEST_TEST_SUITE("Leaf sets")
  {
    DOCTEST_TEST_CASE("Construct empty and single")
    {
      constexpr int dim = 2;

      TreeNode<uint32_t, dim> array[1] = {};
      CHECK( LeafListView<dim>().none() );
      CHECK( LeafListView<dim>( array, array + 1 ).is_singleton() );

      CHECK( LeafRange<dim>().none() );
      CHECK( LeafRange<dim>::make( {}, {} ).is_singleton() );

      TreeNode<uint32_t, dim> range_list[2] = {};
      CHECK( LeafRangeListView<dim>().none() );
      CHECK( LeafRangeListView<dim>( array, array + 2 ).is_singleton() );
    }

    DOCTEST_TEST_CASE("Construct duplicated singleton")
    {
      constexpr int dim = 2;
      std::array<TreeNode<uint32_t, dim>, 6> array = {};
      CHECK( LeafListView<dim>( &(*array.begin()), &(*array.end()) )
          .is_singleton() );
      CHECK( LeafRangeListView<dim>( &(*array.begin()), &(*array.end()) )
          .is_singleton() );
    }

    DOCTEST_TEST_CASE("Subdivide empty")
    {
      constexpr int dim = 2;
      _InitializeHcurve(dim);

      LeafListView<dim> owned_cells = {};
      CHECK( owned_cells.subdivide(sfc::SubIndex(0)).none() );
      CHECK( owned_cells.subdivide(sfc::SubIndex(0)).root().getLevel() == 1 );

      LeafRange<dim> neighbor_range = {};
      CHECK( neighbor_range.subdivide(sfc::SubIndex(0)).none() );
      CHECK( neighbor_range.subdivide(sfc::SubIndex(0)).root().getLevel() == 1 );

      LeafRangeListView<dim> range_list = {};
      /// CHECK( range_list.subdivide(sfc::SubIndex(0)).none() );
      /// CHECK( range_list.subdivide(sfc::SubIndex(0)).root().getLevel() == 1 );
      CHECK( range_list.fair_segment(0, 2).none() );
      CHECK( range_list.fair_segment(1, 2).none() );

      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("Drill to descendant on construct")
    {
      constexpr int dim = 2;
      _InitializeHcurve(dim);
      TreeNode<uint32_t, dim> root = {};
      std::array<TreeNode<uint32_t, dim>, 2> singleton = {
        morton_lineage(root, {1, 2, 2, 2}),
        morton_lineage(root, {1, 2, 2, 2})
      };

      LeafListView<dim> leaf_list(&(*singleton.begin()), &(*singleton.end()));
      REQUIRE( leaf_list.is_singleton() );
      CHECK( leaf_list.is_singleton_of(leaf_list.root()) );
      CHECK( leaf_list.root().getLevel() == 4 );

      LeafRange<dim> leaf_range = LeafRange<dim>::make(
          singleton.front(), singleton.back() );
      REQUIRE( leaf_range.is_singleton() );
      CHECK( leaf_range.is_singleton_of(leaf_range.root()) );
      CHECK( leaf_range.root().getLevel() == 4 );

      LeafRangeListView<dim> range_list = LeafRangeListView<dim>(
          &(*singleton.begin()), &(*singleton.end()));
      REQUIRE( range_list.is_singleton() );
      CHECK( range_list.is_singleton_of(range_list.root()) );
      CHECK( range_list.root().getLevel() == 4 );

      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("Drill to descendant")
    {
      constexpr int dim = 2;
      _InitializeHcurve(dim);
      TreeNode<uint32_t, dim> root = {};

      // LeafListView
      const std::array<TreeNode<uint32_t, dim>, 2> leaf_list_array = {
        morton_lineage(root, {0, 2, 2, 2}),
        morton_lineage(root, {3, 1, 1, 1})
      };
      LeafListView<dim> leaf_list(
          &(*leaf_list_array.begin()), &(*leaf_list_array.end()));
      REQUIRE( leaf_list.any() );
      REQUIRE_FALSE( leaf_list.is_singleton() );
      CHECK( leaf_list.child(sfc::ChildNum(0)).is_singleton() );
      CHECK( leaf_list.child(sfc::ChildNum(0)).root().getLevel() == 4 );
      CHECK( leaf_list.child(sfc::ChildNum(1)).none() );
      CHECK( leaf_list.child(sfc::ChildNum(2)).none() );
      CHECK( leaf_list.child(sfc::ChildNum(3)).is_singleton() );
      CHECK( leaf_list.child(sfc::ChildNum(3)).root().getLevel() == 4 );

      // LeafRange
      LeafRange<dim> leaf_range = LeafRange<dim>::make(
        morton_lineage(root, {1, 3, 3, 3}),
        morton_lineage(root, {2, 0, 0, 0}) );
      REQUIRE( leaf_range.any() );
      REQUIRE_FALSE( leaf_range.is_singleton() );
      CHECK( leaf_range.child(sfc::ChildNum(0)).none() );
      CHECK( leaf_range.child(sfc::ChildNum(1)).is_singleton() );
      CHECK( leaf_range.child(sfc::ChildNum(1)).root().getLevel() == 4 );
      CHECK( leaf_range.child(sfc::ChildNum(2)).is_singleton() );
      CHECK( leaf_range.child(sfc::ChildNum(2)).root().getLevel() == 4 );
      CHECK( leaf_range.child(sfc::ChildNum(3)).none() );

      /// // LeafRangeListView
      /// const std::array<TreeNode<uint32_t, dim>, 4> sample = {
      ///   morton_lineage(root, {1, 3, 3, 3}),
      ///   morton_lineage(root, {1, 3, 3, 3}),
      ///   morton_lineage(root, {2, 0, 0, 0}),
      ///   morton_lineage(root, {2, 0, 0, 0})
      /// };

      /// LeafRangeListView<dim> range_list_one( &sample[1], &sample[3] );
      /// REQUIRE( range_list_one.any() );
      /// REQUIRE_FALSE( range_list_one.is_singleton() );
      /// CHECK( range_list_one.child(sfc::ChildNum(0)).none() );
      /// CHECK( range_list_one.child(sfc::ChildNum(1)).is_singleton() );
      /// CHECK( range_list_one.child(sfc::ChildNum(1)).root().getLevel() == 4 );
      /// CHECK( range_list_one.child(sfc::ChildNum(2)).is_singleton() );
      /// CHECK( range_list_one.child(sfc::ChildNum(2)).root().getLevel() == 4 );
      /// CHECK( range_list_one.child(sfc::ChildNum(3)).none() );

      /// LeafRangeListView<dim> range_list_two( &sample[0], &sample[4] );
      /// REQUIRE( range_list_two.any() );
      /// REQUIRE_FALSE( range_list_two.is_singleton() );
      /// CHECK( range_list_two.child(sfc::ChildNum(0)).none() );
      /// CHECK( range_list_two.child(sfc::ChildNum(1)).is_singleton() );
      /// CHECK( range_list_two.child(sfc::ChildNum(1)).root().getLevel() == 4 );
      /// CHECK( range_list_two.child(sfc::ChildNum(2)).is_singleton() );
      /// CHECK( range_list_two.child(sfc::ChildNum(2)).root().getLevel() == 4 );
      /// CHECK( range_list_two.child(sfc::ChildNum(3)).none() );

      _DestroyHcurve();
    }

    DOCTEST_TEST_CASE("LeafRangeListVew subdivision reduces set")
    {
      constexpr int dim = 2;
      _InitializeHcurve(dim);

      // neat_ranges
      std::vector<TreeNode<uint32_t, dim>> neat_ranges = {
        morton_lineage<dim>({0, 0}),  morton_lineage<dim>({0, 1}),
        morton_lineage<dim>({1, 0}),  morton_lineage<dim>({1, 1}),
        morton_lineage<dim>({2, 0}),  morton_lineage<dim>({2, 1}),
        morton_lineage<dim>({3, 0}),  morton_lineage<dim>({3, 1})
      };
      const LeafRangeListView<dim> neat_range_list_view(
          &(*neat_ranges.begin()), &(*neat_ranges.end()));

      for (ot::sfc::ChildNum child(0); child < ot::nchild(dim); ++child)
      {
        /// CHECK( neat_range_list_view.child(child) != neat_range_list_view );
        CHECK( neat_range_list_view.fair_segment(child, ot::nchild(dim))
            != neat_range_list_view );
      }

      // trans_ranges
      std::vector<TreeNode<uint32_t, dim>> trans_ranges = {
        morton_lineage<dim>({0, 0}),  morton_lineage<dim>({1, 0}),
        morton_lineage<dim>({1, 1}),  morton_lineage<dim>({2, 0}),
      };
      const LeafRangeListView<dim> trans_range_list_view(
          &(*trans_ranges.begin()), &(*trans_ranges.end()));

      for (ot::sfc::ChildNum child(0); child < ot::nchild(dim); ++child)
      {
        INFO("child == ", int(child));
        /// CHECK( trans_range_list_view.child(child) != trans_range_list_view );
        CHECK( trans_range_list_view.fair_segment(child, ot::nchild(dim))
            != trans_range_list_view );
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
  // DescendantSet::child()
  template<int dim>
  DescendantSet<dim> DescendantSet<dim>::child(sfc::ChildNum c) const
  {
    return { m_root.getChildMorton(c), m_sfc.child_curve(c) };
  }

  // DescendantSet::subdivide()
  template<int dim>
  DescendantSet<dim> DescendantSet<dim>::subdivide(sfc::SubIndex s) const
  {
    return this->child(m_sfc.child_num(s));
  }

  // DescendantSet::select()
  template<int dim>
  DescendantSet<dim> DescendantSet<dim>::select(TreeNode<uint32_t, dim> subtree) const
  {
    DescendantSet result = *this;
    for (int level = result.m_root.getLevel(); level < subtree.getLevel(); ++level)
      result = result.child(sfc::ChildNum(subtree.getMortonIndex(level + 1)));
    return result;
  }


  // LeafListView::LeafListView()
  template <int dim>
  LeafListView<dim>::LeafListView( const TreeNode<uint32_t, dim> *begin,
                           const TreeNode<uint32_t, dim> *end,
                           DescendantSet<dim> scope )
    : Base({(begin < end ?
              scope.select(common_ancestor(*begin, *(end - 1)))  // Drill down
              : scope)}),
      m_begin(begin), m_end(end)
  { }

  // LeafListView::any()
  template <int dim>
  bool LeafListView<dim>::any() const
  {
    return m_begin < m_end;
  }

  // LeafListView::none()
  template <int dim>
  bool LeafListView<dim>::none() const
  {
    return not any();
  }

  // LeafListView::is_singleton()
  template <int dim>
  bool LeafListView<dim>::is_singleton() const
  {
    const bool is_singleton = m_begin < m_end and *m_begin == *(m_end - 1);
    if (is_singleton)
    {
      assert(this->root() == *m_begin);
    }
    return is_singleton;
  }

  // LeafListView::is_singleton_of()
  template <int dim>
  bool LeafListView<dim>::is_singleton_of(const TreeNode<uint32_t, dim> &member) const
  {
    return is_singleton() and *m_begin == member;
  }

  // LeafListView::operator==()
  template <int dim>
  bool LeafListView<dim>::operator==(const LeafListView &other) const
  {
    return std::tie(m_begin, m_end) == std::tie(other.m_begin, other.m_end);
  }

  // LeafListView::operator!=()
  template <int dim>
  bool LeafListView<dim>::operator!=(const LeafListView &other) const
  {
    return not (*this == other);
  }

  // LeafListView::child()
  template <int dim>
  LeafListView<dim> LeafListView<dim>::child(sfc::ChildNum c) const
  {
    assert(( (not this->is_singleton())
          or (this->root().getLevel() < m_begin->getLevel()) ));

    const TreeNode<uint32_t, dim> child_key = this->root().getChildMorton(c);

    // Slice as two binary searches: [Exclusive rank, inclusive rank)
    // Equate descendants finer than key level.

    const size_t begin = 0,  end = m_end - m_begin;
    const SFC_Region<uint32_t, dim> region = {this->root(), this->sfc()};
    const auto key_level = child_key.getLevel();
    const auto equal = [key_level](
        TreeNode<uint32_t, dim> a, TreeNode<uint32_t, dim> b)
    {
      a = (a.getLevel() < key_level ? a : a.getAncestor(key_level));
      b = (b.getLevel() < key_level ? b : b.getAncestor(key_level));
      return a == b;
    };

    const size_t new_begin = sfc_binary_search<uint32_t, dim>(
        child_key, m_begin, begin, end, RankType::exclusive, equal);
    const size_t new_end = sfc_binary_search<uint32_t, dim>(
        child_key, m_begin, new_begin, end, RankType::inclusive, equal);

    // New slice
    LeafListView result(
      m_begin + new_begin, m_begin + new_end, this->m_scope.child(c));

    return result;
  }

  // LeafListView::subdivide()
  template <int dim>
  LeafListView<dim> LeafListView<dim>::subdivide(sfc::SubIndex s) const
  {
    return this->child(this->sfc().child_num(s));
  }

  // LeafListView::begin()
  template <int dim>
  const TreeNode<uint32_t, dim> *LeafListView<dim>::begin() const
  {
    return m_begin;
  }

  // LeafListView::end()
  template <int dim>
  const TreeNode<uint32_t, dim> *LeafListView<dim>::end() const
  {
    return m_end;
  }


  // vec_leaf_list_view()
  template <int dim, typename V>
  LeafListView<dim> vec_leaf_list_view(const V &vector)
  {
    // Need random access.
    return LeafListView<dim>(&vector[0], &vector[vector.size()]);
  }


  // LeafRangeListView::LeafRangeListView()
  template <int dim>
  LeafRangeListView<dim>::LeafRangeListView( AsList,
      LeafListView<dim> sublist)
  :
    Base( sublist.scope() ),
    m_sublist( sublist ),
    m_subrange(
        sublist.any() ?  LeafRange<dim>::make(
          *sublist.begin(), *(sublist.end() - 1), sublist.scope()) :
        LeafRange<dim>::make_empty(sublist.scope()))
  {
    assert((sublist.end() - sublist.begin()) % 2 == 0);
  }

  // LeafRangeListView::LeafRangeListView()
  template <int dim>
  LeafRangeListView<dim>::LeafRangeListView( AsRange,
      LeafListView<dim> sublist,
      LeafRange<dim> subrange)
  :
    Base( subrange.scope() ),
    m_sublist( sublist ),
    m_subrange( subrange )
  {
    assert((sublist.end() - sublist.begin()) == 2);
  }

  // LeafRangeListView::is_subrange()
  template <int dim>
  bool LeafRangeListView<dim>::is_subrange() const
  {
    return (m_sublist.end() - m_sublist.begin() == 2);
  }

  // LeafRangeListView::LeafRangeListView()
  template <int dim>
  LeafRangeListView<dim>::LeafRangeListView(
      const TreeNode<uint32_t, dim> *begin,
      const TreeNode<uint32_t, dim> *end,
      DescendantSet<dim> scope)
  :
    LeafRangeListView(AsList{}, LeafListView<dim>(begin, end, scope))
  { }

  // LeafRangeListView::any()
  template <int dim>
  bool LeafRangeListView<dim>::any() const
  {
    if (this->is_subrange())
      return m_subrange.any();
    else
      return m_sublist.any();
  }

  // LeafRangeListView::none()
  template <int dim>
  bool LeafRangeListView<dim>::none() const
  {
    return not any();
  }

  // LeafRangeListView::is_singleton()
  template <int dim>
  bool LeafRangeListView<dim>::is_singleton() const
  {
    const bool is_singleton = (this->is_subrange() ?
        m_subrange.is_singleton() : m_sublist.is_singleton());
    if (is_singleton)
    {
      if (this->is_subrange())
      {
        assert(m_subrange.is_singleton_of(this->root()));
      }
      else
      {
        assert(m_sublist.is_singleton_of(this->root()));
      }
    }
    return is_singleton;
  }

  // LeafRangeListView::is_singleton_of()
  template <int dim>
  bool LeafRangeListView<dim>::is_singleton_of(const TreeNode<uint32_t, dim> &member) const
  {
    if (this->is_subrange())
      return m_subrange.is_singleton_of(member);
    else
      return m_sublist.is_singleton_of(member);
  }

  // LeafRangeListView::operator==()
  template <int dim>
  bool LeafRangeListView<dim>::operator==(const LeafRangeListView &other) const
  {
    return std::tie(m_sublist, m_subrange)
        == std::tie(other.m_sublist, other.m_subrange);
  }

  // LeafRangeListView::operator!=()
  template <int dim>
  bool LeafRangeListView<dim>::operator!=(const LeafRangeListView &other) const
  {
    return not (*this == other);
  }

  /// // LeafRangeListView::child()
  /// template <int dim>
  /// LeafRangeListView<dim> LeafRangeListView<dim>::child(sfc::ChildNum c) const
  /// {
  ///   if (this->is_subrange())
  ///   {
  ///     LeafRange<dim> new_range = m_subrange.child(c);
  ///     return LeafRangeListView(AsRange{}, m_sublist, new_range);
  ///   }
  ///   else
  ///   {
  ///     LeafListView<dim> new_list = m_sublist.child(c);

  ///     const TreeNode<uint32_t, dim> *begin = new_list.begin();
  ///     const TreeNode<uint32_t, dim> *end = new_list.end();

  ///     // The search usually cuts a range, but we need the whole range.
  ///     bool adjusted = false;
  ///     if ((begin - m_sublist.begin()) % 2 != 0)
  ///     {
  ///       --begin;
  ///       adjusted = true;
  ///     }
  ///     if ((end - m_sublist.begin()) % 2 != 0)
  ///     {
  ///       ++end;
  ///       adjusted = true;
  ///     }

  ///     return LeafRangeListView(
  ///         begin, end, (adjusted? this->m_scope : new_list.scope()));
  ///   }
  /// }

  /// // LeafRangeListView::subdivide()
  /// template <int dim>
  /// LeafRangeListView<dim> LeafRangeListView<dim>::subdivide(sfc::SubIndex s) const
  /// {
  ///   return this->child(this->sfc().child_num(s));
  /// }

  // LeafRangeListView::fair_segment()
  template <int dim>
  LeafRangeListView<dim> LeafRangeListView<dim>::fair_segment(int pick, int count) const
  {
    const size_t total_length = (this->end() - this->begin()) / 2;
    const auto *begin = (total_length * pick / count) * 2     + this->begin();
    const auto *end = (total_length * (pick + 1) / count) * 2 + this->begin();
    return LeafRangeListView(begin, end, this->scope());
  }

  // LeafRangeListView::begin()
  template <int dim>
  const TreeNode<uint32_t, dim> *LeafRangeListView<dim>::begin() const
  {
    return m_sublist.begin();
  }

  // LeafRangeListView::end()
  template <int dim>
  const TreeNode<uint32_t, dim> *LeafRangeListView<dim>::end() const
  {
    return m_sublist.end();
  }

  // LeafRangeListView::is_single_range()
  template <int dim>
  bool LeafRangeListView<dim>::is_single_range() const
  {
    return this->is_subrange();
  }

  // LeafRangeListView::range()
  template <int dim>
  LeafRange<dim> LeafRangeListView<dim>::range() const
  {
    return m_subrange;
  }

  // LeafRangeListView::shrink_begin()
  template <int dim>
  LeafRangeListView<dim> LeafRangeListView<dim>::shrink_begin() const
  {
    return LeafRangeListView(this->begin() + 2, this->end(), this->scope());
  }

  // LeafRangeListView::shrink_end()
  template <int dim>
  LeafRangeListView<dim> LeafRangeListView<dim>::shrink_end() const
  {
    return LeafRangeListView(this->begin(), this->end() - 2, this->scope());
  }


  // vec_leaf_range_list_view()
  template <int dim, typename V>
  LeafRangeListView<dim> vec_leaf_range_list_view(const V &vector)
  {
    // Need random access.
    return LeafRangeListView<dim>(&vector[0], &vector[vector.size()]);
  }


  // LeafRange()
  template <int dim>
  LeafRange<dim>::LeafRange( TreeNode<uint32_t, dim> first,
                             TreeNode<uint32_t, dim> last,
                             bool nonempty,
                             DescendantSet<dim> scope )
    : Base({(nonempty?
              scope.select(common_ancestor(first, last))  // Drill down
              : scope)}),
      m_first(first), m_last(last), m_nonempty(nonempty)
  { }

  // LeafRange::make_empty()
  template <int dim>
  LeafRange<dim> LeafRange<dim>::make_empty(DescendantSet<dim> scope)
  {
    return LeafRange( {}, {}, false, scope );
  }

  // LeafRange::make()
  template <int dim>
  LeafRange<dim> LeafRange<dim>::make(TreeNode<uint32_t, dim> first,
                 TreeNode<uint32_t, dim> last,
                 DescendantSet<dim> scope)
  {
    assert(scope.m_root.isAncestorInclusive(first));
    assert(scope.m_root.isAncestorInclusive(last));
    return LeafRange( first, last, true, scope );
  }

  // LeafRange::any()
  template <int dim>
  bool LeafRange<dim>::any() const
  {
    return m_nonempty;
  }

  // LeafRange::none()
  template <int dim>
  bool LeafRange<dim>::none() const
  {
    return not any();
  }

  // LeafRange::is_singleton()
  template <int dim>
  bool LeafRange<dim>::is_singleton() const
  {
    const bool is_singleton = any() and m_first == m_last;
    if (is_singleton)
    {
      assert(this->root() == m_first);
    }
    return is_singleton;
  }

  // LeafRange::is_singleton_of()
  template <int dim>
  bool LeafRange<dim>::is_singleton_of(const TreeNode<uint32_t, dim> &member) const
  {
    return is_singleton() and m_first == member;
  }

  // LeafRange::operator==()
  template <int dim>
  bool LeafRange<dim>::operator==(const LeafRange &other) const
  {
    return this->none() and other.none()
        or std::tie(m_first, m_last, m_nonempty)
        == std::tie(other.m_first, other.m_last, other.m_nonempty);
  }

  // LeafRange::operator!=()
  template <int dim>
  bool LeafRange<dim>::operator!=(const LeafRange &other) const
  {
    return not (*this == other);
  }

  // LeafRange::child()
  template <int dim>
  LeafRange<dim> LeafRange<dim>::child(sfc::ChildNum c) const
  {
    assert(( (this->none())
          or (this->root().getLevel() < m_first.getLevel()
            and this->root().getLevel() < m_last.getLevel()) ));

    if (not this->any())
    {
      return LeafRange::make_empty(this->m_scope.child(c));
    }

    const TreeNode<uint32_t, dim> child_key = this->root().getChildMorton(c);
    const int level = child_key.getLevel();
    const SFC_State<dim> sfc = this->sfc();

    // Non-ancestors: either precede, succeed, or are descendants of child_key.
    const auto sfc_rank = [sfc, level](TreeNode<uint32_t, dim> t) {
      return sfc.child_rank(sfc::ChildNum(t.getMortonIndex(level)));
    };
    const sfc::SubIndex first_rank = sfc_rank(m_first);
    const sfc::SubIndex last_rank = sfc_rank(m_last);
    const sfc::SubIndex child_rank = sfc.child_rank(sfc::ChildNum(c));

    // If last precedes descendants, or first succeeds descendants: Empty.
    if (last_rank < child_rank or child_rank < first_rank)
    {
      return LeafRange::make_empty(this->m_scope.child(c));
    }

    // Default: If a bound (first or last) is a descendant, keep it that way.
    TreeNode<uint32_t, dim> new_first = m_first;
    TreeNode<uint32_t, dim> new_last = m_last;

    // If first precedes, clamp; divide until not (proper) ancestor of last.
    if (first_rank < child_rank)
    {
      new_first = child_key;

      constexpr sfc::SubIndex front(0);
      SFC_State<dim> descendant_sfc = sfc.child_curve(c);
      while (new_first.isAncestor(m_last))
      {
        new_first = new_first.getChildMorton(descendant_sfc.child_num(front));
        descendant_sfc = descendant_sfc.subcurve(front);
      }
    }

    // If last succeeds, clamp; divide until not (proper) ancestor of first.
    if (last_rank > child_rank)
    {
      new_last = child_key;

      constexpr sfc::SubIndex back(nchild(dim) - 1);
      SFC_State<dim> descendant_sfc = sfc.child_curve(c);
      while (new_last.isAncestor(m_first))
      {
        new_last = new_last.getChildMorton(descendant_sfc.child_num(back));
        descendant_sfc = descendant_sfc.subcurve(back);
      }
    }

    return LeafRange::make(new_first, new_last, this->m_scope.child(c));
  }

  // LeafRange::subdivide()
  template <int dim>
  LeafRange<dim> LeafRange<dim>::subdivide(sfc::SubIndex s) const
  {
    return this->child(this->sfc().child_num(s));
  }


  // vec_leaf_range()
  template <int dim, typename V>
  LeafRange<dim> vec_leaf_range(const V &vector)
  {
    if (vector.empty())
      return LeafRange<dim>::make_empty();
    else
      return LeafRange<dim>::make(vector.front(), vector.back());
  }

}


#endif//DENDRO_KT_LEAF_SETS_HPP
