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

      LeafRange subdivide(sfc::SubIndex s) const;
      LeafRange child(sfc::ChildNum c) const;

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


  // TODO make interface LeafListView: if is_singleton() then root() == member.

  // LeafListView::LeafListView()
  template <int dim>
  LeafListView<dim>::LeafListView( const TreeNode<uint32_t, dim> *begin,
                           const TreeNode<uint32_t, dim> *end,
                           DescendantSet<dim> scope )
    : Base({scope}), m_begin(begin), m_end(end)
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
    return m_begin + 1 == m_end;
  }

  // LeafListView::is_singleton_of()
  template <int dim>
  bool LeafListView<dim>::is_singleton_of(const TreeNode<uint32_t, dim> &member) const
  {
    return is_singleton() and *m_begin == member;
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

    // Drill down to deepest common ancestor.
    if (new_begin < new_end)
    {
      const TreeNode<uint32_t, dim> first = *(result.m_begin);
      const TreeNode<uint32_t, dim> last = *(result.m_end - 1);
      const TreeNode<uint32_t, dim> common
          = first.getAncestor(first.getCommonAncestorDepth(last));
      result.m_scope = result.m_scope.select(common);
    }

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


  // TODO make interface LeafRange: if is_singleton() then root() == member.

  // LeafRange()
  template <int dim>
  LeafRange<dim>::LeafRange( TreeNode<uint32_t, dim> first,
                             TreeNode<uint32_t, dim> last,
                             bool nonempty,
                             DescendantSet<dim> scope )
    : Base({scope}), m_first(first), m_last(last), m_nonempty(nonempty)
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
    return any() and m_first == m_last;
  }

  // LeafRange::is_singleton_of()
  template <int dim>
  bool LeafRange<dim>::is_singleton_of(const TreeNode<uint32_t, dim> &member) const
  {
    return is_singleton() and m_first == member;
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

    // Drill down DescendantSet to deepest common ancestor.
    const TreeNode<uint32_t, dim> common =
        new_first.getAncestor(new_first.getCommonAncestorDepth(new_last));
    return LeafRange::make(new_first, new_last, this->m_scope.select(common));
  }

  // LeafRange::subdivide()
  template <int dim>
  LeafRange<dim> LeafRange<dim>::subdivide(sfc::SubIndex s) const
  {
    return this->child(this->sfc().child_num(s));
  }

}


#endif//DENDRO_KT_LEAF_SETS_HPP
