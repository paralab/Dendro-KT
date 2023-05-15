/**
 * @author Masado Ishii
 * @date   2023-05-12
 * @brief  Find things that are adjacent to other things.
 */

#ifndef DENDRO_KT_SUBDIVISION_SEARCH_HPP
#define DENDRO_KT_SUBDIVISION_SEARCH_HPP

// Function declaration for linkage purposes.
inline void link_subdivision_search_tests() {};

#include "include/treeNode.h"

namespace ot
{
  template <unsigned dim>
  constexpr bool adjoins(
      const TreeNode<uint32_t, dim> &a,
      const TreeNode<uint32_t, dim> &b)
  {
    const auto a_range = a.range();
    const auto b_range = b.range();
    const periodic::IntersectionMagnitude intersection =
        periodic::intersect_magnitude(a_range, b_range);
    return intersection.nonempty and intersection.dimension < dim;
  }

  template <unsigned dim>
  constexpr bool overlaps(
      const TreeNode<uint32_t, dim> &a,
      const TreeNode<uint32_t, dim> &b)
  {
    return a.isAncestorInclusive(b) or b.isAncestorInclusive(a);
  }

  template <unsigned dim>
  constexpr bool covers(
      const TreeNode<uint32_t, dim> &a,
      const TreeNode<uint32_t, dim> &b)
  {
    return a.isAncestorInclusive(b);
  }

  namespace adjacency
  {
    template <int dim> struct BoundingBox;

    // =========================================================================
    // List<> and Set<> interfaces
    // -------------------------------------------------------------------------
    // Derive from these classes using CRTP, provide member functions, and
    // ensure that the atoms returned atom() are supported by overloads of
    // adjoins(), overlaps(), and covers() in the same namespace.
    // -------------------------------------------------------------------------
    // adjoins (const X &x, const Y &y)    adjoins (const X &x, BoundingBox b)
    // overlaps(const X &x, const Y &y)    overlaps(const X &x, BoundingBox b)
    //                                     covers  (const X &x, BoundingBox b)
    // =========================================================================
    template <int dim, typename ListType>
    struct List
    {
      // Interface
      //   bool none() const;
      //   bool any() const;
      //   BoundingBox<dim> bounds() const;
      //   [Iterable of ListType] list_split() const;
      //   bool is_single_item() const;
      //   typename ListType::item_type item() const;

      // Down-cast.
      ListType       & cast()       { return static_cast<ListType &>(*this); }
      const ListType & cast() const { return static_cast<const ListType &>(*this); }
    };

    template <int dim, typename SetType>
    struct Set
    {
      // Interface
      //   bool none() const;
      //   bool any() const;
      //   BoundingBox<dim> bounds() const;
      //   [Iterable of SetType] set_split() const;
      //   SetType restrict(BoundingBox<dim> box) const;
      //     // ^ Choose any subset containing all atoms that meet box.
      //   bool is_single_atom() const;
      //   typename SetType::atom_type atom() const;

      // Down-cast.
      SetType       & cast()       { return static_cast<SetType &>(*this); }
      const SetType & cast() const { return static_cast<const SetType &>(*this); }
    };
    // =========================================================================


    // =========================================================================
    // BoundingBox
    // =========================================================================
    template <int dim>
    struct BoundingBox
    {
      public:
        uint32_t size() const { return octant.range().side(); }
      public:
        TreeNode<uint32_t, dim> octant;
    };

    template <int dim>
    constexpr bool adjoins(BoundingBox<dim> a, BoundingBox<dim> b)
    {
      return adjoins(a.octant, b.octant);
    }

    template <int dim>
    constexpr bool overlaps(BoundingBox<dim> a, BoundingBox<dim> b)
    {
      return a.octant.isAncestorInclusive(b.octant) or b.octant.isAncestorInclusive(a.octant);
    }

    template <int dim>
    constexpr bool covers(BoundingBox<dim> a, BoundingBox<dim> b)
    {
      // covers() is an absolute test that should only be applied to an atom.
      // The bounding box is not enough to determine if an atom covers b.
      return false;
    }
    // =========================================================================


    // =========================================================================
    // Meets: Subdivision search with where_meet() and overloads of meets().
    // =========================================================================
    enum OverlapOption { WithoutOverlap, MaybeOverlap };

    template <int dim, OverlapOption how_overlap> class Meets;

    template <int dim>
    Meets<dim, WithoutOverlap> meets_without_overlap() { return {}; }

    template <int dim>
    Meets<dim, MaybeOverlap> meets_maybe_overlap() { return {}; }

    template <bool>
    struct Presume { };

    template <int dim, OverlapOption how_overlap>
    class Meets
    {
      public:
        //future: make everything constexpr (requires c++17 lambdas)

        // includes_overlaps()  (class property)
        static bool includes_overlaps() { return how_overlap != WithoutOverlap; }

        // meets(box, box)  (base case)
        static bool meets(BoundingBox<dim> a,
                          BoundingBox<dim> b)
        {
          return adjoins(a, b) or overlaps(a, b);
        }

        // meets(primitive, box)  (base case)
        template <typename X>
        static bool meets(const X &x,
                          BoundingBox<dim> b)
        {
          return adjoins(x, b) or overlaps(x, b);
        }

        // meets(box, primitive)  (base case)
        template <typename X>
        static bool meets(BoundingBox<dim> b,
                          const X &x)
        {
          return adjoins(x, b) or overlaps(x, b);
        }

        // meets(primitive, primitive)  (base case)
        template <typename X, typename Y>
        static bool meets(const X &x, const Y &y)
        {
          bool result;
          switch (how_overlap)
          {
            case WithoutOverlap: return adjoins(x, y);
            case MaybeOverlap:   return adjoins(x, y) or overlaps(x, y);
            default:             assert(!"Unhandled how_overlap enumerator");
            return false;
          }
        }


        // meets(atom|box, set): Divide set until find an atom::atom meeting.
        template <typename X, typename SetType, bool presume = false>
        static bool meets_set(const X &x, const Set<dim, SetType> &set_, Presume<presume> = {})
        {
          const SetType &set = set_.cast();
          // Preconditions: set nonempty and x meets bound(set)
          if (not presume)
          {
            if (set.none())
              return false;
            if (not meets(x, set.bounds()))
              return false;
          }

          // base
          if (covers(x, set.bounds()))
            if (includes_overlaps())
              return true;
            else
              return false;

          // base
          if (set.is_single_atom())
            return meets(x, set.atom());
          // recurse
          else
          {
            for (SetType subset : set.set_split())
              if (subset.any() and meets(x, subset.bounds()))
                if (meets_set(x, subset, Presume<true>()))
                  return true;
            return false;
          }
        }


        // meets(set, set): Divide both sets until reach (atom, set) case.
        template <typename PType, typename QType, bool presume = false>
        static bool meets_set(const Set<dim, PType> &P_, const Set<dim, QType> &Q_, Presume<presume> = {})
        {
          const PType &P = P_.cast();
          const QType &Q = Q_.cast();
          // Preconditions: both nonempty and bounds(P) meets bounds(Q).
          if (not presume)
          {
            if (P.none() or Q.none() or not meets(P.bounds(), Q.bounds()))
              return false;
          }

          if (is_single_atom(P))
            if (is_single_atom(Q))              // base
              return meets(P.atom(), Q.atom());
            else                                // base
              return meets_set(P.atom(), Q);
          else
            if (is_single_atom(Q))              // base
              return meets_set(Q.atom(), P);
            else
              if (P.bounds().size() >= Q.bounds().size())     // recurse
              {
                for (PType subset : P.set_split())
                  if (subset.any() and meets(subset.bounds(), Q.bounds()))
                    if (meets_set(subset, Q, Presume<true>()))
                      return true;
                return false;
              }
              else                                            // recurse
              {
                for (QType subset : Q.set_split())
                  if (subset.any() and meets(P.bounds(), subset.bounds()))
                    if (meets_set(P, subset, Presume<true>()))
                      return true;
                return false;
              }
        }

        // where_meet(): Emit every item of list that meets set.
        template <typename ListType,
                  typename SetType,
                  typename Emit,
                  bool presume = false>
        static void where_meet(
            const List<dim, ListType> &list_,
            const Set<dim, SetType> &set_,
            Emit &&emit,
            Presume<presume> = {})
        {
          const ListType &list = list_.cast();
          const SetType &set = set_.cast();
          // Preconditions: both nonempty, set is reduced to bound(list),
          //                and bound(list) meets set
          if (not presume)
          {
            if (list.none() or set.none())
              return;
            const SetType &set_reduced = set.reduce(list.bounds());
            if (set_reduced.none())
              return;
            else if (not meets_set(list.bounds(), set_reduced, Presume<true>()))
              return;
            // Preconditions are now met.
            return where_meet(list, set_reduced, std::forward<Emit>(emit), Presume<true>());
          }

          // base
          if (list.is_single_item())
          {
            const auto &item = list.item();
            if (meets_set(item, set, Presume<true>()))
              emit(item);
          }
          // recurse
          else
          {
            for (ListType section : list.list_split())
              if (section.any())
              {
                const SetType &subset = set.reduce(section.bounds());
                if (subset.any() and meets_set(section.bounds(), subset, Presume<true>()))
                  where_meet(section, subset, std::forward<Emit>(emit), Presume<true>());
              }
          }
        }
    };
    // =========================================================================


    // =========================================================================
    // Concepts and proofs.
    // =========================================================================

    // Distinguish between list items and geometric primitives.
    //
    // A set is considered as a union of atoms, and can always be divided into atoms.
    // An atom cannot be divided.
    // A bounding box need not be divided.

    // For subdivision search to work on a list, it must be possible to derive
    // the bounding box of an entire list from the bounding boxes of the first
    // and last primitives in the list.

    // - In the default case, when the primitives are nonoverlapping octants that
    //   equal their bounding boxes, the ancestor common to the first and last
    //   octants is guaranteed to bound all octants in between.

    // (hanging case)
    // - This also works if the bounding box of an octant is defined as its parent.
    //   - Let x < y < z be nonoverlapping octants, p(x), p(y), p(z) their parents.
    //   - Define B := CommonAncestor( p(x), p(z) ).
    //   - Show that B is also an ancestor of p(y).
    //     - B and p(y) are either nested or volume-disjoint.
    //       - p(y) cannot be volume disjoint from B, for then so would y, but x < y < z.
    //       - So, either B is an ancestor of p(y), or p(y) is a (proper) ancestor of B.
    //         - But p(y) cannot be a proper ancestor of B, for then y would be
    //           an ancestor of B, overlapping x and z.
    //   QED

  }//namespace adjacency

}//namespace ot



// =============================================================================
// Tests
// =============================================================================
#ifdef DOCTEST_LIBRARY_INCLUDED
#include "include/tnUtils.h"
#include "include/leaf_sets.hpp"
#include <array>
#include <vector>
namespace ot
{
  DOCTEST_TEST_SUITE("Subdivision search")
  {
    namespace normal_search
    {
      template <int dim>
      struct Leaf
      {
        TreeNode<uint32_t, dim> me;
      };

      template <int dim>
      bool adjoins(Leaf<dim> a, Leaf<dim> b)
      { return adjoins(a.me, b.me); }

      template <int dim>
      bool adjoins(Leaf<dim> a, adjacency::BoundingBox<dim> b)
      { return adjoins(a.me, b.octant); }

      template <int dim>
      bool overlaps(Leaf<dim> a, Leaf<dim> b)
      { return overlaps(a.me, b.me); }

      template <int dim>
      bool overlaps(Leaf<dim> a, adjacency::BoundingBox<dim> b)
      { return overlaps(a.me, b.octant); }

      template <int dim>
      bool covers(Leaf<dim> a, adjacency::BoundingBox<dim> b)
      { return covers(a.me, b.octant); }


      template <int dim>
      struct NormalList: public ::ot::adjacency::List<dim, NormalList<dim>>,
                                ::ot::adjacency::Set<dim, NormalList<dim>>
      {
        public:
          NormalList(const LeafListView<dim> &view) : m_view(view) { }
          bool none() const                { return m_view.none(); }
          bool any() const                 { return m_view.any(); }
          adjacency::BoundingBox<dim> bounds() const  { return { m_view.scope().m_root }; }
          std::array<NormalList, 2> list_split() const
          {
            const TreeNode<uint32_t, dim> *ptr = m_view.begin();
            const size_t length = m_view.end() - m_view.begin();
            const size_t half = length / 2;
            return {LeafListView<dim>{ptr, ptr + half},
                    LeafListView<dim>{ptr + half, ptr + length}};
          }

          bool is_single_item() const { return m_view.begin() + 1 == m_view.end(); }
          Leaf<dim> item() const { return {*m_view.begin()}; }

          const NormalList &reduce(adjacency::BoundingBox<dim>) const { return *this; }  // waste
          std::array<NormalList, 2> set_split() const { return list_split(); }
          bool is_single_atom() const { return m_view.begin() + 1 == m_view.end(); }
          Leaf<dim> atom() const { return {*m_view.begin()}; }

        public:
          LeafListView<dim> m_view;
      };
    }

    namespace hanging_search
    {
      template <int dim>
      struct Leaf
      {
        TreeNode<uint32_t, dim> me;
      };

      template <int dim>
      bool adjoins(Leaf<dim> a, Leaf<dim> b)
      { return adjoins(a.me, b.me)
        or (a.me.getLevel() + 1 == b.me.getLevel() and adjoins(a.me, b.me.getParent()))
        or (a.me.getLevel() == b.me.getLevel() + 1 and adjoins(a.me.getParent(), b.me)); }

      template <int dim>
      bool adjoins(Leaf<dim> a, adjacency::BoundingBox<dim> b)
      { return adjoins(a.me, b.octant)
        or (a.me.getLevel() > b.octant.getLevel() and adjoins(a.me.getParent(), b.octant)); }

      template <int dim>
      bool overlaps(Leaf<dim> a, Leaf<dim> b)
      { return overlaps(a.me, b.me); }

      template <int dim>
      bool overlaps(Leaf<dim> a, adjacency::BoundingBox<dim> b)
      { return overlaps(a.me, b.octant); }

      template <int dim>
      bool covers(Leaf<dim> a, adjacency::BoundingBox<dim> b)
      { return covers(a.me, b.octant); }


      template <int dim>
      struct HangList: public ::ot::adjacency::List<dim, HangList<dim>>,
                                ::ot::adjacency::Set<dim, HangList<dim>>
      {
        public:
          HangList(const LeafListView<dim> &view) : m_view(view) { }
          bool none() const                { return m_view.none(); }
          bool any() const                 { return m_view.any(); }
          adjacency::BoundingBox<dim> bounds() const
          {
            const TreeNode<uint32_t, dim> p_first = m_view.begin()->getParent();
            const TreeNode<uint32_t, dim> p_last = (m_view.end() - 1)->getParent();
            return { common_ancestor(p_first, p_last) };
          }
          std::array<HangList, 2> list_split() const
          {
            const TreeNode<uint32_t, dim> *ptr = m_view.begin();
            const size_t length = m_view.end() - m_view.begin();
            const size_t half = length / 2;
            return {LeafListView<dim>{ptr, ptr + half},
                    LeafListView<dim>{ptr + half, ptr + length}};
          }

          bool is_single_item() const { return m_view.begin() + 1 == m_view.end(); }
          Leaf<dim> item() const { return {*m_view.begin()}; }

          const HangList &reduce(adjacency::BoundingBox<dim>) const { return *this; }  // waste
          std::array<HangList, 2> set_split() const { return list_split(); }
          bool is_single_atom() const { return m_view.begin() + 1 == m_view.end(); }
          Leaf<dim> atom() const { return {*m_view.begin()}; }

        public:
          LeafListView<dim> m_view;
      };
    }


    DOCTEST_TEST_CASE("basic ranges")
    {
      constexpr int dim = 2;
      using Octant = TreeNode<uint32_t, dim>;
      _InitializeHcurve(dim);

      const std::vector<Octant> red = {
        morton_lineage<dim>({0, 3, 3, 0}),
        morton_lineage<dim>({0, 3, 3, 1}),
        morton_lineage<dim>({0, 3, 3, 2}),
        morton_lineage<dim>({0, 3, 3, 3, 0}),
        morton_lineage<dim>({0, 3, 3, 3, 1}),
      };

      const std::vector<Octant> blue = {
        morton_lineage<dim>({0, 3, 3, 3, 2}),
        morton_lineage<dim>({0, 3, 3, 3, 3}),
      };

      normal_search::NormalList<dim> red_list  = {vec_leaf_list_view<dim>(red)};
      normal_search::NormalList<dim> blue_list = {vec_leaf_list_view<dim>(blue)};
      size_t count = 0;
      const auto M = adjacency::meets_without_overlap<dim>();
      M.where_meet(red_list, blue_list, [&](auto) { ++count; });
      CHECK( count == 3 );
    }

    DOCTEST_TEST_CASE("weird hanging case")
    {
      constexpr int dim = 2;
      using Octant = TreeNode<uint32_t, dim>;
      _InitializeHcurve(dim);

      const std::vector<Octant> red = {
        morton_lineage<dim>({0, 3, 3, 0}),
        morton_lineage<dim>({0, 3, 3, 1}),
        morton_lineage<dim>({0, 3, 3, 2}),
        morton_lineage<dim>({0, 3, 3, 3, 0}),
        morton_lineage<dim>({0, 3, 3, 3, 1}),
      };

      const std::vector<Octant> blue = {
        morton_lineage<dim>({0, 3, 3, 3, 2}),
        morton_lineage<dim>({0, 3, 3, 3, 3}),
      };

      hanging_search::HangList<dim> red_list  = {vec_leaf_list_view<dim>(red)};
      hanging_search::HangList<dim> blue_list = {vec_leaf_list_view<dim>(blue)};
      size_t count = 0;
      const auto M = adjacency::meets_without_overlap<dim>();
      M.where_meet(red_list, blue_list, [&](auto) { ++count; });
      CHECK( count == 5 );

      _DestroyHcurve();
    }
  }
}
#endif//DOCTEST_LIBRARY_INCLUDED



#endif//DENDRO_KT_SUBDIVISION_SEARCH_HPP

