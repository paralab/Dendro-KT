//
// Created by masado on 10/28/22.
//

#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro

/// #include "test/octree/multisphere.h"

#include <include/treeNode.h>
#include <include/tsort.h>
/// #include <include/distTree.h> // for convenient uniform grid partition
#include <include/sfc_search.h>

#include "test/octree/gaussian.hpp"

#include <vector>

// -------------------------------------------
// TODO
// ✔ Define intervals (less/leq/greater/geq to key)
// _ Interval proximity depth-first search
// _ Log the number of octant intersection tests
// _ Accelerate search by diving to common ancestor of endpoints
// _ Test: Every interval has proximity to itself
// _ Test: Empty intervals have proximity to none
// _ Test: Two sibling singleton intervals have proximity
// _ Test: Every search returns a pair, that are contained in the intervals and are adjacent
// -------------------------------------------

// -----------------------------
// Typedefs
// -----------------------------
using uint = unsigned int;
using LLU = long long unsigned;

template <int dim>
using Oct = ot::TreeNode<uint, dim>;

// -----------------------------
// Helper classes
// -----------------------------
struct SfcTableScope
{
  SfcTableScope(int dim) { _InitializeHcurve(dim); }
  ~SfcTableScope() { _DestroyHcurve(); }
};

template <class Class>
struct Constructors       // Callable object wrapping constructor overloads
{
  template <typename ... T>
  Class operator()(T && ... ts) const {
    return Class(std::forward<T>(ts)...);
  }
};

template <typename T, unsigned dim>
ot::SFC_Region<T, dim> octant_to_region(ot::TreeNode<T, dim> octant);


using uiCoord = unsigned;

/**
 * @brief Intersection of an upper half-line, lower half-line, and set of octant descendants,
 *  in the octant total ordering.
 */
template <int dim>
struct IntervalSubRegion
{
  public:
    // By default, all octants, that are descendants of the root (still all).
    ot::TreeNode<uiCoord, dim> begin = {};
    ot::TreeNode<uiCoord, dim> end   = {};
    ot::SFC_Region<uiCoord, dim> subregion = {};
  private:
    ot::SFC_Region<uiCoord, dim> r_begin = {};  // to compare with begin
    ot::SFC_Region<uiCoord, dim> r_end = {};    // to compare with end
    bool region_bounded_below = true;           // inform splits, detect subset
    bool region_bounded_above = true;           // inform splits, detect subset
  public:
    bool exclude_begin_descendants = false;
    bool exclude_end_descendants   = false;


  public:
    // Create default, then piecewise construct on temporary.
    // The half interval conditions match possible bucket boundaries.
    // nlt/ngt: Not less than, not greater than a region and its descendants.
    // gt/lt: Greater than, less than a region and its descendants.
    constexpr IntervalSubRegion() = default;
    constexpr IntervalSubRegion nlt(ot::TreeNode<uiCoord, dim> oct) &&;
    constexpr IntervalSubRegion ngt(ot::TreeNode<uiCoord, dim> oct) &&;
    constexpr IntervalSubRegion gt(ot::TreeNode<uiCoord, dim> oct) &&;
    constexpr IntervalSubRegion lt(ot::TreeNode<uiCoord, dim> oct) &&;

    // Split subregion.
    constexpr IntervalSubRegion split_region(ot::sfc::SubIndex i) const;
    constexpr int first_segment() const;  // signed index
    constexpr int last_segment() const;   // signed index

    // Test membership.
    constexpr bool contains_descendants() const;
    constexpr bool contains_octant() const;
};


// Assume that I and J have octants that overlap or touch.
template <int dim>
bool interval_proximity(const IntervalSubRegion<dim> &I, const IntervalSubRegion<dim> &J)
{
  const bool border = ::border(I.subregion.octant.range(),
                               J.subregion.octant.range());
  const int i_level = I.subregion.octant.getLevel();
  const int j_level = J.subregion.octant.getLevel();

  if (I.contains_descendants() and J.contains_descendants())
    return true;  // already assume I and J intersect

  else if (border and I.contains_octant() and J.contains_octant()) //parent
    return true;

  else if (I.contains_descendants() or j_level < i_level)
  {
    for (int s = J.first_segment(), last = J.last_segment(); s <= last; ++s)
    {
      const auto J_s = J.split_region(ot::sfc::SubIndex(s));
      const bool intersect = closed_overlap(I.subregion.octant.range(),
                                            J_s.subregion.octant.range());
      if (intersect and interval_proximity(I, J_s))  // recurse on J's children
        return true;
    }
  }
  else if (J.contains_descendants() or i_level < j_level)
  {
    for(int s = I.first_segment(), last = I.last_segment(); s <= last; ++s)
    {
      const auto I_s = J.split_region(ot::sfc::SubIndex(s));
      const bool intersect = closed_overlap(I_s.subregion.octant.range(),
                                            J.subregion.octant.range());
      if (intersect and interval_proximity(I_s, J))  // recurse on I's children
        return true;
    }
  }
  else
  {
    //future: Use a heuristic to pick pairs most likely to intersect
    IntervalSubRegion<dim> I_sub[ot::nchild(dim)];
    IntervalSubRegion<dim> J_sub[ot::nchild(dim)];
    for(int s = I.first_segment(), last = I.last_segment(); s <= last; ++s)
      I_sub[s] = I.split_region(ot::sfc::SubIndex(s));
    for (int s = J.first_segment(), last = J.last_segment(); s <= last; ++s)
      J_sub[s] = J.split_region(ot::sfc::SubIndex(s));
    for(int s = I.first_segment(), s_last = I.last_segment(); s <= s_last; ++s)
      for (int t = J.first_segment(), t_last = J.last_segment(); t <= t_last; ++t)
        if (closed_overlap(I_sub[s].subregion.octant.range(),
                           J_sub[t].subregion.octant.range())
            and interval_proximity(I_sub[s], J_sub[t]))
          return true;
  }
  return false;
}




TEST_CASE("Every singleton interval has proximity to itself")
{
  constexpr int DIM = 3;
  using Oct = Oct<DIM>;
  const SfcTableScope _(DIM);
  {
    const size_t n_octants = 1000;
    std::vector<Oct> octants = test::gaussian<uint, DIM>(0, n_octants, Constructors<Oct>{});
    for (auto oct : octants)
    {
      /// ot::SFC_Region<uint, DIM> region = octant_to_region(oct);
      auto I = IntervalSubRegion<DIM>().nlt(oct).ngt(oct);
      CHECK(interval_proximity(I, I));
    }
  }
}

TEST_CASE("Every empty interval has proximity to none")
{
  constexpr int DIM = 3;
  using Oct = Oct<DIM>;
  const SfcTableScope _(DIM);
  {
    const size_t n_octants = 1000;
    std::vector<Oct> octants = test::gaussian<uint, DIM>(0, n_octants, Constructors<Oct>{});
    for (auto oct_singleton : octants)
      for (auto oct_empty : octants)
      {
        /// ot::SFC_Region<uint, DIM> region_singleton = octant_to_region(oct_singleton);
        /// ot::SFC_Region<uint, DIM> region_empty = octant_to_region(oct_empty);
        auto singleton = IntervalSubRegion<DIM>().nlt(oct_singleton).ngt(oct_singleton);
        auto empty = IntervalSubRegion<DIM>().gt(oct_empty).lt(oct_empty);
        CHECK_FALSE(interval_proximity(singleton, empty));
      }
  }
}




template <typename T, unsigned dim>
ot::SFC_Region<T, dim> octant_to_region(ot::TreeNode<T, dim> octant)
{
  ot::SFC_Region<T, dim> region;
  while (region.octant != octant)
    region = region.subdivide(region.locate_segment(octant));
  return region;
}





template <int dim>
constexpr IntervalSubRegion<dim> IntervalSubRegion<dim>::nlt(ot::TreeNode<uiCoord, dim> oct) &&
{
  begin = oct;
  exclude_begin_descendants = false;

  int compare;
  std::tie(compare, r_begin) = ot::sfc_compare<uiCoord,dim>(begin, subregion.octant, r_begin);

  region_bounded_below = (compare <= 0);
  return std::move(*this);
}

template <int dim>
constexpr IntervalSubRegion<dim> IntervalSubRegion<dim>::gt(ot::TreeNode<uiCoord, dim> oct) &&
{
  begin = oct;
  exclude_begin_descendants = true;

  int compare;
  std::tie(compare, r_begin) = ot::sfc_compare<uiCoord,dim>(begin, subregion.octant, r_begin);

  region_bounded_below = (compare < 0);
  return std::move(*this);
}

template <int dim>
constexpr IntervalSubRegion<dim> IntervalSubRegion<dim>::lt(ot::TreeNode<uiCoord, dim> oct) &&
{
  end = oct;
  exclude_end_descendants = true;

  int compare;
  std::tie(compare, r_end) = ot::sfc_compare<uiCoord,dim>(subregion.octant, end, r_end);

  region_bounded_below = (compare < 0);
  return std::move(*this);
}

template <int dim>
constexpr IntervalSubRegion<dim> IntervalSubRegion<dim>::ngt(ot::TreeNode<uiCoord, dim> oct) &&
{
  end = oct;
  exclude_end_descendants = false;

  int compare;
  std::tie(compare, r_end) = ot::sfc_compare<uiCoord,dim>(subregion.octant, end, r_end);

  region_bounded_below = (compare <= 0);
  return std::move(*this);
}

template <int dim>
constexpr IntervalSubRegion<dim> IntervalSubRegion<dim>::split_region(ot::sfc::SubIndex i) const
{
  // Don't create empty split, otherwise, in child, assumption breaks to locate().
  assert(this->first_segment() <= i);
  assert(i <= this->last_segment());

  // Don't need to change:
  //   begin, end, r_begin, r_end, exclude_begin_descendants, exclude_end_descendants
  IntervalSubRegion subdivided = *this;

  int locate_begin, locate_end;

  // Update: subregion, region_bounded_below, region_bounded_above
  subdivided.subregion = subregion.subdivide(i);

  // All descendants are included by begin endpoint if any of the following.
  subdivided.region_bounded_below = region_bounded_below
      or i > (locate_begin = subregion.locate_segment(begin))
      or ((not exclude_begin_descendants)
          and i == locate_begin and subdivided.subregion.octant == begin);

  // All descendants are included by begin endpoint if any of the following.
  subdivided.region_bounded_above = region_bounded_above
      or i < (locate_end = subregion.locate_segment(end))
      or ((not exclude_end_descendants)
          and i == locate_end and subdivided.subregion.octant == end);

  return subdivided;
}

template <int dim>
constexpr int IntervalSubRegion<dim>::first_segment() const
{
  if (region_bounded_below)
    return 0;
  else
  {
    // Assume overlap. If region not contained from left, then begin is a descendant.
    const ot::sfc::SubIndex locate_begin = subregion.locate_segment(begin);
    const int level = subregion.octant.getLevel();
    const bool exclude = (exclude_begin_descendants and begin.getLevel() == level + 1);
    return int(locate_begin) + exclude;
  }
}

template <int dim>
constexpr int IntervalSubRegion<dim>::last_segment() const
{
  if (region_bounded_above)
    return ot::nchild(dim) - 1;
  else
  {
    // Assume overlap. If region not contained from right, then end is a descendant.
    const ot::sfc::SubIndex locate_end = subregion.locate_segment(end);
    const int level = subregion.octant.getLevel();
    const bool exclude = (exclude_end_descendants and end.getLevel() == level + 1);
    return int(locate_end) - exclude;
  }
}

template <int dim>
constexpr bool IntervalSubRegion<dim>::contains_descendants() const
{
  return region_bounded_below and region_bounded_above;
}

template <int dim>
constexpr bool IntervalSubRegion<dim>::contains_octant() const
{
  // Sufficient; in the pre-order traversal, also necessary.
  // Note that by assumption there is region intersection from the right.
  return region_bounded_below;
}


















/// template <int dim>
/// constexpr IntervalSubRegion<dim> IntervalSubRegion<dim>::cannonical() &&
/// {
/// 
///   bool nonempty = true;
///   const int comp = (begin == end) ? 0 : sfc_compare(begin, end, {}).first;
///   if (not descend_begin and not descend_end)
///     nonempty = (comp < 0 or (comp == 0 and (include_begin and include_end)));
///   else if (not descend_end)
///   {
///     if (not begin.isAncestorInclusive(end))
///       nonempty = (comp < 0);
///     else
///     {
///     }
/// 
/// 
///     if (include_begin)  // first(begin) <=/< end  (include_end)/(not include_end)
/// 
///     else                // last(begin) < end
/// 
/// 
///   }
///   else if (not descend_begin)
///   {
///     if (not end.isAncestorInclusive(begin))
///       nonempty = (comp < 0);
///     else
///     {
///     }
///   }
///   else  // both are intervals of descendants
///   {
///     if (begin.isAncestorInclusive(end))
///     {
///       nonempty = (include_begin and (include_end or first(end) != first(begin)));
///     }
///     else if (end.isAncestorInclusive(begin))
///     {
///       nonempty = (include_end and (include_begin or last(begin) != last(end)));
///     }
///     else
///     {
///       nonempty = (comp < 0);
///     }
///   }
/// 
/// 
/// 
///   // Make sure empty set has standard representation.
///   if (not nonempty)
///   {
///     (*this) = {};
///     include_begin = false;
///     include_end = false;
///   }
/// 
///   int compare_begin = sfc_compare(oct, begin).first;
///   int compare_end = sfc_compare(oct.end).first;
///   if (descend_begin and begin.isAncestorInclusive(oct))
///     compare_begin = 0;
///   if (descend_end and end.isAncestorInclusive(oct))
///     compare_end = 0;
///   const bool contained =
///       (compare_begin > 0 or (include_begin and compare_begin == 0))
///       and (compare_end < 0 or (include_end and compare_end == 0));
///   return {contained, r};
/// 
/// 
/// }



/// template <int dim>
/// constexpr std::pair<bool, ot::SFC_Region<uiCoord, dim>>
///   IntervalSubRegion<dim>::contains(
///     ot::TreeNode<uiCoord, dim> oct,
///     ot::SFC_Region<uiCoord, dim> r = {}) const
/// {
///   //future: Refine r to accelerate future containment tests on subdivisions
///   int compare_begin = sfc_compare(oct, begin).first;
///   int compare_end = sfc_compare(oct.end).first;
///   if (descend_begin and begin.isAncestorInclusive(oct))
///     compare_begin = 0;
///   if (descend_end and end.isAncestorInclusive(oct))
///     compare_end = 0;
///   const bool contained =
///       (compare_begin > 0 or (include_begin and compare_begin == 0))
///       and (compare_end < 0 or (include_end and compare_end == 0));
///   return {contained, r};
/// }

/// template <int dim>
/// constexpr std::pair<bool, ot::SFC_Region<uiCoord, dim>>
///   IntervalSubRegion<dim>::contains_descendants(
///     ot::TreeNode<uiCoord, dim> oct,
///     ot::SFC_Region<uiCoord, dim> r = {}) const
/// {
///   //future: Refine r to accelerate future containment tests on subdivisions
///   int compare_begin = sfc_compare(oct, begin).first;
///   int compare_end = sfc_compare(oct.end).first;
///   if (descend_begin and begin.isAncestorInclusive(oct))
///     compare_begin = 0;
///   if (descend_end and end.isAncestorInclusive(oct))
///     compare_end = 0;
///   const bool contained =
///       (compare_begin > 0 or (include_begin and compare_begin == 0))
///       and (compare_end < 0 or (include_end and compare_end == 0));
///   return {contained, r};
/// }







