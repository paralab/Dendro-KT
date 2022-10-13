
#ifndef DENDRO_KT_SFC_SEARCH_H
#define DENDRO_KT_SFC_SEARCH_H

#include "include/tsort.h"

namespace ot
{
  enum RankType { exclusive, inclusive };

  template <typename T, int dim, typename Equals = std::equal_to<TreeNode<T, dim>>>
  size_t sfc_binary_search(
      TreeNode<T, dim> key,
      const TreeNode<T, dim> *list,
      size_t begin,
      size_t end,
      RankType rank_type,
      SFC_Region<T, dim> ra = {},   // future: single region, two levels
      SFC_Region<T, dim> rb = {},
      Equals equals = {})
  {
    // begin <= rank <= end
    if (begin == end)
      return begin;

    const size_t i = (end - begin) / 2 + begin;
    const auto x = list[i];

    // future: min and max of level parameters
    const auto rc = (ra.octant.getLevel() <= rb.octant.getLevel() ? ra : rb);
    const auto rf = (ra.octant.getLevel() <= rb.octant.getLevel() ? rb : ra);

    int comp;
    SFC_Region<T, dim> rk;
    std::tie(comp, rk) = sfc_compare<T, dim>(x, key, (rf.octant.isAncestorInclusive(x) ? rf : rc));
    // future: condition on whether the finer level is less-or-equal to level(x)
    // future: sfc_compare takes region and level that is less-or-equal level of region

    const bool replace_a = (comp == -1) or ((rank_type == inclusive) and (comp == 0));
    /// const bool replace_b = (comp == 1) or ((rank_type == exclusive) and (comp == 0));

    if (replace_a)
      return sfc_binary_search<T, dim>(key, list, i+1, end, rank_type, rk, rb, equals);
    else
      return sfc_binary_search<T, dim>(key, list, begin, i, rank_type, ra, rk, equals);
  }

}//namespace ot

#endif//DENDRO_KT_SFC_SEARCH_H
