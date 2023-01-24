#ifndef DENDRO_KT_TEST_SUBDOMAIN_VOLUME_H
#define DENDRO_KT_TEST_SUBDOMAIN_VOLUME_H

#include "include/filterFunction.h"
#include "include/treeNode.h"
#include "include/distTree.h"

namespace test
{
  // return: pair of {number of OUT octants, number of INTERCEPTED octants}

  template <int dim, bool pre_tested = false>

  std::pair<long long unsigned, long long unsigned>
  count_subdomain_octants(
      ot::TreeNode<unsigned, dim> subtree_root,
      int ending_level,
      ibm::DomainDecider filter,
      ibm::Partition pre_test = ibm::INTERCEPTED)  //ignored unless pre_tested==true
  {
    assert(subtree_root.getLevel() <= ending_level);
    ibm::Partition filter_result = pre_tested ? pre_test : ot::decide_octant(subtree_root, filter);
    if (filter_result == ibm::Partition::IN)
      return {};
    if (subtree_root.getLevel() == ending_level)
      return {filter_result == ibm::OUT, filter_result == ibm::INTERCEPTED};

    // Morton traversal
    std::pair<long long unsigned, long long unsigned> summation = {};
    std::pair<long long unsigned, long long unsigned> child_values;
    for (int child_id = 0; child_id < (1u << dim); ++child_id)
    {
      const ot::TreeNode<unsigned, dim> child = subtree_root.getChildMorton(child_id);
      filter_result = ot::decide_octant(child, filter);
      switch (filter_result)
      {
        case ibm::Partition::OUT:
          summation.first += 1u << (dim * (ending_level - child.getLevel()));
          break;
        case ibm::Partition::IN:
          break;
        default:
          child_values = count_subdomain_octants<dim, true>(child, ending_level, filter, filter_result);
          summation.first += child_values.first;
          summation.second += child_values.second;
      }
    }

    return summation;
  }
}

#endif//DENDRO_KT_TEST_SUBDOMAIN_VOLUME_H
