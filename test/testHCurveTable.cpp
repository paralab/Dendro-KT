/*
 * testHCurveTable.cpp
 *   Test properties of the SFC defined by the generated HCurve data.
 *
 * Masado Ishii  --  UofU SoC, 2019-01-18
 */

#include "treeNode.h"
#include "hcurvedata.h"
/// #include "octUtils.h"

#include <iostream>
#include <stdio.h>


//
// depthFirstTraversal()
//
template <unsigned int K, typename T = unsigned int, bool printData = false>
void depthFirstTraversal(std::vector<ot::TreeNode<T,K>> &ordering,
                         unsigned int eLev,
                         ot::TreeNode<T,K> parent = ot::TreeNode<T,K>{},
                         unsigned int pRot = 0)
{
  const int numChildren = ot::TreeNode<T,K>::numChildren;

  if (parent.getLevel() > eLev)
    return;

  if (parent.getLevel() == eLev)  // 'Leaf' of the traversal.
  {
    if (printData)
      std::cout << "\t\t\t" << parent.getBase32Hex().data() << "\n";
    ordering.push_back(parent);
  }

  else
  {
    if (printData)
      std::cout << "(intrnl) " << parent.getBase32Hex().data() << "\n";
    for (unsigned char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      unsigned char child = rotations[pRot * 2*numChildren + child_sfc];
      unsigned int cRot = HILBERT_TABLE[pRot * numChildren + child];
      ot::TreeNode<T,K> cNode = parent.getChildMorton(child);
      depthFirstTraversal<K,T>(ordering, eLev, cNode, cRot);
    }
  }
}

//
// test_depthFirstTraversal()
//
template <unsigned int K, bool printData = false>
void test_depthFirstTraversal(unsigned int eLev)
{
  _InitializeHcurve(K);

  using T = unsigned int;
  std::vector<ot::TreeNode<T,K>> ordering;
  depthFirstTraversal<K,T,printData>(ordering, eLev);

  const unsigned int dist = 1u << (m_uiMaxDepth - eLev);

  // Check spatial adjacency.
  typename std::vector<ot::TreeNode<T,K>>::const_iterator prev, cur;
  prev = ordering.begin();
  cur = prev+1;

  bool success = true;
  while (cur < ordering.end())
  {
    // Check each dimension coordinate one by one.
    bool foundMvDim = false;
    for (int d = 0; d < K; d++)
    {
      unsigned int cx = cur->getX(d), px = prev->getX(d);
      unsigned diff = (cx > px ? cx - px : px - cx);
      if (diff == dist && foundMvDim)
        success = false;
      else if (diff == dist)
        foundMvDim = true;
      else if (diff > dist)
        success = false;
    }
    if (!success || !foundMvDim)
    {
      success = false;
      break;
    }

    prev++;
    cur++;
  }
  if (success)
  {
    std::cout << "Adjacency test succeeded.\n";
  }
  else
  {
    std::cout << "Adjacency test failed with elements "
              << (prev - ordering.begin())
              << " - "
              << (cur - ordering.begin())
              << "\n";
    std::cout << prev->getBase32Hex().data() << " ... ";
    std::cout << cur->getBase32Hex().data() << "\n";
  }

  _DestroyHcurve();
}



//
// printTableData()
//
template <unsigned int K>
void printTableData()
{
  _InitializeHcurve(K);

  const unsigned int numChildren = 1u << K;

  //
  // Output the rotation tables.
  //
  int counter = 0;
  while (counter < _KD_ROTATIONS_SIZE(K))
  {
    for (int ch = 0; ch < numChildren; ch++, counter++)
      std::cout << (int) rotations[counter] << ",";
    std::cout << "  ";
    for (int ch = 0; ch < numChildren; ch++, counter++)
      std::cout << (int) rotations[counter] << ",";
    std::cout << "\n";
  }

  _DestroyHcurve();
}


//
// main()
//
int main(int argc, char *argv[])
{
  /* Do MPI stuff here if needed. */

  /// printTableData<2>();  std::cout << "\n\n";
  /// printTableData<3>();  std::cout << "\n\n";
  /// printTableData<4>();  std::cout << "\n\n";

  test_depthFirstTraversal<2>(8);  std::cout << "\n";
  test_depthFirstTraversal<3>(7);  std::cout << "\n";
  test_depthFirstTraversal<4>(4);  std::cout << "\n";

  return 0;
}
