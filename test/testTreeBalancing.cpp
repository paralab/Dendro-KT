/*
 * testTreeBalancing.cpp
 *   Test creation of balancing nodes and subsequent complete balanced tree.
 *
 * Masado Ishii  --  UofU SoC, 2019-01-10
 */


#include "treeNode.h"
#include "tsort.h"
#include "octUtils.h"

#include "hcurvedata.h"

#include "octUtils.h"
#include <vector>

#include <assert.h>
#include <mpi.h>

//------------------------
// test_propagateNeighbours()
//------------------------
void test_propagateNeighbours(int numPoints)
{
  using T = unsigned int;
  const unsigned int dim = 2; ///4;
  const unsigned int numChildren = 1u << dim;
  using TreeNode = ot::TreeNode<T,dim>;

  _InitializeHcurve(dim);

  std::vector<TreeNode> points = ot::getPts<T,dim>(numPoints);

  const unsigned int maxPtsPerRegion = 8;
  /// const unsigned int maxPtsPerRegion = 3;

  const T leafLevel = m_uiMaxDepth;

  std::cout << "----------------------------------------\n";
  std::cout << "BEFORE: Source points.\n";
  std::cout << "----------------------------------------\n";

  for (const TreeNode &tn : points)
    std::cout << tn.getBase32Hex().data() << "\n";

  std::cout << "\n";
  std::cout << "\n";

  // Perform the function propagateNeighbours().
  ot::SFC_Tree<T,dim>::propagateNeighbours(points);

  /// std::cout << "----------------------------------------\n";
  /// std::cout << "AFTER: With added auxiliary nodes.\n";
  /// std::cout << "----------------------------------------\n";

  /// for (const TreeNode &tn : points)
  ///   std::cout << tn.getBase32Hex().data() << "\n";

  /// std::cout << "\n";
  /// std::cout << "\n";
}




int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int ptsPerProc = 200;
  if (argc > 1)
    ptsPerProc = strtol(argv[1], NULL, 0);

  test_propagateNeighbours(ptsPerProc);

  MPI_Finalize();

  return 0;
}


