
#include "eleTreeIterator.h"
#include "hcurvedata.h"

#include <stdio.h>


bool testRandomPoints();


int main(int argc, char *argv[])
{
  bool success = testRandomPoints();
  std::cout << "Result: " << (success ? "success" : "failure") << "\n";
  return !success;
}


/**
 * testRandomPoints()
 *
 * Create points, generate neighboring cells, check that traversal includes
 * all the neighboring cells.
 *
 */
bool testRandomPoints()
{
  const int numPoints = 100;
  const unsigned int depthMin = 3;
  const unsigned int depthMax = 6;

  m_uiMaxDepth = depthMax;

  constexpr unsigned int dim = 4;
  using C = unsigned int;
  using T = float;

  _InitializeHcurve(dim);

  // Test:
  // while ./tstEleTreeIterator > /dev/null ; do echo ; done

  // Pseudo-random number generators for point coordinates.
  std::random_device rd;
  unsigned int seed = rd();
  /// unsigned int seed = 2716830963;  // Seeds that used to cause seg fault.
  /// unsigned int seed = 3163620652;
  /// unsigned int seed = 3230132189;
  /// unsigned int seed = 1115998293;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<C> coordDis(0, 1u << m_uiMaxDepth);
  std::uniform_int_distribution<unsigned int> levDis(depthMin, depthMax);

  std::cerr << "Seed: " << seed << "\n";

  constexpr unsigned int numNeighbours = (1u << dim);

  std::vector<T> startVals(numPoints, 1.0);
  std::vector<T> expectedVals(numPoints, 0.0);

  // Fill sample nodes and their neighbors.
  std::vector<ot::TreeNode<C,dim>> nodeCoords;
  std::vector<ot::TreeNode<C,dim>> nodeNeighbours;
  for (int ii = 0; ii < numPoints; ii++)
  {
    const unsigned int lev = levDis(gen);
    const C mask = (1u << (m_uiMaxDepth + 1)) - (1u << (m_uiMaxDepth - lev));
    std::array<C,dim> coords;
    for (int d = 0; d < dim; d++)
      coords[d] = coordDis(gen) & mask;

    nodeCoords.emplace_back(1, coords, lev);

    std::array<C,dim> ncoords;
    for (int n = 0; n < numNeighbours; n++)
    {
      bool insideDomain = true;
      for (int d = 0; d < dim; d++)
      {
        ncoords[d] = ( (n & (1u<<d)) ? coords[d] : coords[d] - (1u<<(m_uiMaxDepth-lev)) );
        insideDomain &= (ncoords[d] < (1u << m_uiMaxDepth));
      }

      if (insideDomain)
      {
        nodeNeighbours.emplace_back(1, ncoords, lev);
        expectedVals[ii] += 1.0;
      }
    }
  }

  /// std::cout << "nodeCoords.size()==" << nodeCoords.size() << "\n";
  /// std::cout << "nodeCoords: ";
  /// for (const auto &x : nodeCoords)
  /// {
  ///   std::cout << "{(" << x.getLevel() << ")";
  ///   for (int d = 0; d < dim; d++)
  ///     std::cout << " " << x.getX(d);
  ///   std::cout << "} ";
  /// }
  /// std::cout << "\n";

  /// std::cout << "(before sort) nodeNeighbours.size()==" << nodeNeighbours.size() << "\n";
  /// std::cout << "(before sort) nodeNeighbours: ";
  /// for (const auto &x : nodeNeighbours)
  /// {
  ///   std::cout << "{(" << x.getLevel() << ")";
  ///   for (int d = 0; d < dim; d++)
  ///     std::cout << " " << x.getX(d);
  ///   std::cout << "} ";
  /// }
  /// std::cout << "\n";


  // Convert list of neighbor cells into list of unique cells.
  ot::SFC_Tree<C,dim>::locTreeSort(
      &(*nodeNeighbours.begin()),
      0,
      (ot::RankI) nodeNeighbours.size(),
      1,
      m_uiMaxDepth,
      0);

  ot::SFC_Tree<C,dim>::locRemoveDuplicates(nodeNeighbours);

  /// std::cout << "nodeNeighbours.size()==" << nodeNeighbours.size() << "\n";
  /// std::cout << "nodeNeighbours: ";
  /// for (const auto &x : nodeNeighbours)
  /// {
  ///   std::cout << "{(" << x.getLevel() << ")";
  ///   for (int d = 0; d < dim; d++)
  ///     std::cout << " " << x.getX(d);
  ///   std::cout << "}\n";
  /// }
  /// std::cout << "\n";

  // Verify that each cell is visited (in order) as a leaf in matvec-style traversal.
  ElementLoop<C, dim, T> loop(
      numPoints,
      &(*nodeCoords.begin()),
      1,
      nodeNeighbours.front(),
      nodeNeighbours.back());
  loop.initialize(&(*startVals.begin()));

  auto neighbourIt = nodeNeighbours.begin();

  int loopCount = 0;
  /// std::cout << "Loop:\n";
  for (ELIterator<C, dim, T> it = loop.begin(); it != loop.end(); ++it)
  {
    /// auto tn = it.getElemTreeNode();
    /// std::cout << "{(" << tn.getLevel() << ")";
    /// for (int d = 0; d < dim; d++)
    ///   std::cout << " " << tn.getX(d);
    /// std::cout << "}\n";

    // Use the original 1's (duplicated for each neighbour) in the summation.
    (*it).submitElement();

    if (*neighbourIt == it.getElemTreeNode())
      ++neighbourIt;

    loopCount++;
  }
  loop.finalize(&(*startVals.begin()));
  std::cout << "\n";

  std::cout << "loopCount==" << loopCount << "\n";

  int nonmatchingNodes = 0;
  for (int nIdx = 0; nIdx < numPoints; nIdx++)
  {
    /// std::cout << "startVals[" << nIdx << "]==" << startVals[nIdx] << " \t"
    ///           << "expectedVals[" << nIdx << "]==" << expectedVals[nIdx] << " \n";
    if (!(startVals[nIdx] == expectedVals[nIdx]))
      nonmatchingNodes++;
  }
  std::cout << "nonmatchingNodes==" << nonmatchingNodes << "/" << numPoints << "\n";

  return (neighbourIt == nodeNeighbours.end() && nonmatchingNodes == 0);

  _DestroyHcurve();
}
