
#include "eleTreeIterator.h"
#include "hcurvedata.h"

#include <stdio.h>


bool testRandomPoints();

bool testInterpolation();

bool testNull();


/**
 * main()
 */
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  /// bool success = testRandomPoints();  // No longer works because original test did not account for interpolation.
  /// bool success = testInterpolation();
  bool success = testNull();
  std::cout << "Result: " << (success ? "success" : "failure") << "\n";

  MPI_Finalize();

  return !success;
}


/**
 * testNull()
 *
 * What happens when either the tree or the node list is empty?
 */
bool testNull()
{
  constexpr unsigned int dim = 2;
  using C = unsigned int;
  using T = float;

  const unsigned int eleOrder = 1;

  _InitializeHcurve(dim);

  const ot::TreeNode<C, dim> treeRoot;
  std::vector<ot::TreeNode<C, dim>> nonemptyTree;
  nonemptyTree.push_back(treeRoot.getChildMorton(0));
  nonemptyTree.push_back(treeRoot.getChildMorton(1));
  nonemptyTree.push_back(treeRoot.getChildMorton(2));
  nonemptyTree.push_back(treeRoot.getChildMorton(3));

  const unsigned int lev = 1;
  const C u = (1u << m_uiMaxDepth - lev);

  std::vector<ot::TreeNode<C, dim>> nonemptyNodeList;
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{0*u, 0*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{1*u, 0*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{2*u, 0*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{0*u, 1*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{1*u, 1*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{2*u, 1*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{0*u, 2*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{1*u, 2*u}, lev);
  nonemptyNodeList.emplace_back(0, std::array<C,dim>{2*u, 2*u}, lev);

  std::vector<ot::TreeNode<C, dim>> emptyTree;
  std::vector<ot::TreeNode<C, dim>> emptyNodeList;

  std::vector<T> nonemptyBuffer(nonemptyNodeList.size(), 0.0);
  std::vector<T> emptyBuffer;

  /// for (int caseIdx = 0; caseIdx < 4; caseIdx++)
  for (int caseIdx = 3; caseIdx >= 0; caseIdx--)
  {
    fprintf(stderr, "Testing %s tree, %s nodeList.\n",
        (caseIdx % 2 ? "nonempty" : "empty"),
        (caseIdx / 2 ? "nonempty" : "empty"));

    std::vector<ot::TreeNode<C, dim>> & tree = (caseIdx % 2 ? nonemptyTree : emptyTree);
    std::vector<ot::TreeNode<C, dim>> & nodeList = (caseIdx / 2 ? nonemptyNodeList : emptyNodeList);
    std::vector<T> & buffer = (caseIdx / 2 ? nonemptyBuffer : emptyBuffer);

    ElementLoop<C, dim, T> loop( nodeList.size(), &(*nodeList.begin()), eleOrder, tree.front(), tree.back());
    loop.initialize(&(*nonemptyBuffer.begin()));

    for (ELIterator<C, dim, T> it = loop.begin(); it != loop.end(); ++it)
    {
      ElementNodeBuffer<C, dim, T> leafBuf = *it;
      leafBuf.submitElement();
    }
    loop.finalize(&(*nonemptyBuffer.begin()));
  }



  _DestroyHcurve();

  return true;
}


/**
 * testInterpolation()
 *
 * Construct a sequence of very simple trees, each showcasing
 * a different number or type of hanging nodes.
 */
bool testInterpolation()
{
  constexpr unsigned int dim = 4;
  using C = unsigned int;
  using T = float;

  const unsigned int eleOrder = 3;

  m_uiMaxDepth = 20;

  _InitializeHcurve(dim);

  const unsigned int NumChildren = (1u << dim);

  // +----.----+----.----+
  // |         |         |
  // |         |         |
  // .    2    .    3    .
  // |         |         |
  // |         |         |
  // +----+----+----.----+
  // |    |HHHH|         |
  // |    |HHHH|         |
  // +----+----+    1    .
  // |    |    |         |
  // |    |    |         |
  // +----+----+----.----+

  // We can get different numbers of hanging nodes on the element marked 'H',
  // by progressively subdividing elements 1..(2^dim - 1).

  const ot::TreeNode<C, dim> treeRoot;
  int numCasesFailed = 0;

  for (int numSubdNeighbours = 0; numSubdNeighbours < NumChildren; numSubdNeighbours++)
  {
    std::cout << "Start case " << numSubdNeighbours << "/" << NumChildren-1 << "\n";

    std::vector<ot::TreeNode<C, dim>> tree;

    // The first sublist of children will be subdivided down to l2.
    for (unsigned int child_l1 = 0; child_l1 <= numSubdNeighbours; child_l1++)
    {
      const ot::TreeNode<C, dim> tn_l1 = treeRoot.getChildMorton(child_l1);
      for (unsigned int child_l2 = 0; child_l2 < NumChildren; child_l2++)
        tree.push_back(tn_l1.getChildMorton(child_l2));
    }

    // The latter sublist of children will not be subdivided; remain at l1.
    for (unsigned int child_l1 = numSubdNeighbours + 1; child_l1 < NumChildren; child_l1++)
      tree.push_back(treeRoot.getChildMorton(child_l1));

    /// //DEBUG
    /// fprintf(stdout, "Tree (%u):\n", tree.size());
    /// for (ot::RankI tIdx = 0; tIdx < tree.size(); tIdx++)
    /// {
    ///   fprintf(stdout, "tree[%03u] == %s(%lu)\n",
    ///       tIdx,
    ///       tree[tIdx].getBase32Hex(2).data(),
    ///       tree[tIdx].getLevel());
    /// }
    /// fprintf(stdout, "\n");

    // Generate all nodes.
    std::vector<ot::TreeNode<C, dim>> nodeCoords;
    {
      std::vector<ot::TNPoint<C, dim>> nodeList;
      for (const ot::TreeNode<C, dim> & treeNode : tree)
        ot::Element<C, dim>(treeNode).template appendNodes<ot::TNPoint<C, dim>>(eleOrder, nodeList);

      // Make unique, remove hanging nodes.
      ot::RankI numUniqueNodes = ot::SFC_NodeSort<C, dim>::dist_countCGNodes(
          nodeList, eleOrder, &tree.front(), &tree.back(), MPI_COMM_WORLD);

      // Convert vector of TNPoint to vector of TreeNode.
      nodeCoords = std::vector<ot::TreeNode<C, dim>>(nodeList.begin(), nodeList.end());
    }
    ot::RankI numUniqueNodes = nodeCoords.size();

    /// //DEBUG
    /// fprintf(stdout, "Node list (%lu):\n", numUniqueNodes);
    /// for (ot::RankI nIdx = 0; nIdx < numUniqueNodes; nIdx++)
    /// {
    ///   fprintf(stdout, "nodeCoords[%03u] == %s(%lu)\n",
    ///       nIdx,
    ///       nodeCoords[nIdx].getBase32Hex(5).data(),
    ///       nodeCoords[nIdx].getLevel());
    /// }
    /// fprintf(stdout, "\n");

    // The test: Evaluate a function that is linear in all node coordinates.
    // For each hanging node, the value will be polynomially interpolated
    // from parent nodes, so it should agree with the known linear function.

    // This function is linear in each component of a nodal coordinate.
    auto myF = [dim](const double *nodeCoord) {
      T coeff = 5.0;
      T v = 0.0;
      for (int d = 0; d < dim; d++)
      {
        v += nodeCoord[d] * coeff;
        coeff = (coeff - 1.3) * (d+3);
      }
      return v;
    };

    // Evaluate myF over all the non-hanging nodes.
    std::vector<T> nodeVals(numUniqueNodes, 0.0);
    for (unsigned int nIdx = 0; nIdx < numUniqueNodes; nIdx++)
    {
      double c[dim];
      for (int d = 0; d < dim; d++)
        c[d] = 1.0 / (1u << m_uiMaxDepth) * nodeCoords[nIdx].getX(d);
      nodeVals[nIdx] = myF(c);
    }

    // Loop over all elements and test the interpolated function values.
    ElementLoop<C, dim, T> loop(
        numUniqueNodes,
        &(*nodeCoords.begin()),
        eleOrder,
        tree.front(),
        tree.back());
    loop.initialize(&(*nodeVals.begin()));

    int loopCount = 0;
    T maxDiff = 0.0;
    for (ELIterator<C, dim, T> it = loop.begin(); it != loop.end(); ++it)
    {
      ElementNodeBuffer<C, dim, T> leafBuf = *it;
      const unsigned int npe = leafBuf.getNodesPerElement();

      for (unsigned int n = 0; n < npe; n++)  // All nodes of an element.
      {
        const T observedValue = leafBuf.getNodeBuffer()[n];
        const T desiredValue = myF(leafBuf.getNodeCoords() + dim*n);
        /// fprintf(stdout, "observedValue (%02u.%02u):\t %f\n", loopCount, n, observedValue);
        T diff = observedValue - desiredValue;
        maxDiff = fmax(maxDiff, fabs(diff));
      }

      leafBuf.submitElement();

      loopCount++;
    }
    loop.finalize(&(*nodeVals.begin()));

    bool caseSuccess = maxDiff < 1e-3;

    std::cout << "Result of case " << numSubdNeighbours << ": \t"
              << (caseSuccess ? "Succeeded" : "Failed")
              << "\n";

    if (!caseSuccess)
      numCasesFailed++;
  }

  _DestroyHcurve();

  std::cout << "Interpolation cases failed: " << numCasesFailed << "/" << NumChildren << "\n";

  return (numCasesFailed == 0);
}


/**
 * testRandomPoints() - No longer works because original test did not account for interpolation.
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
