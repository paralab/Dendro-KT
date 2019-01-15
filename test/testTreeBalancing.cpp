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


// ...........................................................................
template <typename T, unsigned int D>
bool checkLocalCompleteness(std::vector<ot::TreeNode<T,D>> &points,
                            std::vector<ot::TreeNode<T,D>> &tree,
                            bool entireTree,
                            bool printData);

template <typename T, unsigned int D>
bool checkBalancingConstraint(const std::vector<ot::TreeNode<T,D>> &tree, bool printData);
// ...........................................................................



//------------------------
// test_propagateNeighbours()
//------------------------
void test_propagateNeighbours(int numPoints)
{
  using T = unsigned int;
  const unsigned int dim = 4;
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

  // These lines are commented out because there are too many points to print.
  /// std::cout << "----------------------------------------\n";
  /// std::cout << "AFTER: With added auxiliary nodes.\n";
  /// std::cout << "----------------------------------------\n";

  /// for (const TreeNode &tn : points)
  ///   std::cout << tn.getBase32Hex().data() << "\n";

  /// std::cout << "\n";
  /// std::cout << "\n";
}


//------------------------
// test_locTreeBalancing()
//------------------------
template <unsigned int dim>
void test_locTreeBalancing(int numPoints)
{
  using T = unsigned int;
  /// const unsigned int dim = 4;
  const unsigned int numChildren = 1u << dim;
  using TreeNode = ot::TreeNode<T,dim>;

  _InitializeHcurve(dim);

  std::vector<TreeNode> points = ot::getPts<T,dim>(numPoints);
  std::vector<TreeNode> tree;

  const unsigned int maxPtsPerRegion = 8;

  const T leafLevel = m_uiMaxDepth;

  ot::SFC_Tree<T,dim>::locTreeBalancing(points, tree, maxPtsPerRegion);

  checkLocalCompleteness<T,dim>(points, tree, true, true);

  bool balanceSuccess = checkBalancingConstraint(tree, true);
  std::cout << "Balancing constraint " << (balanceSuccess ? "succeeded" : "FAILED") << "\n";

  // Make sure there is something there.
  std::cout << "Final.... points.size() == " << points.size() << "  tree.size() == " << tree.size() << "\n";

  ///std::cout << "Points:      |-----\n";
  ///for (const TreeNode &tn : points)
  ///  std::cout << tn.getBase32Hex().data() << "\n";
  ///std::cout << "        -----|\n";

  ///std::cout << "Tree:      |-----\n";
  ///for (const TreeNode &tn : tree)
  ///  std::cout << tn.getBase32Hex().data() << "\n";
  ///std::cout << "      -----|\n";
}


//------------------------
// test_distTreeConstruction()
//
// Notes:
//   - For this test we expect the local and global adjacency criteria to succeed.
//   - However we expect the points-in-buckets criterion to probably fail, since
//     buckets are redistributed again after points are partitioned.
//------------------------
void test_distTreeBalancing(int numPoints, MPI_Comm comm = MPI_COMM_WORLD)
{
  int nProc, rProc;
  MPI_Comm_size(comm, &nProc);
  MPI_Comm_rank(comm, &rProc);

  using T = unsigned int;
  const unsigned int dim = 4;
  const unsigned int numChildren = 1u << dim;
  using TreeNode = ot::TreeNode<T,dim>;

  _InitializeHcurve(dim);

  std::vector<TreeNode> points = ot::getPts<T,dim>(numPoints);
  std::vector<TreeNode> treePart;

  const unsigned int maxPtsPerRegion = 32;

  const double loadFlexibility = 0.2;

  const T leafLevel = m_uiMaxDepth;
  const T firstVariableLevel = 1;      // Not sure about this whole root thing...

  std::cerr << "Starting distTreeBalancing()...\n";
  ot::SFC_Tree<T,dim>::distTreeBalancing(points, treePart, maxPtsPerRegion, loadFlexibility, comm);
  std::cerr << "Finished distTreeBalancing().\n\n";

  // Local adjacency test. Ignore messages about unaccounted points.
  int myLocAdjacency = checkLocalCompleteness<T,dim>(points, treePart, false, false);

  const bool printGlobData = true;

  int myGlobAdjacency = true;

  // Exchange left to right to test adjacency.
  TreeNode prevEnd;
  MPI_Request request;
  MPI_Status status;
  if (rProc < nProc-1)
    par::Mpi_Isend<TreeNode>(&treePart.back(), 1, rProc+1, 0, comm, &request);
  if (rProc > 0)
    par::Mpi_Recv<TreeNode>(&prevEnd, 1, rProc-1, 0, comm, &status);
  
  // Completeness at boundaries.
  if (rProc == 0)
  {
    const TreeNode &tn = treePart.front();
    for (int l = firstVariableLevel; l <= tn.getLevel(); l++)
      if (tn.getMortonIndex(l) != 0)
      {
        myGlobAdjacency = false;
        if (printGlobData)
          std::cout << "Global completeness failed, bdry start (rank " << rProc << ")\n";
      }
  }
  if (rProc == nProc - 1)
  {
    const TreeNode &tn = treePart.back();
    for (int l = firstVariableLevel; l <= tn.getLevel(); l++)   // < not <=, since level should be 0?
      if (tn.getMortonIndex(l) != numChildren - 1)
      {
        myGlobAdjacency = false;
        if (printGlobData)
          std::cout << "Global completeness failed, bdry end (rank " << rProc << ")"
                    << "  (index[" << l << "] == " << (int) tn.getMortonIndex(l) << ")\n";
      }
  }

  // Inter-processor adjacency.
  //TODO there is probably a clever bitwise Morton-adjacency test. make a class member function.
  if (rProc > 0)
  {
    // Verify that our beginning is adjacent to previous end.
    const TreeNode &myFront = treePart.front();
    int l_match = 0;
    while (l_match <= m_uiMaxDepth && myFront.getMortonIndex(l_match) == prevEnd.getMortonIndex(l_match))
      l_match++;

    if (myFront.getMortonIndex(l_match) != 1 + prevEnd.getMortonIndex(l_match))
    {
      myGlobAdjacency = false;
      if (printGlobData)
        std::cout << "Global completeness failed, digit increment (rank " << rProc << ")\n";
    }

    for (int l = l_match + 1; l <= myFront.getLevel(); l++)
      if (myFront.getMortonIndex(l) != 0)
      {
        myGlobAdjacency = false;
        if (printGlobData)
          std::cout << "Global completeness failed, local front nonzero (rank " << rProc << ")\n";
      }

    for (int l = l_match + 1; l <= prevEnd.getLevel(); l++)
      if (prevEnd.getMortonIndex(l) != numChildren - 1)
      {
        myGlobAdjacency = false;
        if (printGlobData)
          std::cout << "Global completeness failed, prev back nonfull (rank " << rProc << ")"
                    << "  (" << prevEnd.getBase32Hex().data() << ":" << myFront.getBase32Hex().data() << ")\n";
      }
  }

  if (rProc < nProc-1)
    MPI_Wait(&request, &status);

  int recvLocAdjacency, recvGlobAdjacency;

  MPI_Reduce(&myLocAdjacency, &recvLocAdjacency, 1, MPI_INT, MPI_LAND, 0, comm);
  MPI_Reduce(&myGlobAdjacency, &recvGlobAdjacency, 1, MPI_INT, MPI_LAND, 0, comm);
  MPI_Barrier(comm);
  if (rProc == 0)
  {
    std::cout << "\n\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << "Local adjacencies " << (recvLocAdjacency ? "succeeded" : "FAILED") << "\n";
    std::cout << "Global adjacency " << (recvGlobAdjacency ? "succeeded" : "FAILED") << "\n";
  }

  //TODO checkBalancingConstraint(tree);

  // Make sure there is something there.
  std::cout << "Final.... (rank " << rProc << ") points.size() == " << points.size() << "  tree.size() == " << treePart.size() << "\n";
}



// ----------------------
// checkLocalCompleteness
// ----------------------
template <typename T, unsigned int D>
bool checkLocalCompleteness(std::vector<ot::TreeNode<T,D>> &points,
                            std::vector<ot::TreeNode<T,D>> &tree,
                            bool entireTree,
                            bool printData)
{
  const unsigned int dim = D;
  const unsigned int numChildren = 1u << dim;
  using TreeNode = ot::TreeNode<T,dim>;

  if (printData)
  {
    for (TreeNode pt : points)
    {
      std::cout << pt.getBase32Hex().data() << "\n";
    }
    std::cout << "\n";
  }

  std::vector<int> address(m_uiMaxDepth+1, 0);
  bool completeness = true;

  if (!entireTree)
  {
    // Set address to match the first TreeNode.
    const TreeNode &front = tree.front();
    for (int l = 0; l <= front.getLevel(); l++)
      address[l] = front.getMortonIndex(l);
  }

  int lev = 0, prevLev = 0;
  typename std::vector<TreeNode>::const_iterator pIt = points.begin();
  const char continueStr[] = "  ";
  const char expandStr[]   = " [_]";
  const char newBlockStr[] = "__";
  const int beginTextPos = 40;
  std::streamsize oldWidth = std::cout.width();
  if (printData)
  {
    std::cout << "    (Buckets)                           (Points)\n";
  }
  for (const TreeNode tn : tree)
  {
    prevLev = lev;
    lev = tn.getLevel();

    // Check completeness.
    // `address' must match address at this node.
    for (int l = 0; l <= lev; l++)
    {
      if (address[l] != tn.getMortonIndex(l))
      {
        completeness = false;

        if (printData)
        {
          std::cout << "Completeness failure here. Level [" << l << "/" << lev <<"], counter == "
            << address[l] << ", but node address == " << (int) tn.getMortonIndex(l) << "  (" << tn.getBase32Hex().data() << ")\n";
        }
      }
      ///else if (printData)
      ///{
      ///  std::cout << "Completeness not violated here. level=["<<l<<"/"<<lev<<"] (" << tn.getBase32Hex().data() << ")\n";
      ///}
    }
    // Now that we've visited this node, add 1 at this level.
    // Remember that addition propagates.
    for (int l = lev; l >= 0 && ++address[l] == numChildren; address[l] = 0, l--);

    // Print buckets.
    if (printData)
    {
      if (lev != prevLev)
      {
        for (int ii = 0; ii < lev; ii++)
          std::cout << continueStr;
        std::cout << "\n";
      }

      if (tn.getMortonIndex() == 0)
      {
        for (int ii = 0; ii < lev; ii++)
          std::cout << newBlockStr;
        std::cout << "\n";
      }

      for (int ii = 0; ii < lev; ii++)
        std::cout << continueStr;
      std::cout << expandStr << "    " << tn.getBase32Hex().data() << "\n";
    }

    // Print points.
    while (tn.isAncestor(pIt->getDFD()) || tn == pIt->getDFD())
    {
      if (printData)
      {
        for (int ii = 0; ii < lev; ii++)
          std::cout << continueStr;
        
        std::cout << std::setw(beginTextPos - 2*lev) << ' '
                  << std::setw(oldWidth) << pIt->getBase32Hex().data()
                  << "\n";
      }

      pIt++;
    }
  }

  // Final check on completeness.
  if (entireTree)
  {
    for (int lev = 1; lev < address.size(); lev++)   // See note at calling of locTreeConstruction().
      if (address[lev] != 0)
      {
        completeness = false;
        if (printData)
        {
          std::cout << "Completeness failure here. Previous count [" << lev << "] == " << address[lev] << "\n";
        }
      }
  }

  if (completeness)
    std::cout << "Completeness criterion successful, all addresses adjacent.\n";
  else
    std::cout << "Completeness criterion FAILED.\n";

  if (pIt == points.end())
    std::cout << "Counting points: All points accounted for.\n";
  else
    std::cout << "Counting points: FAILED - SOME POINTS WERE NOT LISTED.\n";

  return completeness;

}
//------------------------


template <typename TN>
struct BalanceSearch
{
  std::vector<int> ancCountStack;
  bool success;
  bool printData;

  BalanceSearch() { ancCountStack.resize(m_uiMaxDepth+2, 0); success = true; printData = false; }

  // tree1 is the source tree.
  // tree2 is the neighbor list derived from the source tree.
  bool operator() (const TN *tree1, const std::array<ot::RankI, TN::numChildren+2> &split1,
                   const TN *tree2, const std::array<ot::RankI, TN::numChildren+2> &split2,
                   ot::LevI level)
  {
    if (!success)
      return false;

    if (split1[1] - split1[0] > 0)  // Leaf of the source tree as an ancestor.
    {
      // Since the source tree only stores leaves, the other buckets should be empty.
      // Also we assume that the source tree contains no duplicates.
      ot::LevI sourceLeafLevel = tree1[split1[0]].getLevel();
      assert(sourceLeafLevel == level-1);

      if (printData)
      {
        std::cout << "'Downward-facing' branch.\n";
        std::cout << "sourceLeaf:  " << tree1[split1[0]].getBase32Hex().data() << "\n";
        std::cout << "descendant hypothetical neighbors:\n";
      }

      // Downward-facing component of the balancing criterion:
      //   An existing neighbor should be no higher than 1 level above a hypothetical neighbor.
      for (const TN *t2It = tree2 + split2[0]; t2It < tree2 + split2[TN::numChildren+1]; t2It++)
      {
        if (printData)
          std::cout << "\t" << t2It->getBase32Hex().data() << "\n";

        if (t2It->getLevel() - sourceLeafLevel > 1)
        {
          return success = false;
        }
      }
    }
    else
    {
      // Upward-facing component of the balancing criterion:
      //   An existing neighbor should be beneath no more than 1 ancestor (parent).
      ancCountStack[level] = ancCountStack[level-1] + (split2[1] - split2[0]);
      if (printData)
      {
        std::cout << "'Upward-facing' branch.\n";
        std::cout << "ancCount[" << level << "] == " << ancCountStack[level] << "\n";
      }
      if (ancCountStack[level] > 1)
      {
        if (printData)
        {
          std::cout << "Ancestor hypothetical neighbor:  "
              << tree2[split2[1]-1].getBase32Hex().data() << "\n";
          std::cout << "Existing descendants:\n";
          for (const TN *t1It = tree1 + split1[1]; t1It < tree1 + split1[TN::numChildren+1]; t1It++)
            std::cout << t1It->getBase32Hex().data() << "\n";
        }
        return success = false;
      }
    }

    return true;
  }
};


//
// checkBalancingConstraint()
//
// Notes:
//   - Assumes that TreeNode::appendAllNeighbours() works.
//
template <typename T, unsigned int D>
bool checkBalancingConstraint(const std::vector<ot::TreeNode<T,D>> &tree, bool printData)
{
  std::vector<ot::TreeNode<T,D>> nList;
  for (const ot::TreeNode<T,D> &tn : tree)
    tn.appendAllNeighbours(nList);

  ot::SFC_Tree<T,D>::locTreeSort(&(*nList.begin()),
                                 0, (unsigned int) nList.size(),
                                 1, m_uiMaxDepth,
                                 0);
  ot::SFC_Tree<T,D>::locRemoveDuplicatesStrict(nList);

  if (printData)
  {
    std::cout << "Begin tree:\n";
    for (const ot::TreeNode<T,D> &tn : tree)
      std::cout << tn.getBase32Hex().data() << "\n";
    std::cout << "End tree.\n\n";

    std::cout << "Begin neighbors:\n";
    for (const ot::TreeNode<T,D> &tn : nList)
      std::cout << tn.getBase32Hex().data() << "\n";
    std::cout << "End neighbors.\n\n";
  }

  // Now nList is a list of unique neighbors (different levels considered distinct).
  // Since it is sorted in the same order as tree (presumably tree was sorted),
  // searching can be accomplished in a single pass comparison (almost).
  // 
  // nList represents neighbors which may or may not exist in the tree.
  // (The completess property would ensure that an ancestor or descendant of
  // every element of nList appears in tree, but we can test the balancing
  // constraint without assuming completeness.) The balancing constraint succeeds
  // iff, for every node in tree, any possible ancestor or descendant of that node
  // that does appear in nList differs in level by no more than 1.
  //
  // As a consequence, if the balancing criterion is met, then each node in tree
  // will have no more than three levels of matches (parent, self, and/or children) in nList.

  BalanceSearch<ot::TreeNode<T,D>> inspector;
  inspector.printData = printData;
  ot::SFC_Tree<T,D>::dualTraversal(&(*tree.begin()), 0, (ot::RankI) tree.size(),
                                   &(*nList.begin()), 0, (ot::RankI) nList.size(),
                                   1, m_uiMaxDepth,
                                   0,
                                   inspector);

  return inspector.success;
}


template <unsigned int dim>
void
testTheTest()
{
  using T = unsigned int;
  /// const unsigned int dim = 4;
  const unsigned int numChildren = 1u << dim;
  using TreeNode = ot::TreeNode<T,dim>;

  _InitializeHcurve(dim);

  std::vector<TreeNode> unbalancedTree;

  TreeNode rootNode;
  TreeNode d0   = rootNode.getFirstChildMorton();
  TreeNode d100 = d0.getNeighbour(0, 1).getFirstChildMorton().getFirstChildMorton();

  unbalancedTree.push_back(d0);
  unbalancedTree.push_back(d100);
  std::cout << "(dim == " << dim << ")\n";
  /// for (const TreeNode &tn : unbalancedTree)
  ///   std::cout << "\t" << tn.getBase32Hex().data() << "\n";

  bool balanceSuccess = checkBalancingConstraint(unbalancedTree, true);
  std::cout << "Balancing constraint " << (balanceSuccess ? "succeeded" : "FAILED") << "\n";
  std::cout << "\n";
}




int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int ptsPerProc = 200;
  if (argc > 1)
    ptsPerProc = strtol(argv[1], NULL, 0);

  ///test_propagateNeighbours(ptsPerProc);

  ///testTheTest<2>();
  ///testTheTest<3>();
  ///testTheTest<4>();

  test_locTreeBalancing<2>(ptsPerProc);
  ///test_locTreeBalancing<4>(ptsPerProc);
  ///test_distTreeBalancing(ptsPerProc, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}


