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

#include <iostream>
#include <fstream>

// ...........................................................................
template <typename T, int Base, unsigned int L>
struct DigitString;
using DType = int;

template <typename T, unsigned int D>
bool checkLocalCompleteness(const std::vector<ot::TreeNode<T,D>> &tree,
                            bool entireTree,
                            bool printData,
                            DigitString<DType, (1<<D), 32> &counter);

template <typename T, unsigned int D, bool printData>
bool checkPointsContainedInorder(const std::vector<ot::TreeNode<T,D>> &points,
                                 const std::vector<ot::TreeNode<T,D>> &tree);

template <typename T, unsigned int D>
bool checkLocalCompletenessMorton(std::vector<ot::TreeNode<T,D>> &points,
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

  // Note: In order for the point check to succeed,
  // the points must be totally sorted, so this needs to be 1.
  const unsigned int maxPtsPerRegion = 1;

  const T leafLevel = m_uiMaxDepth;

  ot::SFC_Tree<T,dim>::locTreeBalancing(points, tree, maxPtsPerRegion);

  bool pointSuccess = checkPointsContainedInorder<T,dim,false>(points, tree);
  std::cout << "<dim==" << dim << "> Point check             " << (pointSuccess ? "succeeded" : "FAILED") << "\n";

  DigitString<DType, numChildren, 32> counter = DigitString<DType, numChildren, 32>::zero();
  bool completenessSuccess = checkLocalCompleteness<T,dim>(tree, true, false, counter);
  std::cout << "<dim==" << dim << "> Completeness constraint " << (completenessSuccess ? "succeeded" : "FAILED") << "\n";

  bool balanceSuccess = checkBalancingConstraint(tree, false);
  std::cout << "<dim==" << dim << "> Balancing constraint    " << (balanceSuccess ? "succeeded" : "FAILED") << "\n";

  // Make sure there is something there.
  std::cout << "Final.... points.size() == " << points.size() << "  tree.size() == " << tree.size() << "\n";

  _DestroyHcurve();
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
  int myLocAdjacency = checkLocalCompletenessMorton<T,dim>(points, treePart, false, false);

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


template <typename T, int Base, unsigned int L>
struct DigitString
{
  std::array<T,L> digits;

  static DigitString zero()
  {
    DigitString ret;
    #pragma unroll(L)
    for (unsigned int p = 0; p < L; p++)
      ret.digits[p] = 0;
    return ret;
  }

  void add(const DigitString &other)
  {
    #pragma unroll(L)
    for (unsigned int p = 0; p < L; p++)
      addToPlace(p, other.digits[p]);
  }

  void addToPlace(unsigned int p, T val)
  {
    assert(p < L);
    val = val % Base;
    do
    {
      digits[p] += val;
      val = (digits[p] >= Base);
      if (val)
        digits[p] -= Base;
    }
    while (++p < L && val);
  }

  unsigned int lowestNonzero() const
  {
    for (unsigned int p = 0; p < L; p++)
      if (digits[p])
        return p;
    return L;
  }

  void print(std::ostream &out, unsigned int loPlace, unsigned int hiPlace) const
  {
    for (unsigned int p = hiPlace; hiPlace >= p && p >= loPlace; p--)  // workaround for unsigned decrementing
      out << digits[p] << " ";
  }
};

//
// checkLocalCompleteness()
//
// Notes:
//   - This method sums up the volume occupied by all the nodes of the tree.
//   - Additionally, checks that each subtree occupies the proper volume.
//     (Assuming that the tree is sorted, contains only leaves, and no duplicates.)
//     Because of this, if `entireTree' is set to false, the counter will be
//     initialized based on what would be the immediate predecessor of the
//     first listed node in a partition.
//   - Assumes that level 0 corresponds to root, so total volume should be 1 at root level.
//
template <typename T, unsigned int D>
bool checkLocalCompleteness(const std::vector<ot::TreeNode<T,D>> &tree,
                            bool entireTree,
                            bool printData,
                            DigitString<DType, (1<<D), 32> &counter)
{
  const int numChildren = 1<<D;
  if (!entireTree)
  {
    counter = DigitString<DType, (1<<D), 32>::zero();
    int pRot = 0;
    for (unsigned int l = 0; l <= m_uiMaxDepth; l++)
    {
      unsigned char child = tree.front().getMortonIndex(l);
      unsigned char child_sfc = rotations[pRot * 2*numChildren + numChildren + child];
      counter.digits[m_uiMaxDepth - l] = child_sfc;    // A 0-based index is also a predecessor count.
      pRot = HILBERT_TABLE[pRot * numChildren + child];
    }
  }

  // Sum the tree volume.
  // Check that before moving to a higher level, the lower levels are so-far complete.
  unsigned int lev = (entireTree ? 0 : tree.front().getLevel());
  for (const ot::TreeNode<T,D> &tn : tree)
  {
    if (printData)
    {
      counter.print(std::cout, 0, m_uiMaxDepth);
      std::cout << "  \t ++ " << tn.getBase32Hex().data() << "\n";
    }

    if (lev > (lev = tn.getLevel()) && (m_uiMaxDepth - counter.lowestNonzero()) > lev)
      return false;
    counter.addToPlace(m_uiMaxDepth - lev, 1);
  }
  if (entireTree)
    return (counter.lowestNonzero() == m_uiMaxDepth &&
           counter.digits[m_uiMaxDepth] == 1);
  else
    return true;
}


template <typename T, unsigned int D, bool printData>
bool checkPointsContainedInorder(const std::vector<ot::TreeNode<T,D>> &points,
                                 const std::vector<ot::TreeNode<T,D>> &tree)
{
  int lev = 0, prevLev = 0;
  typename std::vector<ot::TreeNode<T,D>>::const_iterator pIt = points.begin();
  const char continueStr[] = "  ";
  const char expandStr[]   = " [_]";
  const char newBlockStr[] = "__";
  const int beginTextPos = 40;
  std::streamsize oldWidth = std::cout.width();
  if (printData)
    std::cout << "    (Buckets)                           (Points)\n";

  for (const ot::TreeNode<T,D> tn : tree)
  {
    prevLev = lev;
    lev = tn.getLevel();

    // Print buckets.
    if (printData)
    {
      if (lev != prevLev)
      {
        for (int ii = 0; ii < lev; ii++)
          std::cout << continueStr;
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

  bool success = (pIt == points.end());

  // Print any remaining points.
  if (printData)
    while (pIt != points.end())
    {
      for (int ii = 0; ii < lev; ii++)
        std::cout << continueStr;
        
      std::cout << std::setw(beginTextPos - 2*lev) << ' '
                << std::setw(oldWidth) << pIt->getBase32Hex().data()
                << "\n";
      pIt++;
    }

  return success;
}



// ----------------------
// checkLocalCompletenessMorton
// ----------------------
template <typename T, unsigned int D>
bool checkLocalCompletenessMorton(std::vector<ot::TreeNode<T,D>> &points,
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


//
// checkBalancingConstraint()
//
// Notes:
//   - Assumes that TreeNode::appendAllNeighbours() works.
//   - This version assumes that the tree is complete.
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
  // The balancing constraint is satisfied iff, for each x in tree and y in nList
  // such that (x == y or x.isAncestor(y)), we have (x == y or x.isParent(y)).
  //
  // By assuming that the tree is complete, we avoid needing a rotation stack
  // to traverse the domain, and we can finish the balancing check in
  // a single pass of both tree and nList.

  if (printData)
    std::cout << "Finish balancing check - interleaved.\n";

  typename std::vector<ot::TreeNode<T,D>>::const_iterator nPtr = nList.begin();
  typename std::vector<ot::TreeNode<T,D>>::const_iterator const nEnd = nList.end();
  for (const ot::TreeNode<T,D> &tn : tree)
  {
    if (printData)
      std::cout << tn.getBase32Hex().data() << "\n";

    while (nPtr != nEnd && nPtr->isAncestor(tn))
    {
      if (printData)
        std::cout << "\t\t" << nPtr->getBase32Hex().data() << " (ancestor)\n";
      nPtr++;
    }

    ot::LevI tnLevel = tn.getLevel();

    while (nPtr != nEnd && (tn == *nPtr || tn.isAncestor(*nPtr)))
    {
      if (printData)
        std::cout << "\t\t" << nPtr->getBase32Hex().data() << "\n";
      if (nPtr->getLevel() - tnLevel > 1)
        return false;
      nPtr++;
    }
  }

  return true;
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
  test_locTreeBalancing<3>(ptsPerProc);
  test_locTreeBalancing<4>(ptsPerProc);
  ///test_distTreeBalancing(ptsPerProc, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}


