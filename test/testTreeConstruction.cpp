/*
 * testTreeConstruction.cpp
 *   Test local and distributed TreeConstruction methods from tsort.h / tsort.cpp
 *
 * Masado Ishii  --  UofU SoC, 2019-01-08
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
void checkLocalCompleteness(std::vector<ot::TreeNode<T,D>> &points,
                            std::vector<ot::TreeNode<T,D>> &tree,
                            bool startAtZero,
                            bool printData);
// ...........................................................................


//------------------------
// test_locTreeConstruction()
//------------------------
void test_locTreeConstruction(int numPoints)
{
  using T = unsigned int;
  const unsigned int dim = 4;
  const unsigned int numChildren = 1u << dim;
  using TreeNode = ot::TreeNode<T,dim>;

  _InitializeHcurve(dim);

  std::vector<TreeNode> points = ot::getPts<T,dim>(numPoints);
  std::vector<TreeNode> tree;

  const unsigned int maxPtsPerRegion = 8;
  /// const unsigned int maxPtsPerRegion = 3;

  const T leafLevel = m_uiMaxDepth;

  // TODO The fact that this starts at level 1 and it works,
  //      might indicate I am confused about what constitutes `root'
  //      and what is the range of allowable addresses.
  ot::SFC_Tree<T,dim>::locTreeConstruction(
      &(*points.begin()), tree,
      maxPtsPerRegion,
      0, (unsigned int) points.size(),
      1, leafLevel,
      0,
      TreeNode());

  checkLocalCompleteness<T,dim>(points, tree, true, true);
}


//------------------------
// test_distTreeConstruction()
//------------------------
void test_distTreeConstruction(int numPoints, MPI_Comm comm = MPI_COMM_WORLD)
{
  using T = unsigned int;
  const unsigned int dim = 4;
  const unsigned int numChildren = 1u << dim;
  using TreeNode = ot::TreeNode<T,dim>;

  _InitializeHcurve(dim);

  std::vector<TreeNode> points = ot::getPts<T,dim>(numPoints);
  std::vector<TreeNode> treePart;

  const unsigned int maxPtsPerRegion = 8;
  /// const unsigned int maxPtsPerRegion = 3;

  const double loadFlexibility = 0.2;

  const T leafLevel = m_uiMaxDepth;

  std::cerr << "Starting distTreeConstruction()...\n";
  ot::SFC_Tree<T,dim>::distTreeConstruction(points, treePart, maxPtsPerRegion, loadFlexibility, comm);
  std::cerr << "Finished distTreeConstruction().\n\n";

  checkLocalCompleteness<T,dim>(points, treePart, false, true);
}


// ----------------------
// checkLocalCompleteness
// ----------------------
template <typename T, unsigned int D>
void checkLocalCompleteness(std::vector<ot::TreeNode<T,D>> &points,
                            std::vector<ot::TreeNode<T,D>> &tree,
                            bool startAtZero,
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

  if (!startAtZero)
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
      if (address[l] != tn.getMortonIndex(l))
      {
        completeness = false;

        if (printData)
        {
          std::cout << "Completeness failure here. Level [" << lev << "], counter == "
            << address[l] << ", but node address == " << (int) tn.getMortonIndex(l) << "\n";
        }
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
  for (int lev = 1; lev < address.size(); lev++)   // See note at calling of locTreeConstruction().
    if (address[lev] != 0)
    {
      completeness = false;
      if (printData)
      {
        std::cout << "Completeness failure here. Previous count [" << lev << "] == " << address[lev] << "\n";
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

}
//------------------------






int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int ptsPerProc = 200;
  if (argc > 1)
    ptsPerProc = strtol(argv[1], NULL, 0);

  //test_locTreeConstruction(ptsPerProc);
  test_distTreeConstruction(ptsPerProc, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}


