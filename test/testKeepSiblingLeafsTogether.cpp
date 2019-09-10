
/**
 * @file testKeepSiblingLeafsTogether.cpp
 * @author Masado Ishii
 *
 * Test the utility function keepSiblingLeafsTogether().
 *
 * We will construct a complete tree based on a random Gaussian point cloud,
 * count the number of ranks with broken siblings. If there are children of
 * a sibling closer to the middle of the local partition, it doesn't count.
 */


#include "treeNode.h"
#include "tsort.h"
#include "parUtils.h"
#include "octUtils.h"
#include "hcurvedata.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <vector>



template <unsigned int dim>
bool testRandTree(MPI_Comm comm);

// Returns either 0 or 1 (local means just one rank).
template <unsigned int dim>
ot::RankI countSeparations(const std::vector<ot::TreeNode<unsigned int, dim>> &treePart, MPI_Comm comm);

// ==============================
// main()
// ==============================
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const char * usageString = "Usage: %s dim\n";
  unsigned int inDim;

  if (argc < 2)
  {
    if (!rProc)
      printf(usageString, argv[0]);
    exit(1);
  }
  else
  {
    inDim   = strtol(argv[1], NULL, 0);
  }

  _InitializeHcurve(inDim);

  if (!rProc)
    printf("Test results: ");

  const char * resultColor;
  const char * resultName;

  int totalSuccess = true;

  // testRandTree
  int result_testRandTree, globResult_testRandTree;
  switch (inDim)
  {
    case 2: result_testRandTree = testRandTree<2>(comm); break;
    case 3: result_testRandTree = testRandTree<3>(comm); break;
    case 4: result_testRandTree = testRandTree<4>(comm); break;
    default: if (!rProc) printf("Dimension not supported.\n"); exit(1); break;
  }
  par::Mpi_Reduce(&result_testRandTree, &globResult_testRandTree, 1, MPI_SUM, 0, comm);
  totalSuccess = totalSuccess && !globResult_testRandTree;
  resultColor = globResult_testRandTree ? RED : GRN;
  resultName = globResult_testRandTree ? "FAILURE" : "success";
  if (!rProc)
    printf("\t[testRandTree](%s%s %d%s)", resultColor, resultName, globResult_testRandTree, NRM);

  if(!rProc)
    printf("\n");

  _DestroyHcurve();

  MPI_Finalize();

  return (!totalSuccess);
}





template <unsigned int dim>
bool testRandTree(MPI_Comm comm)
{
  // Test:
  // while mpirun -np <NP> ./tstKeepSiblingLeafsTogether <dim> > /dev/null ; do echo ; done

  using C = unsigned int;

  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const unsigned int totalNumPoints = 10;
  const unsigned int numMyPoints = (totalNumPoints / nProc)
                                   + (rProc < totalNumPoints % nProc);
  const unsigned int numPrevPoints = (totalNumPoints / nProc) * rProc
                                     + (rProc < totalNumPoints % nProc ? rProc : totalNumPoints % nProc);

  unsigned int seed;
  ot::RankI countInitial_glob;
  std::vector<ot::TreeNode<C,dim>> tree;


  // Repeat until we get a test case where siblings are actually split.
  int trialSeeds = 0;
  const bool justOnce = true;
  do
  {
    trialSeeds++;
    tree.clear();

    if (!rProc)
    {
      // Pseudo-random number generators for point coordinates.
      std::random_device rd;
      seed = rd();
      /// seed = 2142908055;
      /// seed = 2450139245;
      /// seed = 3106312564;
      /// seed = 2884066049;
      /// seed = 1622234473;
      seed = 4023431161;
      // Can also set seed manually if needed.

      /// std::cerr << "Seed: " << seed << "\n";  // Wait till we know it's a good test.
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, comm);
    std::mt19937 gen(seed);
    gen.discard((dim + 1) * numPrevPoints);

    std::uniform_int_distribution<C> coordDis(0, (1u << m_uiMaxDepth) - 1);
    std::uniform_int_distribution<unsigned int> levDis((m_uiMaxDepth + 1)/4, (m_uiMaxDepth + 1)/2);

    std::vector<ot::TreeNode<C,dim>> pointCoords;

    for (int ii = 0; ii < numMyPoints; ii++)
    {
      const unsigned int lev = levDis(gen);
      const C mask = (1u << (m_uiMaxDepth + 1)) - (1u << (m_uiMaxDepth - lev));

      std::array<C,dim> coords;
      for (int d = 0; d < dim; d++)
        coords[d] = coordDis(gen) & mask;

      pointCoords.emplace_back(1, coords, lev);
    }

    // Make points at least locally unique.
    ot::SFC_Tree<C,dim>::locTreeSort(
        &(*pointCoords.begin()),
        0,
        (ot::RankI) pointCoords.size(),
        1,
        m_uiMaxDepth,
        0);
    ot::SFC_Tree<C,dim>::locRemoveDuplicates(pointCoords);

    // Distributed tree construction.
    ot::SFC_Tree<C,dim>::distTreeConstruction(pointCoords, tree, 1, 0.0, comm);

    // Count num separations right after construction and partition.
    ot::RankI countInitial = countSeparations(tree, comm);

    /// fprintf(stderr, "%*s[g%d] Finished first countSeparations()\n", 40*rProc, "\0", rProc);

    par::Mpi_Reduce(&countInitial, &countInitial_glob, 1, MPI_SUM, 0, comm);
  }
  while (!justOnce && trialSeeds < 1000 && !countInitial_glob);

  if (!rProc)
    std::cerr << "Seed: " << seed << " (trial " << trialSeeds << ")\n";  // Now we'll use the test.

  if (!rProc)
  {
    std::cout << "countInitial_glob==" << countInitial_glob << " \n";
  }

  ot::keepSiblingLeafsTogether<C,dim>(tree, comm);

  /// //DEBUG
  /// for (int ii = 0; ii < tree.size(); ii++)
  /// {
  ///   fprintf(stdout, "%*c[%03d] == (%-17s).%u\n",
  ///       rProc * 40, ' ',
  ///       ii, tree[ii].getBase32Hex().data(), tree[ii].getLevel());
  /// }

  /// fprintf(stderr, "%*s[g%d] Finished keepSiblingLeafsTogether()\n", 40*rProc, "\0", rProc);

  // Count num separations after filtering.
  ot::RankI countFinal = countSeparations(tree, comm);
  ot::RankI countFinal_glob;

  /// fprintf(stderr, "%*s[g%d] Finished second countSeparations()\n", 40*rProc, "\0", rProc);

  par::Mpi_Reduce(&countFinal, &countFinal_glob, 1, MPI_SUM, 0, comm);
  if (!rProc)
  {
    std::cout << "countFinal_glob==" << countFinal_glob << " \n";
  }

  return countFinal;
}




template <unsigned int dim>
ot::RankI countSeparations(const std::vector<ot::TreeNode<unsigned int, dim>> &treePart, MPI_Comm comm)
{
  // DISREGARD, The below story still allows false negatives
  // (case that doesn't pass but should)..  e.g.,
  //      P0      |   P1    |  P2
  //              |         |
  //              |  [] []  |  []
  // [] [] [] []  |         |
  //
  //
  // Classify end segments, communicate, and combine
  // classifications of adjacent end segments.
  //
  // An end segment on the front end (resp. back end)
  // is the longest consecutive run of TreeNodes that share the same parent
  // as the front (resp. back) TreeNode. (These are possibly the same run).
  //
  // What I mean by classification:
  // There are two classes, independent (I) and dependent (D).
  // - An end segment is independent if it contains all NUM_CHILDREN TreeNodes,
  //   or it is terminated by non-sibling TreeNodes, for whom the parent of the
  //   front/back is an ancestor.
  // - Otherwise the end segment is dependent.
  //
  // We exchange the classification and shared parent of each end segment
  // with the adjacent rank. Adjacent end segments are combined into
  // 'joint segments.'
  //
  // Two independent end segments, or an independent and a dependent end segment,
  // combine to form an *unbroken* segment.
  // Two dependent end segments, if they share a parent, combine to form a *broken* segment
  // (otherwise it is again an *unbroken* segment).

  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  bool isFrontBroken = false, isBackBroken = false;

  /// fprintf(stderr, "%*s[g%d] Before Comm_split.\n", 40*rProc, "\0", rProc);

  // If some ranks, have no TreeNodes, exclude them from the new communicator.
  MPI_Comm nonemptys;
  MPI_Comm_split(comm, (treePart.size() > 0 ? 1 : MPI_UNDEFINED), rProc, &nonemptys);

  /// fprintf(stderr, "%*s[g%d] After Comm_split.\n", 40*rProc, "\0", rProc);

  if (treePart.size())
  {
    int nNE, rNE;
    MPI_Comm_rank(nonemptys, &rNE);
    MPI_Comm_size(nonemptys, &nNE);

    const unsigned int NUM_CHILDREN = 1u << dim;

    const ot::TreeNode<unsigned int, dim> *treePtr;
    const ot::TreeNode<unsigned int, dim> * const treeBegin = &(*treePart.begin());
    const ot::TreeNode<unsigned int, dim> * const treeEnd = &(*treePart.end());

    MPI_Request requestBegin, requestEnd;
    MPI_Status status;

    ot::TreeNode<unsigned int, dim> sendFrontParent = treePart.front().getParent();
    ot::TreeNode<unsigned int, dim> sendBackParent = treePart.back().getParent();
    ot::TreeNode<unsigned int, dim> prevBackParent, nextFrontParent;

    constexpr int DEPENDENT = 0;
    constexpr int INDEPENDENT = 1;
    int sendFrontClass, sendBackClass, prevBackClass, nextFrontClass;

    /// fprintf(stderr, "%*s[%d] Before Classify Front.\n", 40*rNE, "\0", rNE);

    // Classify front.
    treePtr = treeBegin + 1;
    while (treePtr < treeEnd && treePtr->getParent() == treeBegin->getParent())
      treePtr++;
    if (treePtr - treeBegin == NUM_CHILDREN ||
        (treePtr < treeEnd &&
         treePtr->getLevel() > treeBegin->getLevel() &&
         treeBegin->getParent().isAncestor(*treePtr)))
      sendFrontClass = INDEPENDENT;
    else
      sendFrontClass = DEPENDENT;

    /// fprintf(stderr, "%*s[%d] Before Classify Back.\n", 40*rNE, "\0", rNE);

    // Classify back.
    treePtr = treeEnd - 2;
    while (treePtr >= treeBegin && treePtr->getParent() == treeEnd[-1].getParent())
      treePtr--;
    if (treeEnd - (treePtr+1) == NUM_CHILDREN ||
        (treePtr >= treeBegin &&
         treePtr->getLevel() > treeEnd[-1].getLevel() &&
         treeEnd[-1].getParent().isAncestor(*treePtr)))
      sendBackClass = INDEPENDENT;
    else
      sendBackClass = DEPENDENT;


    /// fprintf(stderr, "%*s[%d] Before any sendrecv.\n", 40*rNE, "\0", rNE);

    // Exchange parents and classes of end segments.
    if (rNE > 0)
    {
      par::Mpi_Sendrecv<ot::TreeNode<unsigned int, dim>,
                        ot::TreeNode<unsigned int, dim>>(&sendFrontParent, 1, rNE-1, 0,
                                                         &prevBackParent, 1, rNE-1, 0,
                                                         nonemptys, &status);
      par::Mpi_Sendrecv<int, int>(&sendFrontClass, 1, rNE-1, 0,
                                  &prevBackClass, 1, rNE-1, 0,
                                  nonemptys, &status);
      if (sendFrontClass == DEPENDENT &&
          prevBackClass == DEPENDENT &&
          sendFrontParent == prevBackParent)
        isFrontBroken = true;

      /// fprintf(stderr, "%*s[g%d] sfClass==%d, prevBackClass==%d.\n", 40*rProc, "\0", rProc, sendFrontClass, prevBackClass);
    }
    if (rNE < nNE - 1)
    {
      par::Mpi_Sendrecv<ot::TreeNode<unsigned int, dim>,
                        ot::TreeNode<unsigned int, dim>>(&sendBackParent, 1, rNE+1, 0,
                                                         &nextFrontParent, 1, rNE+1, 0,
                                                         nonemptys, &status);
      par::Mpi_Sendrecv<int, int>(&sendBackClass, 1, rNE+1, 0,
                                  &nextFrontClass, 1, rNE+1, 0,
                                  nonemptys, &status);
      if (sendBackClass == DEPENDENT &&
          nextFrontClass == DEPENDENT &&
          sendBackParent == nextFrontParent)
        isFrontBroken = true;

      /// fprintf(stderr, "%*s[g%d] sbClass==%d, nextFrontClass==%d.\n", 40*rProc, "\0", rProc, sendBackClass, nextFrontClass);
    }

    /// fprintf(stderr, "%*s[%d] After all sendrecv.\n", 40*rNE, "\0", rNE);
  }

  MPI_Comm_free(&nonemptys);

  return (isFrontBroken || isBackBroken);
}
