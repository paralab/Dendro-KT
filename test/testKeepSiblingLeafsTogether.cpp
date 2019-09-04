
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
bool testRandTree(MPI_Comm comm, unsigned int depth, unsigned int order);

// Returns either 0 or 1 (local means just one rank).
template <unsigned int dim>
ot::RankI countLocalSeparations(const std::vector<ot::TreeNode<unsigned int, dim>> &treePart);

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

  const char * usageString = "Usage: %s dim depth order\n";
  unsigned int inDim, inDepth, inOrder;

  if (argc < 4)
  {
    if (!rProc)
      printf(usageString, argv[0]);
    exit(1);
  }
  else
  {
    inDim   = strtol(argv[1], NULL, 0);
    inDepth = strtol(argv[2], NULL, 0);
    inOrder = strtol(argv[3], NULL, 0);
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
    case 2: result_testRandTree = testRandTree<2>(comm, inDepth, inOrder); break;
    case 3: result_testRandTree = testRandTree<3>(comm, inDepth, inOrder); break;
    case 4: result_testRandTree = testRandTree<4>(comm, inDepth, inOrder); break;
    default: if (!rProc) printf("Dimension not supported.\n"); exit(1); break;
  }
  par::Mpi_Reduce(&result_testRandTree, &globResult_testRandTree, 1, MPI_SUM, 0, comm);
  totalSuccess = totalSuccess && globResult_testRandTree;
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
bool testRandTree(MPI_Comm comm, unsigned int depth, unsigned int order)
{
  // Test:
  // while ./tstKeepSiblingLeafsTogether > /dev/null ; do echo ; done

  using C = unsigned int;

  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const unsigned int totalNumPoints = 50;
  const unsigned int numMyPoints = (totalNumPoints / nProc)
                                   + (rProc < totalNumPoints % nProc);
  const unsigned int numPrevPoints = (totalNumPoints / nProc) * rProc
                                     + (rProc < totalNumPoints % nProc ? rProc : totalNumPoints % nProc);

  unsigned int seed;

  if (!rProc)
  {
    // Pseudo-random number generators for point coordinates.
    std::random_device rd;
    seed = rd();
    // Can also set seed manually if needed.

    std::cerr << "Seed: " << seed << "\n";
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
  std::vector<ot::TreeNode<C,dim>> tree;

  ot::SFC_Tree<C,dim>::distTreeConstruction(pointCoords, tree, 1, 0.0, comm);

  // Count num separations right after construction and partition.
  ot::RankI countInitial = countLocalSeparations(tree);
  ot::RankI countInitial_glob;

  par::Mpi_Reduce(&countInitial, &countInitial_glob, 1, MPI_SUM, 0, comm);
  if (!rProc)
  {
    std::cout << "countInitial_glob==" << countInitial_glob << " \n";
  }

  ot::keepSiblingLeafsTogether<C,dim>(tree, comm);

  // Count num separations after filtering.
  ot::RankI countFinal = countLocalSeparations(tree);
  ot::RankI countFinal_glob;

  par::Mpi_Reduce(&countFinal, &countFinal_glob, 1, MPI_SUM, 0, comm);
  if (!rProc)
  {
    std::cout << "countFinal_glob==" << countFinal_glob << " \n";
  }

  return (countFinal == 0);
}




template <unsigned int dim>
ot::RankI countLocalSeparations(const std::vector<ot::TreeNode<unsigned int, dim>> &treePart)
{
  if (!treePart.size())
    return 0;

  const unsigned int NUM_CHILDREN = 1u << dim;

  const ot::TreeNode<unsigned int, dim> *treePtr;
  const ot::TreeNode<unsigned int, dim> * const treeBegin = &(*treePart.begin());
  const ot::TreeNode<unsigned int, dim> * const treeEnd = &(*treePart.end());

  bool isFrontBroken = false, isBackBroken = false;

  // Test front.
  treePtr = treeBegin + 1;
  while (treePtr < treeEnd && treePtr->getParent() == treeBegin->getParent())
    treePtr++;
  if (treePtr - treeBegin == NUM_CHILDREN ||
      (treePtr < treeEnd && treePtr->getParent().getParent() == treeBegin->getParent()))
    isFrontBroken = false;
  else
    isFrontBroken = true;

  if (isFrontBroken)
    return 1;

  if (treePtr == treeEnd)
    return isFrontBroken;

  // If front doesn't extend to the back, test the back too.
  treePtr = treeEnd - 2;
  while (treePtr >= treeBegin && treePtr->getParent() == treeEnd[-1].getParent())
    treePtr--;
  if (treeEnd - (treePtr+1) == NUM_CHILDREN ||
      (treePtr >= treeBegin && treePtr->getParent().getParent() == treeEnd[-1].getParent()))
    isBackBroken = false;
  else
    isBackBroken = true;

  return (isFrontBroken || isBackBroken);
}
