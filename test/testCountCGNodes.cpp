/*
 * testCountCGNodes.cpp
 *   Test sequential node enumeration methods.
 *
 * Masado Ishii  --  UofU SoC, 2019-02-19
 */


#include "testAdaptiveExamples.h"

#include "treeNode.h"
#include "mathUtils.h"
#include "nsort.h"

#include "hcurvedata.h"

#include <bitset>
#include <vector>

#include <iostream>


/// using T = unsigned int;
/// 
/// template <unsigned int dim>
/// using Tree = std::vector<ot::TreeNode<T,dim>>;
/// 
/// template <unsigned int dim>
/// using NodeList = std::vector<ot::TNPoint<T,dim>>;

template<typename X>
void distPrune(std::vector<X> &list, MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const int listSize = list.size();
  const int baseSeg = listSize / nProc;
  const int remainder = listSize - baseSeg * nProc;
  const int myStart = rProc * baseSeg + (rProc < remainder ? rProc : remainder);
  const int mySeg = baseSeg + (rProc < remainder ? 1 : 0);

  /// fprintf(stderr, "[%d] listSize==%d, myStart==%d, mySeg==%d\n", rProc, listSize, myStart, mySeg);

  list.erase(list.begin(), list.begin() + myStart);
  list.resize(mySeg);
}



int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);

  const bool RunDistributed = true;  // Switch between sequential and distributed.
  int nProc, rProc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rProc);
  MPI_Comm_size(MPI_COMM_WORLD, &nProc);
  MPI_Comm comm = MPI_COMM_WORLD;

  constexpr unsigned int dim = 3;
  const unsigned int endL = 3;
  const unsigned int order = 2;

  double tol = 0.05;

  _InitializeHcurve(dim);

  unsigned int numPoints;
  Tree<dim> tree;
  NodeList<dim> nodeListExterior;
  NodeList<dim> nodeListInterior;

  ot::RankI numUniqueInteriorNodes;
  ot::RankI numUniqueExteriorNodes;
  ot::RankI numUniqueNodes;

  ot::ScatterMap scatterMap;
  ot::GatherMap gatherMap;

  // -------------

  if (rProc == 0)
  {
    numPoints = Example1<dim>::num_points(endL, order);
    Example1<dim>::fill_tree(endL, tree);
    printf("Example1: numPoints==%u, numElements==%lu.\n", numPoints, tree.size());
    tree.clear();

    numPoints = Example2<dim>::num_points(endL, order);
    Example2<dim>::fill_tree(endL, tree);
    printf("Example2: numPoints==%u, numElements==%lu.\n", numPoints, tree.size());
    tree.clear();

    numPoints = Example3<dim>::num_points(endL, order);
    Example3<dim>::fill_tree(endL, tree);
    printf("Example3: numPoints==%u, numElements==%lu.\n", numPoints, tree.size());
    tree.clear();
  }

  // -------------

  // Example1
  Example1<dim>::fill_tree(endL, tree);
  if (RunDistributed)
  {
    distPrune(tree, comm);
    ot::SFC_Tree<T,dim>::distTreeSort(tree, tol, comm);
  }
  for (const ot::TreeNode<T,dim> &tn : tree)
  {
    ot::Element<T,dim>(tn).appendInteriorNodes(order, nodeListInterior);
    ot::Element<T,dim>(tn).appendExteriorNodes(order, nodeListExterior);
  }
  /// for (auto &&n : nodeListExterior)
  /// {
  ///   if (!n.isOnDomainBoundary())
  ///     std::cout << n.getFinestOpenContainer().getBase32Hex().data() << "\n";
  /// }
  numUniqueInteriorNodes = nodeListInterior.size();
  if (RunDistributed)
  {
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeListExterior, order, tree.data(), scatterMap, gatherMap, comm);
    ot::RankI globInterior = 0;
    par::Mpi_Allreduce(&numUniqueInteriorNodes, &globInterior, 1, MPI_SUM, comm);
    numUniqueInteriorNodes = globInterior;
  }
  else
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::countCGNodes(&(*nodeListExterior.begin()), &(*nodeListExterior.end()), order);
  numUniqueNodes = numUniqueInteriorNodes + numUniqueExteriorNodes;
  /// for (auto &&n : nodeListExterior)
  ///   std::cout << n.getBase32Hex().data() << " \t " << n.getBase32Hex(5).data() << "\n";
  printf("Example1: Algorithm says # points == %u \t [Int:%u] [Ext:%u].\n", numUniqueNodes, numUniqueInteriorNodes, numUniqueExteriorNodes);
  tree.clear();
  nodeListInterior.clear();
  nodeListExterior.clear();

  // Example2
  Example2<dim>::fill_tree(endL, tree);
  if (RunDistributed)
  {
    distPrune(tree, comm);
    ot::SFC_Tree<T,dim>::distTreeSort(tree, tol, comm);
  }
  for (const ot::TreeNode<T,dim> &tn : tree)
  {
    ot::Element<T,dim>(tn).appendInteriorNodes(order, nodeListInterior);
    ot::Element<T,dim>(tn).appendExteriorNodes(order, nodeListExterior);
  }
  numUniqueInteriorNodes = nodeListInterior.size();
  if (RunDistributed)
  {
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeListExterior, order, tree.data(), scatterMap, gatherMap, comm);
    ot::RankI globInterior = 0;
    par::Mpi_Allreduce(&numUniqueInteriorNodes, &globInterior, 1, MPI_SUM, comm);
    numUniqueInteriorNodes = globInterior;
  }
  else
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::countCGNodes(&(*nodeListExterior.begin()), &(*nodeListExterior.end()), order);
  numUniqueNodes = numUniqueInteriorNodes + numUniqueExteriorNodes;
  /// for (auto &&n : nodeListExterior)
  ///   std::cout << n.getBase32Hex().data() << " \t " << n.getBase32Hex(5).data() << "\n";
  printf("Example2: Algorithm says # points == %u \t [Int:%u] [Ext:%u].\n", numUniqueNodes, numUniqueInteriorNodes, numUniqueExteriorNodes);
  tree.clear();
  nodeListInterior.clear();
  nodeListExterior.clear();

  // Example3
  Example3<dim>::fill_tree(endL, tree);
  if (RunDistributed)
  {
    distPrune(tree, comm);
    ot::SFC_Tree<T,dim>::distTreeSort(tree, tol, comm);
  }
  for (const ot::TreeNode<T,dim> &tn : tree)
  {
    ot::Element<T,dim>(tn).appendInteriorNodes(order, nodeListInterior);
    ot::Element<T,dim>(tn).appendExteriorNodes(order, nodeListExterior);
  }
  numUniqueInteriorNodes = nodeListInterior.size();
  if (RunDistributed)
  {
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeListExterior, order, tree.data(), scatterMap, gatherMap, comm);
    ot::RankI globInterior = 0;
    par::Mpi_Allreduce(&numUniqueInteriorNodes, &globInterior, 1, MPI_SUM, comm);
    numUniqueInteriorNodes = globInterior;
  }
  else
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::countCGNodes(&(*nodeListExterior.begin()), &(*nodeListExterior.end()), order);
  numUniqueNodes = numUniqueInteriorNodes + numUniqueExteriorNodes;
  /// for (auto &&n : nodeListExterior)
  ///   std::cout << n.getBase32Hex().data() << " \t " << n.getBase32Hex(5).data() << "\n";
  printf("Example3: Algorithm says # points == %u \t [Int:%u] [Ext:%u].\n", numUniqueNodes, numUniqueInteriorNodes, numUniqueExteriorNodes);
  tree.clear();
  nodeListInterior.clear();
  nodeListExterior.clear();

  _DestroyHcurve();

  MPI_Finalize();

  return 0;
}
