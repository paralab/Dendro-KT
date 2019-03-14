/*
 * testCountCGNodes.cpp
 *   Test sequential node enumeration methods.
 *
 * Masado Ishii  --  UofU SoC, 2019-02-19
 */


#include "treeNode.h"
#include "mathUtils.h"
#include "nsort.h"
/// #include "tsort.h"
/// #include "octUtils.h"

#include "hcurvedata.h"

#include <bitset>
#include <vector>

#include <iostream>


using T = unsigned int;

template <unsigned int dim>
using Tree = std::vector<ot::TreeNode<T,dim>>;

template <unsigned int dim>
using NodeList = std::vector<ot::TNPoint<T,dim>>;

/**
 * @brief Example 1 is the minimal balanced tree in which the very center of the
 *        domain has pow(2,dim) elements of level endL.
 */ 
template <unsigned int dim>
struct Example1
{
  public:
    //
    // num_points()
    static constexpr unsigned int num_points(unsigned int endL, unsigned int order)
    {
      return (endL-2)*(intPow(4*order-1, dim) - intPow(2*order-1, dim)) + intPow(4*order+1, dim);
    }

    //
    // fill_tree()
    static void fill_tree(unsigned int endL, Tree<dim> &outTree)
    {
      constexpr unsigned char numCh = ot::TreeNode<T,dim>::numChildren;
      const ot::TreeNode<T,dim> root;
      for (unsigned char ch = 0; ch < numCh; ch++)
      {
        generate_corner(root.getChildMorton(ch), numCh - 1 - ch, endL, outTree);
      }
    }

  private:
    /**@note Recursive method to generate a corner of the domain. */
    static void generate_corner(ot::TreeNode<T,dim> e, unsigned char ch, unsigned int endL, Tree<dim> &outTree)
    {
      constexpr unsigned char numCh = ot::TreeNode<T,dim>::numChildren;
      if (e.getLevel() >= endL)
        outTree.push_back(e);
      else
      {
        for (unsigned char otherCh = 0; otherCh < numCh; otherCh++)
        {
          if (otherCh != ch)
            outTree.push_back(e.getChildMorton(otherCh));
        }
        generate_corner(e.getChildMorton(ch), ch, endL, outTree);
      }
    }
};


/**
 * @brief Example 2 is the uniform grid with elements at level endL.
 */
template <unsigned int dim>
struct Example2
{
  public:
    //
    // num_points()
    static constexpr unsigned int num_points(unsigned int endL, unsigned int order)
    {
      return intPow(intPow(2,endL)*order + 1, dim);
    }

    //
    // fill_tree()
    static void fill_tree(unsigned int endL, Tree<dim> &outTree)
    {
      ot::TreeNode<T,dim> root;
      fill_tree(root, endL, outTree);
    }

  private:
    static void fill_tree(ot::TreeNode<T,dim> parent, unsigned int endL, Tree<dim> &outTree)
    {
      constexpr unsigned char numCh = ot::TreeNode<T,dim>::numChildren;
      if (parent.getLevel() >= endL)
        outTree.push_back(parent);
      else
      {
        for (unsigned char ch = 0; ch < numCh; ch++)
        {
          fill_tree(parent.getChildMorton(ch), endL, outTree);
        }
      }
    }
};


/**
 * @brief Example 3 is the minimal balanced tree with a fringe of elements of
 *        level endL all around the domain boundary.
 */
template <unsigned int dim>
struct Example3
{
  public:
    //
    // num_points()
    static unsigned int num_points(unsigned int endL, unsigned int order)
    {
      // Starts with a uniform grid of the finest level.
      unsigned int total = Example2<dim>::num_points(endL, order);

      // Summation (negative): Intermediate shells take points away.
      for (unsigned int l = 2; l <= endL - 1; l++)
      {
        total += intPow((intPow(2,l) - 2)*order + 1, dim);
        total -= intPow((intPow(2,l+1) - 4)*order + 1, dim);
      }

      return total;
    }

    //
    // fill_tree()
    static void fill_tree(unsigned int endL, Tree<dim> &outTree)
    {
      constexpr unsigned char numCh = ot::TreeNode<T,dim>::numChildren;
      ot::TreeNode<T,dim> root;
      for (unsigned char ch = 0; ch < numCh; ch++)
        subdivide_element(root.getChildMorton(ch), endL, outTree);
    }

  private:
    static void subdivide_element(ot::TreeNode<T,dim> parent, unsigned int endL, Tree<dim> &outTree)
    {
      constexpr unsigned char numCh = ot::TreeNode<T,dim>::numChildren;
      if (parent.getLevel() >= endL)
        outTree.push_back(parent);
      else
      {
        for (unsigned char ch = 0; ch < numCh; ch++)
        {
          ot::TreeNode<T,dim> f = parent.getChildMorton(ch);
          if (f.isTouchingDomainBoundary())
            subdivide_element(f, endL, outTree);
          else
            outTree.push_back(f);
        }
      }
    }
};



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
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeListExterior, order, tree.data(), comm);
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
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeListExterior, order, tree.data(), comm);
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
    numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeListExterior, order, tree.data(), comm);
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
