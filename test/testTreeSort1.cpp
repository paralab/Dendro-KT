/*
 * test_SFC_Tree.cpp
 *   Test the methods in SFC_Tree.h / SFC_Tree.cpp
 *
 *
 * Masado Ishii  --  UofU SoC, 2018-12-03
 */


#include "treeNode.h"
#include "tsort.h"
#include "octUtils.h"
#include <iostream>

#include "hcurvedata.h"

#include "octUtils.h"
#include <vector>

#include <assert.h>
#include <mpi.h>

//------------------------
// test_distTreeSort()
//------------------------
void test_distTreeSort1(int numPoints, MPI_Comm comm = MPI_COMM_WORLD)
{
  int nProc, rProc;
  MPI_Comm_size(comm, &nProc);
  MPI_Comm_rank(comm, &rProc);

  using T = unsigned int;
  const unsigned int dim = 2;
  using TreeNode = ot::TreeNode<T,dim>;

  /// const int numPoints = 23;    // Now made a parameter.

  _InitializeHcurve(dim);

  std::vector<TreeNode> points = ot::getPts<T,dim>(numPoints);

  // Sort!
  ot::SFC_Tree<T,dim>::distTreeSort(points, 0.2, MPI_COMM_WORLD);
//   std::cout << "abcdsef";
  ///ot::SFC_Tree<T,dim>::distTreeSort(points, 0.0, comm);

}
//------------------------


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();

  int ptsPerProc = 200;
  if (argc > 1)
    ptsPerProc = strtol(argv[1], NULL, 0);

  //test_locTreeSort();

  test_distTreeSort1(ptsPerProc, MPI_COMM_WORLD);

  //test_locTreeConstruction(ptsPerProc);

  DendroScopeEnd();
  MPI_Finalize();

  return 0;
}


