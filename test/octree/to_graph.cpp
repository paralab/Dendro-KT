
#include <vector>
#include <array>

#include "test/octree/multisphere.h"
#include "include/distTree.h"
#include "include/filterFunction.h"
#include "include/octUtils.h"
#include "include/tnUtils.h"
#include "include/treeNode.h"

#include "include/octree_to_graph.hpp"

using uint = unsigned int;
constexpr int DIM = 3;

// --------------------------------------------------------------------
using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;
using DistTree = ot::DistTree<uint, DIM>;

int maxLevel(const OctList &octList);
int maxLevel(const DistTree &dtree);

DendroIntL mpi_sum(DendroIntL x, MPI_Comm comm);
bool mpi_and(bool x, MPI_Comm comm);
// --------------------------------------------------------------------

//
// main()
//
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int commRank, commSize;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  const double sfc_tol = 0.1;

  test::SphereSet<DIM> sphereSet;
  sphereSet.carveSphere(0.10, {0.15, 0.6, 0.4});
  sphereSet.carveSphere(0.10, {0.7, 0.4, 0.5});
  sphereSet.carveSphere(0.210, {0.45, 0.55, 0.6});

  if (argc < 2)
  {
    fprintf(stderr, "usage: %s fine_level{1..8}\n", argv[0]);
    exit(1);
  }

  const int fineLevel = atol(argv[1]);
  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::minimalSubdomainDistTree(
          fineLevel, sphereSet, comm, sfc_tol);

  /// ot::quadTreeToGnuplot(distTree.getTreePartFiltered(), fineLevel, "quadtree", comm);

  const graph::ElementGraph e2e = ot::octree_to_graph<DIM>(distTree, comm, graph::SelfLoop::Remove);

  std::cout << e2e;

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}

