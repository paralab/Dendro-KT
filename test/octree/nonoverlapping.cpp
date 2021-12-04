
#include <vector>
#include <array>

#include "multisphere.h"
#include "distTree.h"
#include "filterFunction.h"
#include "octUtils.h"
#include "tnUtils.h"
#include "treeNode.h"


using uint = unsigned int;
constexpr int DIM = 2;


int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int commRank, commSize;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  test::SphereSet<DIM> sphereSet;
  sphereSet.carveSphere(0.10, {0.15, 0.6});
  sphereSet.carveSphere(0.10, {0.7, 0.4});
  sphereSet.carveSphere(0.210, {0.45, 0.55});

  const int fineLevel = 7;
  const double sfc_tol = 0.1;

  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::constructSubdomainDistTree(
          fineLevel, sphereSet, comm, sfc_tol);

  ot::quadTreeToGnuplot(distTree.getTreePartFiltered(), fineLevel, "sphereSet", comm);

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}
