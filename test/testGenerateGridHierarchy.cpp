///#include "meshLoop.h"
#include "distTree.h"
#include "octUtils.h"

#include <vector>
#include <iostream>

#include "hcurvedata.h"



int main(int argc, char * argv[])
{
  using T = unsigned int;
  constexpr unsigned int dim = 3;

  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  _InitializeHcurve(dim);

  const unsigned int numPoints = 300;
  /// const unsigned int sLev = 5;
  /// const unsigned int eLev = 5;
  const unsigned int sLev = 2;
  const unsigned int eLev = 2;

  // Normally distributed collection of points.
  /// std::vector<ot::TreeNode<T, dim>> tnlist = ot::getPts<T, dim>(numPoints, sLev, eLev);

  // Complete regular grid.
  std::vector<ot::TreeNode<T, dim>> tnlist;
  ot::createRegularOctree(tnlist, eLev, comm);


  // Give DistTree ownership of the octree.
  ot::DistTree<T,dim> dtree(tnlist);

  const std::vector<ot::TreeNode<T, dim>> &stratum0 = dtree.getTreePartFiltered(0);

  std::cout << "---------------------------------\n";
  std::cout << "Stratum0 (size==" << stratum0.size() << ")\n\n";
  for (const ot::TreeNode<T, dim> &tn : stratum0)
  {
    ot::printtn(tn, eLev);
    std::cout << "\n";
  }
  std::cout << "\n\n";

  dtree.generateGridHierarchy(true, 2, comm);

  const std::vector<ot::TreeNode<T, dim>> &stratum1 = dtree.getTreePartFiltered(1);

  std::cout << "---------------------------------\n";
  std::cout << "Stratum1 (size==" << stratum1.size() << ")\n\n";
  for (const ot::TreeNode<T, dim> &tn : stratum1)
  {
    ot::printtn(tn, eLev);
    std::cout << "\n";
  }
  std::cout << "\n\n";

  _DestroyHcurve();

  MPI_Finalize();

  return 0;
}

