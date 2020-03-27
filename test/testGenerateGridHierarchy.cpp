#include "distTree.h"
#include "octUtils.h"
#include "oda.h"

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
  const unsigned int sLev = 3;
  const unsigned int eLev = 3;

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

  /// dtree.generateGridHierarchyUp(true, 3, 0.1, comm);
  dtree.generateGridHierarchyDown(3, 0.1, comm);

  const std::vector<ot::TreeNode<T, dim>> &stratum1 = dtree.getTreePartFiltered(1);

  std::cout << "---------------------------------\n";
  std::cout << "Stratum1 (size==" << stratum1.size() << ")\n\n";
  for (const ot::TreeNode<T, dim> &tn : stratum1)
  {
    ot::printtn(tn, eLev);
    std::cout << "\n";
  }
  std::cout << "\n\n";


  const unsigned int order = 2;

  std::vector<ot::DA<dim>> multiDA;
  ot::DA<dim>::multiLevelDA(multiDA, dtree, comm, order);
  for (int l = 0; l < multiDA.size(); ++l)
  {
    fprintf(stdout, "%*s[%2d]---DA[%2d] localNodalSz==%u\n",
        10*rProc, "",
        rProc,
        l,
        (unsigned int) multiDA[l].getLocalNodalSz());
  }

  _DestroyHcurve();

  MPI_Finalize();

  return 0;
}

