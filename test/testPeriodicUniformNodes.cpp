

#include "dendro.h"

#include "pcoord.h"
#include "distTree.h"
#include "oda.h"

#include <mpi.h>
#include <array>
#include <iostream>
#include <sstream>

constexpr int DIM = 2;
using uint = unsigned int;
using RankI = ot::RankI;


ot::DistTree<uint, DIM> refineOnBoundary(const ot::DistTree<uint, DIM> &distTree);

// main()
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();

  _InitializeHcurve(DIM);
  periodic::PCoord<uint, DIM>::periods({(1u<<m_uiMaxDepth), periodic::NO_PERIOD});
  /// periodic::PCoord<uint, DIM>::periods({periodic::NO_PERIOD, periodic::NO_PERIOD});

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  const int fineLevel = 4;

  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::constructSubdomainDistTree(fineLevel, comm);

  ot::quadTreeToGnuplot(distTree.getTreePartFiltered(), fineLevel, "uniformPeriodic", comm);

  ot::DA<DIM> * da = new ot::DA<DIM>(distTree, comm, 1);

  {
    const RankI cellDims = 1u << fineLevel;
    const RankI vertexDims = (1u << fineLevel) + 1;
    const RankI expectedNodes = cellDims * vertexDims;  // 2D

    const RankI measuredNodes = da->getGlobalNodeSz();

    fprintf(stdout, "expectedNodes==%llu  %smeasuredNodes==%llu%s\n",
        expectedNodes,
        (expectedNodes == measuredNodes ? GRN : RED),
        measuredNodes,
        NRM);
  }

  // TODO iterate over the nodes in an element loop

  delete da;

  _DestroyHcurve();

  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}


// refineOnBoundary()
ot::DistTree<uint, DIM> refineOnBoundary(const ot::DistTree<uint, DIM> &distTree)
{
  const std::vector<ot::TreeNode<uint, DIM>> oldTree = distTree.getTreePartFiltered();
  const size_t oldSz = oldTree.size();
  std::vector<ot::OCT_FLAGS::Refine> refines(oldSz, ot::OCT_FLAGS::OCT_NO_CHANGE);
  for (size_t ii = 0; ii < oldSz; ++ii)
    if (oldTree[ii].getIsOnTreeBdry())
      refines[ii] = ot::OCT_FLAGS::OCT_REFINE;

  ot::DistTree<uint, DIM> newDistTree;
  ot::DistTree<uint, DIM> surrDistTree;
  ot::DistTree<uint, DIM>::distRemeshSubdomain(
      distTree, refines, newDistTree, surrDistTree, ot::SurrogateInByOut, 0.3);
  return newDistTree;
}

