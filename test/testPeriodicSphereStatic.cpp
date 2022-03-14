

#include "dendro.h"

#include "pcoord.h"
#include "distTree.h"
#include "oda.h"
#include "filterFunction.h"

#include <mpi.h>
#include <array>
#include <iostream>
#include <sstream>

constexpr int DIM = 2;
using uint = unsigned int;

struct DomainDecider
{ ibm::Partition operator()(const double *elemPhysCoords, double elemPhysSize) const; };

ot::DistTree<uint, DIM> refineOnBoundary(const ot::DistTree<uint, DIM> &distTree);
double distMeasureVolume(const ot::DistTree<uint, DIM> &distTree, MPI_Comm comm);

constexpr int numSpheres = 2;
/// constexpr int numSpheres = 1;

// spheres()
std::array<double, DIM> spheres(int i)
{
  const std::array<double, DIM> spheres[numSpheres] = {
    {0.0, 0.5}, 
    {0.5, 0.5}
  };
  return spheres[i];
}

// radii()
double radii(int i)
{
  const double radius = 0.125;
  return radius;
}


// main()
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();

  _InitializeHcurve(DIM);
  periodic::PCoord<uint, DIM>::periods({(1u<<m_uiMaxDepth)/2, periodic::NO_PERIOD});
  /// periodic::PCoord<uint, DIM>::periods({(1u<<m_uiMaxDepth), periodic::NO_PERIOD});
  /// periodic::PCoord<uint, DIM>::periods({periodic::NO_PERIOD, periodic::NO_PERIOD});

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::constructSubdomainDistTree(0, DomainDecider(), comm);

  const int fineLevel = 8;
  for (int level = 0; level <= fineLevel; ++level)
    distTree = refineOnBoundary(distTree);

  const bool balanced = ot::is2to1Balanced(distTree.getTreePartFiltered(), comm);

  /// ot::quadTreeToGnuplot(distTree.getTreePartFiltered(), fineLevel+1, "sphereMesh", comm);

  const double measureVolume = distMeasureVolume(distTree, comm);
  if (comm_rank == 0)
  {
    if (not balanced)
      printf(RED "Not balanced\n" NRM);

    const double expectedMissing = M_PI * (radii(0) * radii(0));
    const double measureMissing = 0.5 - measureVolume;
    const bool isExpected = fabs(measureMissing - expectedMissing) / expectedMissing < 0.05;
    printf("Expected=%f  %sMeasure=%f%s\n",
        expectedMissing, (isExpected ? GRN : RED), measureMissing, NRM);
  }

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

// DomainDecider::operator()
ibm::Partition DomainDecider::operator()(
    const double *elemPhysCoords, double elemPhysSize) const
{

  // ------------------------------------

  // For each sphere,
  //   find nearest point on box and test distance from center.
  //   find furtherst point on box and test distance from center.

  bool isIn = false, isOut = true;
  for (int i = 0; i < numSpheres; ++i)
  {
    double originToCenter[DIM];
    for (int d = 0; d < DIM; ++d)
      originToCenter[d] = spheres(i)[d] - elemPhysCoords[d];

    double nearest[DIM];
    for (int d = 0; d < DIM; ++d)
    {
      double clamped = originToCenter[d];
      if (clamped < 0)
        clamped = 0;
      else if (clamped > elemPhysSize)
        clamped = elemPhysSize;
      nearest[d] = clamped;
    }
    double nearestDist2 = 0;
    for (int d = 0; d < DIM; ++d)
    {
      const double dist = nearest[d] - originToCenter[d];
      nearestDist2 += dist * dist;
    }

    double furthest[DIM];
    for (int d = 0; d < DIM; ++d)
    {
      double a = fabs(originToCenter[d] - 0);
      double b = fabs(originToCenter[d] - elemPhysSize);
      furthest[d] = (a >= b ? 0 : elemPhysSize);
    }
    double furthestDist2 = 0;
    for (int d = 0; d < DIM; ++d)
    {
      const double dist = furthest[d] - originToCenter[d];
      furthestDist2 += dist * dist;
    }

    const double r2 = radii(i) * radii(i);
    isIn |= furthestDist2 <= r2;
    isOut &= nearestDist2 > r2;
  }

  // ------------------------------------

  ibm::Partition result;
  if (isIn && !isOut)
    result = ibm::IN;
  else if (isOut && !isIn)
    result = ibm::OUT;
  else
    result = ibm::INTERCEPTED;
  return result;
}


// measureVolume()
double measureVolume(const ot::DistTree<uint, DIM> &distTree)
{
  double sum = 0;
  for (const ot::TreeNode<uint, DIM> &tn : distTree.getTreePartFiltered())
  {
    std::array<double, DIM> physCoords;
    double physSize;
    ot::treeNode2Physical(tn, physCoords.data(), physSize);

    // Octant volume;
    double vol = 1;
    for (int d = 0; d < DIM; ++d)
      vol *= physSize;

    sum += vol;
  }
  return sum;
}


// distMeasureVolume()
double distMeasureVolume(const ot::DistTree<uint, DIM> &distTree, MPI_Comm comm)
{
  double localMeasure = measureVolume(distTree);
  double globalMeasure = 0;
  par::Mpi_Allreduce(&localMeasure, &globalMeasure, 1, MPI_SUM, comm);
  return globalMeasure;
}


