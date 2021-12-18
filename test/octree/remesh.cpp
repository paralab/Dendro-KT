/// #include "dollar.hpp"

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

// --------------------------------------------------------------------
using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;
using DistTree = ot::DistTree<uint, DIM>;

std::vector<ot::OCT_FLAGS::Refine> rngRefine(size_t localSz);

size_t size(const DistTree &distTree);
size_t size(const OctList &octList);
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

  // Any octree.
  test::SphereSet<DIM> sphereSet;
  sphereSet.carveSphere(0.10, {0.15, 0.6});
  sphereSet.carveSphere(0.10, {0.7, 0.4});
  sphereSet.carveSphere(0.210, {0.45, 0.55});
  const int fineLevel = 3;
  const double sfc_tol = 0.1;
  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::minimalSubdomainDistTree(
          fineLevel, sphereSet, comm, sfc_tol);

  const std::vector<ot::OCT_FLAGS::Refine> refineFlags = rngRefine(size(distTree));

  // Old: remesh subdomain via Whole.
  {
    DistTree outTree, surrogate;
    {///$
      DistTree::distRemeshSubdomainViaWhole(
          distTree, refineFlags, outTree, surrogate,
          ot::SurrogateInByOut, sfc_tol);
    }
    size_t sizeDistTree = mpi_sum(size(distTree), comm);
    size_t sizeSurrogate = mpi_sum(size(surrogate), comm);
    size_t sizeOutTree = mpi_sum(size(outTree), comm);
    if (commRank == 0)
      printf("via whole: size(%lu)-->surr(%lu), size(%lu)\n",
          size(distTree), size(surrogate), size(outTree));
  }

  // New: remesh subdomain without communicating void.
  {
    DistTree outTree, surrogate;
    {///$
      DistTree::distRemeshSubdomain(
          distTree, refineFlags, outTree, surrogate,
          ot::SurrogateInByOut, sfc_tol);
    }
    size_t sizeDistTree = mpi_sum(size(distTree), comm);
    size_t sizeSurrogate = mpi_sum(size(surrogate), comm);
    size_t sizeOutTree = mpi_sum(size(outTree), comm);
    if (commRank == 0)
      printf("subdomain: size(%lu)-->surr(%lu), size(%lu)\n",
          size(distTree), size(surrogate), size(outTree));
  }

  /// if (commRank == 0)
  ///   dollar::text(std::cout);
  /// dollar::clear();

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}


// ------------------------------------------

std::vector<ot::OCT_FLAGS::Refine> rngRefine(size_t localSz)
{
  std::vector<ot::OCT_FLAGS::Refine> refines;
  std::mt19937 gen;
  std::uniform_int_distribution<int> dist(0, 2);  // 0, 1, 2
  for (size_t i = 0; i < localSz; ++i)
    refines.push_back(ot::OCT_FLAGS::Refine(dist(gen)));
  return refines;
}

// size()
size_t size(const DistTree &distTree)
{
  return size(distTree.getTreePartFiltered());
}

// size()
size_t size(const OctList &octList)
{
  return octList.size();
}

// mpi_sum()
DendroIntL mpi_sum(DendroIntL x, MPI_Comm comm)
{
  DendroIntL sum = 0;
  par::Mpi_Allreduce(&x, &sum, 1, MPI_SUM, comm);
  return sum;
}

// mpi_and()
bool mpi_and(bool x_, MPI_Comm comm)
{
  int x = x_, global = true;
  par::Mpi_Allreduce(&x, &global, 1, MPI_LAND, comm);
  return bool(global);
}


