#include <dollar.hpp>
#include "dollar_stat.h"

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
const OctList & octList(const DistTree &dtree);

int maxLevel(const OctList &octList);
int maxLevel(const DistTree &dtree);

template <typename T>
T mpi_max(T x, MPI_Comm comm);

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
  const double sfc_tol = 0.1;
  /// const int fineLevel = 3;
  /// ot::DistTree<uint, DIM> distTree =
  ///     ot::DistTree<uint, DIM>::minimalSubdomainDistTree(
  ///         fineLevel, sphereSet, comm, sfc_tol);
  const size_t grain = 1000;
  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::minimalSubdomainDistTreeGrain(
          grain, sphereSet, comm, sfc_tol);

  const int refinements = 1;
  const int fineLevel = maxLevel(distTree) + refinements;

  const int repetitions = 10;

  struct RemeshTrees
  {
    DistTree inTree;
    DistTree surrogate;
    DistTree outTree;
  }
  viaWhole, noVoid;

  for (int repIt = 0; repIt < repetitions; ++repIt)
  {
    viaWhole.outTree = noVoid.outTree = distTree;

    for (int refIt = 0; refIt < refinements; ++refIt)
    {
      const size_t refSize = std::max(size(viaWhole.outTree), size(noVoid.outTree));
      const std::vector<ot::OCT_FLAGS::Refine> refineFlags = rngRefine(refSize);

      // Old: remesh subdomain via Whole.
      {
        RemeshTrees &set = viaWhole;
        set.inTree = set.outTree;

        {DOLLAR("distRemeshSubdomainViaWhole()")
          DistTree::distRemeshSubdomainViaWhole(
              set.inTree, refineFlags, set.outTree, set.surrogate,
              ot::SurrogateInByOut, sfc_tol);
        }
        /// size_t sizeDistTree = size(set.inTree);
        /// size_t sizeSurrogate = size(set.surrogate);
        /// size_t sizeOutTree = size(set.outTree);
        /// if (commRank == 0)
        ///   printf("via whole: size(%lu)-->surr(%lu), size(%lu)\n",
        ///       size(set.inTree), size(set.surrogate), size(set.outTree));

        /// ot::quadTreeToGnuplot(octList(set.outTree), fineLevel, "viaWhole" + std::to_string(refIt), comm);
      }

      // New: remesh subdomain without communicating void.
      {
        RemeshTrees &set = noVoid;
        set.inTree = set.outTree;

        DistTree outTree, surrogate;
        {DOLLAR("distRemeshSubdomain()")
          DistTree::distRemeshSubdomain(
              set.inTree, refineFlags, set.outTree, set.surrogate,
              ot::SurrogateInByOut, sfc_tol);
        }
        /// size_t sizeDistTree = size(set.inTree);
        /// size_t sizeSurrogate = size(set.surrogate);
        /// size_t sizeOutTree = size(set.outTree);
        /// if (commRank == 0)
        ///   printf("subdomain: size(%lu)-->surr(%lu), size(%lu)\n",
        ///       size(set.inTree), size(set.surrogate), size(set.outTree));

        /// ot::quadTreeToGnuplot(octList(set.outTree), fineLevel, "noVoid" + std::to_string(refIt), comm);
      }

      /// const size_t sizeViaWhole = size(viaWhole.outTree);
      /// const size_t sizeNoVoid   = size(noVoid.outTree);
      /// if (commRank == 0)
      ///   printf("[rep %d  it %d] via whole: size==%lu  subdomain: size==%lu\n",
      ///       repIt, refIt, sizeViaWhole, sizeNoVoid);
      /// if (!mpi_and(viaWhole.outTree.getTreePartFiltered() == noVoid.outTree.getTreePartFiltered(), comm))
      ///   if (commRank == 0)
      ///     std::cout << RED "Different trees produced on iteration " << refIt << NRM "\n";
    }
  }

  dollar::DollarStat dollar_stat(comm);
  dollar::clear();
  dollar::DollarStat dollar_mean = dollar_stat.mpi_reduce_mean();
  /// dollar::DollarStat dollar_min = dollar_stat.mpi_reduce_min();
  /// dollar::DollarStat dollar_max = dollar_stat.mpi_reduce_max();
  if (commRank == 0)
  {
    /// std::ofstream file("mean_chrome.json");
    /// dollar_mean.chrome(file);

    std::cout << "\n" << "[Mean]\n";
    dollar_mean.text(std::cout);
  }

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
  return mpi_sum(size(distTree.getTreePartFiltered()), distTree.getComm());
}

// size()
size_t size(const OctList &octList)
{
  return octList.size();
}

// octList()
const OctList & octList(const DistTree &dtree)
{
  return dtree.getTreePartFiltered();
}

// maxLevel(OctList)
int maxLevel(const OctList &octList)
{
  int maxLevel = 0;
  for (const Oct &oct : octList)
    if (maxLevel < oct.getLevel())
      maxLevel = oct.getLevel();
  return maxLevel;
}

// maxLevel(DistTree)
int maxLevel(const DistTree &dtree)
{
  return mpi_max(maxLevel(octList(dtree)), dtree.getComm());
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

template <typename T>
T mpi_max(T x, MPI_Comm comm)
{
  T global;
  par::Mpi_Allreduce(&x, &global, 1, MPI_MAX, comm);
  return global;
}


