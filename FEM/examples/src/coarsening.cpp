
#include <iostream>
#include <mpi.h>
#include <stdio.h>

// Dendro
// bug: if petsc is included (with logging)
//  before dendro parUtils then it triggers
//  dendro macros that use undefined variables.
#include "dendro.h"
#include "filterFunction.h"
#include "oda.h"
#include "octUtils.h"
/// #include "point.h"
/// #include "poissonMat.h"
/// #include "poissonGMG.h"

#ifdef BUILD_WITH_PETSC
#include <petsc.h>
#endif

#include "aMat.hpp"
#include "aMatBased.hpp"
#include "aMatFree.hpp"
#include "aVec.hpp"
#include "constraintRecord.hpp"
#include "enums.hpp"
#include "fe_vector.hpp"
#include "integration.hpp"
#include "ke_matrix.hpp"
#include "maps.hpp"
#include "solve.hpp"

#include <Eigen/Dense>

using Eigen::Matrix;
using uint = unsigned int;

// ==========================================
template <int dim>
ibm::DomainDecider domainCube();

template <int dim>
ibm::DomainDecider domainChannel(int widthLevel);

template <int dim>
ibm::DomainDecider domainLeg(int widthLevel);


// ==========================================

////////////////////////////////////////////
template <int dim>
int main_dim(int argc, char *argv[]);
static void usage(const char *arg0);
inline static int str2int(const char *str);
////////////////////////////////////////////


int main(int argc, char *argv[])
{
#ifdef BUILD_WITH_PETSC
  PetscInitialize(&argc, &argv, NULL, NULL);
#endif

  if (argc < 2)
    usage(argv[0]);

  // Dispatch template dim.
  const int dim = str2int(argv[1]);
  int ret;
  switch (dim)
  {
    case 2: ret = main_dim<2>(argc, argv); break;
    case 3: ret = main_dim<3>(argc, argv); break;
    case 4: ret = main_dim<4>(argc, argv); break;
    default: usage(argv[1]);
  }
  
#ifdef BUILD_WITH_PETSC
  PetscFinalize();
#endif
  return ret;
}

template <int dim>
int main_dim(int argc, char *argv[])
{
  DendroScopeBegin();
  _InitializeHcurve(dim);

  MPI_Comm comm = PETSC_COMM_WORLD;
  const int eleOrder = 1;
  const unsigned int ndofs = 1;
  const double sfc_tol = 1.0/32;
  using DTree_t = ot::DistTree<uint, dim>;
  /// using DA_t = ot::DA<dim>;

  if (argc < 4)
    usage(argv[0]);
  const int domainOption = str2int(argv[2]);
  const int widthLevel = str2int(argv[3]);

  ibm::DomainDecider domainDecider;
  int fineLevel = widthLevel;
  switch(domainOption)
  {
    case 0: domainDecider = domainCube<dim>();
            break;
    case 1: domainDecider = domainChannel<dim>(widthLevel);
            break;
    case 2: domainDecider = domainChannel<dim>(widthLevel); fineLevel++;
            break;
    case 3: domainDecider = domainLeg<dim>(widthLevel);
            break;
    case 4: domainDecider = domainLeg<dim>(widthLevel); fineLevel++;
            break;
    default: usage(argv[0]);
  }

  // Finest level octree (stratum=0).
  DTree_t dtree = DTree_t::constructSubdomainDistTree(
      fineLevel, domainDecider, comm, sfc_tol);
  const size_t localFineOctants = dtree.getFilteredTreePartSz(0);

  // Add a coarser grid (stratum=1).
  DTree_t surrogateDTree;
  const std::vector<ot::OCT_FLAGS::Refine> coarsen(
      localFineOctants, ot::OCT_FLAGS::OCT_COARSEN);
  DTree_t::defineCoarsenedGrid(
      dtree, surrogateDTree, coarsen, ot::GridAlignment::CoarseByFine, sfc_tol);

  _DestroyHcurve();
  DendroScopeEnd();
  return 0;
}

template <int dim>
ibm::DomainDecider domainCube()
{
  std::array<double, dim> sizes;
  sizes.fill(1.0);
  return typename ot::DistTree<uint, dim>::BoxDecider(sizes);
}

template <int dim>
ibm::DomainDecider domainChannel(int widthLevel)
{
  const double width = 1.0 / (1u << widthLevel);
  std::array<double, dim> sizes;
  sizes.fill(width);
  sizes[0] = 1.0;
  return typename ot::DistTree<uint, dim>::BoxDecider(sizes);
}


template <int dim_>
ibm::DomainDecider domainLeg(int widthLevel)
{
  const int dim = dim_;
  const double width = 1.0 / (1u << widthLevel);

  using Boxer = typename ot::DistTree<uint, dim>::BoxDecider;
  std::vector<Boxer> boxers;
  boxers.reserve(dim);
  for (int d = 0; d < dim; ++d)
  {
    std::array<double, dim> sizes;
    sizes.fill(width);
    sizes[d] = 1.0;
    boxers.emplace_back(sizes);
  }

  return [=](const double *coords, double sz) -> ibm::Partition
  {
    bool isIn = true,  isOut = false;
    for (int d = 0; d < dim; ++d)
    {
      const ibm::Partition legPartition = boxers[d](coords, sz);
      isIn &= (legPartition == ibm::IN);
      isOut |= (legPartition == ibm::OUT);
    }
    return (isIn ? ibm::IN : isOut ? ibm::OUT : ibm::INTERCEPTED);
  };
}


///////////////////////////////////////////////////

static void usage(const char *arg0)
{
  int mpiRank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpiRank);

  if (mpiRank == 0)
    std::cout << "Usage: " << arg0
              << "\n    dim(2-4)"
              << "\n    domain(0=cube,1=channelBase,2=channelSubdiv1,3=legsBases,4=legsSubdiv1)"
              << "\n    widthLevel"
              << "\n";

#ifdef BUILD_WITH_PETSC
  PetscEnd();
#else
  exit(1);
#endif
}

inline static int str2int(const char *str)
{
  return static_cast<int>(strtoul(str, NULL, 0));
}
