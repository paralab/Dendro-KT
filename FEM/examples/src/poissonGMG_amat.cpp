#include <iostream>
#include <mpi.h>
#include <stdio.h>

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

// Dendro
#include "dendro.h"
#include "oda.h"
#include "poissonGMG.h"

using Eigen::Matrix;

constexpr int dim = 3;

//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  _InitializeHcurve(dim);

  MPI_Comm comm = PETSC_COMM_WORLD;
  const int eleOrder = 1;
  const unsigned int ndofs = 1;
  const double sfc_tol = 0.3;

  using uint = unsigned int;
  using DTree_t = ot::DistTree<uint, dim>;
  using DA_t = ot::DA<dim>;

  // Fine grid.
  // For simplicity define using a uniform grid.
  const int finestLevel = 4;
  const int numGrids = 2;
  DTree_t dtree = DTree_t::constructSubdomainDistTree(
      finestLevel, comm, sfc_tol);

  // Define coarse grids based on the fine grid.
  // Also need surrogate grids.
  DTree_t surrogateDTree =
    dtree.generateGridHierarchyUp(true, numGrids, sfc_tol);

  // Create multi-level DA (we own), needed by gmgMat.
  ot::MultiDA<dim> multiDA, surrMultiDA;
  const ot::GridAlignment gridAlignment = ot::GridAlignment::CoarseByFine;
  DA_t::multiLevelDA(multiDA, dtree, comm, eleOrder, 100, sfc_tol);
  DA_t::multiLevelDA(surrMultiDA, surrogateDTree, comm, eleOrder, 100, sfc_tol);

  // Create the gmgMat.
  // (The elemental operators are supplied by the derived type,
  // which owns an array of poissonMat)
  PoissonEq::PoissonGMGMat<dim> poissonGMG(
      &dtree, &multiDA, &surrogateDTree, &surrMultiDA, gridAlignment, ndofs);

  std::cout << "dtree number of grids == " << dtree.getNumStrata() << "\n";
  std::cout << "fine grid number of elements == " << multiDA[0].getGlobalElementSz() << "\n";

  // -----------------------------
  // aMat
  // -----------------------------

  // Typedefs for aMat derived types.
  typedef par::aMat<par::aMatBased<double, unsigned long, unsigned int>,
                    double, unsigned long, unsigned int> aMatBased;
                    // ^ aMat type taking aMatBased as derived class
  typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>,
                    double, unsigned long, unsigned int>  aMatFree;
                    // ^ aMat type taking aMatFree as derived class
  typedef par::Maps<double,unsigned long,unsigned int> aMatMaps;

  // Use the aMat interface of gmgMat to define a chain of aMat.
  aMatFree** stMatFree_strata = nullptr;
  aMatMaps** meshMaps_strata = nullptr;
  poissonGMG.allocAMatMapsStrata(meshMaps_strata);
  poissonGMG.createAMatStrata(stMatFree_strata, meshMaps_strata);
  std::cout << "Created aMat chain\n";

  // Custom poissonGMG operator calls getAssembledAMat() on each level.
  poissonGMG.getAssembledAMatStrata(stMatFree_strata);

  // Pestc begins and completes assembling the global stiffness matrices.
  for (int stratum = 0; stratum < numGrids; ++stratum)
    stMatFree_strata[stratum]->finalize();

  std::cout << "Assembled aMat on each level.\n";

  // TODO define vcycle() for AMat

  // Once done, use aMat interface of gmgMat to deallocate.
  poissonGMG.destroyAMatStrata(stMatFree_strata);
  poissonGMG.deallocAMatMapsStrata(meshMaps_strata);

  _DestroyHcurve();
  PetscFinalize();
  return 0;
}
