// Dendro
#include "distTree.h"
#include "oda.h"
#include "intergridTransfer.h"
#include "gmgMat.h"
#include "point.h"
#include "refel.h"
#include "tensor.h"

// Dendro examples
#include "poissonMat.h"
#include "poissonVec.h"

// PETSc
#include "petsc.h"

// AMat  (assembly-free matvec)
#include "../external/aMat/include/aMatBased.hpp"
#include "../external/aMat/include/aMatFree.hpp"

// stdlib
#include <iostream>

constexpr int dim = 2;

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

  using DTree_t = ot::DistTree<uint, dim>;
  using DA_t = ot::DA<dim>;

  // Static uniform refinement to depth coarseLev.
  const int coarseLev = 3;
  DTree_t dtree = DTree_t::constructSubdomainDistTree(
      coarseLev, comm, sfc_tol);

  // DA
  DA_t * da = new DA_t(dtree, comm, eleOrder, 100, sfc_tol);

  // poissonMat and poissonVec
  const double d_min=-0.5;
  const double d_max=0.5;
  Point<dim> pt_min(d_min,d_min,d_min);
  Point<dim> pt_max(d_max,d_max,d_max);
  PoissonEq::PoissonMat<dim> poissonMat(da, &dtree.getTreePartFiltered(), 1);
  poissonMat.setProblemDimensions(pt_min,pt_max);
  PoissonEq::PoissonVec<dim> poissonVec(da, &dtree.getTreePartFiltered(), 1);
  poissonVec.setProblemDimensions(pt_min,pt_max);


  // -----------------------------
  // aMat
  // -----------------------------
  typedef par::aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatBased; // aMat type taking aMatBased as derived class
  typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>  aMatFree; // aMat type taking aMatBased as derived class
  typedef par::Maps<double,unsigned long,unsigned int> aMatMaps;

  aMatFree*   stMatFree=NULL;
  /// aMatBased* stMatBased=NULL;
  aMatMaps* meshMaps=NULL;
  da->allocAMatMaps(meshMaps, dtree.getTreePartFiltered(), 1);
  da->createAMat(stMatFree,meshMaps);

  poissonMat.getAssembledAMat(stMatFree);

  printf("Assembled poissonMat stMatFree\n");

  delete da;
  _DestroyHcurve();
  PetscFinalize();

  return 0;
}
