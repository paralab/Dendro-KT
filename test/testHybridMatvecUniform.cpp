// Dendro
#include "distTree.h"
#include "oda.h"
#include "subset.h"
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
using uint = unsigned int;

template <typename NodeT>
std::ostream & print(const ot::DA<dim> *da,
                     const NodeT *vec,
                     std::ostream & out = std::cout);

template <typename C, unsigned int dim>
std::vector<bool> boundaryFlags(
    const ot::DistTree<C, dim> &distTree,
    const int stratum = 0);

template <typename NodeT>
NodeT * ptr(std::vector<NodeT> &vec) { return vec.data(); }

template <typename NodeT>
const NodeT * const_ptr(const std::vector<NodeT> &vec) { return vec.data(); }


//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  _InitializeHcurve(dim);

  MPI_Comm comm = PETSC_COMM_WORLD;
  int mpiRank, mpiSize;
  MPI_Comm_rank(comm, &mpiRank);
  MPI_Comm_size(comm, &mpiSize);

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

  // Subset
  std::vector<bool> bdryFlags = boundaryFlags(dtree);
  ot::GlobalSubset<dim> subsetExp(da, &dtree.getTreePartFiltered(), bdryFlags, true);
  ot::GlobalSubset<dim> subsetImp(da, &dtree.getTreePartFiltered(), bdryFlags, false);

  {
    // Sanity checks. Should all match due to partitioning from simpleDA.
    ot::GlobalSubset<dim> subsetAll(da, &dtree.getTreePartFiltered(), std::vector<bool>(bdryFlags.size(), true), true);

#define ASSERT_EQUAL_DA(method) assert(subsetAll.method() == da->method());
#define PRINT_DIFF(method) { if (subsetAll.method() != da->method()) {\
  fprintf(stderr, "%s:\t%u:%u (%d)\n",\
      #method,\
      (unsigned) subsetAll.method(),\
      (unsigned) da->method(),\
      int(subsetAll.method() - da->method()));\
  }}
  PRINT_DIFF( getLocalElementSz );              ASSERT_EQUAL_DA( getLocalElementSz );
  PRINT_DIFF( getLocalNodalSz );                ASSERT_EQUAL_DA( getLocalNodalSz );
  PRINT_DIFF( getLocalNodeBegin );              ASSERT_EQUAL_DA( getLocalNodeBegin );
  PRINT_DIFF( getLocalNodeEnd );                ASSERT_EQUAL_DA( getLocalNodeEnd );
  PRINT_DIFF( getPreNodalSz );                  ASSERT_EQUAL_DA( getPreNodalSz );
  PRINT_DIFF( getPostNodalSz );                 ASSERT_EQUAL_DA( getPostNodalSz );
  PRINT_DIFF( getTotalNodalSz );                ASSERT_EQUAL_DA( getTotalNodalSz );
  PRINT_DIFF( getGlobalNodeSz );                ASSERT_EQUAL_DA( getGlobalNodeSz );
  PRINT_DIFF( getGlobalNodeBegin );             ASSERT_EQUAL_DA( getGlobalNodeBegin );
  PRINT_DIFF( getGlobalElementSz );             ASSERT_EQUAL_DA( getGlobalElementSz );
  PRINT_DIFF( getGlobalElementBegin );          ASSERT_EQUAL_DA( getGlobalElementBegin );
#undef PRINT_DIFF
#undef ASSERT_EQUAL_DA
  }


  // poissonMat and poissonVec
  const double d_min=-0.5;
  const double d_max=0.5;
  Point<dim> pt_min(d_min,d_min,d_min);
  Point<dim> pt_max(d_max,d_max,d_max);
  PoissonEq::PoissonMat<dim> poissonMat(da, &dtree.getTreePartFiltered(), 1);
  poissonMat.setProblemDimensions(pt_min,pt_max);
  PoissonEq::PoissonVec<dim> poissonVec(da, &dtree.getTreePartFiltered(), 1);
  poissonVec.setProblemDimensions(pt_min,pt_max);

  // vectors
  std::vector<double> u, v_pure, v_hybrid;
  da->createVector(u, false, false, 1);
  da->createVector(v_pure, false, false, 1);
  da->createVector(v_hybrid, false, false, 1);
  std::fill(v_pure.begin(), v_pure.end(), 0);
  std::fill(v_hybrid.begin(), v_hybrid.end(), 0);

  // set u = x*x + y*y
  for (size_t i = 0; i < da->getLocalNodalSz(); ++i)
  {
    // unit cube coords
    double x[dim];
    ot::treeNode2Physical((da->getTNCoords() + da->getLocalNodeBegin())[i], eleOrder, x);

    // scale
    for (int d = 0; d < dim; ++d)
      x[d] *= pt_max.x(d) - pt_min.x(d);

    // shift
    for (int d = 0; d < dim; ++d)
      x[d] += pt_min.x(d);

    u[i] = x[0]*x[0] + x[1]*x[1];
  }


  // -----------------------------
  // aMat
  // -----------------------------
  typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>  aMatFree; // aMat type taking aMatBased as derived class
  typedef par::Maps<double,unsigned long,unsigned int> aMatMaps;

  //TODO subset: boundary, allocAMatMaps, createAMat, createVector, split/merge

  std::vector<double> dirichletZeros(da->getBoundaryNodeIndices().size() * ndofs, 0.0);

  aMatFree* stMatFree=NULL;
  aMatMaps* meshMaps=NULL;
  da->allocAMatMaps(meshMaps, dtree.getTreePartFiltered(), dirichletZeros.data(), ndofs);
  da->createAMat(stMatFree,meshMaps);
  stMatFree->set_matfree_type((par::MATFREE_TYPE)1);
  poissonMat.getAssembledAMat(stMatFree);
  stMatFree->finalize();

  stMatFree->matvec(ptr(v_pure), const_ptr(u), false);

  print(da, v_pure.data(), std::cout) << "------------------------------------\n";

  delete da;
  _DestroyHcurve();
  PetscFinalize();

  return 0;
}


template <typename NodeT>
std::ostream & print(const ot::DA<dim> *da,
                     const NodeT *vec,
                     std::ostream & out)
{
  return ot::printNodes(
      da->getTNCoords() + da->getLocalNodeBegin(),
      da->getTNCoords() + da->getLocalNodeBegin() + da->getLocalNodalSz(),
      vec,
      da->getElementOrder(),
      out);
}


template <typename C, unsigned int dim>
std::vector<bool> boundaryFlags(
    const ot::DistTree<C, dim> &distTree,
    const int stratum)
{
  const std::vector<ot::TreeNode<C, dim>> &elements =
      distTree.getTreePartFiltered(stratum);
  const size_t size = elements.size();

  std::vector<bool> boundaryFlags(size, false);

  for (size_t ii = 0; ii < size; ++ii)
    boundaryFlags[ii] = elements[ii].getIsOnTreeBdry();

  return boundaryFlags;
}

