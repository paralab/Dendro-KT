// Dendro
#include "distTree.h"
#include "oda.h"
#include "subset.h"
#include "intergridTransfer.h"
#include "gmgMat.h"
#include "point.h"
#include "refel.h"
#include "tensor.h"
#include <set>
#include "subset_amat.h"

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
std::ostream & printLocal(const ot::DA<dim> *da,
                          const NodeT *vec,
                          std::ostream & out = std::cout);

template <typename NodeT>
std::ostream & printGhosted(const ot::DA<dim> *da,
                            const NodeT *vec,
                            std::ostream & out = std::cout);

// Note: subset is drawn from da ghosted vec.
template <typename NodeT>
std::ostream & print(const ot::LocalSubset<dim> subset,
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


namespace ot
{

  template <unsigned int dim, typename DT>
  std::vector<DT> ghostRead(const DA<dim> *da, const int ndofs, const std::vector<DT> &localVec);

  template <unsigned int dim, typename DT>
  std::vector<DT> ghostWriteAccumulate(const DA<dim> *da, const int ndofs, const std::vector<DT> &ghostedVec);

  template <unsigned int dim, typename DT>
  std::vector<DT> ghostWriteOverwrite(const DA<dim> *da, const int ndofs, const std::vector<DT> &ghostedVec);
}



//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  DendroScopeBegin();
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

  // Domain
  const double d_min=-0.5;
  const double d_max=0.5;
  Point<dim> pt_min(d_min,d_min,d_min);
  Point<dim> pt_max(d_max,d_max,d_max);

  // Static uniform refinement to depth coarseLev.
  const int coarseLev = 3;
  DTree_t dtree = DTree_t::constructSubdomainDistTree(
      coarseLev, comm, sfc_tol);

  // DA
  DA_t * da = new DA_t(dtree, comm, eleOrder, 100, sfc_tol);

  // Subset
  std::vector<bool> bdryFlags = boundaryFlags(dtree);
  ot::LocalSubset<dim> subsetExp(da, &dtree.getTreePartFiltered(), bdryFlags, true);
  ot::LocalSubset<dim> subsetImp(da, &dtree.getTreePartFiltered(), bdryFlags, false);

  // vectors
  std::vector<double> u, v_pure;
  da->createVector(u, false, false, 1);
  da->createVector(v_pure, false, false, 1);
  std::fill(v_pure.begin(), v_pure.end(), 0);

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

  // poissonMat and poissonVec
  PoissonEq::PoissonMat<dim> poissonMat(da, &dtree.getTreePartFiltered(), 1);
  poissonMat.setProblemDimensions(pt_min,pt_max);
  PoissonEq::PoissonVec<dim> poissonVec(da, &dtree.getTreePartFiltered(), 1);
  poissonVec.setProblemDimensions(pt_min,pt_max);


  // -----------------------------
  // aMat
  // -----------------------------
  typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>  aMatFree; // aMat type taking aMatBased as derived class
  typedef par::Maps<double,unsigned long,unsigned int> aMatMaps;

  // whole da
  aMatFree* stMatFree=NULL;
  aMatMaps* meshMaps=NULL;
  std::vector<double> dirichletZeros(da->getGhostedBoundaryNodeIndices().size() * ndofs, 0.0);
  da->allocAMatMaps(meshMaps, dtree.getTreePartFiltered(), dirichletZeros.data(), ndofs);
  da->createAMat(stMatFree,meshMaps);
  stMatFree->set_matfree_type((par::MATFREE_TYPE)1);
  poissonMat.getAssembledAMat(stMatFree);
  stMatFree->finalize();

  stMatFree->matvec(ptr(v_pure), const_ptr(u), false);

  // subset explicit
  aMatMaps *meshMapsExp = nullptr;
  aMatFree *matFreeExp = nullptr;
  std::vector<double> dirichletZeros_exp(subsetExp.getBoundaryNodeIndices().size() * ndofs, 0.0);
  allocAMatMaps(subsetExp, meshMapsExp, dirichletZeros_exp.data(), ndofs);
  createAMat(subsetExp, matFreeExp, meshMapsExp);
  matFreeExp->set_matfree_type((par::MATFREE_TYPE) 1);
  fem::getAssembledAMat(subsetExp, poissonMat, matFreeExp);
  matFreeExp->finalize();


  //   vector -> ghost read -> split[A, B]
  //          A -> resultA          B -> resultB
  //   merge[resultA, resultB] -> ghost write -> result

  // Ghost read:
  const std::vector<double> gh_u = ghostRead(da, ndofs, u);
  std::vector<double> gh_v_hybrid(gh_u.size(), 0);

  // Split:
  std::vector<double> u_exp = gather_ndofs(const_ptr(gh_u), subsetExp, ndofs);
  std::vector<double> u_imp = gather_ndofs(const_ptr(gh_u), subsetImp, ndofs);

  // Explicit matvec:
  std::vector<double> v_exp(u_exp.size(), 0);
  matFreeExp->matvec(ptr(v_exp), const_ptr(u_exp), false);

  // Implicit matvec:
  std::vector<double> v_imp(u_imp.size(), 0);
  const auto eleOp = [&poissonMat] (
      const VECType *in,    VECType *out, unsigned int ndofs,
      const double *coords, double scale, bool isElementBoundary)
  {
    return poissonMat.elementalMatVec(in, out, ndofs, coords, scale, isElementBoundary);
  };
  fem::matvec(
      u_imp.data(), v_imp.data(), ndofs,
      subsetImp.relevantNodes().data(), subsetImp.relevantNodes().size(),
      subsetImp.relevantOctList().data(), subsetImp.relevantOctList().size(),
      fem::EleOpT<double>(eleOp), 1.0, da->getReferenceElement());

  // Unsplit [accumulate]:
  scatter_ndofs_accumulate(const_ptr(v_exp), ptr(gh_v_hybrid), subsetExp, ndofs);
  scatter_ndofs_accumulate(const_ptr(v_imp), ptr(gh_v_hybrid), subsetImp, ndofs);
  // Ghost write [accumulate]:
  std::vector<double> v_hybrid = ghostWriteAccumulate(da, ndofs, gh_v_hybrid);

  /// printLocal(da, v_hybrid.data(), std::cout) << "------------------------------------\n";
  /// printLocal(da, v_pure.data(), std::cout) << "------------------------------------\n";

  // Compare hybrid vs pure.
  double maxDiff = 0;
  for (size_t ii = 0; ii < v_pure.size(); ++ii)
    if (fabs(v_hybrid[ii] - v_pure[ii]) > maxDiff)
      maxDiff = fabs(v_hybrid[ii] - v_pure[ii]);
  const bool success = (maxDiff < 1e-10);
  printf("Difference is %s%f (%.0e)%s\n", (success ? GRN : RED), maxDiff, maxDiff, NRM);

  delete da;
  _DestroyHcurve();
  DendroScopeEnd();
  PetscFinalize();

  return 0;
}


template <typename NodeT>
std::ostream & printLocal(const ot::DA<dim> *da,
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

template <typename NodeT>
std::ostream & printGhosted(const ot::DA<dim> *da,
                            const NodeT *vec,
                            std::ostream & out)
{
  return ot::printNodes(
      da->getTNCoords(),
      da->getTNCoords() + da->getTotalNodalSz(),
      vec,
      da->getElementOrder(),
      out);
}


// Note: subset is drawn from da ghosted vec.
template <typename NodeT>
std::ostream & print(const ot::LocalSubset<dim> subset,
                     const NodeT *vec,
                     std::ostream & out)
{
  return ot::printNodes(
      &(*subset.relevantNodes().begin()),
      &(*subset.relevantNodes().end()),
      vec,
      subset.da()->getElementOrder(),
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




namespace ot
{

  template <unsigned int dim, typename DT>
  std::vector<DT> ghostRead(const DA<dim> *da, const int ndofs, const std::vector<DT> &localVec)
  {
    const size_t localSz = da->getLocalNodalSz();
    const size_t localBegin = da->getLocalNodeBegin();
    std::vector<DT> ghostedVec(ndofs * da->getTotalNodalSz(), 0);
    std::copy_n(localVec.begin(), ndofs * localSz, ghostedVec.begin() + ndofs * localBegin);
    da->template readFromGhostBegin<DT>(ghostedVec.data(), ndofs);
    da->template readFromGhostEnd<DT>(ghostedVec.data(), ndofs);
    return ghostedVec;
  }

  template <unsigned int dim, typename DT>
  std::vector<DT> ghostWriteAccumulate(const DA<dim> *da, const int ndofs, const std::vector<DT> &ghostedVec)
  {
    const bool useAccumulate = true;
    const size_t localBegin = da->getLocalNodeBegin();
    const size_t localEnd = da->getLocalNodeEnd();
    std::vector<DT> localVec(ghostedVec.begin(), ghostedVec.end());
    da->template writeToGhostsBegin<VECType>(localVec.data(), ndofs);
    da->template writeToGhostsEnd<VECType>(localVec.data(), ndofs, useAccumulate);
    localVec.erase(localVec.begin() + ndofs * localEnd, localVec.end());
    localVec.erase(localVec.begin(), localVec.begin() + ndofs * localBegin);
    return localVec;
  }

  template <unsigned int dim, typename DT>
  std::vector<DT> ghostWriteOverwrite(const DA<dim> *da, const int ndofs, const std::vector<DT> &ghostedVec)
  {
    const bool useAccumulate = false;
    const size_t localBegin = da->getLocalNodeBegin();
    const size_t localEnd = da->getLocalNodeEnd();
    std::vector<DT> localVec(ghostedVec.begin(), ghostedVec.end());
    da->template writeToGhostsBegin<VECType>(localVec.data(), ndofs);
    da->template writeToGhostsEnd<VECType>(localVec.data(), ndofs, useAccumulate);
    localVec.erase(localVec.begin() + ndofs * localEnd, localVec.end());
    localVec.erase(localVec.begin(), localVec.begin() + ndofs * localBegin);
    return localVec;
  }
}