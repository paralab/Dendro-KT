/**
 * GMG v-cycle on uniform grid with AMat + Subset interface.
 */

#include <iostream>
#include <mpi.h>
#include <stdio.h>

// Dendro
#include "dendro.h"
#include "oda.h"
#include "subset.h"
#include "octUtils.h"
#include "point.h"
#include "poissonMat.h"
#include "poissonVec.h"
#include "subset_amat.h"
#include "gridWrapper.h"
#include "vector.h"
#include "petscVector.h"
/// #include "poissonGMG.h"
// bug: if petsc is included (with logging)
//  before dendro parUtils then it triggers
//  dendro macros that use undefined variables.


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

constexpr int DIM = 2;

typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>,
                  double, unsigned long, unsigned int>  aMatFree;
typedef par::Maps<double, unsigned long, unsigned int>  aMatMaps;

using idx::LocalIdx;
using idx::GhostedIdx;
using ot::LocalVector;
using ot::GhostedVector;
using ot::LocalPetscVector;
using GridWrapper = ot::GridWrapper<DIM>;

template <typename NodeT> NodeT * ptr(std::vector<NodeT> &vec);
template <typename NodeT> NodeT * ptr(LocalVector<NodeT> &vec);
template <typename NodeT> NodeT * ptr(GhostedVector<NodeT> &vec);
template <typename NodeT> const NodeT * const_ptr(const std::vector<NodeT> &vec);
template <typename NodeT> const NodeT * const_ptr(const LocalVector<NodeT> &vec);
template <typename NodeT> const NodeT * const_ptr(const GhostedVector<NodeT> &vec);

template <unsigned int dim, typename DT>
GhostedVector<DT> ghostRead( const ot::GridWrapper<dim> &grid,
                             const int ndofs,
                             const LocalVector<DT> &localVec);

template <unsigned int dim, typename DT>
LocalVector<DT> ghostWriteAccumulate( const ot::GridWrapper<dim> &grid,
                                      const int ndofs,
                                      const GhostedVector<DT> &ghostedVec);

template <unsigned int dim, typename DT>
LocalVector<DT> ghostWriteOverwrite( const ot::GridWrapper<dim> &grid,
                                     const int ndofs,
                                     const GhostedVector<DT> &ghostedVec);

template <typename C, unsigned int dim>
std::vector<bool> boundaryFlags(
    const ot::DistTree<C, dim> &distTree,
    const int stratum = 0);

ot::LocalSubset<DIM> subsetAll(const GridWrapper &grid);
ot::LocalSubset<DIM> subsetBoundary(const GridWrapper &grid);

template <int dim>
ibm::DomainDecider domainChannel(int widthLevel);


void allocAMatMaps(const GridWrapper &grid, aMatMaps* &maps, const double *ghosted_bdry, int ndofs);
void createAMatFree(const GridWrapper &grid, aMatFree* &amatFree, aMatMaps* &maps);
void createAMatFree(const ot::LocalSubset<DIM> &subset, aMatFree* &amatFree, aMatMaps* &maps);

PoissonEq::PoissonMat<DIM>
createPoissonMat(
    const GridWrapper &grid, const AABB<DIM> &bounds, const double *ghosted_bdry, int ndofs);
PoissonEq::PoissonVec<DIM>
createPoissonVec(
    const GridWrapper &grid, const AABB<DIM> &bounds, int ndofs);


template <typename NodeT>
std::ostream & printLocal(const ot::DA<DIM> *da, const NodeT *vec, std::ostream & out = std::cout);
template <typename NodeT>
std::ostream & printGhosted(const ot::DA<DIM> *da, const NodeT *vec, std::ostream & out = std::cout);


class CoarseSolver
{
  protected:
    CoarseSolver() = default;

    aMatFree *m_amat;
    KSP m_ksp;
    LocalPetscVector<double> m_Mfrhs, m_ux;
    int m_ndofs;
    double m_tol;
    int m_max_iter;

    bool nonnull() const { return m_amat != nullptr; }

  public:
    // Expects aMatFree is parallel, and has been assembled and finalized.
    CoarseSolver(const GridWrapper &grid, aMatFree *aMatFree, int ndofs, double tol, int max_iter);

    int solve(const LocalVector<double> &rhs, LocalVector<double> &sol);

    ~CoarseSolver();
};



template <int dim = DIM>
class HybridGridChain
{
  protected:
    // GridWrapper provides simplified access to DistTree and DA.
    std::vector<GridWrapper> m_grid;
    std::vector<GridWrapper> m_surrGrid;
    std::vector<ot::LocalSubset<dim>> m_subsets;
    std::vector<ot::LocalSubset<dim>> m_surrSubsets;

  public:
    HybridGridChain(
        const ot::DistTree<unsigned int, dim> *dtree,
        const ot::DistTree<unsigned int, dim> *surrDtree,
        const ot::MultiDA<dim> *multiDA,
        const ot::MultiDA<dim> *surrMultiDA,
        size_t numGrids);

    const std::vector<GridWrapper> & grid()     const { return m_grid; }
    const std::vector<GridWrapper> & surrGrid() const { return m_surrGrid; }
    const GridWrapper & grid(int s)     const { return grid()[s]; }
    const GridWrapper & surrGrid(int s) const { return surrGrid()[s]; }

    const ot::LocalSubset<dim> & subset(int s)     const { return m_subsets[s]; }
    const ot::LocalSubset<dim> & surrSubset(int s) const { return m_surrSubsets[s]; }

    void fineNonGeom(ot::LocalSubset<dim> &&subset);

  protected:
};


template <unsigned int dim>
class Restriction
{
  protected:
    Restriction() = default;
    const HybridGridChain<dim> *m_chain;
    int m_fine_stratum;

  public:
    Restriction(const HybridGridChain<dim> *chain, int fine_stratum)
      : m_chain(chain), m_fine_stratum(fine_stratum) { }
    //TODO create amat fineGrid->fineGrid

    template <typename DofT>
    ot::LocalVector<DofT> operator()(const ot::LocalVector<DofT> &fineVec);

    template <typename DofT>
    void operator()(ot::LocalVector<DofT> &coarseVec, const ot::LocalVector<DofT> &fineVec);

  protected:
};


template <unsigned int dim>
class Prolongation
{
  protected:
    Prolongation() = default;
    const HybridGridChain<dim> *m_chain;
    int m_fine_stratum;

  public:
    Prolongation(const HybridGridChain<dim> *chain, int fine_stratum)
      : m_chain(chain), m_fine_stratum(fine_stratum) { }
    //TODO create amat fineGrid->fineGrid

    template <typename DofT>
    ot::LocalVector<DofT> operator()(const ot::LocalVector<DofT> &coarseVec);

    template <typename DofT>
    void operator()(ot::LocalVector<DofT> &fineVec, const ot::LocalVector<DofT> &coarseVec);

  protected:
};


template <unsigned int dim>
template <typename DofT>
ot::LocalVector<DofT> Restriction<dim>::operator()(
    const ot::LocalVector<DofT> &fineVec)
{
  const GridWrapper &coarseGrid = m_chain->grid(m_fine_stratum + 1);
  ot::LocalVector<DofT> coarseVec(coarseGrid, fineVec.ndofs());
  this->operator()(coarseVec, fineVec);
  return coarseVec;
}


template <unsigned int dim>
template <typename DofT>
void Restriction<dim>::operator()(
    ot::LocalVector<DofT> &coarseVec, const ot::LocalVector<DofT> &fineVec)
{
  // Duplicate inputs, write to owned node outputs.

  assert(coarseVec.ndofs() == fineVec.ndofs());
  const GridWrapper &surrGrid = m_chain->surrGrid(m_fine_stratum + 1);
  ot::LocalVector<DofT> surrVec(surrGrid, fineVec.ndofs());

  //TODO

  // Geom


  // NonGeom

}


template <unsigned int dim>
template <typename DofT>
ot::LocalVector<DofT> Prolongation<dim>::operator()(
    const ot::LocalVector<DofT> &coarseVec)
{
  const GridWrapper &fineGrid = m_chain->grid(m_fine_stratum);
  ot::LocalVector<DofT> fineVec(fineGrid, coarseVec.ndofs());
  this->operator()(fineVec, coarseVec);
  return fineVec;
}


template <unsigned int dim>
template <typename DofT>
void Prolongation<dim>::operator()(
    ot::LocalVector<DofT> &fineVec, const ot::LocalVector<DofT> &coarseVec)
{
  // Duplicate inputs, write to owned node outputs.

  assert(coarseVec.ndofs() == fineVec.ndofs());
  const GridWrapper &surrGrid = m_chain->surrGrid(m_fine_stratum + 1);
  ot::LocalVector<DofT> surrVec(surrGrid, fineVec.ndofs());

  // Geom


  // NonGeom

}


template <typename DofT>
DofT coordinateProduct(const GridWrapper &mesh,
                       const LocalVector<DofT> &vecA,
                       const LocalVector<DofT> &vecB);

template <typename DofT>
LocalVector<DofT> pointwiseProduct(
    const GridWrapper &mesh,
    const LocalVector<DofT> &vecA,
    const LocalVector<DofT> &vecB);

template <typename DofT>
DofT mpi_sum(const LocalVector<DofT> &vec, MPI_Comm comm);


//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = PETSC_COMM_WORLD;
  const int eleOrder = 2;
  const double sfc_tol = 0.1;
  using uint = unsigned int;
  using DTree_t = ot::DistTree<uint, DIM>;
  using DA_t = ot::DA<DIM>;

  const int fineLevel = 3;
  const size_t dummyInt = 100;
  const size_t singleDof = 1;

  const int numGrids = 2;

  int commSize, commRank;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  // Domain scaling.  (compatible with nodeCoord())
  /// AABB<DIM> bounds(Point<DIM>(-1.0), Point<DIM>(1.0));
  AABB<DIM> bounds(Point<DIM>(-0.5), Point<DIM>(0.5));

  // Functions.
  //
  // u_exact function
  const double coefficient[] = {1, 2, 5, 3};  // up to 4D allowed.
  const double sum_coeff = std::accumulate(coefficient, coefficient + DIM, 0);
  const auto u_exact = [=] (const double *x) {
    double expression = 1;
    for (int d = 0; d < DIM; ++d)
      expression += coefficient[d] * x[d] * x[d];
    return expression;
  };
  // ... is the solution to -div(grad(u)) = f, where f is
  const auto f = [=] (const double *x) {
    return -2*sum_coeff;
  };
  // ... and boundary is prescribed (matching u_exact)
  const auto u_bdry = [=] (const double *x) {
    return u_exact(x);
  };


  // Fine grid (uniform).
  /// DTree_t dtree = DTree_t::constructSubdomainDistTree(fineLevel, comm, sfc_tol);
  DTree_t dtree = DTree_t::constructSubdomainDistTree(fineLevel, domainChannel<DIM>(3), comm, sfc_tol);

  // Define coarse grids based on the fine grid.
  // Also need surrogate grids.
  DTree_t surrogateDTree =
    dtree.generateGridHierarchyUp(true, numGrids, sfc_tol);

  // Create multi-level DA.
  ot::MultiDA<DIM> multiDA, surrMultiDA;
  const ot::GridAlignment gridAlignment = ot::GridAlignment::CoarseByFine;
  DA_t::multiLevelDA(multiDA, dtree, comm, eleOrder, 100, sfc_tol);
  DA_t::multiLevelDA(surrMultiDA, surrogateDTree, comm, eleOrder, 100, sfc_tol);

  std::cout << "dtree number of grids == " << dtree.getNumStrata() << "\n";
  std::cout << "fine grid number of elements == " << multiDA[0].getGlobalElementSz() << "\n";

  HybridGridChain<DIM> chain(
      &dtree, &surrogateDTree, &multiDA, &surrMultiDA, numGrids);

  const GridWrapper & fineGrid = chain.grid()[0];
  const std::vector<size_t> &localBdryIdx = fineGrid.da()->getBoundaryNodeIndices();
  const std::vector<size_t> &ghostedBdryIdx = fineGrid.da()->getGhostedBoundaryNodeIndices();
  const size_t ghostedBdrySz = ghostedBdryIdx.size();

  // Exact numerical solution to later evaluate error.
  LocalVector<double> u_exact_vec(fineGrid, singleDof);
  for (size_t ii = 0; ii < u_exact_vec.size(); ++ii)
    u_exact_vec[LocalIdx(ii)] = u_exact(
        fineGrid.nodeCoord(LocalIdx(ii), bounds).data());

  const auto sol_err_linf = [=, &u_exact_vec](const LocalVector<double> &approx)
  {
      double err_max = 0.0;
      for (size_t ii = 0; ii < u_exact_vec.size(); ++ii)
      {
        const double err_ii = abs(
            approx[LocalIdx(ii)] -
            u_exact_vec[LocalIdx(ii)] );
        err_max = fmax(err_max, err_ii);
      }
      double glob_err_max = 0.0;
      par::Mpi_Allreduce(&err_max, &glob_err_max, 1, MPI_MAX, comm);
      return glob_err_max;
  };


  LocalVector<double> u_vec(fineGrid, singleDof);
  LocalVector<double> f_vec(fineGrid, singleDof);

  // Initialize u=Dirichlet  and  f={f function}
  for (size_t ii = 0; ii < u_vec.size(); ++ii)
    u_vec[LocalIdx(ii)] = 0;
  for (size_t bdryIdx : localBdryIdx)
    u_vec[LocalIdx(bdryIdx)] = u_bdry(
        fineGrid.nodeCoord(LocalIdx(bdryIdx), bounds).data());

  for (size_t ii = 0; ii < f_vec.size(); ++ii)
    f_vec[LocalIdx(ii)] = f( fineGrid.nodeCoord(LocalIdx(ii), bounds).data() );

  std::vector<double> ghosted_bdry(ghostedBdrySz);
  for (size_t bii = 0; bii < ghostedBdrySz; ++bii)
    ghosted_bdry[bii] = u_bdry(
        fineGrid.nodeCoord(GhostedIdx(ghostedBdryIdx[bii]), bounds).data());


  const double tol=1e-12;
  const unsigned int max_iter=50;

  // Discretized operators for fine grid.
  PoissonEq::PoissonMat<DIM> poissonMat =
      createPoissonMat(fineGrid, bounds, const_ptr(ghosted_bdry), singleDof);
  PoissonEq::PoissonVec<DIM> poissonVec =
      createPoissonVec(fineGrid, bounds, singleDof);

  // Compute rhs of weak formulation.
  LocalVector<double> rhs_vec(fineGrid, singleDof);
  poissonVec.computeVec(const_ptr(f_vec), ptr(rhs_vec));

  // Elemental operators are defined for the finest grid.
  // We will derive the coarser grid operators as
  // the Galerkin Coarse Grid Operator.

  // Multi-level aMat
  std::vector<aMatMaps*> meshMaps(numGrids, nullptr);
  std::vector<aMatFree*> amatFree(numGrids, nullptr);

          // Happens to be all for this test. Future: boundary elems.
  chain.fineNonGeom(subsetAll(fineGrid));

  /// chain.fineNonGeom(subsetBoundary(fineGrid));
  const ot::LocalSubset<DIM> & subset = chain.subset(0);  // nongeom

  // Create the fine-grid aMat directly.
  /// allocAMatMaps(fineGrid, meshMaps[0], const_ptr(ghosted_bdry), singleDof);
  /// createAMatFree(fineGrid, amatFree[0], meshMaps[0]);
  ot::allocAMatMaps(subset, meshMaps[0], const_ptr(ghosted_bdry), singleDof);
  createAMatFree(subset, amatFree[0], meshMaps[0]);

  // Store Eigen matrices for later, before assemble.
  std::vector<fem::EigenMat> e_mats = fem::subsetEigens(subset, poissonMat);

  // Assemble for aMat.
  fem::getAssembledAMat(e_mats, amatFree[0]);
  if (amatFree[0])
    amatFree[0]->finalize();

  /// CoarseSolver coarse_solver(fineGrid, amatFree[0], singleDof, tol, max_iter);  // TODO coarse amat, not fine
  /// int iteration = coarse_solver.solve(rhs_vec, u_vec);
  /// printf("[iteration=%2d]  |error|=%.2e\n",
  ///     iteration, sol_err_linf(u_vec));


  // 2-grid method
  const int nvcycle = 1;
  for (int cycle = 0; cycle < nvcycle; ++cycle)
  {
    /// printf("[cycle=%2d]  |error|=%.2e\n",
    ///     cycle, sol_err_linf(u_vec));

    // (prototype for vcycle)
    //
    // Matvec: Ghost read, split, op, unsplit, ghost write
    GhostedVector<double> u_gh = ghostRead(fineGrid, singleDof, u_vec);
    GhostedVector<double> v_gh(fineGrid, singleDof);
    // First subset --
    std::vector<double> u_sub = ot::gather_ndofs(const_ptr(u_gh), subset, singleDof);
    std::vector<double> v_sub(u_sub.size(), 0);
        // matvec(): ghosted=true if created(DA), ghosted=false if created(subset)
    if (amatFree[0])
      amatFree[0]->matvec(ptr(v_sub), const_ptr(u_sub), false);
    ot::scatter_ndofs_accumulate(const_ptr(v_sub), ptr(v_gh), subset, singleDof);
    // -- If there was a second subset, do the same with it
    //
    LocalVector<double> v_vec = ghostWriteAccumulate(fineGrid, singleDof, v_gh);
    /// printLocal(fineGrid.da(), const_ptr(v_vec));

    LocalVector<double> residual = v_vec - rhs_vec;

    using Poisson = PoissonEq::PoissonMat<DIM>;

    Restriction<DIM> restriction(&chain, 0);
    Prolongation<DIM> prolongation(&chain, 0);
    /// Operator<Poisson> coarse_op = galerkinCoarseOp(fine_op, restriction, prolongation);


    // test: [P(err_2h), res_h] == [err_2h, R(res_h)]
    LocalVector<double> rhs_2h = restriction(residual);
    /// LocalVector<double> err_2h = solve_2h(grid[1], rhs_2h); //TODO need operators..
    LocalVector<double> err_2h_dummy(chain.grid(1), singleDof);
    std::fill(err_2h_dummy.data().begin(), err_2h_dummy.data().end(), 1);
    LocalVector<double> err_h = prolongation(err_2h_dummy);

    /// const double product_p = coordinateProduct(fineGrid, err_h, residual);
    /// const double product_r = coordinateProduct(chain.grid(1), err_2h_dummy, rhs_2h);

    const LocalVector<double> v_product_p = pointwiseProduct(fineGrid, err_h, residual);
    const LocalVector<double> v_product_r = pointwiseProduct(chain.grid(1), err_2h_dummy, rhs_2h);
    const double product_p = mpi_sum(v_product_p, fineGrid.da()->getGlobalComm());
    const double product_r = mpi_sum(v_product_r, chain.grid(1).da()->getGlobalComm());

    if (product_p == product_r)
      printf("product_p " GRN "[%.3f] == [%.3f]" NRM " product_r\n", product_p, product_r);
    else
    {
      printf("product_p " RED "[%.3f] != [%.3f]" NRM " product_r\n", product_p, product_r);

      printf("\nproduct_p:\n");
      printLocal(fineGrid.da(), const_ptr(v_product_p));
      printf("\nproduct_r:\n");
      printLocal(chain.grid(1).da(), const_ptr(v_product_r));
    }

    /// u_vec = u_vec - err_h;
  }



  printf("[cycle=%2d]  |error|=%.2e\n",
      nvcycle, sol_err_linf(u_vec));

  _DestroyHcurve();
  DendroScopeEnd();
  PetscFinalize();
  return 0;
}//main()



// --------------------------------------------------
// Definitions of helper functions in driver.
// --------------------------------------------------

// ptr() std::vector
template <typename NodeT>
NodeT * ptr(std::vector<NodeT> &vec)
{
  return vec.data();
}

// ptr() LocalVector
template <typename NodeT>
NodeT * ptr(LocalVector<NodeT> &vec)
{
  return vec.data().data();
}

// ptr() GhostedVector
template <typename NodeT>
NodeT * ptr(GhostedVector<NodeT> &vec)
{
  return vec.data().data();
}

// const_ptr() std::vector
template <typename NodeT>
const NodeT * const_ptr(const std::vector<NodeT> &vec)
{
  return vec.data();
}

// const_ptr() LocalVector
template <typename NodeT>
const NodeT * const_ptr(const LocalVector<NodeT> &vec)
{
  return vec.data().data();
}

// const_ptr() GhostedVector
template <typename NodeT>
const NodeT * const_ptr(const GhostedVector<NodeT> &vec)
{
  return vec.data().data();
}


// ghostRead()
template <unsigned int dim, typename DT>
GhostedVector<DT> ghostRead(const ot::GridWrapper<dim> &grid, const int ndofs, const LocalVector<DT> &localVec)
{
  LocalVector<DT> locCopy = localVec;
  GhostedVector<DT> ghCopy = ot::makeGhosted(grid, std::move(locCopy));
  grid.da()->template readFromGhostBegin<DT>(ghCopy.data().data(), ndofs);
  grid.da()->template readFromGhostEnd<DT>(ghCopy.data().data(), ndofs);
  return ghCopy;
}

// ghostWriteAccumulate()
template <unsigned int dim, typename DT>
LocalVector<DT> ghostWriteAccumulate(const ot::GridWrapper<dim> &grid, const int ndofs, const GhostedVector<DT> &ghostedVec)
{
  const bool useAccumulate = true;
  GhostedVector<DT> ghCopy = ghostedVec;
  grid.da()->template writeToGhostsBegin<VECType>(ghCopy.data().data(), ndofs);
  grid.da()->template writeToGhostsEnd<VECType>(ghCopy.data().data(), ndofs, useAccumulate);
  return ot::makeLocal(grid, std::move(ghCopy));
}

// ghostWriteOverwrite()
template <unsigned int dim, typename DT>
LocalVector<DT> ghostWriteOverwrite(const ot::GridWrapper<dim> &grid, const int ndofs, const GhostedVector<DT> &ghostedVec)
{
  const bool useAccumulate = false;
  GhostedVector<DT> ghCopy = ghostedVec;
  grid.da()->template writeToGhostsBegin<VECType>(ghCopy.data().data(), ndofs);
  grid.da()->template writeToGhostsEnd<VECType>(ghCopy.data().data(), ndofs, useAccumulate);
  return ot::makeLocal(grid, std::move(ghCopy));
}


// boundaryFlags()
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


// subsetBoundary()
ot::LocalSubset<DIM> subsetBoundary(const GridWrapper &grid)
{
  return ot::LocalSubset<DIM>(
      grid.da(),
      &grid.octList(),
      boundaryFlags(*grid.distTree(), grid.stratum()),
      true);
}

// subsetAll()
ot::LocalSubset<DIM> subsetAll(const GridWrapper &grid)
{
  return ot::LocalSubset<DIM>(
      grid.da(),
      &grid.distTree()->getTreePartFiltered(),
      std::vector<bool>(grid.distTree()->getTreePartFiltered().size(), true),
      true);
}


// domainChannel()
template <int dim>
ibm::DomainDecider domainChannel(int widthLevel)
{
  const double width = 1.0 / (1u << widthLevel);
  std::array<double, dim> sizes;
  sizes.fill(width);
  sizes[0] = 1.0;
  return typename ot::DistTree<uint, dim>::BoxDecider(sizes);
}



// allocAMatMaps()
void allocAMatMaps(const GridWrapper &grid, aMatMaps* &maps, const double *ghosted_bdry, int ndofs)
{
  grid.da()->allocAMatMaps(
      maps,
      grid.distTree()->getTreePartFiltered(grid.stratum()),
      ghosted_bdry,
      ndofs);
}

// createAMatFree()
void createAMatFree(const GridWrapper &grid, aMatFree* &amatFree, aMatMaps* &maps)
{
  grid.da()->createAMat(amatFree, maps);
  if (amatFree)
    amatFree->set_matfree_type((par::MATFREE_TYPE)1);
}

// createAMatFree()
void createAMatFree(const ot::LocalSubset<DIM> &subset, aMatFree* &amatFree, aMatMaps* &maps)
{
  ot::createAMat(subset, amatFree, maps);
  if (amatFree)
    amatFree->set_matfree_type((par::MATFREE_TYPE)1);
}


// createPoissonMat()
PoissonEq::PoissonMat<DIM> createPoissonMat(
    const GridWrapper &grid,
    const AABB<DIM> &bounds,
    const double *ghosted_bdry,
    int ndofs)
{
  PoissonEq::PoissonMat<DIM> poissonMat(
      grid.da(),
      &grid.distTree()->getTreePartFiltered(grid.stratum()),
      ndofs,
      ghosted_bdry);
  poissonMat.setProblemDimensions(bounds.min(), bounds.max());
  return poissonMat;
}

// createPoissonVec()
PoissonEq::PoissonVec<DIM> createPoissonVec(
    const GridWrapper &grid,
    const AABB<DIM> &bounds,
    int ndofs)
{
  PoissonEq::PoissonVec<DIM> poissonVec(
      const_cast<ot::DA<DIM>*>(grid.da()),// violate const for weird feVec non-const necessity
      &grid.distTree()->getTreePartFiltered(grid.stratum()),
      ndofs);
  poissonVec.setProblemDimensions(bounds.min(), bounds.max());
  return poissonVec;
}



// printLocal()
template <typename NodeT>
std::ostream & printLocal(const ot::DA<DIM> *da,
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

// printGhosted()
template <typename NodeT>
std::ostream & printGhosted(const ot::DA<DIM> *da,
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



template <typename DofT>
DofT coordinateProduct(const GridWrapper &mesh,
                       const LocalVector<DofT> &vecA,
                       const LocalVector<DofT> &vecB)
{
  DofT localProduct = 0.0f;
  const size_t localSize = vecA.size();
  for (size_t i = 0; i < localSize; ++i)
    localProduct += vecA[LocalIdx(i)] * vecB[LocalIdx(i)];

  DofT globalProduct = 0.0f;
  MPI_Comm comm = mesh.da()->getGlobalComm();
  par::Mpi_Allreduce(&localProduct, &globalProduct, 1, MPI_SUM, comm);

  return globalProduct;
}

template <typename DofT>
LocalVector<DofT> pointwiseProduct(
    const GridWrapper &mesh,
    const LocalVector<DofT> &vecA,
    const LocalVector<DofT> &vecB)
{
  LocalVector<DofT> product(mesh, vecA.ndofs());
  const size_t localSize = vecA.size();
  for (size_t i = 0; i < localSize; ++i)
    product[LocalIdx(i)] = vecA[LocalIdx(i)] * vecB[LocalIdx(i)];
  return product;
}


template <typename DofT>
DofT mpi_sum(const LocalVector<DofT> &vec, MPI_Comm comm)
{
  DofT localSum = 0.0f;
  const size_t localSize = vec.size();
  for (size_t i = 0; i < localSize; ++i)
    localSum += vec[LocalIdx(i)];

  DofT globalSum = 0.0f;
  par::Mpi_Allreduce(&localSum, &globalSum, 1, MPI_SUM, comm);
  return globalSum;
}






// CoarseSolver::CoarseSolver()
CoarseSolver::CoarseSolver(const GridWrapper &grid, aMatFree *amat, int ndofs, double tol, int max_iter)
  :
    m_amat(amat), m_tol(tol), m_max_iter(max_iter),
    m_ndofs(ndofs),
    m_Mfrhs(grid, ndofs), m_ux(grid, ndofs)
{
  if (this->nonnull())
  {
    // PETSc solver context: Create and set KSP
    KSPCreate(m_amat->get_comm(), &m_ksp);
    KSPSetType(m_ksp, KSPCG);
    KSPSetFromOptions(m_ksp);

    // Set tolerances.
    KSPSetTolerances(m_ksp, m_tol, PETSC_DEFAULT, PETSC_DEFAULT, m_max_iter);

    // Set operators.
    Mat petscMat = m_amat->get_matrix();
    KSPSetOperators(m_ksp, petscMat, petscMat);

    // Set preconditioner.
    PC pc;
    KSPGetPC(m_ksp, &pc);
    PCSetType(pc, PCJACOBI);
    PCSetFromOptions(pc);
  }
}


// CoarseSolver::solve()
int CoarseSolver::solve(const LocalVector<double> &rhs, LocalVector<double> &sol)
{
  PetscInt numIterations = -1;

  if (this->nonnull())
  {
    assert(rhs.ndofs() == m_ndofs);
    assert(sol.ndofs() == m_ndofs);

    // Compensate r.h.s. of weak formulation. Amat subtracts boundary from rhs.
    copyToPetscVector(m_Mfrhs, rhs);
    m_amat->apply_bc(m_Mfrhs.vec());

    VecAssemblyBegin(m_Mfrhs.vec());
    VecAssemblyEnd(m_Mfrhs.vec());

    // Copy vectors to Petsc vectors.
    copyToPetscVector(m_ux, sol);

    // Solve the system.
    KSPSolve(m_ksp, m_Mfrhs.vec(), m_ux.vec());
    VecAssemblyBegin(m_ux.vec());
    VecAssemblyEnd(m_ux.vec());

    KSPGetIterationNumber(m_ksp, &numIterations);

    // Copy back from Petsc vector.
    copyFromPetscVector(sol, m_ux);
  }

  return numIterations;
}


// CoarseSolver::~CoarseSolver()
CoarseSolver::~CoarseSolver()
{
  KSPDestroy(&m_ksp);
}



// HybridGridChain::HybridGridChain()
template <int dim>
HybridGridChain<dim>::HybridGridChain(
        const ot::DistTree<unsigned int, dim> *dtree,
        const ot::DistTree<unsigned int, dim> *surrDtree,
        const ot::MultiDA<dim> *multiDA,
        const ot::MultiDA<dim> *surrMultiDA,
        size_t numGrids)
{
  for (int s = 0; s < numGrids; ++s)
  {
    m_grid.emplace_back(dtree, multiDA, s);
    m_surrGrid.emplace_back(surrDtree, surrMultiDA, s);

    m_subsets.emplace_back(ot::LocalSubset<dim>::makeNone(grid(s).da()));
    m_surrSubsets.emplace_back(ot::LocalSubset<dim>::makeNone(surrGrid(s).da()));
  }
}


// HybridGridChain::fineNonGeom()
template <int dim>
void HybridGridChain<dim>::fineNonGeom(ot::LocalSubset<dim> &&subset)
{
  m_subsets[0] = subset;

  int commSize, commRank;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  for (int s = 1; s < m_grid.size(); ++s)
  {
    const ot::LocalSubset<dim> &fineSubset = m_subsets[s-1];
    const ot::LocalSubset<dim> &surrSubset = m_surrSubsets[s];
    const ot::LocalSubset<dim> &coarseSubset = m_subsets[s];
    const GridWrapper &surrGrid = m_surrGrid[s];
    const GridWrapper &coarseGrid = m_grid[s];

    const auto overlaps = [&](size_t s, size_t f) {
      return surrGrid.octList()[s].isAncestorInclusive(
          fineSubset.relevantOctList()[f]);
    };

    std::vector<char> surrIncluded(surrGrid.numElements(), false);
    std::vector<char> coarseIncluded(coarseGrid.numElements(), false);

    const size_t fineSize = fineSubset.getLocalElementSz();
    for (size_t fine_i = 0,  surr_i = 0;
             fine_i < fineSize
         and surr_i < surrIncluded.size(); )
    {
      if (overlaps(surr_i, fine_i))
        surrIncluded[surr_i] = true;
      while (fine_i < fineSize and overlaps(surr_i, fine_i))
        ++fine_i;
      ++surr_i;
    }

    par::shift(
        coarseGrid.da()->getGlobalComm(),
        surrIncluded.data(),
        surrIncluded.size(),
        surrGrid.da()->getGlobalElementBegin(),
        coarseIncluded.data(),
        coarseIncluded.size(),
        coarseGrid.da()->getGlobalElementBegin(),
        1);

    m_subsets[s] = ot::LocalSubset<dim>(
        coarseGrid.da(), &coarseGrid.octList(), coarseIncluded, char(true));
    m_surrSubsets[s] = ot::LocalSubset<dim>(
        surrGrid.da(), &surrGrid.octList(), surrIncluded, char(true));
  }
}









