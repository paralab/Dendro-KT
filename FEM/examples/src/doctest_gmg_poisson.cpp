//
// Created by masado on 10/19/22.
//

#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro
#include <mpi.h>

#include <include/treeNode.h>
#include <include/tsort.h>
#include <include/distTree.h> // for convenient uniform grid partition

#include <include/dendro.h>
#include <include/octUtils.h>
#include <FEM/include/refel.h>
#include <FEM/examples/include/poissonMat.h>
#include <FEM/examples/include/poissonVec.h>

#include <vector>
#include <functional>


// -----------------------------
// TODO
// ✔ Define VectorPool member functions
// ✔ Define smoother (start with Jacobi smoother)
// ✔ Define restriction (adapt from gmgMat)
// _ Define prolongation (follow restriction, adapt from gmgMat)
// _ Finish vcycle() definition
// _ Refactor MatVec to wrap a new MatVecGhosted, which can be called in the solver
// _ future: 4th kind Chebyshev smoother
// -----------------------------

// -----------------------------
// Typedefs
// -----------------------------
using uint = unsigned int;
using LLU = long long unsigned;

template <int dim>
using Oct = ot::TreeNode<uint, dim>;

template <unsigned int dim, typename NodeT>
std::ostream & print(const ot::DA<dim> &da,
                     const std::vector<NodeT> &local_vec,
                     std::ostream & out = std::cerr);

// -----------------------------
// Structs
// -----------------------------

// DA_Pair
template <int dim>
struct DA_Pair
{
  const ot::DA<dim> *primary;
  const ot::DA<dim> *surrogate;
};

#include <map>

template <typename X>
class VectorPool
{
  public:
    VectorPool() = default;

    ~VectorPool() { std::cerr << "checkouts==" << m_log_checkouts << " \tallocs==" << m_log_allocs << "\n"; }

    void free_all();
    std::vector<X> checkout(size_t size);
    void checkin(std::vector<X> &&vec);
      //idea: multimap find, assign moved, erase empty, return assigned; if not found, return new.

    VectorPool(VectorPool && other) : m_idle(std::move(other.m_idle)) {}
    VectorPool & operator=(VectorPool &&other) { m_idle = std::move(other.m_idle); return *this; }
    VectorPool(const VectorPool &) = delete;
    VectorPool & operator=(const VectorPool &) = delete;

  private:
    std::multimap<int, std::vector<X>> m_idle;
    long long unsigned m_log_checkouts = 0;
    long long unsigned m_log_allocs = 0;
};

template <typename X>
void VectorPool<X>::free_all()
{
  m_idle.clear();
}

template <typename X>
std::vector<X> VectorPool<X>::checkout(size_t size)
{
  ++m_log_checkouts;
  auto it = m_idle.find(size);
  if (it == m_idle.end())
  {
    ++m_log_allocs;
    return std::vector<X>(size);
  }
  std::vector<X> extracted = std::move(it->second);
  m_idle.erase(it);
  return extracted;
}

template <typename X>
void VectorPool<X>::checkin(std::vector<X> &&vec)
{
  m_idle.insert({vec.size(), std::move(vec)});
}

// -----------------------------
// Functions
// -----------------------------

// cgSolver()
template <unsigned int dim, typename MatMult>
int cgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, VECType * u, const VECType * rhs, int max_iter, double relResErr);

// matvec_base_in()
template <typename X, unsigned dim>
ot::MatvecBaseIn<dim, X> matvec_base_in(
    const ot::DA<dim> *da, int ndofs, const X *ghosted, int extra_depth = 0);

// matvec_base_ou_accumulatet()
template <typename X, unsigned dim>
ot::MatvecBaseOut<dim, X, true> matvec_base_out_accumulate(
    const ot::DA<dim> *da, int ndofs, int extra_depth = 0);

// restrict_fine_to_coarse()
template <int dim>
void restrict_fine_to_coarse(
    DA_Pair<dim> fine_da, const VECType *fine_vec_ghosted,
    DA_Pair<dim> coarse_da, VECType *coarse_vec_ghosted,
    int ndofs,
    VectorPool<VECType> &vector_pool)
{
  const unsigned int nPe = fine_da.primary->getNumNodesPerElement();

  std::vector<VECType> coarse_surr_ghosted = vector_pool.checkout(
      coarse_da.surrogate->getTotalNodalSz() * ndofs);

  std::vector<VECType> leafBuffer = vector_pool.checkout(ndofs * nPe);
  leafBuffer.assign(leafBuffer.size(), 42);

  // Surrogate is coarse grid partitioned by fine
  // Interpolate^T from the fine primary grid to coarse surrogate.

  // readFromGhost*(fine_vec_ghosted) must precede restrict_fine_to_coarse().

  // Fine ghosted elemental owners.
  using OwnershipT = DendroIntL;
  const OwnershipT * ownersGhostedPtr = fine_da.primary->getNodeOwnerElements();

  if (fine_da.primary->getLocalNodalSz() > 0)
  {
    // Index fine grid elements as we loop.
    OwnershipT globElementId = fine_da.primary->getGlobalElementBegin();

    unsigned eleOrder = fine_da.primary->getElementOrder();

    // Fine and coarse element-to-node loops.
    ot::MatvecBaseIn<dim, OwnershipT>
        loopOwners = matvec_base_in(fine_da.primary, ndofs, ownersGhostedPtr, 0);
    ot::MatvecBaseIn<dim, VECType>
        loopFine = matvec_base_in(fine_da.primary, ndofs, fine_vec_ghosted, 0);
    ot::MatvecBaseOut<dim, VECType, true>
        loopCoarse = matvec_base_out_accumulate<VECType>(coarse_da.surrogate, ndofs, 1);

    // Traverse fine and coarse grids simultaneously.
    while (!loopFine.isFinished())
    {
      // Depth controlled by fine.
      if (loopFine.isPre() && loopFine.subtreeInfo().isLeaf())
      {
        const VECType * fineLeafIn = loopFine.subtreeInfo().readNodeValsIn();
        const OwnershipT * fineOwners = loopOwners.subtreeInfo().readNodeValsIn();
        for (size_t nIdx = 0; nIdx < nPe; ++nIdx)
        {
          if (loopFine.subtreeInfo().readNodeNonhangingIn()[nIdx])
          {
            // Only transfer a node to parent from the owning element.
            if (fineOwners[nIdx] == globElementId)
            {
              for (int dof = 0; dof < ndofs; ++dof)
                leafBuffer[ndofs * nIdx + dof] = fineLeafIn[ndofs * nIdx + dof];
            }
            else
            {
              for (int dof = 0; dof < ndofs; ++dof)
                leafBuffer[ndofs * nIdx + dof] = 0;
            }
          }
          else
          {
            for (int dof = 0; dof < ndofs; ++dof)
              leafBuffer[ndofs * nIdx + dof] = 0.0f;
          }
        }

        loopCoarse.subtreeInfo().overwriteNodeValsOut(leafBuffer.data());

        loopFine.next();
        loopCoarse.next();
        loopOwners.next();

        globElementId++;
      }
      else
      {
        loopFine.step();
        loopCoarse.step();
        loopOwners.step();
      }
    }
    const size_t writtenSz = loopCoarse.finalize(coarse_surr_ghosted.data());
  }

  // Coarse ghost write.
  coarse_da.surrogate->writeToGhostsBegin(coarse_surr_ghosted.data(), ndofs);
  coarse_da.surrogate->writeToGhostsEnd(coarse_surr_ghosted.data(), ndofs);

  // Shift in the coarse grid from surrogate to primary.
  ot::distShiftNodes(
      *coarse_da.surrogate,
      coarse_surr_ghosted.data() + coarse_da.surrogate->getLocalNodeBegin() * ndofs,
      *coarse_da.primary,
      coarse_vec_ghosted + coarse_da.primary->getLocalNodeBegin() * ndofs,
      ndofs);

  vector_pool.checkin(std::move(leafBuffer));
  vector_pool.checkin(std::move(coarse_surr_ghosted));
}

// prolongate_coarse_to_fine()
    /// VectorPool &vector_pool)


// -----------------------------
// Main test cases
// -----------------------------

MPI_TEST_CASE("Poisson problem on a uniformly refined cube with 5 processes, should converge to exact solution", 5)
{
  MPI_Comm comm = test_comm;

  DendroScopeBegin();
  constexpr int dim = 2;
  _InitializeHcurve(dim);

  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  using Oct = Oct<dim>;
  const double partition_tolerance = 0.1;
  const int polynomial_degree = 1;

  Point<dim> min_corner(-1.0);
  Point<dim> max_corner(1.0);

  // u_exact function
  const double coefficient[] = {1, 2, 5, 3};
  const double sum_coeff = std::accumulate(coefficient, coefficient + dim, 0);
  const auto u_exact = [=] (const double *x) {
    double expression = 1;
    for (int d = 0; d < dim; ++d)
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

  // Mesh (uniform grid)
  const int refinement_level = 7;
  ot::DistTree<uint, dim> base_tree = ot::DistTree<uint, dim>::
      constructSubdomainDistTree(refinement_level, comm, partition_tolerance);
  ot::DA<dim> base_da(base_tree, comm, polynomial_degree, int{}, partition_tolerance);

  // ghosted_node_coordinate()
  const auto ghosted_node_coordinate = [&](const ot::DA<dim> &da, size_t idx) -> Point<dim>
  {
    const ot::TreeNode<uint, dim> node = da.getTNCoords()[idx];   //integer
    std::array<double, dim> coordinates;
    ot::treeNode2Physical(node, polynomial_degree, coordinates.data());  //fixed

    for (int d = 0; d < dim; ++d)
      coordinates[d] = min_corner.x(d) * (1 - coordinates[d]) // scale and shift
                     + max_corner.x(d) * coordinates[d];

    return Point<dim>(coordinates);
  };

  // local_node_coordinate()
  const auto local_node_coordinate = [&](const ot::DA<dim> &da, size_t idx) -> Point<dim>
  {
    return ghosted_node_coordinate(da, idx + da.getLocalNodeBegin());
  };

  // local_vector()
  const auto local_vector = [&](const ot::DA<dim> &da, auto type, int dofs)
  {
    std::vector<std::remove_cv_t<decltype(type)>> local;
    const bool ghosted = false;  // local means not ghosted
    da.createVector(local, false, ghosted, dofs);
    return local;
  };

  // Vectors
  const int single_dof = 1;
  std::vector<double> u_vec = local_vector(base_da, double{}, single_dof);
  std::vector<double> v_vec = local_vector(base_da, double{}, single_dof);
  std::vector<double> f_vec = local_vector(base_da, double{}, single_dof);
  std::vector<double> rhs_vec = local_vector(base_da, double{}, single_dof);

  using PoissonMat = PoissonEq::PoissonMat<dim>;

  // fe_matrix()
  //  future: return PoissonMat without dynamic memory, after finish move constructors
  const auto fe_matrix = [&](const ot::DA<dim> &da) -> PoissonMat *
  {
    auto *mat = new PoissonMat(
        &da, &da.dist_tree()->getTreePartFiltered(), single_dof);
    mat->setProblemDimensions(min_corner, max_corner);
    return mat;
  };

  // fe_vector()
  const auto fe_vector = [&](const ot::DA<dim> &da) -> PoissonEq::PoissonVec<dim>
  {
    PoissonEq::PoissonVec<dim> vec(
        const_cast<ot::DA<dim> *>(&da),   // violate const for weird feVec non-const necessity
        &da.dist_tree()->getTreePartFiltered(),
        single_dof);
    vec.setProblemDimensions(min_corner, max_corner);
    return vec;
  };

  // Boundary condition and 0 on interior (needed to subtract A u^bdry from rhs)
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    u_vec[i] = 0;
  for (size_t bdyIdx : base_da.getBoundaryNodeIndices())
    u_vec[bdyIdx] = u_bdry(&local_node_coordinate(base_da, bdyIdx).x(0));

  // Evaluate forcing term.
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    f_vec[i] = f(&local_node_coordinate(base_da, i).x(0));

  // Linear system oracles encapsulating the operators.
  PoissonEq::PoissonMat<dim> &base_mat = *fe_matrix(base_da);  //future: not a pointer
  PoissonEq::PoissonVec<dim> base_vec = fe_vector(base_da);

  // Compute right hand side, subtracting boundary data.
  // (See preMatVec, postMatVec, preComputeVec, postComputeVec).
  base_mat.matVec(u_vec.data(), v_vec.data());
  base_vec.computeVec(f_vec.data(), rhs_vec.data());
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    rhs_vec[i] -= v_vec[i];
  base_mat.zero_boundary(true);

  // Precompute exact solution to evaluate error of an approximate solution.
  std::vector<double> u_exact_vec = local_vector(base_da, double{}, single_dof);
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    u_exact_vec[i] = u_exact(&local_node_coordinate(base_da, i).x(0));

  // Function to check solution error.
  const auto sol_err_max = [&](const std::vector<double> &u) {
      double err_max = 0.0;
      for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
        err_max = fmax(err_max, abs(u[i] - u_exact_vec[i]));
      err_max = par::mpi_max(err_max, comm);
      return err_max;
  };

  // Multigrid setup
  //  future: distCoarsen to coarsen by 1 or more levels
  //  future: coarse_to_fine and fine_to_coarse without surrogates
  //  future: Can do 2:1-balance all-at-once instead of iteratively.
  const int n_grids = 2;
  ot::DistTree<uint, dim> &trees = base_tree;
  ot::DistTree<uint, dim> surrogate_trees;
  surrogate_trees = trees.generateGridHierarchyUp(true, n_grids, partition_tolerance);
  assert(trees.getNumStrata() == n_grids);
  /// struct DA_Pair { const ot::DA<dim> *primary; const ot::DA<dim> *surrogate; };
  std::vector<DA_Pair<dim>> das(n_grids, DA_Pair<dim>{nullptr, nullptr});
  das[0].primary = &base_da;
  for (int g = 1; g < n_grids; ++g)
  {
    das[g].primary = new ot::DA<dim>(
        trees, g, comm, polynomial_degree, size_t{}, partition_tolerance);
    das[g].surrogate = new ot::DA<dim>(
        surrogate_trees, g, comm, polynomial_degree, size_t{}, partition_tolerance);
  }
  std::vector<std::vector<double>> u_ghosted(n_grids);
  std::vector<std::vector<double>> r_ghosted(n_grids);
  for (int g = 0; g < n_grids; ++g)
  {
    u_ghosted[g].resize(das[g].primary->getTotalNodalSz() * single_dof);
    r_ghosted[g].resize(das[g].primary->getTotalNodalSz() * single_dof);
  }

  std::vector<PoissonMat *> mats(n_grids, nullptr);
  mats[0] = &base_mat;
  for (int g = 1; g < n_grids; ++g)
    mats[g] = fe_matrix(*das[g].primary);

  std::vector<std::vector<double>> a_diag_ghosted(n_grids);
  for (int g = 0; g < n_grids; ++g)
  {
    a_diag_ghosted[g].resize(das[g].primary->getTotalNodalSz() * single_dof);
    mats[g]->setDiag(a_diag_ghosted[g].data() + das[g].primary->getLocalNodeBegin() * single_dof);
    das[g].primary->readFromGhostBegin(a_diag_ghosted[g].data(), single_dof);
    das[g].primary->readFromGhostEnd(a_diag_ghosted[g].data(), single_dof);
  }

  VectorPool<double> vector_pool;

  // Jacobi relaxation (one iteration)
  const auto jacobi = [&](PoissonMat *mat, double *u_ghost, double *r_ghost, const double damp, const double *diag_ghosted) -> void
  {
    const ot::DA<dim> *da = mat->da();
    std::vector<double> Dinv_r = vector_pool.checkout(da->getTotalNodalSz() * single_dof);

    for (size_t i = 0; i < da->getTotalNodalSz() * single_dof; ++i)
    {
      const double update = damp / diag_ghosted[i] * r_ghost[i];
      Dinv_r[i] = update;
      u_ghost[i] += update;
    }

    // future: matVecGhosted directly on fresh ghosted data
    mat->matVec(
        Dinv_r.data() + da->getLocalNodeBegin() * single_dof,
        Dinv_r.data() + da->getLocalNodeBegin() * single_dof);
    auto & ADinv_r = Dinv_r;
    da->readFromGhostBegin(ADinv_r.data(), single_dof);
    da->readFromGhostEnd(ADinv_r.data(), single_dof);

    for (size_t i = 0; i < da->getTotalNodalSz() * single_dof; ++i)
      r_ghost[i] -= ADinv_r[i];

    vector_pool.checkin(std::move(Dinv_r));
  };

  // Multigrid v-cycle
  //  Parameters: [in-out] u  initial guess and final solution
  //              [in-out] r  residual of initial guess and of final solution
  const auto vcycle = [&](double *u, double *r) -> void
  {
    // Copy u, r to u_ghosted, r_ghosted.
    base_da.nodalVecToGhostedNodal(r, r_ghosted[0].data(), single_dof);
    base_da.nodalVecToGhostedNodal(u, u_ghosted[0].data(), single_dof);
    //
    base_da.readFromGhostBegin(r_ghosted[0].data(), single_dof);
    base_da.readFromGhostBegin(u_ghosted[0].data(), single_dof);
    base_da.readFromGhostEnd(r_ghosted[0].data(), single_dof);
    base_da.readFromGhostEnd(u_ghosted[0].data(), single_dof);

    const double damp = 2.0/3.0;

    for (int height = 0; height < n_grids - 1; ++height)
    {
      // pre-smoothing (on ghosted vectors)
      jacobi(mats[height], u_ghosted[height].data(), r_ghosted[height].data(), damp, a_diag_ghosted[height].data());

      // restriction (fine-to-coarse) (on ghosted vectors)
      restrict_fine_to_coarse(
          das[height], r_ghosted[height].data(),
          das[height+1], r_ghosted[height+1].data(),
          single_dof, vector_pool);
    }

    // Coarse solve

    for (int height = n_grids - 1; height > 0; --height)
    {
      // prolongation (coarse-to-fine) (on ghosted vectors)

      // post-smoothing (on ghosted vectors)
      jacobi(mats[height-1], u_ghosted[height-1].data(), r_ghosted[height-1].data(), damp, a_diag_ghosted[height-1].data());
    }

    // Copy u_ghosted, r_ghosted to u, r.
    base_da.ghostedNodalToNodalVec(r_ghosted[0].data(), r, single_dof);
    base_da.ghostedNodalToNodalVec(u_ghosted[0].data(), u, single_dof);
  };

  // Preconditioned right-hand side.
  std::vector<double> pc_rhs_vec = rhs_vec;
  std::vector<double> v_temporary = pc_rhs_vec;
  pc_rhs_vec.assign(pc_rhs_vec.size(), 0);  // reset to zero
  vcycle(pc_rhs_vec.data(), v_temporary.data());

  // Preconditioned matrix multiplication.
  const auto pc_mat = [&base_mat, &vcycle, &v_temporary](const double *u, double *v) -> void {
    base_mat.matVec(u, v_temporary.data());

    std::fill_n(v, v_temporary.size(), 0); // reset to zero
    vcycle(v, v_temporary.data());
  };

  // Solve equation.
  const double tol=1e-14;
  const unsigned int max_iter=1500;

  double relative_residual = tol;
  /// const int steps = cgSolver(
  ///     &base_da, [&base_mat](const double *u, double *v){ base_mat.matVec(u, v); },
  ///     &(*u_vec.begin()), &(*rhs_vec.begin()), max_iter, relative_residual);
  const int steps = cgSolver(
      &base_da, pc_mat,
      &(*u_vec.begin()), &(*pc_rhs_vec.begin()), max_iter, relative_residual);
  const double err = sol_err_max(u_vec);

  // Multigrid teardown.
  for (int g = 1; g < n_grids; ++g)
  {
    delete das[g].primary;
    delete das[g].surrogate;
    delete mats[g];
  }
  das.clear();
  mats.clear();

  // future: base_mat can stay in stack memory, don't need to new/delete
  delete &base_mat;

  if(!rank)
  {
    std::cout << "___________End of poissonEq: " 
              << "Finished "
              << "in " << steps << " iterations. "
              << "Final error==" << err << "\n";
  }

  // Debug
  /*
  if (!rank)
    std::cerr << "\n\nExact solution:\n";
  print(base_da, u_exact_vec, std::cerr);
  if (!rank)
    std::cerr << "\n\nApproximation:\n";
  print(base_da, u_vec, std::cerr);
  std::vector<double> err_vec = u_vec;
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    err_vec[i] -= u_exact_vec[i];
  if (!rank)
    std::cerr << "\n\nError:\n";
  print(base_da, err_vec, std::cerr);
  */

  MPI_CHECK(0, err < 1e-10);

  _DestroyHcurve();
  DendroScopeEnd();
}
// ==============================================================


template <unsigned int dim, typename NodeT>
std::ostream & print(const ot::DA<dim> &da,
                     const std::vector<NodeT> &local_vec,
                     std::ostream & out)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::ostream & return_out = ot::printNodes(
      da.getTNCoords() + da.getLocalNodeBegin(),
      da.getTNCoords() + da.getLocalNodeBegin() + da.getLocalNodalSz(),
      local_vec.data(),
      da.getElementOrder(),
      out);
  MPI_Barrier(MPI_COMM_WORLD);
  return return_out;
}



// cgSolver()
template <unsigned int dim, typename MatMult>
int cgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, VECType * u, const VECType * rhs, int max_iter, double relResErr)
{
  // only single dof per node supported here

  // Vector norm and dot
  const auto vec_norm_linf = [da](const VECType *v) -> VECType {
    return normLInfty(v, da->getLocalNodalSz(), da->getCommActive());
  };
  const auto vec_dot = [da](const VECType *u, const VECType *v) -> VECType {
    return dot(u, v, da->getLocalNodalSz(), da->getCommActive());
  };

  // residual
  const auto residual = [da](auto &&matrix, VECType *r, const VECType *u, const VECType *rhs) -> void {
    matrix(u, r);
    subt(rhs, r, da->getLocalNodalSz(), r);  // r = rhs - r
  };

  const double normb = vec_norm_linf(rhs);
  const double thresh = relResErr * normb;

  static std::vector<VECType> r;
  static std::vector<VECType> p;
  static std::vector<VECType> Ap;
  const size_t localSz = da->getLocalNodalSz();
  r.resize(localSz);
  /// p.resize(localSz);
  Ap.resize(localSz);

  int step = 0;
  residual(mat_mult, &r[0], u, rhs);
  VECType rmag = vec_norm_linf(&r[0]);
  fprintf(stdout, "step==%d  normb==%e  res==%e \n", step, normb, rmag);
  if (rmag < thresh)
    return step;
  VECType rProd = vec_dot(&r[0], &r[0]);

  VECType iterLInf = 0.0f;

  p = r;
  while (step < max_iter)
  {
    mat_mult(&p[0], &Ap[0]);  // Ap
    const VECType pProd = vec_dot(&p[0], &Ap[0]);

    const VECType alpha = rProd / pProd;
    iterLInf = alpha * vec_norm_linf(&p[0]);
    for (size_t ii = 0; ii < localSz; ++ii)
      u[ii] += alpha * p[ii];
    ++step;

    const VECType rProdPrev = rProd;

    residual(mat_mult, &r[0], u, rhs);
    rmag = vec_norm_linf(&r[0]);
    if (rmag < thresh)
      break;
    rProd = vec_dot(&r[0], &r[0]);

    const VECType beta = rProd / rProdPrev;
    for (size_t ii = 0; ii < localSz; ++ii)
      p[ii] = r[ii] + beta * p[ii];

    if (step % 10 == 0)
      fprintf(stdout, "step==%d  res==%e  diff==%e  rProd==%e  pProd==%e  a==%e  b==%e\n", step, rmag, iterLInf, rProd, pProd, alpha, beta);
  }
  fprintf(stdout, "step==%d  normb==%e  res==%e \n", step, normb, rmag);

  return step;
}


// matvec_base_in()
template <typename X, unsigned dim>
ot::MatvecBaseIn<dim, X> matvec_base_in(
    const ot::DA<dim> *da, int ndofs, const X *ghosted, int extra_depth)
{
  return ot::MatvecBaseIn<dim, X>(
        da->getTotalNodalSz(), ndofs, da->getElementOrder(),
        extra_depth > 0, extra_depth,
        da->getTNCoords(), ghosted,
        da->dist_tree()->getTreePartFiltered(da->stratum()).data(),
        da->dist_tree()->getTreePartFiltered(da->stratum()).size(),
        *da->getTreePartFront(),
        *da->getTreePartBack());
}

// matvec_base_ou_accumulatet()
template <typename X, unsigned dim>
ot::MatvecBaseOut<dim, X, true> matvec_base_out_accumulate(
    const ot::DA<dim> *da, int ndofs, int extra_depth)
{
  const bool accumulate = true;
  return ot::MatvecBaseOut<dim, X, accumulate>(
      da->getTotalNodalSz(), ndofs, da->getElementOrder(),
      extra_depth > 0, extra_depth,
      da->getTNCoords(),
      da->dist_tree()->getTreePartFiltered(da->stratum()).data(),
      da->dist_tree()->getTreePartFiltered(da->stratum()).size(),
      *da->getTreePartFront(),
      *da->getTreePartBack());
}


