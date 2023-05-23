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
#include <FEM/include/intergridTransfer.h>

#include <vector>
#include <functional>


// -----------------------------
// ✔ Define VectorPool member functions
// ✔ Define smoother (start with Jacobi smoother)
// ✔ Define restriction (adapt from gmgMat)
// ✔ Define prolongation (follow restriction, adapt from gmgMat)
// ✔ Add doctest that restriction and prolongation match
// ✔ Try postMatVec() at end of restriction, preMatVec() at start of prolongation
// ✔ Get an appropriate coarse solver (or make the iterative method converge)
// ✔ Finish vcycle() definition
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

template <unsigned int dim, typename NodeT>
std::ostream & print(const ot::DA<dim> &da,
                     const NodeT *local_vec,
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
int cgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, VECType * u, const VECType * rhs, int max_iter, double relResErr, bool print_progress);

// matvec_base_in()
template <typename X, unsigned dim>
ot::MatvecBaseIn<dim, X> matvec_base_in(
    const ot::DA<dim> *da, int ndofs, const X *ghosted, int extra_depth = 0);

// matvec_base_ou_accumulatet()
template <typename X, unsigned dim>
ot::MatvecBaseOut<dim, X, true> matvec_base_out_accumulate(
    const ot::DA<dim> *da, int ndofs, int extra_depth = 0);

// restrict_fine_to_coarse()
template <int dim, typename PostRestriction>
void restrict_fine_to_coarse(
    DA_Pair<dim> fine_da, const VECType *fine_vec_ghosted,
    DA_Pair<dim> coarse_da, VECType *coarse_vec_ghosted,
    PostRestriction post_restriction,  // on local vector
    int ndofs,
    VectorPool<VECType> &vector_pool);

// prolongate_coarse_to_fine()
template <int dim, typename PreProlongation>
void prolongate_coarse_to_fine(
    DA_Pair<dim> coarse_da, const VECType *coarse_vec_ghosted,
    DA_Pair<dim> fine_da, VECType *fine_vec_ghosted,
    PreProlongation pre_prolongation,  // on local vector
    int ndofs,
    VectorPool<VECType> &vector_pool);


struct CycleSettings
{
  int pre_smooth()     const { return m_pre_smooth; }
  int post_smooth()    const { return m_post_smooth; }
  double damp_smooth() const { return m_damp_smooth; }

  void pre_smooth(int pre_smooth)      { m_pre_smooth = pre_smooth; }
  void post_smooth(int post_smooth)    { m_post_smooth = post_smooth; }
  void damp_smooth(double damp_smooth) { m_damp_smooth = damp_smooth; }

  int m_pre_smooth = 1;
  int m_post_smooth = 1;
  double m_damp_smooth = 1.0;
};


template <typename MatType>
struct VCycle
{
  public:
    static constexpr int dim()
    {
      return std::remove_pointer_t<decltype(std::declval<MatType>().da())>::template_dim;
    }

  // Temporary data
  public:
    // Inputs
    int n_grids = 1;
    MatType * *mats = nullptr;
    CycleSettings settings = {};
    int ndofs = 1;

    std::vector<const ot::DA<dim()> *> surrogate_das;

    // PETSc
    Mat coarse_mat;
    Vec coarse_u, coarse_rhs;
    KSP coarse_ksp;
    PC coarse_pc;

    // Memory resources
    VectorPool<double> vector_pool;
    std::vector<std::vector<double>> e_ghosted;
    std::vector<std::vector<double>> r_ghosted;
    std::vector<std::vector<double>> u_ghosted;
    std::vector<std::vector<double>> a_diag_ghosted;

  // Public methods
  public:
    VCycle(const VCycle &) = delete;
    VCycle(VCycle &&) = default;

    VCycle(int n_grids,
        const std::vector<DA_Pair<dim()>> &da_pairs,
        MatType * *mats, CycleSettings settings, int ndofs)
      : n_grids(n_grids), mats(mats), settings(settings), ndofs(ndofs),
        r_ghosted(n_grids),
        u_ghosted(n_grids),
        e_ghosted(n_grids),
        a_diag_ghosted(n_grids)
    {
      // Save pointers to surrogate DAs. (Primary DAs already stored in mats).
      surrogate_das.reserve(n_grids);
      for (DA_Pair<dim()> pair: da_pairs)
        surrogate_das.push_back(pair.surrogate);
      assert(surrogate_das.size() == n_grids);
      assert(surrogate_das[0] == nullptr);

      // Allocate temporary ghosted vectors.
      for (int g = 0; g < n_grids; ++g)
      {
        const auto *da = mats[g]->da();
        u_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
        r_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
        e_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
      }

      // Initialilze a_diag_ghosted.
      for (int g = 0; g < n_grids; ++g)
      {
        const auto *da = mats[g]->da();
        a_diag_ghosted[g].resize(da->getTotalNodalSz() * ndofs);
        mats[g]->setDiag(a_diag_ghosted[g].data() + da->getLocalNodeBegin() * ndofs);
        da->readFromGhostBegin(a_diag_ghosted[g].data(), ndofs);
        da->readFromGhostEnd(a_diag_ghosted[g].data(), ndofs);
      }

      // ===========================================================================
      // Try KSPSolve() with Coarse grid system.
      // ===========================================================================
      const auto *coarse_da = mats[n_grids - 1]->da();

      // Assemble the coarse grid matrix (assuming one-time assembly).
      coarse_da->createMatrix(coarse_mat, MATAIJ, ndofs);
      mats[n_grids - 1]->getAssembledMatrix(&coarse_mat, {});
      MatAssemblyBegin(coarse_mat, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(coarse_mat, MAT_FINAL_ASSEMBLY);

      // Placeholder Vec's to which user array will be bound in coarse grid solve.
      const MPI_Comm coarse_comm = coarse_da->getCommActive();
      const size_t coarse_local_size = coarse_da->getLocalNodalSz() * ndofs;
      const size_t coarse_global_size = coarse_da->getGlobalNodeSz() * ndofs;
      VecCreateMPIWithArray(coarse_comm, 1, coarse_local_size, coarse_global_size, nullptr, &coarse_u);
      VecCreateMPIWithArray(coarse_comm, 1, coarse_local_size, coarse_global_size, nullptr, &coarse_rhs);

      std::vector<int> coarse_global_ids_of_local_boundary_nodes(coarse_da->getBoundaryNodeIndices().size());
      std::copy(coarse_da->getBoundaryNodeIndices().cbegin(),
                coarse_da->getBoundaryNodeIndices().cend(),
                coarse_global_ids_of_local_boundary_nodes.begin());
      for (int &id : coarse_global_ids_of_local_boundary_nodes)
        id += coarse_da->getGlobalRankBegin();
      // If ndofs != 1 then need to duplicate and zip.

      MatZeroRows(
          coarse_mat,
          coarse_global_ids_of_local_boundary_nodes.size(),
          coarse_global_ids_of_local_boundary_nodes.data(),
          1.0,
          NULL, NULL);

      /// MatView(coarse_mat, PETSC_VIEWER_STDOUT_(coarse_da->getCommActive()));

      // Coarse solver setup with PETSc.
      KSPCreate(coarse_da->getCommActive(), &coarse_ksp);
      KSPSetOperators(coarse_ksp, coarse_mat, coarse_mat);
      KSPSetType(coarse_ksp, KSPPREONLY);  // Do not use iteration.
      KSPGetPC(coarse_ksp, &coarse_pc);
      PCSetType(coarse_pc, PCGAMG);  // Direct solver choice.
      KSPSetUp(coarse_ksp);
    }

    ~VCycle()
    {
      VecDestroy(&coarse_u);
      VecDestroy(&coarse_rhs);
      KSPDestroy(&coarse_ksp);
      MatDestroy(&coarse_mat);
    }

    int coarse_solver(double *u_local, const double *rhs_local)
    {
      // Bind u and rhs to Vec
      VecPlaceArray(coarse_u, u_local);
      VecPlaceArray(coarse_rhs, rhs_local);

      // Solve.
      KSPSolve(coarse_ksp, coarse_rhs, coarse_u);

      // Debug for solution magnitude.
      PetscReal sol_norm = 0.0;
      VecNorm(coarse_u, NORM_INFINITY, &sol_norm);

      // Unbind from Vec.
      VecResetArray(coarse_u);
      VecResetArray(coarse_rhs);

      // Get residual norm and number of iterations.
      int iterations = 0;
      PetscReal res_norm = 0.0;
      KSPGetIterationNumber(coarse_ksp, &iterations);
      KSPGetResidualNorm(coarse_ksp, &res_norm);
      const int rank = par::mpi_comm_rank(this->mats[0]->da()->getCommActive());
      if (rank == 0)
      {
        std::cout
          << "\t"
          << "iterations=" << iterations
          << "  res=" << res_norm
          << "  sol=" << sol_norm << "\n";
      }
      return iterations;
    }

    void pre_smoother(int height, double *u_ghosted, double *r_ghosted)
    {
      jacobi(
          this->mats[height],
          u_ghosted,
          r_ghosted,
          this->settings.damp_smooth(),
          this->a_diag_ghosted[height].data());
    }

    void post_smoother(int height, double *u_ghosted, double *r_ghosted)
    {
      jacobi(
          this->mats[height],
          u_ghosted,
          r_ghosted,
          this->settings.damp_smooth(),
          this->a_diag_ghosted[height].data());
    }

    void vcycle(double *u_local, double *r_local)
    {
      const auto &base_da = *this->mats[0]->da();
      const int ndofs = this->ndofs;
      const int n_grids = this->n_grids;
      const int pre_smooth = this->settings.pre_smooth();
      const int post_smooth = this->settings.post_smooth();
      const double damp = this->settings.damp_smooth();

      std::vector<std::vector<double>> &r_ghosted = this->r_ghosted;
      std::vector<std::vector<double>> &u_ghosted = this->u_ghosted;
      MatType * *mats = this->mats;

      // Multigrid v-cycle
      //  Parameters: [in-out] u  initial guess and final guess
      //              [in-out] r  residual of initial guess and of final guess

      // Copy u, r to u_ghosted, r_ghosted.
      base_da.nodalVecToGhostedNodal(r_local, r_ghosted[0].data(), ndofs);
      base_da.nodalVecToGhostedNodal(u_local, u_ghosted[0].data(), ndofs);
      //
      base_da.readFromGhostBegin(r_ghosted[0].data(), ndofs);
      base_da.readFromGhostBegin(u_ghosted[0].data(), ndofs);
      base_da.readFromGhostEnd(r_ghosted[0].data(), ndofs);
      base_da.readFromGhostEnd(u_ghosted[0].data(), ndofs);

      // reset initial guess for coarser grids to 0.
      for (int height = 1; height < n_grids; ++height)
        u_ghosted[height].assign(u_ghosted[height].size(), 0);

      for (int height = 0; height < n_grids - 1; ++height)
      {
        // pre-smoothing (on ghosted vectors)
        for (int i = 0; i < pre_smooth; ++i)
          this->pre_smoother(height, u_ghosted[height].data(), r_ghosted[height].data());

        // restriction (fine-to-coarse) (on ghosted vectors)
        restrict_fine_to_coarse<dim()>(
            {mats[height]->da(), this->surrogate_das[height]}, r_ghosted[height].data(),
            {mats[height+1]->da(), this->surrogate_das[height+1]}, r_ghosted[height+1].data(),
            [mat=mats[height+1]](double *vec) { mat->postMatVec(vec, vec); },
            ndofs, vector_pool);
      }

      // Coarse solve
      // Direct method.
      const int coarse_steps = this->coarse_solver(
          u_ghosted[n_grids-1].data() + mats[n_grids-1]->da()->getLocalNodeBegin() * ndofs,
          r_ghosted[n_grids-1].data() + mats[n_grids-1]->da()->getLocalNodeBegin() * ndofs);

      mats[n_grids-1]->da()->readFromGhostBegin(u_ghosted[n_grids-1].data(), ndofs);
      mats[n_grids-1]->da()->readFromGhostEnd(u_ghosted[n_grids-1].data(), ndofs);

      for (int height = n_grids - 1; height > 0; --height)
      {
        // prolongation (coarse-to-fine) (on ghosted vectors)
        prolongate_coarse_to_fine<dim()>(
            {mats[height]->da(), this->surrogate_das[height]}, u_ghosted[height].data(),
            {mats[height-1]->da(), this->surrogate_das[height-1]}, e_ghosted[height-1].data(),
            [mat=mats[height]](double *vec) { mat->preMatVec(vec, vec); },
            ndofs, vector_pool);

        // Accumulate into u[h-1] and r[h-1]
        for (size_t i = 0; i < mats[height-1]->da()->getTotalNodalSz() * ndofs; ++i)
          u_ghosted[height-1][i] += e_ghosted[height-1][i];
        //future: matVecGhosted directly on fresh ghosted data
        mats[height-1]->matVec(
            e_ghosted[height-1].data() + mats[height-1]->da()->getLocalNodeBegin() * ndofs,
            e_ghosted[height-1].data() + mats[height-1]->da()->getLocalNodeBegin() * ndofs);
        for (size_t i = 0; i < mats[height-1]->da()->getTotalNodalSz() * ndofs; ++i)
          r_ghosted[height-1][i] -= e_ghosted[height-1][i];

        // post-smoothing (on ghosted vectors)
        for (int i = 0; i < post_smooth; ++i)
          this->post_smoother(height - 1, u_ghosted[height-1].data(), r_ghosted[height-1].data());
      }

      // Copy u_ghosted, r_ghosted to u, r.
      base_da.ghostedNodalToNodalVec(r_ghosted[0].data(), r_local, ndofs);
      base_da.ghostedNodalToNodalVec(u_ghosted[0].data(), u_local, ndofs);
    }


  // Private methods
  private:

    // Jacobi relaxation (one iteration)
    void jacobi(MatType *mat, double *u_ghost, double *r_ghost, const double damp, const double *diag_ghosted)
    {
      const auto *da = mat->da();
      std::vector<double> Dinv_r = this->vector_pool.checkout(da->getTotalNodalSz() * this->ndofs);

      for (size_t i = 0; i < da->getTotalNodalSz() * this->ndofs; ++i)
      {
        const double update = damp / diag_ghosted[i] * r_ghost[i];
        Dinv_r[i] = update;
        u_ghost[i] += update;
      }

      // future: matVecGhosted directly on fresh ghosted data
      mat->matVec(
          Dinv_r.data() + da->getLocalNodeBegin() * this->ndofs,
          Dinv_r.data() + da->getLocalNodeBegin() * this->ndofs);
      auto & ADinv_r = Dinv_r;
      da->readFromGhostBegin(ADinv_r.data(), this->ndofs);
      da->readFromGhostEnd(ADinv_r.data(), this->ndofs);

      for (size_t i = 0; i < da->getTotalNodalSz() * this->ndofs; ++i)
        r_ghost[i] -= ADinv_r[i];

      this->vector_pool.checkin(std::move(Dinv_r));
    };



};//class VCycle




// -----------------------------
// Main test cases
// -----------------------------

// ========================================================================== //
MPI_TEST_CASE("(R r_fine, u_coarse) == (r_fine, P u_coarse) on uniform grid", 2)
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

  // Mesh (uniform grid)
  const int refinement_level = 4;
  ot::DistTree<uint, dim> trees = ot::DistTree<uint, dim>::
      constructSubdomainDistTree(refinement_level, comm, partition_tolerance);
  const int n_grids = 2;
  ot::DistTree<uint, dim> surrogate_trees;
  surrogate_trees = trees.generateGridHierarchyUp(true, n_grids, partition_tolerance);
  DA_Pair<dim> das[2];
  for (int g = 0; g < n_grids; ++g)
    das[g] = {
      new ot::DA<dim>(trees, g, comm, polynomial_degree, size_t{}, partition_tolerance),
      new ot::DA<dim>(surrogate_trees, g, comm, polynomial_degree, size_t{}, partition_tolerance)
    };

  // Vectors
  const int single_dof = 1;
  std::vector<double> r_fine, u_fine, r_coarse, u_coarse;
  das[0].primary->createVector(r_fine, false, true, single_dof);
  das[0].primary->createVector(u_fine, false, true, single_dof);
  das[1].primary->createVector(r_coarse, false, true, single_dof);
  das[1].primary->createVector(u_coarse, false, true, single_dof);
  VectorPool<double> vector_pool;

  // Initialize vectors
  u_fine.assign(u_fine.size(), 0);
  r_coarse.assign(r_coarse.size(), 0);
  auto random = [gen = std::mt19937_64(), dist = std::uniform_int_distribution<>(1, 25)]() mutable {
    return dist(gen);
  };
  for (double & ri : r_fine)
    ri = random();
  for (double & ui : u_coarse)
    ui = random();
  das[0].primary->readFromGhostBegin(r_fine.data(), single_dof);
  das[0].primary->readFromGhostEnd(r_fine.data(), single_dof);
  das[1].primary->readFromGhostBegin(u_coarse.data(), single_dof);
  das[1].primary->readFromGhostEnd(u_coarse.data(), single_dof);

  // Coarse operator
  Point<dim> min_corner(-1.0);
  Point<dim> max_corner(1.0);
  PoissonEq::PoissonMat<dim> coarse_mat(das[n_grids-1].primary, {}, single_dof);
  coarse_mat.setProblemDimensions(min_corner, max_corner);
  coarse_mat.zero_boundary(true);

  // restriction (fine-to-coarse) (on ghosted vectors)
  restrict_fine_to_coarse(
      das[0], r_fine.data(),
      das[1], r_coarse.data(),
      [&](double *vec) { coarse_mat.postMatVec(vec, vec); },
      single_dof, vector_pool);

  // prolongation (coarse-to-fine) (on ghosted vectors)
  prolongate_coarse_to_fine(
      das[1], u_coarse.data(),
      das[0], u_fine.data(),
      [&](double *vec) { coarse_mat.preMatVec(vec, vec); },
      single_dof, vector_pool);

  const double Rr_dot_u = dot(
      r_coarse.data() + das[1].primary->getLocalNodeBegin() * single_dof,
      u_coarse.data() + das[1].primary->getLocalNodeBegin() * single_dof,
      das[1].primary->getLocalNodalSz() * single_dof,
      comm);

  const double r_dot_Pu = dot(
      r_fine.data() + das[0].primary->getLocalNodeBegin() * single_dof,
      u_fine.data() + das[0].primary->getLocalNodeBegin() * single_dof,
      das[0].primary->getLocalNodalSz() * single_dof,
      comm);

  delete das[0].primary;
  delete das[0].surrogate;
  delete das[1].primary;
  delete das[1].surrogate;

  /// if (rank == 0)
  ///   fprintf(stderr, "Rr_dot_u==%f, r_dot_Pu==%f\n", Rr_dot_u, r_dot_Pu);
  MPI_CHECK(0, Rr_dot_u == r_dot_Pu);

  _DestroyHcurve();
  DendroScopeEnd();
}


#include <fenv.h>

// ========================================================================== //
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
  const int refinement_level = 5;
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
        &da, {}, single_dof);
    mat->setProblemDimensions(min_corner, max_corner);
    return mat;
  };

  // fe_vector()
  const auto fe_vector = [&](const ot::DA<dim> &da) -> PoissonEq::PoissonVec<dim>
  {
    PoissonEq::PoissonVec<dim> vec(
        const_cast<ot::DA<dim> *>(&da),   // violate const for weird feVec non-const necessity
        {},
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

  std::vector<PoissonMat *> mats(n_grids, nullptr);
  mats[0] = &base_mat;
  for (int g = 1; g < n_grids; ++g)
  {
    mats[g] = fe_matrix(*das[g].primary);
    mats[g]->zero_boundary(true);
  }

  CycleSettings cycle_settings;
  cycle_settings.pre_smooth(2);
  cycle_settings.post_smooth(1);
  cycle_settings.damp_smooth(2.0 / 3.0);

  VCycle<PoissonMat> vcycle(n_grids, das, mats.data(), cycle_settings, single_dof);

  // Preconditioned right-hand side.
  std::vector<double> pc_rhs_vec = rhs_vec;
  std::vector<double> v_temporary = pc_rhs_vec;
  pc_rhs_vec.assign(pc_rhs_vec.size(), 0);  // reset to zero
  vcycle.vcycle(pc_rhs_vec.data(), v_temporary.data());

  // Preconditioned matrix multiplication.
  const auto pc_mat = [&vcycle, &base_mat, &v_temporary](const double *u, double *v) -> void {
    base_mat.matVec(u, v_temporary.data());

    std::fill_n(v, v_temporary.size(), 0); // reset to zero
    vcycle.vcycle(v, v_temporary.data());
  };

  // Solve equation.
  const double tol=1e-14;
  const unsigned int max_iter=300;

  /// feenableexcept(FE_INVALID);

  double relative_residual = tol;

  /// const int steps = cgSolver(
  ///     &base_da, [&base_mat](const double *u, double *v){ base_mat.matVec(u, v); },
  ///     &(*u_vec.begin()), &(*rhs_vec.begin()), max_iter, relative_residual, true);

  /// const int steps = cgSolver(
  ///     &base_da, pc_mat,
  ///     &(*u_vec.begin()), &(*pc_rhs_vec.begin()), max_iter, relative_residual, true);

  // Multigrid V-Cycle iteration as solver.
  const int steps = [&, steps=0]() mutable {
    base_mat.matVec(u_vec.data(), v_vec.data());
    for (size_t i = 0; i < v_vec.size(); ++i)
      v_vec[i] = rhs_vec[i] - v_vec[i];
    double err;
    err = sol_err_max(u_vec);
    if (rank == 0)
    {
      fprintf(stdout, "steps==%2d  err==%e\n", steps, err);
      fprintf(stdout, "\n");
    }
    while (steps < max_iter and err > 1e-14)
    {
      vcycle.vcycle(u_vec.data(), v_vec.data());
      base_mat.matVec(u_vec.data(), v_vec.data());
      for (size_t i = 0; i < v_vec.size(); ++i)
        v_vec[i] = rhs_vec[i] - v_vec[i];
      ++steps;
      err = sol_err_max(u_vec);
      if (rank == 0)
      {
        fprintf(stdout, "steps==%2d  err==%e\n", steps, err);
        fprintf(stdout, "\n");
      }
    }
    return steps;
  }();

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

  /// // Debug
  /// if (!rank)
  ///   std::cerr << "\n\nExact solution:\n";
  /// print(base_da, u_exact_vec, std::cerr);
  /// if (!rank)
  ///   std::cerr << "\n\nApproximation:\n";
  /// print(base_da, u_vec, std::cerr);
  /// std::vector<double> err_vec = u_vec;
  /// for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
  ///   err_vec[i] -= u_exact_vec[i];
  /// if (!rank)
  ///   std::cerr << "\n\nError:\n";
  /// print(base_da, err_vec, std::cerr);

  MPI_CHECK(0, err < 1e-10);

  _DestroyHcurve();
  DendroScopeEnd();
}
// ========================================================================== //


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

template <unsigned int dim, typename NodeT>
std::ostream & print(const ot::DA<dim> &da,
                     const NodeT *local_vec,
                     std::ostream & out)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::ostream & return_out = ot::printNodes(
      da.getTNCoords() + da.getLocalNodeBegin(),
      da.getTNCoords() + da.getLocalNodeBegin() + da.getLocalNodalSz(),
      local_vec,
      da.getElementOrder(),
      out);
  MPI_Barrier(MPI_COMM_WORLD);
  return return_out;
}



// cgSolver()
template <unsigned int dim, typename MatMult>
int cgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, VECType * u, const VECType * rhs, int max_iter, double relResErr, bool print_progress)
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
  const VECType rmag0 = rmag;
  fprintf(stdout, "step==%d  normb==%e  res==%e \n", step, normb, rmag);
  if (rmag <= thresh)
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
    if (rmag <= thresh)
      break;
    rProd = vec_dot(&r[0], &r[0]);

    const VECType beta = rProd / rProdPrev;
    for (size_t ii = 0; ii < localSz; ++ii)
      p[ii] = r[ii] + beta * p[ii];

    if (print_progress and step % 10 == 0)
      fprintf(stdout, "step==%d  res==%e  reduce==%e  diff==%e  rProd==%e  pProd==%e  a==%e  b==%e\n", step, rmag, rmag/rmag0, iterLInf, rProd, pProd, alpha, beta);
  }
  fprintf(stdout, "step==%d  normb==%e  res==%e  reduce==%e\n", step, normb, rmag, rmag/rmag0);

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


// restrict_fine_to_coarse()
template <int dim, typename PostRestriction>
void restrict_fine_to_coarse(
    DA_Pair<dim> fine_da, const VECType *fine_vec_ghosted,
    DA_Pair<dim> coarse_da, VECType *coarse_vec_ghosted,
    PostRestriction post_restriction,
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

  post_restriction(coarse_vec_ghosted + coarse_da.primary->getLocalNodeBegin() * ndofs);

  coarse_da.primary->readFromGhostBegin(coarse_vec_ghosted, ndofs);
  coarse_da.primary->readFromGhostEnd(coarse_vec_ghosted, ndofs);

  vector_pool.checkin(std::move(leafBuffer));
  vector_pool.checkin(std::move(coarse_surr_ghosted));
}


// prolongate_coarse_to_fine()
template <int dim, typename PreProlongation>
void prolongate_coarse_to_fine(
    DA_Pair<dim> coarse_da, const VECType *coarse_vec_ghosted,
    DA_Pair<dim> fine_da, VECType *fine_vec_ghosted,
    PreProlongation pre_prolongation,  // on local vector
    int ndofs,
    VectorPool<VECType> &vector_pool)
{
  const unsigned int nPe = fine_da.primary->getNumNodesPerElement();

  std::vector<VECType> coarse_surr_ghosted = vector_pool.checkout(
      coarse_da.surrogate->getTotalNodalSz() * ndofs);

  std::vector<VECType> leafBuffer = vector_pool.checkout(ndofs * nPe);
  leafBuffer.assign(leafBuffer.size(), 42);

  std::vector<VECType> coarse_copy_ghosted = vector_pool.checkout(
      coarse_da.primary->getTotalNodalSz() * ndofs);
  std::copy_n(coarse_vec_ghosted, coarse_copy_ghosted.size(), coarse_copy_ghosted.begin());
  pre_prolongation(coarse_copy_ghosted.data() + coarse_da.primary->getLocalNodeBegin() * ndofs);
  coarse_vec_ghosted = coarse_copy_ghosted.data();

  // Shift in the coarse grid from primary to surrogate.
  ot::distShiftNodes(
      *coarse_da.primary,
      coarse_vec_ghosted + coarse_da.primary->getLocalNodeBegin() * ndofs,
      *coarse_da.surrogate,
      coarse_surr_ghosted.data() + coarse_da.surrogate->getLocalNodeBegin() * ndofs,
      ndofs);
  coarse_da.surrogate->readFromGhostBegin(coarse_surr_ghosted.data(), ndofs);
  coarse_da.surrogate->readFromGhostEnd(coarse_surr_ghosted.data(), ndofs);

  // Surrogate is coarse grid partitioned by fine
  // Interpolate from the coarse surrogate grid to fine primary.

  using TN = ot::TreeNode<uint, dim>;

  fem::MeshFreeInputContext<VECType, TN>
      inctx{ coarse_surr_ghosted.data(),
             coarse_da.surrogate->getTNCoords(),
             (unsigned) coarse_da.surrogate->getTotalNodalSz(),
             coarse_da.surrogate->dist_tree()->getTreePartFiltered(coarse_da.surrogate->stratum()).data(),
             coarse_da.surrogate->dist_tree()->getTreePartFiltered(coarse_da.surrogate->stratum()).size(),
             *coarse_da.surrogate->getTreePartFront(),
             *coarse_da.surrogate->getTreePartBack() };

  fem::MeshFreeOutputContext<VECType, TN>
      outctx{fine_vec_ghosted,
             fine_da.primary->getTNCoords(),
             (unsigned) fine_da.primary->getTotalNodalSz(),
             fine_da.primary->dist_tree()->getTreePartFiltered(fine_da.primary->stratum()).data(),
             fine_da.primary->dist_tree()->getTreePartFiltered(fine_da.primary->stratum()).size(),
             *fine_da.primary->getTreePartFront(),
             *fine_da.primary->getTreePartBack() };

  const RefElement * refel = fine_da.primary->getReferenceElement();

  std::vector<char> outDirty(fine_da.primary->getTotalNodalSz(), 0);
  fem::locIntergridTransfer(inctx, outctx, ndofs, refel, &(*outDirty.begin()));
  // The outDirty array is needed when wrwiteToGhosts useAccumulation==false (hack).
  fine_da.primary->template writeToGhostsBegin<VECType>(fine_vec_ghosted, ndofs, &(*outDirty.cbegin()));
  fine_da.primary->template writeToGhostsEnd<VECType>(fine_vec_ghosted, ndofs, false, &(*outDirty.cbegin()));

  fine_da.primary->readFromGhostBegin(fine_vec_ghosted, ndofs);
  fine_da.primary->readFromGhostEnd(fine_vec_ghosted, ndofs);

  vector_pool.checkin(std::move(leafBuffer));
  vector_pool.checkin(std::move(coarse_surr_ghosted));
  vector_pool.checkin(std::move(coarse_copy_ghosted));
}


// =============================================================================

