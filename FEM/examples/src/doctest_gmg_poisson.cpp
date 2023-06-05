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

#include "FEM/include/mg.hpp"
#include "FEM/include/solver_utils.hpp"

#include <vector>
#include <functional>

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
// Functions
// -----------------------------

// cgSolver()
template <unsigned int dim, typename MatMult>
int cgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, VECType * u, const VECType * rhs, int max_iter, double relResErr, bool print_progress);


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
  mg::DA_Pair<dim> das[2];
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
  detail::VectorPool<double> vector_pool;

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
  const int n_grids = 3;
  ot::DistTree<uint, dim> &trees = base_tree;
  ot::DistTree<uint, dim> surrogate_trees;
  surrogate_trees = trees.generateGridHierarchyUp(true, n_grids, partition_tolerance);
  assert(trees.getNumStrata() == n_grids);
  /// struct DA_Pair { const ot::DA<dim> *primary; const ot::DA<dim> *surrogate; };
  std::vector<mg::DA_Pair<dim>> das(n_grids, mg::DA_Pair<dim>{nullptr, nullptr});
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

  mg::CycleSettings cycle_settings;
  cycle_settings.pre_smooth(2);
  cycle_settings.post_smooth(1);
  cycle_settings.damp_smooth(2.0 / 3.0);
  cycle_settings.print(false);
  cycle_settings.n_grids(n_grids);

  mg::VCycle<PoissonMat> vcycle(das, mats.data(), cycle_settings, single_dof);

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

  if(rank == 0)
  {
    std::cout << "___________Begin poissonEq\n";
  }

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
    /// if (rank == 0)
    /// {
    ///   fprintf(stdout, "steps==%2d  err==%e\n", steps, err);
    ///   fprintf(stdout, "\n");
    /// }
    while (steps < max_iter and err > 1e-14)
    {
      vcycle.vcycle(u_vec.data(), v_vec.data());
      base_mat.matVec(u_vec.data(), v_vec.data());
      for (size_t i = 0; i < v_vec.size(); ++i)
        v_vec[i] = rhs_vec[i] - v_vec[i];
      ++steps;
      err = sol_err_max(u_vec);
      /// if (rank == 0)
      /// {
      ///   fprintf(stdout, "steps==%2d  err==%e\n", steps, err);
      ///   fprintf(stdout, "\n");
      /// }
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


// ========================================================================== //
MPI_TEST_CASE("Nonuniform Poisson gmg sinusoid", 5)
{
  MPI_Comm comm = test_comm;
  DendroScopeBegin();
  constexpr int dim = 2;
  _InitializeHcurve(dim);
  const int rank = par::mpi_comm_rank(comm);
  using Oct = Oct<dim>;
  const double partition_tolerance = 0.1;
  const int polynomial_degree = 1;
  constexpr double pi = { M_PI };
  static_assert(3.0 < pi, "The macro M_PI is not defined as a value > 3!");

  // Domain is a cube from -1 to 1 in each axis.
  Point<dim> min_corner(-1.0),  max_corner(1.0);

  // Solve "-div(grad(u)) = f"
  // Sinusoid: f(x) = sin(pi * x)    //future/c++17: std::transform_reduce()
  const auto f = [](const double *x, double *y = nullptr) {
    double result = 1.0;
    for (int d = 0; d < dim; ++d)
      result *= std::sin(pi * x[d]);
    if (y != nullptr)
      *y = result;
    return result;
  };
  // Solution: u(x) = 1/(pi^2) sin(pi * x)
  const auto u_exact = [] (const double *x) {
    double result = 1.0 / (dim * pi * pi);
    for (int d = 0; d < dim; ++d)
      result *= std::sin(pi * x[d]);
    return result;
  };
  // Dirichlet boundary condition (matching u_exact).
  const auto u_bdry = [=] (const double *x) {
    return u_exact(x);
  };

  // Mesh (nonuniform grid)
  const double interpolation = 1e-3;
  /// const int refinement_level = 5;
  ot::DistTree<uint, dim> base_tree = ot::DistTree<uint, dim>::
      constructDistTreeByFunc<double>(
          f, 1, comm, polynomial_degree, interpolation, partition_tolerance);
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
  const int n_grids = 3;
  ot::DistTree<uint, dim> &trees = base_tree;
  ot::DistTree<uint, dim> surrogate_trees;
  surrogate_trees = trees.generateGridHierarchyUp(true, n_grids, partition_tolerance);
  assert(trees.getNumStrata() == n_grids);
  /// struct DA_Pair { const ot::DA<dim> *primary; const ot::DA<dim> *surrogate; };
  std::vector<mg::DA_Pair<dim>> das(n_grids, mg::DA_Pair<dim>{nullptr, nullptr});
  das[0].primary = &base_da;
  for (int g = 1; g < n_grids; ++g)
  {
    das[g].primary = new ot::DA<dim>(
        trees, g, comm, polynomial_degree, size_t{}, partition_tolerance);
    das[g].surrogate = new ot::DA<dim>(
        surrogate_trees, g, comm, polynomial_degree, size_t{}, partition_tolerance);
  }

  /// ot::quadTreeToGnuplot(trees.getTreePartFiltered(0), 10, "base", comm);
  /// ot::quadTreeToGnuplot(surrogate_trees.getTreePartFiltered(1), 10, "surr", comm);
  /// ot::quadTreeToGnuplot(trees.getTreePartFiltered(1), 10, "coarse", comm);


  std::vector<PoissonMat *> mats(n_grids, nullptr);
  mats[0] = &base_mat;
  for (int g = 1; g < n_grids; ++g)
  {
    mats[g] = fe_matrix(*das[g].primary);
    mats[g]->zero_boundary(true);
  }

  mg::CycleSettings cycle_settings;
  cycle_settings.pre_smooth(2);
  cycle_settings.post_smooth(1);
  cycle_settings.damp_smooth(2.0 / 3.0);
  cycle_settings.n_grids(n_grids);

  mg::VCycle<PoissonMat> vcycle(das, mats.data(), cycle_settings, single_dof);

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

  if(rank == 0)
  {
    std::cout << "___________Begin poissonEq ("
              << par::mpi_comm_size(comm)
              << " processes)\n";
  }

  // Solve equation.
  const double tol=1e-14;
  const unsigned int max_iter=300;
  const int max_vcycles = 100;

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
    util::ConvergenceRate convergence(3);
    util::ConvergenceRate residual_convergence(3);
    base_mat.matVec(u_vec.data(), v_vec.data());
    for (size_t i = 0; i < v_vec.size(); ++i)
      v_vec[i] = rhs_vec[i] - v_vec[i];
    double err;
    double res;
    err = sol_err_max(u_vec);
    res = normLInfty(v_vec.data(), v_vec.size(), comm);
    convergence.observe_step(err);
    residual_convergence.observe_step(res);
    if (rank == 0)
    {
      fprintf(stdout, "steps==%2d  err==%e\n", steps, err);
    }
    while (steps < max_vcycles and err > 1e-14)
    {
      vcycle.vcycle(u_vec.data(), v_vec.data());
      base_mat.matVec(u_vec.data(), v_vec.data());
      for (size_t i = 0; i < v_vec.size(); ++i)
        v_vec[i] = rhs_vec[i] - v_vec[i];
      ++steps;
      err = sol_err_max(u_vec);
      res = normLInfty(v_vec.data(), v_vec.size(), comm);
      convergence.observe_step(err);
      residual_convergence.observe_step(res);
      const double rate = convergence.rate();
      const double res_rate = residual_convergence.rate();
      if (rank == 0)
      {
        fprintf(stdout, "steps==%2d  err==%e  rate==(%0.1fx)"
            "      res==%e  rate==(%0.1fx)\n",
            steps, err, std::exp(std::abs(std::log(rate))),
            res, std::exp(std::abs(std::log(res_rate))));
      }
      if (res_rate > 0.95)
        break;
    }
    return steps;
  }();

  const double err = sol_err_max(u_vec);
  const double res = normLInfty(v_vec.data(), v_vec.size(), comm);

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

  MPI_CHECK(0, err < interpolation);
  MPI_CHECK(0, res < 1e-10);

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

// =============================================================================

