//
// Created by masado on 10/17/22.
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


MPI_TEST_CASE("Poisson problem on a uniformly refined cube with 5 processes, should converge to exact solution", 5)
{
  MPI_Comm comm = test_comm;

  DendroScopeBegin();
  constexpr int dim = 2;
  _InitializeHcurve(dim);

  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  /// const double wavelet_tol = 0.0001;
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

  // fe_matrix()
  const auto fe_matrix = [&](const ot::DA<dim> &da) -> PoissonEq::PoissonMat<dim>
  {
    PoissonEq::PoissonMat<dim> mat(
        &da, &da.dist_tree()->getTreePartFiltered(), single_dof);
    mat.setProblemDimensions(min_corner, max_corner);
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
  PoissonEq::PoissonMat<dim> base_mat = fe_matrix(base_da);
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

  const double tol=1e-14;
  const unsigned int max_iter=1500;

  // status= 0:solved  1:unsolved
  double relative_residual = tol;
  const int status = base_mat.cgSolve(&(*u_vec.begin()), &(*rhs_vec.begin()), max_iter, relative_residual);
  const double err = sol_err_max(u_vec);

  if(!rank)
  {
    std::cout << "___________End of poissonEq: " 
              << (!status ? "solved " : "not solved ")
              << "in at most " << max_iter << " iterations. "
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


