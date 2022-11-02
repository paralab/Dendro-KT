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
#include <algorithm>

// -----------------------------------------------------------------------------
// TODO
// ✔ Basic test to count the nodes from special quadratic on a uniform grid
// _ Extend cancellation-node methodology with a check for weird hanging
// _ Generate quadratic primary shared nodes
// _ Generate quadratic primary interior nodes
// _ Generate quadratic cancellation nodes?
// _ Check assumptions in _constructInner() about polynomial degree
// _ Check assumptions in getNodeElementOwnership() about polynomial degree
// _ future: new element-based DA construction with nodes literally anywhere...
// -----------------------------------------------------------------------------

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


template <unsigned dim>
LLU count_explicit_hanging_nodes(const ot::DA<dim> &da);


MPI_TEST_CASE("Regular DA construction on uniform grid still works", 1)
{
  MPI_Comm comm = test_comm;
  DendroScopeBegin();
  constexpr int dim = 2;
  _InitializeHcurve(dim);
  const double partition_tolerance = 0.1;
  const int refinement_level = 3;

  // Uniform grid
  ot::DistTree<uint, dim> tree = ot::DistTree<uint, dim>::
      constructSubdomainDistTree(refinement_level, comm, partition_tolerance);

  for (int polynomial_degree = 1; polynomial_degree <= 5; ++polynomial_degree)
  {
    // To construct the node set, the quadratic elements must be known.
    ot::DA<dim> da(tree,
        comm, polynomial_degree, int{}, partition_tolerance);

    INFO("degree==", polynomial_degree);
    MPI_CHECK(0, da.getGlobalNodeSz() == (pow(polynomial_degree * (1 << refinement_level) + 1, dim)));
  }

  _DestroyHcurve();
  DendroScopeEnd();
}


MPI_TEST_CASE("Special quadratic elements give expected # of nodes, "
              "assuming outer p-refinement and inner h-refinement", 1)
{
  MPI_Comm comm = test_comm;

  DendroScopeBegin();
  constexpr int dim = 2;
  _InitializeHcurve(dim);
  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);
  const double partition_tolerance = 0.1;
  const int polynomial_degree = 1;

  // Mesh (initially uniform grid)
  const int refinement_level = 3;
  ot::DistTree<uint, dim> tree = ot::DistTree<uint, dim>::
      constructSubdomainDistTree(refinement_level, comm, partition_tolerance);

  // Refine tree
  {
    std::vector<int> refine(tree.getTreePartFiltered().size(), 0);
    REQUIRE(npes == 1); // if not, need ghost-enabled alternative to get next-outermost shell

    std::vector<ot::TreeNode<uint, dim>> boundary, boundary_neighbor;
    std::copy_if(tree.getTreePartFiltered().begin(),
                 tree.getTreePartFiltered().end(),
                 std::back_inserter(boundary),
                 [](const auto &oct) -> bool { return oct.getIsOnTreeBdry(); });
    for (const auto &oct : boundary)
      oct.appendAllNeighbours(boundary_neighbor);
    ot::SFC_Tree<uint, dim>::locTreeSort(boundary_neighbor);
    boundary_neighbor.erase(
        std::unique(boundary_neighbor.begin(), boundary_neighbor.end()),
        boundary_neighbor.end());

    // Find all cells that are same-level neighbors of boundary cells,
    // but exclude those that are simultaneously boundary cells.
    // This only works because all cells have initially the same level
    // (and don't have to worry about ghost cells if running uniprocess).
    size_t j = 0;
    for (size_t i = 0; i < tree.getTreePartFiltered().size(); ++i)
    {
      if (tree.getTreePartFiltered()[i] == boundary_neighbor[j])
      {
        ++j;
        if (not tree.getTreePartFiltered()[i].getIsOnTreeBdry())
          refine[i] = 1;
      }
    }

    // Replace tree with refined tree
    ot::DistTree<uint, dim> tmp_tree;
    std::swap(tree, tmp_tree);
    tmp_tree.distRefine(tmp_tree, std::move(refine), tree, partition_tolerance);
  }

  // Indicate p-refinement on boundary
  ot::SpecialElements special;  // member `quadratic` is a vector<size_t>
  for (size_t i = 0; i < tree.getTreePartFiltered().size(); ++i)
    if (tree.getTreePartFiltered()[i].getIsOnTreeBdry())
      special.quadratic.push_back(i);
  const long long unsigned expected_q2_elements
    = pow(1 << refinement_level, dim) - pow((1 << refinement_level) - 2, dim);
  REQUIRE(special.quadratic.size() == expected_q2_elements);

  // User could have appended indices of special elements in any order.
  // Repetitions also ok. The DA constructor will sort and remove duplicates.
  // Simulate user putting in random order and the first one duplicated:
  special.quadratic.push_back(special.quadratic[0]);
  std::shuffle(special.quadratic.begin(), special.quadratic.end(), std::mt19937_64{42});

  // To construct the node set, the quadratic elements must be known.
  ot::DA<dim> da(tree,
      std::move(special),
      comm, polynomial_degree, int{}, partition_tolerance);
  CHECK(da.special_elements().quadratic.size() == expected_q2_elements);
  for (size_t i = 0; i < tree.getTreePartFiltered().size(); ++i)
    CHECK(da.getElementOrder(i) == (tree.getTreePartFiltered()[i].getIsOnTreeBdry() ? 2 : 1));

  // Check number of nodes.
  MPI_CHECK(0, da.getGlobalNodeSz() > (pow((1 << refinement_level) + 1, dim)));
  const long long unsigned expected_nodes =
      pow(2*(1 << refinement_level) + 1, dim) // all quadratic
      - pow(2*((1 << refinement_level)-4) + 1, dim) // under 2nd layer not quadratic
      + pow(((1 << refinement_level)-4) + 1, dim); // under 2nd layer is linear
  MPI_CHECK(0, da.getGlobalNodeSz() == expected_nodes);

  // Check no quadratic hanging nodes.
  /// const LLU explicit_hanging_nodes = count_explicit_hanging_nodes(da);
  /// CHECK(explicit_hanging_nodes == 0);

  _DestroyHcurve();
  DendroScopeEnd();
}


//future: solve a PDE
//-------------------
/// MPI_TEST_CASE("Poisson problem on a uniformly refined cube with p-refinement on boundaries (5 processes), should converge to exact solution", 5)
/// {
///   MPI_Comm comm = test_comm;
/// 
///   DendroScopeBegin();
///   constexpr int dim = 2;
///   _InitializeHcurve(dim);
/// 
///   int rank, npes;
///   MPI_Comm_rank(comm, &rank);
///   MPI_Comm_size(comm, &npes);
/// 
///   /// const double wavelet_tol = 0.0001;
///   using Oct = Oct<dim>;
///   const double partition_tolerance = 0.1;
///   const int polynomial_degree = 1;
/// 
///   Point<dim> min_corner(-1.0);
///   Point<dim> max_corner(1.0);
/// 
///   // u_exact function
///   const double coefficient[] = {1, 2, 5, 3};
///   const double sum_coeff = std::accumulate(coefficient, coefficient + dim, 0);
///   const auto u_exact = [=] (const double *x) {
///     double expression = 1;
///     for (int d = 0; d < dim; ++d)
///       expression += coefficient[d] * x[d] * x[d];
///     return expression;
///   };
///   // ... is the solution to -div(grad(u)) = f, where f is
///   const auto f = [=] (const double *x) {
///     return -2*sum_coeff;
///   };
///   // ... and boundary is prescribed (matching u_exact)
///   const auto u_bdry = [=] (const double *x) {
///     return u_exact(x);
///   };
/// 
///   // Mesh (uniform grid)
///   const int refinement_level = 6;
///   ot::DistTree<uint, dim> tree = ot::DistTree<uint, dim>::
///       constructSubdomainDistTree(refinement_level, comm, partition_tolerance);
///   //TODO Do h-refinement on the shell just inside the boundary elements
/// 
///   // Indicate p-refinement on boundary
///   assert(polynomial_degree == 1);
///   ot::SpecialElements special;
///   for (size_t i = 0; i < tree.getTreePartFiltered().size(); ++i)
///     if (tree.getTreePartFiltered()[i].getIsOnTreeBdry())
///       special.quadratic.push_back(i);
///   if (npes == 1)
///     REQUIRE(special.quadratic.size() ==
///         pow(1 << refinement_level, dim) - pow((1 << refinement_level) - 2, dim));
/// 
///   // To construct the node set, the quadratic elements must be known.
///   ot::DA<dim> da(tree,
///       std::move(special),
///       comm, polynomial_degree, int{}, partition_tolerance);
/// 
///   // ghosted_node_coordinate()
///   const auto ghosted_node_coordinate = [&](const ot::DA<dim> &da, size_t idx) -> Point<dim>
///   {
///     const ot::TreeNode<uint, dim> node = da.getTNCoords()[idx];   //integer
///     std::array<double, dim> coordinates;
///     ot::treeNode2Physical(node, polynomial_degree, coordinates.data());  //fixed
/// 
///     for (int d = 0; d < dim; ++d)
///       coordinates[d] = min_corner.x(d) * (1 - coordinates[d]) // scale and shift
///                      + max_corner.x(d) * coordinates[d];
/// 
///     return Point<dim>(coordinates);
///   };
/// 
///   // local_node_coordinate()
///   const auto local_node_coordinate = [&](const ot::DA<dim> &da, size_t idx) -> Point<dim>
///   {
///     return ghosted_node_coordinate(da, idx + da.getLocalNodeBegin());
///   };
/// 
///   // local_vector()
///   const auto local_vector = [&](const ot::DA<dim> &da, auto type, int dofs)
///   {
///     std::vector<std::remove_cv_t<decltype(type)>> local;
///     const bool ghosted = false;  // local means not ghosted
///     da.createVector(local, false, ghosted, dofs);
///     return local;
///   };
/// 
///   // Vectors
///   const int single_dof = 1;
///   std::vector<double> u_vec = local_vector(da, double{}, single_dof);
///   std::vector<double> v_vec = local_vector(da, double{}, single_dof);
///   std::vector<double> f_vec = local_vector(da, double{}, single_dof);
///   std::vector<double> rhs_vec = local_vector(da, double{}, single_dof);
/// 
///   //TODO: Must define custom operator that is aware of mixed degrees
/// 
///   // fe_matrix()
///   const auto fe_matrix = [&](const ot::DA<dim> &da) -> PoissonEq::PoissonMat<dim>
///   {
///     PoissonEq::PoissonMat<dim> mat(
///         &da, &da.dist_tree()->getTreePartFiltered(), single_dof);
///     mat.setProblemDimensions(min_corner, max_corner);
///     return mat;
///   };
/// 
///   // fe_vector()
///   const auto fe_vector = [&](const ot::DA<dim> &da) -> PoissonEq::PoissonVec<dim>
///   {
///     PoissonEq::PoissonVec<dim> vec(
///         const_cast<ot::DA<dim> *>(&da),   // violate const for weird feVec non-const necessity
///         &da.dist_tree()->getTreePartFiltered(),
///         single_dof);
///     vec.setProblemDimensions(min_corner, max_corner);
///     return vec;
///   };
/// 
///   // Boundary condition and 0 on interior (needed to subtract A u^bdry from rhs)
///   for (size_t i = 0; i < da.getLocalNodalSz(); ++i)
///     u_vec[i] = 0;
///   for (size_t bdyIdx : da.getBoundaryNodeIndices())
///     u_vec[bdyIdx] = u_bdry(&local_node_coordinate(da, bdyIdx).x(0));
/// 
///   // Evaluate forcing term.
///   for (size_t i = 0; i < da.getLocalNodalSz(); ++i)
///     f_vec[i] = f(&local_node_coordinate(da, i).x(0));
/// 
///   // Linear system oracles encapsulating the operators.
///   PoissonEq::PoissonMat<dim> base_mat = fe_matrix(da);
///   PoissonEq::PoissonVec<dim> base_vec = fe_vector(da);
/// 
///   // Compute right hand side, subtracting boundary data.
///   // (See preMatVec, postMatVec, preComputeVec, postComputeVec).
///   base_mat.matVec(u_vec.data(), v_vec.data());
///   base_vec.computeVec(f_vec.data(), rhs_vec.data());
///   for (size_t i = 0; i < da.getLocalNodalSz(); ++i)
///     rhs_vec[i] -= v_vec[i];
///   base_mat.zero_boundary(true);
/// 
///   // Precompute exact solution to evaluate error of an approximate solution.
///   std::vector<double> u_exact_vec = local_vector(da, double{}, single_dof);
///   for (size_t i = 0; i < da.getLocalNodalSz(); ++i)
///     u_exact_vec[i] = u_exact(&local_node_coordinate(da, i).x(0));
/// 
///   // Function to check solution error.
///   const auto sol_err_max = [&](const std::vector<double> &u) {
///       double err_max = 0.0;
///       for (size_t i = 0; i < da.getLocalNodalSz(); ++i)
///         err_max = fmax(err_max, abs(u[i] - u_exact_vec[i]));
///       err_max = par::mpi_max(err_max, comm);
///       return err_max;
///   };
/// 
///   const double tol=1e-14;
///   const unsigned int max_iter=1500;
/// 
///   // status= 0:solved  1:unsolved
///   double relative_residual = tol;
///   const int status = base_mat.cgSolve(&(*u_vec.begin()), &(*rhs_vec.begin()), max_iter, relative_residual);
///   const double err = sol_err_max(u_vec);
/// 
///   if(!rank)
///   {
///     std::cout << "___________End of poissonEq: " 
///               << (!status ? "solved " : "not solved ")
///               << "in at most " << max_iter << " iterations. "
///               << "Final error==" << err << "\n";
///   }
/// 
///   MPI_CHECK(0, err < 1e-10);
/// 
///   _DestroyHcurve();
///   DendroScopeEnd();
/// }
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


template <unsigned dim>
LLU count_explicit_hanging_nodes(const ot::DA<dim> &da)
{
  // Test based on list of nodes, list of octants, and tree search (no matvec).

  // future: (stage 2): Collect ghost cells before searching sequentially.
  if (da.getNpesActive() > 1 and da.getRankActive() == 0)
    std::cerr <<
      "Warning: count_explicit_hanging_nodes() called with multiple processes, "
      "but only uniprocess is supported.\n";

  //TODO stage 1:
  //       for each node, tree search to find neighboring cells
  //         if is a hanging node (in a non-nodal position on any element)
  //         then count as EXPLICIT HANGING
  //           if is a quadratic node (in a non-vertex position on a quadratic element)
  //           then, optionally, label as EXPLICIT HANGING QUADRATIC

  throw std::logic_error("count_explicit_hanging_nodes() is not implemented");
  return -1;
}
