
#include <doctest/extensions/doctest_mpi.h>

#include <mpi.h>

#include "include/dendro.h"
#include "include/distTree.h"
#include "include/oda.h"
#include "FEM/examples/include/poissonVec.h"
#include "FEM/examples/include/hybridPoissonMat.h"
#include "FEM/examples/include/poissonMat.h"
#include "include/mathUtils.h"

using LLU = long long unsigned;

template <int dim>
ot::DistTree<uint32_t, dim> gaussian_tree(LLU global_seeds, double partition_tol, MPI_Comm comm);

template <int dim>
ot::DistTree<uint32_t, dim> uniform_tree(LLU global_seeds, double partition_tol, MPI_Comm comm);

// local_vector()
template <unsigned dim, typename X>
auto local_vector(const ot::DA<dim> &da, X type, int dofs)
{
  std::vector<std::remove_cv_t<decltype(type)>> local;
  const bool ghosted = false;  // local means not ghosted
  da.createVector(local, false, ghosted, dofs);
  return local;
};

namespace PoissonEq
{
#warning "HybridPoissonVec not implemented"

  template <unsigned dim>
  using HybridPoissonVec = PoissonVec<dim>;
}



// ========================================================================== //
MPI_TEST_CASE("Hybrid poisson should match gmg poisson", 1)
{
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  DendroScopeBegin();
  constexpr int dim = 2;
  _InitializeHcurve(dim);

  const double partition_tol = 0.1;
  const int polynomial_degree = 1;
  constexpr double pi = { M_PI };
  static_assert(3.0 < pi, "The macro M_PI is not defined as a value > 3!");

  // Domain is a cube from -1 to 1 in each axis.
  Point<dim> min_corner(-1.0),  max_corner(1.0);

  const int single_dof = 1;

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

  // debug
  const auto u_xpy = [](const double *x) -> double {
    double sum = 0.0;
    for (int d = 0; d < dim; ++d)
      sum += x[d];
    return sum;
  };

  // Mesh (nonuniform grid)
  const LLU global_seeds = 1000;
  const int n_grids = 2;
  ot::DistTree<uint32_t, dim> trees = gaussian_tree<dim>(global_seeds, partition_tol, comm);
  /// ot::DistTree<uint32_t, dim> trees = uniform_tree<dim>(global_seeds, partition_tol, comm);
  ot::DistTree<uint32_t, dim> surrogate_trees = trees.generateGridHierarchyUp(true, n_grids, partition_tol);
  std::vector<ot::DA<dim> *> das(n_grids, nullptr);
  std::vector<ot::DA<dim> *> surrogate_das(n_grids, nullptr);
  for (int g = 0; g < n_grids; ++g)
    das[g] = new ot::DA<dim>(trees, g, comm, polynomial_degree, int{}, partition_tol);
  for (int g = 1; g < n_grids; ++g)
    surrogate_das[g] = new ot::DA<dim>(surrogate_trees, g, comm, polynomial_degree, int{}, partition_tol);
  ot::DA<dim> &base_da = *das[0];
  const size_t local_cells = base_da.getLocalElementSz();
  const LLU global_cells = base_da.getGlobalElementSz();
  const LLU global_nodes = base_da.getGlobalNodeSz();


  // node2unit()
  const auto node2unit = [&](const ot::TreeNode<uint32_t, dim> &node) -> std::array<double, dim>
  {
    std::array<double, dim> coordinates;
    ot::treeNode2Physical(node, polynomial_degree, coordinates.data());
    return coordinates;
  };

  // unit2phys()
  const auto unit2phys = [&](const double *x) -> Point<dim>
  {
    std::array<double, dim> coordinates;
    for (int d = 0; d < dim; ++d)
      coordinates[d] = min_corner.x(d) * (1.0 - x[d]) // scale and shift
                     + max_corner.x(d) * x[d];
    return Point<dim>(coordinates);
  };

  // local_phys
  const auto local_phys = [&](const ot::DA<dim> &da, size_t loc_idx) -> Point<dim>
  {
    const size_t ghost_idx = loc_idx + da.getLocalNodeBegin();
    return unit2phys(node2unit(da.getTNCoords()[ghost_idx]).data());
  };

  // Define system bilinear (lhs) and linear (rhs) operators.
  using PoissonMat = PoissonEq::PoissonMat<dim>;
  using PoissonVec = PoissonEq::PoissonVec<dim>;
  using HybridMat = PoissonEq::HybridPoissonMat<dim>;
  PoissonMat fine_geo_matrix(&base_da, nullptr, single_dof);
  PoissonVec fine_geo_vector(&base_da, nullptr, single_dof);  //future: rm {} octlist

  PoissonMat fine_hyb_matrix_def(&base_da, nullptr, single_dof);
  HybridMat fine_hyb_matrix(&fine_hyb_matrix_def);

  fine_geo_matrix.setProblemDimensions(min_corner, max_corner);
  fine_geo_vector.setProblemDimensions(min_corner, max_corner);
  fine_hyb_matrix.setProblemDimensions(min_corner, max_corner);

  // Vector storage
  std::vector<double> u_vec = local_vector(base_da, double{}, single_dof);
  std::vector<double> v_vec = local_vector(base_da, double{}, single_dof);
  std::vector<double> w_vec = local_vector(base_da, double{}, single_dof);
  std::vector<double> f_vec = local_vector(base_da, double{}, single_dof);
  std::vector<double> rhs_vec = local_vector(base_da, double{}, single_dof);

  std::vector<double> w_evald_vec = local_vector(base_da, double{}, single_dof);

  std::vector<double> u_c_vec = local_vector(*das[1], double{}, single_dof);
  std::vector<double> v_c_vec = local_vector(*das[1], double{}, single_dof);
  std::vector<double> w_c_vec = local_vector(*das[1], double{}, single_dof);


  // Boundary condition and 0 on interior (needed to subtract A u^bdry from rhs)
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    u_vec[i] = 0;
  for (size_t bdyIdx : base_da.getBoundaryNodeIndices())
    u_vec[bdyIdx] = u_bdry(&local_phys(base_da, bdyIdx).x(0));

  // Evaluate forcing term.
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    f_vec[i] = f(&local_phys(base_da, i).x(0));

  // Compute right hand side, subtracting boundary data.
  // (See preMatVec, postMatVec, preComputeVec, postComputeVec).
  fine_geo_matrix.matVec(u_vec.data(), v_vec.data());
  fine_geo_vector.computeVec(f_vec.data(), rhs_vec.data());
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    rhs_vec[i] -= v_vec[i];
  fine_geo_matrix.zero_boundary(true);
  fine_hyb_matrix.matdef()->zero_boundary(true);

  // Precompute exact solution to evaluate error of an approximate solution.
  std::vector<double> u_exact_vec = local_vector(base_da, double{}, single_dof);
  for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
    u_exact_vec[i] = u_exact(&local_phys(base_da, i).x(0));

  // Randomize u_vec
  std::mt19937_64 gen(42);                                  // deterministic
  gen.discard(base_da.getGlobalRankBegin() * single_dof);   // quasi-independent
  std::uniform_real_distribution<double> rand(0.0, 1.0);
  for (double &u : u_vec)
    u = rand(gen);
  for (double &u_c : u_c_vec)
    u_c = rand(gen);

  /// // debug: predictable function
  /// for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
  ///   u_vec[i] = u_xpy(&local_phys(base_da, i).x(0));
  /// for (size_t i = 0; i < das[1]->getLocalNodalSz(); ++i)
  ///   u_c_vec[i] = u_xpy(&local_phys(*das[1], i).x(0));

  fine_geo_matrix.matVec(u_vec.data(), v_vec.data());
  fine_hyb_matrix.matVec(u_vec.data(), w_vec.data());

  // Choose a proper subset of elements to be 'evaluated'
  for (size_t i = 0; i < local_cells / 2; ++i)
    fine_hyb_matrix.store_evaluated(i);
  fine_hyb_matrix.matVec(u_vec.data(), w_evald_vec.data());


  // Compare coarse systems.
  PoissonMat coarse_geo_matrix(das[1], nullptr, single_dof);
  PoissonMat coarse_hyb_matrix_def(das[1], nullptr, single_dof);
  HybridMat coarse_hyb_matrix = fine_hyb_matrix.coarsen(&coarse_hyb_matrix_def);

  coarse_geo_matrix.zero_boundary(true);
  coarse_hyb_matrix.matdef()->zero_boundary(true);

  coarse_geo_matrix.matVec(u_c_vec.data(), v_c_vec.data());
  coarse_hyb_matrix.matVec(u_c_vec.data(), w_c_vec.data());

  //
  const double diff_infty = normLInfty(v_vec.data(), w_vec.data(), v_vec.size(), comm);
  MPI_CHECK(0, diff_infty < 1e-12);

  const double diff_evald_infty = normLInfty(v_vec.data(), w_evald_vec.data(), v_vec.size(), comm);
  MPI_CHECK(0, diff_evald_infty < 1e-12);

  const double coarse_diff_infty = normLInfty(v_c_vec.data(), w_c_vec.data(), v_c_vec.size(), comm);
  MPI_CHECK(0, coarse_diff_infty < 1e-12);

  for (ot::DA<dim> *da: das)
    delete da;
  for (ot::DA<dim> *da: surrogate_das)
    delete da;

  _DestroyHcurve();
  DendroScopeEnd();

  MPI_Comm_free(&comm);
}
// ========================================================================== //



// uniform_tree()
template <int dim>
ot::DistTree<uint32_t, dim> uniform_tree(LLU global_seeds, double partition_tol, MPI_Comm comm)
{
  auto result = ot::DistTree<uint32_t, dim>::constructSubdomainDistTree(
      std::ceil(std::log(global_seeds) / std::log(std::pow(2, dim))),
      comm,
      partition_tol);

  std::cerr << "uniform_tree() -> " << result.getTreePartFiltered().size() << " cells\n";

  return result;
}


// gaussian_tree()
template <int dim>
ot::DistTree<uint32_t, dim> gaussian_tree(LLU global_seeds, double partition_tol, MPI_Comm comm)
{
  const int comm_size = par::mpi_comm_size(comm);
  const int comm_rank = par::mpi_comm_rank(comm);

  const LLU local_begin = global_seeds * comm_rank / comm_size;
  const LLU next_begin = global_seeds * (comm_rank + 1) / comm_size;
  const size_t local_count = next_begin - local_begin;

  // Gaussian point distribution.
  std::vector<ot::TreeNode<uint32_t, dim>> points;

  std::mt19937_64 gen(1331);              // deterministic
  gen.discard(local_begin * dim);   // quasi-independent
  std::normal_distribution<double> distCoord((1u << m_uiMaxDepth) / 2, (1u << m_uiMaxDepth) / 25);

  double coordClampLow = 0;
  double coordClampHi = (1u << m_uiMaxDepth);
  std::array<uint32_t, dim> uiCoords;

  for (int i = 0; i < local_count; ++i)
  {
    for (uint32_t &u : uiCoords)
    {
        double dc = distCoord(gen);
        if (dc < coordClampLow)
          dc = coordClampLow;
        if (dc > coordClampHi)
          dc = coordClampHi;
        u = (uint32_t) dc;
    }
    points.push_back(ot::TreeNode<uint32_t, dim>(uiCoords, m_uiMaxDepth));
  }

  // 2:1-balanced octant list.
  std::vector<ot::TreeNode<uint32_t, dim>> tree;
  ot::SFC_Tree<uint32_t, dim>::distTreeBalancing(points, tree, 3, partition_tol, comm);

  // DistTree.
  auto result = ot::DistTree<uint32_t, dim>(tree, comm);

  std::cerr << "gaussian_tree() -> " << result.getTreePartFiltered().size() << " cells\n";

  return result;
}


