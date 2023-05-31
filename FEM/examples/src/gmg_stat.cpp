//
// Created by masado on 10/19/22.
//

#define CRUNCHTIME    // undefine if debugging input/output only

#include <mpi.h>

#include "include/dendro.h"
#include "include/parUtils.h"
#ifdef CRUNCHTIME
#include "include/distTree.h" // for convenient uniform grid partition
#include "include/point.h"
#include "FEM/examples/include/poissonMat.h"
#include "FEM/examples/include/poissonVec.h"
#include "FEM/include/mg.hpp"
#include "FEM/include/solver_utils.hpp"
#include <petsc.h>
#else//CRUNCHTIME
#define PetscInitialize(...)
#define PetscFinalize()
#endif//CRUNCHTIME

#include <vector>
#include <functional>
#include <sstream>

#include "highfive/H5File.hpp"
#include "highfive/H5Easy.hpp"

#include "c4/conf/conf.hpp"
#include "ryml_std.hpp"  // Add this for STL interop with ryml

#include "c4/yml/detail/print.hpp"

// -----------------------------
// Typedefs
// -----------------------------
#ifdef CRUNCHTIME
using uint = unsigned int;
using LLU = long long unsigned;

template <unsigned int dim, typename NodeT>
std::ostream & print(const ot::DA<dim> &da,
                     const std::vector<NodeT> &local_vec,
                     std::ostream & out = std::cerr);

template <unsigned int dim, typename NodeT>
std::ostream & print(const ot::DA<dim> &da,
                     const NodeT *local_vec,
                     std::ostream & out = std::cerr);
#endif//CRUNCHTIME

// -----------------------------
// Functions
// -----------------------------

std::string git_show();

class Configuration;

// Forward-declare tmain()
template <int dim>  int tmain(int argc, char *argv[], Configuration &);
template int tmain<2>(int argc, char *argv[], Configuration &);
template int tmain<3>(int argc, char *argv[], Configuration &);
template int tmain<4>(int argc, char *argv[], Configuration &);



// =============================================================================
// Configuration
// =============================================================================
// Declare default configuration.
extern const char * const initial_config;

// Read command line options (including config filename) using c4conf,
// but only read the config file from mpi process rank 0.

// Extract command line arguments before PETSc reads command line,
// otherwise PETSc will warn about misspellings and unused options.

template <typename X>
X to(const c4::yml::ConstNodeRef &node)
{
  X x;
  node >> x;
  return x;
}

class Configuration
{
  public:
    Configuration() = default;
    Configuration(const std::string & initial);
    Configuration(int *argc, char ***argv);
    inline void eat_command_line(int *argc, char ***argv);
    inline void mpi_apply(MPI_Comm comm);
    void dump(std::ostream &out) const;
    template <typename Index>
    auto operator[](Index &&index);

    template <typename Index>
    auto operator[](Index &&index) const;

    std::string help() const;

  private:
    std::vector<c4::conf::ParsedOpt> m_configs;
    c4::yml::Tree m_yaml_tree;
    std::stringstream m_help;
};

// Configuration::operator[]
template <typename Index>
auto Configuration::operator[](Index &&index)
{
  return m_yaml_tree[std::forward<Index>(index)];
}

// Configuration::operator[]
template <typename Index>
auto Configuration::operator[](Index &&index) const
{
  return m_yaml_tree[std::forward<Index>(index)];
}

// Configuration::eat_command_line()
void Configuration::eat_command_line(int *argc, char ***argv)
{
  using namespace c4::conf;  // Configuration library
  using c4::yml::Tree;       // YAML tree
  using c4::csubstr;         // Constant string range view

  ConfigActionSpec conf_specs[] = {
    spec_for<ConfigAction::load_file>("-yf", "--yaml-file"),
    spec_for<ConfigAction::set_node>("-c"),
    spec_for([](Tree &t, csubstr){t["options"]["help"] << true;}, "-h", "--help", {}, "Print options and exit."),
    spec_for([](Tree &t, csubstr){t["options"]["quiet"] << true;}, "-q", "--quiet", {}, "Do not dump YAML configuration tree on startup."),
  };

  print_help([&, this](csubstr s){this->m_help << s;},
      conf_specs, C4_COUNTOF(conf_specs), "options");

  m_configs = parse_opts<std::vector<ParsedOpt>>(
      argc, argv, conf_specs, C4_COUNTOF(conf_specs));
}
// =============================================================================


// =============================================================================
// RootCollect
// =============================================================================
class RootCollect
{
  public:
    RootCollect(MPI_Comm comm)
      : m_comm(comm), m_is_root(par::mpi_comm_rank(comm) == 0)
    {
      if (m_is_root)
      {
        m_git_metadata = git_show();
      }
    }

    bool is_root() const { return m_is_root; }

    void clear_solver()
    {
      m_vcycle_progress.clear();
      m_matvec_progress.clear();
      m_residual_L2.clear();
      m_residual_Linf.clear();
    }

    // observe()
    void observe(
        int vcycle_progress,
        int matvec_progress,
        double residual_L2,
        double residual_Linf)
    {
      if (m_is_root)
      {
        m_vcycle_progress.push_back(vcycle_progress);
        m_matvec_progress.push_back(matvec_progress);
        m_residual_L2.push_back(residual_L2);
        m_residual_Linf.push_back(residual_Linf);
      }
    }

    // create_hdf5()
    HighFive::File create_hdf5(
        const std::string &filename,
        const Configuration &config) const
    {
      assert(this->is_root());

      auto mode = HighFive::File::ReadWrite;
      if (to<bool>(config["overwrite_all"]))
        mode = HighFive::File::Overwrite;

      int failed_attempts = 0;
      while (failed_attempts < 3)
      {
        try
        {
          HighFive::File file(filename, mode);
          return file;
        }
        catch (const HighFive::Exception &e)
        {
          ++failed_attempts;
          if (failed_attempts == 3)
            throw e;
          std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }
      }
      //future: config
      //future: git reference
      return HighFive::File(filename, mode);
    }

    HighFive::Group & flush_to_hdf5(
        HighFive::Group &group)
    {
      assert(this->is_root());
      group.createDataSet("vcycles", m_vcycle_progress);
      group.createDataSet("matvecs", m_matvec_progress);
      group.createDataSet("res_L2", m_residual_L2);
      group.createDataSet("res_Linf", m_residual_Linf);
      this->clear_solver();
      return group;
    }

    HighFive::Group flush_to_hdf5(
        HighFive::Group &&group)
    {
      assert(this->is_root());
      return std::move(this->flush_to_hdf5(group));
    }

  private:
    MPI_Comm m_comm;
    bool m_is_root;
    std::string m_git_metadata;
    std::vector<int> m_vcycle_progress;
    std::vector<int> m_matvec_progress;
    std::vector<double> m_residual_L2;
    std::vector<double> m_residual_Linf;
};
// =============================================================================



// =============================================================================
// Solver choices
// =============================================================================

// gmg_solver()
template <typename MatType>
int gmg_solver(
    c4::yml::ConstNodeRef run_config,
    RootCollect &collection,
    mg::VCycle<MatType> &vcycle,
    std::vector<double> &u_vec,
    std::vector<double> &v_vec,
    const std::vector<double> &rhs_vec,
    int max_vcycles,
    double relative_residual);


// amg_solver()
template <typename MatType>
int amg_solver(
    c4::yml::ConstNodeRef run_config,
    RootCollect &collection,
    MatType *mat,
    const mg::CycleSettings &cycle_settings,
    std::vector<double> &u_vec,
    std::vector<double> &v_vec,
    const std::vector<double> &rhs_vec,
    int max_vcycles,
    double relative_tolerance);


// =============================================================================




const char * const initial_config = R"(
options:
  help: false
  quiet: false

output: vcycle_data.hdf5

overwrite_all: true

dim: -1
)";


// =============================================================================
// Main
// =============================================================================
int main(int argc, char *argv[])
{
  int return_code = 1;
  MPI_Init(&argc, &argv);

  bool run = true;

  // Make c4conf eat the command line arguments before PetscInitialize().
  Configuration config(initial_config);
  config.eat_command_line(&argc, &argv);
  PetscInitialize(&argc, &argv, NULL, NULL);
  config.mpi_apply(MPI_COMM_WORLD);

  const bool is_mpi_root = (par::mpi_comm_rank(MPI_COMM_WORLD) == 0);
  if (to<bool>(config["options"]["help"]))
  {
    if (is_mpi_root)
      std::cout << config.help();
    run = false;
  }

  if (not to<bool>(config["options"]["quiet"]))
  {
    if (is_mpi_root)
      config.dump(std::cout);
  }

  if (run)
  {
    assert(config[""].has_child("dim"));
    const int dim = to<int>(config["dim"]);
    switch (dim)
    {
      case 2:     return_code = tmain<2>(argc, argv, config); break;
      case 3:     return_code = tmain<3>(argc, argv, config); break;
      case 4:     return_code = tmain<4>(argc, argv, config); break;
      default:    if (is_mpi_root)
                    std::cerr << "Unsupported dim=" << dim << "\n";
    }
  }

  PetscFinalize();
  MPI_Finalize();
  return return_code;
}

template <int dim>
int tmain(int argc, char *argv[], Configuration &config)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  const int rank = par::mpi_comm_rank(comm);
  assert(dim == to<int>(config["dim"]));
  //future: write relvant configurations to log
  //future: remember which configurations were accessed, log those

  enum OverwriteMode { OverwriteAll, OverwriteSome };
  OverwriteMode overwrite_mode = OverwriteAll;
  if (to<bool>(config["overwrite_all"]) == false)
    overwrite_mode = OverwriteSome;

  const std::string out_filename = to<std::string>(config["output"]);

  RootCollect collection(comm);

  HighFive::File *h5_file = nullptr;
  if (collection.is_root())
  {
    h5_file = new auto(collection.create_hdf5(out_filename, config));
  }

#ifdef CRUNCHTIME
  DendroScopeBegin();
  _InitializeHcurve(dim);
  const double partition_tolerance = 0.1;
  constexpr double pi = { M_PI };
  static_assert(3.0 < pi, "The macro M_PI is not defined as a value > 3!");

  // ---------------------------------------------------------------------------
  // Problem definition
  // ---------------------------------------------------------------------------

  // Domain is a cube from -1 to 1 in each axis.
  Point<dim> min_corner(-1.0),  max_corner(1.0);

  const double domain_length = (max_corner - min_corner).x();

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

  int all_runs = -1;
  for (c4::yml::ConstNodeRef setup: config["setups"])
  {
    // -------------------------------------------------------------------------
    // Discrete mesh
    // -------------------------------------------------------------------------
    ot::DistTree<uint, dim> base_tree;
    std::string construction;
    setup["mesh_recipe"]["construct"] >> construction;
    const int polynomial_degree = to<int>(setup["mesh_recipe"]["degree"]);
    std::string mesh_name;
    if (construction ==  "by_func")
    {
      const double interpolation = to<double>(setup["interpolation"]);
      base_tree = ot::DistTree<uint, dim>::template
          constructDistTreeByFunc<double>(
              f, 1, comm, polynomial_degree, interpolation, partition_tolerance);
      mesh_name = "adaptive(" + to<std::string>(setup["interpolation"]) + ")";
    }
    else if (construction == "uniform")
    {
      const int max_depth = to<int>(setup["max_depth"]);
      base_tree = ot::DistTree<uint, dim>::
          constructSubdomainDistTree(max_depth, comm, partition_tolerance);
      mesh_name = "uniform(" + to<std::string>(setup["max_depth"]) + ")";
    }
    ot::DA<dim> base_da(base_tree, comm, polynomial_degree, int{}, partition_tolerance);

    const double spacing = par::mpi_min(
        domain_length * ot::treeNode2Physical(
          std::min_element(
            base_tree.getTreePartFiltered().begin(),
            base_tree.getTreePartFiltered().end(),
            [](const auto &a, const auto &b) {
                return a.range().side() < b.range().side();
            })->range().side()
          ),
        comm);
    std::cout << "spacing = " << spacing << "\n";

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
    const auto fe_matrix = [&](const ot::DA<dim> &da) -> PoissonMat
    {
      PoissonMat mat(&da, {}, single_dof);
      mat.setProblemDimensions(min_corner, max_corner);
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
    PoissonEq::PoissonMat<dim> base_mat = fe_matrix(base_da);
    PoissonEq::PoissonVec<dim> base_vec = fe_vector(base_da);

    // Compute right hand side, subtracting boundary data.
    // (See preMatVec, postMatVec, preComputeVec, postComputeVec).
    base_mat.matVec(u_vec.data(), v_vec.data());
    base_vec.computeVec(f_vec.data(), rhs_vec.data());
    for (size_t i = 0; i < base_da.getLocalNodalSz(); ++i)
      rhs_vec[i] -= v_vec[i];
    base_mat.zero_boundary(true);

    // Copy initial guess to be re-used between solves.
    const std::vector<double> u_vec_initial = u_vec;

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
      mats[g] = new PoissonMat(fe_matrix(*das[g].primary));
      mats[g]->zero_boundary(true);
    }

    int run_idx = -1;
    for (c4::yml::ConstNodeRef run: setup["runs"])
    {
      ++run_idx;
      ++all_runs;

      if (not to<bool>(run["active"]))
      {
        if (collection.is_root())
          std::cout << "Skipping run [" << run_idx << "] (inactive).\n";
        continue;
      }

      bool skip_existing = false;
      const bool force_overwrite =
          run.has_child("force_overwrite") and to<bool>(run["force_overwrite"]);
      if (overwrite_mode == OverwriteSome and not force_overwrite)
      {
        if (collection.is_root())
        {
          skip_existing = h5_file->exist(std::to_string(all_runs));
        }
        par::Mpi_Bcast(&skip_existing, 1, 0, comm);
      }

      if (skip_existing)
      {
        if (collection.is_root())
          std::cout << "Skipping run [" << run_idx << "] (already exists).\n";
        continue;
      }

      if (collection.is_root())
        std::cout << "Executing run [" << run_idx << "].\n";

      std::copy(u_vec_initial.cbegin(), u_vec_initial.cend(), u_vec.begin());

      const double tol=1e-14;
      const unsigned int max_iter=300;
      const int max_vcycles = 40;

      if (run["solver"]["class"].val() == "multigrid")
      {
        mg::CycleSettings cycle_settings;
        cycle_settings.n_grids(n_grids);
        cycle_settings.pre_smooth(2);
        cycle_settings.post_smooth(1);
        cycle_settings.damp_smooth(2.0 / 3.0);

        if (run["solver"]["type"].val() == "GMG")
        {
          mg::VCycle<PoissonMat> vcycle(das, mats.data(), cycle_settings, single_dof);

          const int steps = gmg_solver(
              run, collection,
              vcycle, u_vec, v_vec, rhs_vec, max_vcycles, tol);

          const double err = sol_err_max(u_vec);
          const double res = normLInfty(v_vec.data(), v_vec.size(), comm);
        }
        else if (run["solver"]["type"].val() == "AMG")
        {
          const int steps = amg_solver(
              run, collection,
              &base_mat, cycle_settings, u_vec, v_vec, rhs_vec, max_vcycles, tol);

          const double err = sol_err_max(u_vec);
          const double res = normLInfty(v_vec.data(), v_vec.size(), comm);
        }
      }

      if (collection.is_root())
      {
        const std::string group_name = std::to_string(all_runs);
        if (h5_file->exist(group_name))
          h5_file->unlink(group_name);
        HighFive::Group group = h5_file->createGroup(group_name);
        group.createAttribute("solver", to<std::string>(run["solver"]["name"]));
        group.createAttribute("mesh_family", to<std::string>(setup["mesh_recipe"]["name"]));
        group.createAttribute("mesh", mesh_name);
        group.createAttribute("cells", base_da.getGlobalElementSz());
        group.createAttribute("spacing", spacing);
        group.createAttribute("unknowns", base_da.getGlobalNodeSz() * single_dof);
        collection.flush_to_hdf5(group);
      }
    }

    // Multigrid teardown.
    for (int g = 1; g < n_grids; ++g)
    {
      delete das[g].primary;
      delete das[g].surrogate;
      delete mats[g];
    }
    das.clear();
    mats.clear();

  }



  _DestroyHcurve();
  DendroScopeEnd();

#else//CRUNCHTIME

  std::cout << "Dry run ...............\n";

  if (collection.is_root())
  {
    collection.observe(0, 0, 1.0, 1.0);
    collection.observe(1, 1, 0.5, 0.5);
    collection.flush_to_hdf5(*h5_file, "first");

    collection.observe(0, 0, 1.0, 1.0);
    collection.observe(1, 1, 0.4, 0.4);
    collection.observe(3, 3, 0.25, 0.25);
    collection.flush_to_hdf5(*h5_file, "second");
  }

#endif//CRUNCHTIME

  delete h5_file;

  return 0;
}

// ========================================================================== //


#ifdef CRUNCHTIME


// =============================================================================
// Helper structs for solvers
// =============================================================================

// Residual
struct Residual
{
  double L2;
  double Linf;
};

// compute_residual()
template <typename MatType>
Residual compute_residual(
  MatType *mat,
  int ndofs,
  const std::vector<double> &u_vec,
  std::vector<double> &v_vec,
  const std::vector<double> &rhs_vec,
  MPI_Comm comm)
{
  mat->matVec(u_vec.data(), v_vec.data());
  subt(rhs_vec.data(), v_vec.data(), v_vec.size(), v_vec.data());
  const DendroIntL global_entries = (mat->da()->getGlobalNodeSz() * ndofs);
  Residual res = {};
  /// res.L2 = normL2(v_vec.data(), v_vec.size(), comm) / global_entries;
  res.L2 = normL2(v_vec.data(), v_vec.size(), comm);
  res.Linf = normLInfty(v_vec.data(), v_vec.size(), comm);
  return res;
}



// =============================================================================

// =============================================================================
// gmg_solver()
// =============================================================================
//
template <typename MatType>
int gmg_solver(
    c4::yml::ConstNodeRef run_config,
    RootCollect &collection,
    mg::VCycle<MatType> &vcycle,
    std::vector<double> &u_vec,
    std::vector<double> &v_vec,
    const std::vector<double> &rhs_vec,
    int max_vcycles,
    double relative_tolerance)
{
  //future: count matvecs with a wrapper

  auto &base_mat = *vcycle.mats[0];
  const int ndofs = base_mat.ndofs();
  const MPI_Comm comm = base_mat.da()->getCommActive();

  assert(run_config["solver"]["ksp"].val() == "Stand");

  // Multigrid V-Cycle iteration as solver.
  int steps = 0;
  util::ConvergenceRate residual_convergence(3);
  Residual res;
  res = compute_residual(
      &base_mat, ndofs, u_vec, v_vec, rhs_vec, comm);
  collection.observe(steps, steps, res.L2, res.Linf);
  residual_convergence.observe_step(res.L2);
  if (collection.is_root())
  {
    fprintf(stdout, "steps==%2d  res==%e\n", steps, res.L2);
  }
  const double initial_res = res.L2;
  while (steps < max_vcycles and res.L2 > initial_res * relative_tolerance)
  {
    vcycle.vcycle(u_vec.data(), v_vec.data());
    ++steps;
    res = compute_residual(
        &base_mat, ndofs, u_vec, v_vec, rhs_vec, comm);
    collection.observe(steps, steps, res.L2, res.Linf);
    residual_convergence.observe_step(res.L2);

    const double res_rate = residual_convergence.rate();
    if (collection.is_root())
    {
      fprintf(stdout, "steps==%2d  "
          "      res==%e  rate==(%0.1fx)\n",
          steps, res.L2, std::exp(std::abs(std::log(res_rate))));
    }
    if (res_rate > 0.95)
      break;
  }
  return steps;
}
// =============================================================================




// =============================================================================
// amg_solver()
// =============================================================================
//
template <typename MatType>
int amg_solver(
    c4::yml::ConstNodeRef run_config,
    RootCollect &collection,
    MatType *mat,
    const mg::CycleSettings &cycle_settings,
    std::vector<double> &u_vec,
    std::vector<double> &v_vec,
    const std::vector<double> &rhs_vec,
    int max_vcycles,
    double relative_tolerance)
{
  const auto *da = mat->da();
  const MPI_Comm comm = da->getCommActive();


  // PETSc
  Mat petsc_mat;
  Vec u, rhs;
  KSP ksp;
  PC pc;

  // Assemble the matrix (assuming one-time assembly).
  da->createMatrix(petsc_mat, MATAIJ, mat->ndofs());
  mat->getAssembledMatrix(&petsc_mat, {});
  MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);

  const size_t local_size = da->getLocalNodalSz() * mat->ndofs();
  const size_t global_size = da->getGlobalNodeSz() * mat->ndofs();
  VecCreateMPIWithArray(comm, 1, local_size, global_size, nullptr, &u);
  VecCreateMPIWithArray(comm, 1, local_size, global_size, nullptr, &rhs);

  std::vector<int> global_ids_of_local_boundary_nodes(da->getBoundaryNodeIndices().size());
  std::copy(da->getBoundaryNodeIndices().cbegin(),
            da->getBoundaryNodeIndices().cend(),
            global_ids_of_local_boundary_nodes.begin());
  for (int &id : global_ids_of_local_boundary_nodes)
    id += da->getGlobalRankBegin();
  // If ndofs != 1 then need to duplicate and zip.
  assert(mat->ndofs() == 1);

  MatZeroRows(
      petsc_mat,
      global_ids_of_local_boundary_nodes.size(),
      global_ids_of_local_boundary_nodes.data(),
      1.0,
      NULL, NULL);

  //TODO add from settings
  /// cycle_settings.n_grids();
  /// cycle_settings.pre_smooth();
  /// cycle_settings.post_smooth();
  /// cycle_settings.damp_smooth();

  // Coarse solver setup with PETSc.
  KSPCreate(da->getCommActive(), &ksp);
  KSPSetOperators(ksp, petsc_mat, petsc_mat);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
  KSPConvergedDefaultSetUIRNorm(ksp);
  KSPSetTolerances(ksp, relative_tolerance, 0.0, 10.0, max_vcycles);

  // See also KSPSetPCSide(PC_SYMMETRIC)

  auto monitor_state = [](
      KSP, int iterations, double res_L2, void *collect) -> int
  {
    //future: accurately count the number of matvecs 
    static_cast<RootCollect*>(collect)->observe(iterations, iterations,
        res_L2, std::numeric_limits<PetscReal>::quiet_NaN());
    return 0;
  };
  KSPMonitorSet(ksp, monitor_state, &collection, NULL);

  // Switch outer Krylov method. Standalone M.G. is Richardson.
  const std::string ksp_choice = to<std::string>(run_config["solver"]["ksp"]);
  if (ksp_choice == "Stand")
    KSPSetType(ksp, KSPRICHARDSON);
  else if (ksp_choice == "CG")
    KSPSetType(ksp, KSPCG);
  else
  {
    if (collection.is_root())
      std::cerr << "Warning: Ignoring ksp choice " << ksp_choice << ".\n";
      // Note that the default is GMRES.
  }

  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCGAMG);  // solver choice.
  KSPSetUp(ksp);

  /// if (collection.is_root())
  /// {
  ///   KSPView(ksp, PETSC_VIEWER_STDOUT_SELF);
  /// }

  // Bind u and rhs to Vec
  VecPlaceArray(u, u_vec.data());
  VecPlaceArray(rhs, rhs_vec.data());

  // Solve.
  KSPSolve(ksp, rhs, u);

  /// // Debug for solution magnitude.
  /// PetscReal sol_norm = 0.0;
  /// VecNorm(u, NORM_INFINITY, &sol_norm);

  // Unbind from Vec.
  VecResetArray(u);
  VecResetArray(rhs);

  // Get residual norm and number of iterations.
  int iterations = 0;
  PetscReal res_norm = 0.0;
  KSPGetIterationNumber(ksp, &iterations);
  KSPGetResidualNorm(ksp, &res_norm);

  /// Residual res = compute_residual(
  ///     mat, mat->ndofs(), u_vec, v_vec, rhs_vec, comm);
  /// collection.observe(iterations, iterations, res.L2, res.Linf);

  if (collection.is_root())
  {
    std::cout << "ksp iterations == " << iterations << "\n";
  }

  VecDestroy(&u);
  VecDestroy(&rhs);
  KSPDestroy(&ksp);
  MatDestroy(&petsc_mat);

  return iterations;
}
// =============================================================================



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
#endif//CRUNCHTIME

// git_show()
std::string git_show()
{
  std::string commit_string;
  FILE *stream = popen("git show -s --pretty='format:%H	%ad	%f'", "r");
  if (stream != NULL)
  {
    const size_t extend = 10;
    commit_string.resize(extend, '\0');
    size_t cursor = 0;
    //future: fread() may be more direct
    while (fgets(&commit_string[cursor], commit_string.size() - cursor, stream) != NULL)
    {
      cursor = commit_string.size() - 1;
      commit_string.resize(commit_string.size() + extend, '\0');
      commit_string.resize(commit_string.capacity(), '\0');
    }
    if (ferror(stream))
      commit_string.clear();
    else
      commit_string.resize(strlen(commit_string.data()));

    const int status = pclose(stream);
    // Could check error status here, if interested.
  }
  return commit_string;
}



// =============================================================================
// Configuration
// =============================================================================

// Configuration::Configuration()
Configuration::Configuration(const std::string &initial)
  : m_yaml_tree(c4::yml::parse_in_arena("(initial)", c4::to_csubstr(initial)))
{ }

// Configuration::Configuration()
Configuration::Configuration(int *argc, char ***argv)
  : Configuration()
{
  this->eat_command_line(argc, argv);
}

// Configuration::dump()
void Configuration::dump(std::ostream &out) const
{
  out << m_yaml_tree;
}

// Configuration::mpi_apply()
void Configuration::mpi_apply(MPI_Comm comm)
{
  const bool is_root = (par::mpi_comm_rank(comm) == 0);
  //future: rank 0 apply_opts(), stringify yaml, broadcast, back from string.
  /// std::string final_yaml;  // use emitrs_yaml<std::string>
  /// if (is_root)
  /// {
  ///   //...
  /// }
  /// // broadcast final_yaml.size()
  /// // broadcast on final_yaml

  // for now: apply on all ranks
  c4::conf::Workspace workspace(&m_yaml_tree);
  workspace.apply_opts(m_configs);

  /// c4::yml::print_tree(m_yaml_tree);

  m_yaml_tree.resolve();  // aliases -> subtrees

  m_configs.clear();
}

// Configuration::help()
std::string Configuration::help() const
{
  return m_help.str();
}





      /// const int steps = cgSolver(
      ///     &base_da, [&base_mat](const double *u, double *v){ base_mat.matVec(u, v); },
      ///     &(*u_vec.begin()), &(*rhs_vec.begin()), max_iter, tol, true);

      /// const int steps = cgSolver(
      ///     &base_da, pc_mat,
      ///     &(*u_vec.begin()), &(*pc_rhs_vec.begin()), max_iter, tol, true);


