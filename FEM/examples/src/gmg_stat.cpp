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

// =============================================================================
// Configuration
// =============================================================================
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

    void clear()
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
      HighFive::File file(filename, HighFive::File::Overwrite);
      //TODO config
      return file;
    }

    void flush_to_hdf5(
        HighFive::File &file,
        const std::string &groupname)
    {
      //TODO git reference
      //TODO groups, names, attributes for metadata

      assert(this->is_root());
      H5Easy::dump(file, "/" + groupname + "/vcycles", m_vcycle_progress);
      H5Easy::dump(file, "/" + groupname + "/matvecs", m_matvec_progress);
      H5Easy::dump(file, "/" + groupname + "/res_L2", m_residual_L2);
      H5Easy::dump(file, "/" + groupname + "/res_Linf", m_residual_Linf);

      this->clear();
    }

    // output_hdf5()
    HighFive::File flush_to_hdf5(
        HighFive::File &&file,
        const std::string &groupname)
    {
      assert(this->is_root());
      this->flush_to_hdf5(file, groupname);
      return std::move(file);
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


const char * const initial_config = R"(
options:
  help: false
  quiet: false

dim: 2
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

  RootCollect collection(comm);

  HighFive::File *h5_file = nullptr;
  if (collection.is_root())
  {
    h5_file = new auto(collection.create_hdf5("vcycle_data.hdf5", config));
  }


#ifdef CRUNCHTIME
  DendroScopeBegin();
  _InitializeHcurve(dim);
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
  ot::DistTree<uint, dim> base_tree = ot::DistTree<uint, dim>::template
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

  mg::CycleSettings cycle_settings;
  cycle_settings.pre_smooth(2);
  cycle_settings.post_smooth(1);
  cycle_settings.damp_smooth(2.0 / 3.0);

  mg::VCycle<PoissonMat> vcycle(n_grids, das, mats.data(), cycle_settings, single_dof);

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

  struct Residual
  {
    double L2;
    double Linf;
  };

  const auto compute_residual = [](
      auto *mat,
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
  };

  //future: count matvecs with a wrapper

  // Multigrid V-Cycle iteration as solver.
  const int steps = [&, steps=0]() mutable {
    util::ConvergenceRate convergence(3);
    util::ConvergenceRate residual_convergence(3);
    Residual res;
    double err;
    res = compute_residual(
        &base_mat, single_dof, u_vec, v_vec, rhs_vec, comm);
    err = sol_err_max(u_vec);
    collection.observe(steps, steps, res.L2, res.Linf);
    residual_convergence.observe_step(res.L2);
    convergence.observe_step(err);
    if (rank == 0)
    {
      fprintf(stdout, "steps==%2d  err==%e\n", steps, err);
    }
    while (steps < max_vcycles and err > 1e-14)
    {
      vcycle.vcycle(u_vec.data(), v_vec.data());
      ++steps;
      res = compute_residual(
          &base_mat, single_dof, u_vec, v_vec, rhs_vec, comm);
      err = sol_err_max(u_vec);
      collection.observe(steps, steps, res.L2, res.Linf);
      residual_convergence.observe_step(res.L2);
      convergence.observe_step(err);

      const double rate = convergence.rate();
      const double res_rate = residual_convergence.rate();
      if (rank == 0)
      {
        fprintf(stdout, "steps==%2d  err==%e  rate==(%0.1fx)"
            "      res==%e  rate==(%0.1fx)\n",
            steps, err, std::exp(std::abs(std::log(rate))),
            res.L2, std::exp(std::abs(std::log(res_rate))));
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

  if(!rank)
  {
    std::cout << "___________End of poissonEq: "
              << "Finished "
              << "in " << steps << " iterations. "
              << "Final error==" << err << "\n";
  }

  _DestroyHcurve();
  DendroScopeEnd();

  if (collection.is_root())
  {
    collection.flush_to_hdf5(*h5_file, "first");
  }

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

  return 0;
}

// ========================================================================== //


#ifdef CRUNCHTIME
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

  m_configs.clear();
}

// Configuration::help()
std::string Configuration::help() const
{
  return m_help.str();
}






