// Created by masado on 2024-06-21

#include <iostream>
#include "include/distTree.h"
#include "include/oda.h"

struct Arguments
{
  int argc;
  char **argv;
};

void print_usage(const Arguments &arguments, std::ostream &out);
auto arguments_are_valid(const Arguments &arguments) -> bool;
auto dimension(const Arguments &arguments) -> long;
auto depth(const Arguments &arguments) -> long;

template <int dim>
int tmain(int argc, char * argv[]);

int main(int argc, char * argv[])
{
  const Arguments arguments = {argc, argv};
  if (not arguments_are_valid(arguments))
  {
    print_usage(arguments, std::cerr);
    return 1;
  }

  const int dim = dimension(arguments);
  switch (dim)
  {
    case 2: return tmain<2>(argc, argv);
    case 3: return tmain<3>(argc, argv);
    case 4: return tmain<4>(argc, argv);
    default: return 1;
  }
}

template <int dim>
int tmain(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  _InitializeHcurve(dim);

  const Arguments arguments = {argc, argv};
  MPI_Comm comm = MPI_COMM_SELF;

  // Create uniform hyperoctree to the prescribed depth.
  ot::DistTree<unsigned, dim> tree =
      ot::DistTree<unsigned, dim>::constructSubdomainDistTree(
          depth(arguments), comm);

  const int degree = 2;

  const struct MeshStats {
    long long n_points;
    long long n_cells;
  } mesh_stats = [&] {

    ot::DA<dim> warmup_da(tree, comm, degree);  

    return MeshStats{ warmup_da.getGlobalNodeSz(), warmup_da.getGlobalElementSz() };
  }();

  const int runs = 5;

  using clock = std::chrono::high_resolution_clock;
  const clock::time_point start = clock::now();
  for (int run = 0; run < runs; ++run)
  {
    ot::DA<dim> da(tree, comm, degree);  
  }
  const clock::time_point stop = clock::now();

  const auto total_milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  const double milliseconds_per_run = total_milliseconds.count() / double(runs);

  std::cout << "Counted " << mesh_stats.n_points << " points ("
      << mesh_stats.n_points - mesh_stats.n_cells << " faces) in "
      << milliseconds_per_run << " milliseconds (mean of "
      << runs << " runs)\n";

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}


void print_usage(const Arguments &arguments, std::ostream &out)
{
  out << arguments.argv[0];

  for (const std::string &argument_name:
      {"dimension(2..4)", "tree_depth(1..30)"})
  {
    out << ' ' << argument_name;
  }

  out << "\n";
}

auto arguments_are_valid(const Arguments &arguments) -> bool
{
  const auto dimension_in_range = [](long dim) { return 2 <= dim and dim <= 4; };
  const auto depth_in_range = [](long depth) { return 1 <= depth and depth <= 30; };
  return arguments.argc >= 3 and
      dimension_in_range(dimension(arguments)) and
      depth_in_range(depth(arguments));
}

auto dimension(const Arguments &arguments) -> long
{
  return std::atol(arguments.argv[1]);
};

auto depth(const Arguments &arguments) -> long
{
  return std::atol(arguments.argv[2]);
}


