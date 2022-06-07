//
// Created by masado on 4/27/22.
//

#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro
#include "directories.h"

#include <include/treeNode.h>
#include <include/distTree.h>
#include <IO/hexlist/json_hexlist.h>
#include <include/octUtils.h>

#include <vector>

const static char * bin_dir = DENDRO_KT_DOCTESTS_BIN_DIR;

using uint = unsigned int;

template <int dim>
using Oct = ot::TreeNode<uint, dim>;

template <uint dim> size_t local(const ot::DistTree<uint, dim> &dtree) { return dtree.getTreePartFiltered().size(); }
template <uint dim> const std::vector<Oct<dim>> & octlist(const ot::DistTree<uint, dim> &dtree) { return dtree.getTreePartFiltered(); }

template <int dim>
std::vector<ot::TreeNode<uint, dim>> load_octlist(const std::string &filename, int unit_level);

template <int dim>
std::vector<ot::TreeNode<uint, dim>> str_2_octlist(const std::string &hexlist_str, int unit_level);

template <int dim>
std::vector<ot::TreeNode<uint, dim>> load_and_sort(const std::string &filename, int unit_level, MPI_Comm comm, double sfc_tol = 0.3);

template <int dim>
void gather_and_dump( const std::vector<ot::TreeNode<uint, dim>> &octlist, const std::string &filename_attempt, int unit_level, MPI_Comm comm);

template <int dim>
std::vector<ot::TreeNode<uint, dim>> gather(
    const std::vector<ot::TreeNode<uint, dim>> &octlist,
    MPI_Comm comm);


// tree coarsen/refine:        (Tree) . (Flags) . (Distribution)
// intergrid coarsen/refine    (Finer Tree) . (Finer Distribution) . (Coarser Tree) . (Coarser Distribution) . (Ndofs)

const std::string filepath(const std::string &file) {
  return std::string(bin_dir) + "/assets/" + file;
};

const auto coarsen_all = [](size_t size) {
  return std::vector<ot::OCT_FLAGS::Refine>(size, ot::OCT_FLAGS::OCT_COARSEN);
};

template <unsigned int dim>
unsigned int finest_level(const ot::DistTree<uint, dim> &dt);



MPI_TEST_CASE("Sample hexlist file exists and translates to an octlist of size 3", 1)
{
  constexpr int DIM = 3;
  REQUIRE(load_octlist<DIM>(filepath("tiny3d.hexlist"), 2).size() == 3);
}

MPI_TEST_CASE("Json representation maintains coordinate order and sizes mean multiples of the specified leaf level", 1)
{
  constexpr int DIM = 3;

  const auto data = R"({"data":[[[0,0,0],2],[[2,0,1],1],[[0,0,2],2]],"fields":["from","size"],"format_version":[2022,4,26]})";
  const auto octlist = str_2_octlist<DIM>(data, 2);

  const auto unit = [leaf = 2](std::array<uint, DIM> a) { for (uint &x : a) x <<= (m_uiMaxDepth - leaf);  return a; };
  const std::vector<Oct<DIM>> manual =
      { Oct<DIM>(unit({0,0,0}),1), Oct<DIM>(unit({2,0,1}),2), Oct<DIM>(unit({0,0,2}),1) }; // levels

  REQUIRE(octlist == manual);
}


// Test suite seems to crash prematurely if an assert() SIGABRT gets through.
#if 0
MPI_TEST_CASE("Coarsening with duplicates should fail" * doctest::should_fail(), 1)
{
  MPI_Comm comm = test_comm;
  constexpr int Dim2 = 2;  // causes two of the three boxes to be duplicates
  using DistTree = ot::DistTree<uint, Dim2>;
  const double sfc_tol = 0.3;

  _InitializeHcurve(Dim2);
  auto octree = load_and_sort<Dim2>(filepath("tiny3d.hexlist"), 2, comm, sfc_tol);
  DistTree loaded = {octree, comm},  coarsened, surrogate;
  DistTree::distRemeshSubdomain(
      loaded, coarsen_all(local(loaded)), coarsened, surrogate, ot::SurrogateOutByIn, sfc_tol);
  _DestroyHcurve();
}
#endif


MPI_TEST_CASE("hexlist coarsen uniprocess", 1)
{
  MPI_Comm comm = test_comm;
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  SUBCASE("Dimension 2")
  {
    constexpr int DIM = 2;
    _InitializeHcurve(DIM);
    using OctList = std::vector<Oct<DIM>>;
    using DistTree = ot::DistTree<uint, DIM>;
    const double sfc_tol = 0.3;

    const int unit_level = 4;  // hexlist extents up to 16x16
    std::string base_filename;

    // Run the test function on a different input each time.
    SUBCASE("child of neighbor connected"                      )  { base_filename = "child_of_neighbor_connected.hexlist"; }
    SUBCASE("grandchildren of neighbor connected"              )  { base_filename = "grandchildren_of_neighbor_connected.hexlist"; }
    SUBCASE("grandchildren of neighbor disconnected"           )  { base_filename = "grandchildren_of_neighbor_disconnected.hexlist"; }
    SUBCASE("grandchildren of neighbor of sibling disconnected")  { base_filename = "grandchildren_of_neighbor_of_sibling_disconnected.hexlist"; }
    const std::string filename_input = filepath(base_filename);
    const std::string filename_answer = filepath("coarsen_all+" + base_filename);
    const std::string filename_attempt = "_output/attempt_coarsen_all+" + base_filename;

    // Create DistTree from loaded file.
    OctList octlist_input  = load_and_sort<DIM>(filename_input, unit_level, comm, sfc_tol);
    DistTree dtree_loaded(octlist_input, comm);

    // Apply distRemeshSubdomain coarsening algorithm.
    DistTree dtree_coarsened, dtree_surrogate;
    DistTree::distRemeshSubdomain( dtree_loaded, coarsen_all(local(dtree_loaded)),
        dtree_coarsened, dtree_surrogate,
        ot::SurrogateOutByIn, sfc_tol);

    // Checks that must pass.
    CHECK(finest_level(dtree_coarsened) + 1 == finest_level(dtree_loaded));

    // Emit a warning if the result does not exactly match the expected answer.
    // 2:1-balanced coarsened grid, while preserving domain where uncoarsened.
    OctList octlist_answer;
    if (comm_rank == 0)
      octlist_answer = load_and_sort<DIM>(filename_answer, unit_level, MPI_COMM_SELF);
    OctList octlist_coarsened = gather<DIM>(octlist(dtree_coarsened), comm);
    INFO("Output in ", filename_attempt);
    WARN(octlist_coarsened == octlist_answer);
    gather_and_dump<DIM>(octlist(dtree_coarsened), filename_attempt, unit_level, comm);

    _DestroyHcurve();
  }
}


// load_octlist()
template <int dim>
std::vector<ot::TreeNode<uint, dim>> load_octlist(const std::string &filename, int unit_level)
{
  nlohmann::json json_hexlist;
  std::ifstream in_file(filename);
  REQUIRE_MESSAGE(bool(in_file), "Could not open ", filename);
  in_file >> json_hexlist;
  return io::JSON_Hexlist(json_hexlist).to_octlist<uint, dim>(unit_level);
}

// str_2_octlist()
template <int dim>
std::vector<ot::TreeNode<uint, dim>> str_2_octlist(const std::string &hexlist_str, int unit_level)
{
  nlohmann::json json_hexlist = nlohmann::json::parse(hexlist_str);
  return io::JSON_Hexlist(json_hexlist).to_octlist<uint, dim>(unit_level);
}

// load_and_sort()
template <int dim>
std::vector<ot::TreeNode<uint, dim>> load_and_sort(const std::string &filename, int unit_level, MPI_Comm comm, double sfc_tol)
{
  std::vector<ot::TreeNode<uint, dim>> octlist;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  if (comm_rank == 0)
    octlist = load_octlist<dim>(filename, unit_level);

  ot::SFC_Tree<uint, dim>::distTreeSort(octlist, sfc_tol, comm);

  return octlist;
}


// gather()
template <int dim>
std::vector<ot::TreeNode<uint, dim>> gather(
    const std::vector<ot::TreeNode<uint, dim>> &octlist,
    MPI_Comm comm)
{
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  int local_count = octlist.size();
  std::vector<int> counts;
  if (comm_rank == 0)
    counts.resize(comm_size);

  par::Mpi_Gather(&local_count, counts.data(), 1, 0, comm);

  std::vector<ot::TreeNode<uint, dim>> gathered_octlist;

  std::vector<int> offsets;
  int global_count = 0;
  if (comm_rank == 0)
  {
    for (int count : counts)
    {
      offsets.push_back(global_count);
      global_count += count;
    }
    gathered_octlist.resize(global_count);
  }

  const auto datatype = par::Mpi_datatype<ot::TreeNode<uint, dim>>::value();

  MPI_Gatherv(octlist.data(), octlist.size(), datatype,
      gathered_octlist.data(), counts.data(), offsets.data(), datatype,
      0, comm);

  return gathered_octlist;
}


// gather_and_dump()
template <int dim>
void gather_and_dump(
    const std::vector<ot::TreeNode<uint, dim>> &octlist,
    const std::string &filename_attempt,
    int unit_level,
    MPI_Comm comm)
{
  const std::vector<ot::TreeNode<uint, dim>> gathered_octlist =
      gather<dim>(octlist, comm);

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  if (comm_rank == 0)
  {
    std::ofstream out_file(filename_attempt);
    REQUIRE_MESSAGE(bool(out_file), "Could not open ", filename_attempt);
    out_file << nlohmann::json(
        io::JSON_Hexlist::from_octlist(gathered_octlist, unit_level));
  }
}


// finest_level()
template <unsigned int dim>
unsigned int finest_level(const ot::DistTree<uint, dim> &dt)
{
  MPI_Comm comm = dt.getComm();
  const auto &ot = octlist(dt);
  const auto finer = [](auto &a_lev, auto &b_oct) { return std::max(a_lev, b_oct.getLevel()); };
  return par::mpi_max(std::accumulate(ot.begin(), ot.end(), 0u, finer), comm);
}

