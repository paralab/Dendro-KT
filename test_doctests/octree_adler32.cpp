#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro

#include <include/distTree.h>
#include <IO/hexlist/json_hexlist.h>

using uint = unsigned int;
template <uint dim> auto l_sz(const ot::DistTree<uint, dim> &dt) { return dt.getFilteredTreePartSz(); }
template <uint dim> const auto & l_ot(const ot::DistTree<uint, dim> &dt) { return dt.getTreePartFiltered(); }

template <int dim>
void hexlist_files(
    const std::vector<ot::TreeNode<uint, dim>> &octlist,
    const std::string &filename_prefix,
    // will fill in "prefix_${rank}suffix"
    const std::string &filename_suffix,
    int unit_level,
    MPI_Comm comm);


MPI_TEST_CASE("OctreeAdler32 is partition independent, 3 processes", 3)
{
  MPI_Comm comm = test_comm;
  const int comm_size = par::mpi_comm_size(comm);
  const int comm_rank = par::mpi_comm_rank(comm);

  MPI_Comm sub_comm;
  const int color = (comm_rank == 0 or comm_rank == comm_size - 1) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(comm, color, comm_rank, &sub_comm);

  constexpr int DIM = 3;
  _InitializeHcurve(DIM);

  using DistTree = ot::DistTree<uint, DIM>;
  DistTree tree_gap;
  const auto box_decider = DistTree::BoxDecider({0.25, 1, 0.5});
  const int finest_level = 5;

  // Partition an octree on just the end processes.
  if (sub_comm != MPI_COMM_NULL)
  {
    tree_gap = DistTree::minimalSubdomainDistTree(finest_level, box_decider, sub_comm);
  }

  auto copy = tree_gap.getTreePartFiltered();
  ot::SFC_Tree<uint, DIM>::distTreeSort(copy, 0.3, comm);
  DistTree tree_compact(copy, comm);

  MPI_REQUIRE(0,                                       l_sz(tree_gap) > 0);
  MPI_REQUIRE(test_nb_procs_as_int_constant.value - 1, l_sz(tree_gap) > 0);
  REQUIRE(par::mpi_sum(l_sz(tree_gap), comm) == par::mpi_sum(l_sz(tree_compact), comm));

  // Partitions are dissimilar.
  CHECK(l_sz(tree_gap) != l_sz(tree_compact));

  // Checksum is unaffected by differing partitions.
  using ot::checksum_octree;
  CHECK(checksum_octree(l_ot(tree_gap), comm).code == checksum_octree(l_ot(tree_compact), comm).code);

  /// hexlist_files<DIM>(l_ot(tree_gap), "tmp_oct", ".hexlist.part", finest_level, comm);

  _DestroyHcurve();
}



template <int dim>
void hexlist_files(
    const std::vector<ot::TreeNode<uint, dim>> &octlist,
    const std::string &filename_prefix,
    // will fill in "prefix_${rank}suffix"
    const std::string &filename_suffix,
    int unit_level,
    MPI_Comm comm)
{
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  const int rank_str_len = std::snprintf(nullptr, 0, "%d", comm_size);
  std::string rank_str(1 + rank_str_len, '_');
  std::snprintf(&rank_str[0], 2 + rank_str_len, "_%0*d", rank_str_len, comm_rank);
  // Allowed to write the null terminator to rank_str[size] as of C++11.
  // https://en.cppreference.com/w/cpp/string/basic_string/operator_at

  const std::string filename = filename_prefix + rank_str + filename_suffix;

  std::ofstream out_file(filename);
  REQUIRE_MESSAGE(bool(out_file), "Could not open ", filename);
  out_file << nlohmann::json(
      io::JSON_Hexlist::from_octlist(octlist, unit_level));
}
