//
// Created by masado on 4/27/22.
//

#include <doctest/extensions/doctest_mpi.h>
#include "directories.h"

#include <include/treeNode.h>
#include <IO/hexlist/json_hexlist.h>

#include <fstream>
#include <vector>

const static std::string bin_dir = DENDRO_KT_DOCTESTS_BIN_DIR;

using uint = unsigned int;

template <int dim>
std::vector<ot::TreeNode<uint, dim>> load_octlist(const std::string &filename, int unit_level);


/// MPI_TEST_CASE("hexlist open", 1)
TEST_CASE("hexlist open")
{
  constexpr int DIM = 2;
  using Oct = ot::TreeNode<uint, DIM>;
  using OctList = std::vector<Oct>;

  const std::string input_filename = bin_dir + "/assets/boxes.hexlist";
  const OctList loaded = load_octlist<DIM>(input_filename, 2);
  CHECK(loaded.size() == 3);
}


// load_octlist
template <int dim>
std::vector<ot::TreeNode<uint, dim>> load_octlist(const std::string &filename, int unit_level)
{
  nlohmann::json json_hexlist;
  std::ifstream in_file(filename);
  REQUIRE_MESSAGE(bool(in_file), "Could not open ", filename);
  in_file >> json_hexlist;
  return io::JSON_Hexlist(json_hexlist).to_octlist<uint, dim>(unit_level);
}
