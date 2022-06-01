//
// Created by masado on 4/27/22.
//

#include <doctest/extensions/doctest_mpi.h>
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

template <uint dim> size_t size(const ot::DistTree<uint, dim> &dtree) { return dtree.getTreePartFiltered().size(); }
template <uint dim> const std::vector<Oct<dim>> & octlist(const ot::DistTree<uint, dim> &dtree) { return dtree.getTreePartFiltered(); }

template <int dim>
std::vector<ot::TreeNode<uint, dim>> load_octlist(const std::string &filename, int unit_level);


MPI_TEST_CASE("hexlist open", 1)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  const auto filepath = [=](const std::string &file) {
    return std::string(bin_dir) + "/assets/" + file;
  };

  const auto coarsen_all = [](size_t size) {
    return std::vector<ot::OCT_FLAGS::Refine>(size, ot::OCT_FLAGS::OCT_COARSEN);
  };

  SUBCASE("Dimension 2")
  {
    constexpr int DIM = 2;
    _InitializeHcurve(DIM);
    using OctList = std::vector<Oct<DIM>>;
    using DistTree = ot::DistTree<uint, DIM>;
    OctList loaded;
    const double sfc_tol = 0.3;

    SUBCASE("valid hexlist")
    {
      loaded = load_octlist<DIM>(filepath("boxes.hexlist"), 2);
      CHECK(loaded.size() == 3);
      ot::SFC_Tree<uint, DIM>::distTreeSort(loaded, sfc_tol, comm);
    }

    DistTree dtree_loaded(loaded, comm);
    DistTree dtree_coarsened, dtree_surrogate;
    DistTree::distRemeshSubdomain(
        dtree_loaded,
        coarsen_all(size(dtree_loaded)),
        dtree_coarsened,
        dtree_surrogate,
        ot::SurrogateOutByIn,
        sfc_tol);

    _DestroyHcurve();
  }
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
