//
// Created by masado on 10/13/22.
//

#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro

/// #include "test/octree/multisphere.h"

#include <include/treeNode.h>
#include <include/tsort.h>
/// #include <include/distTree.h> // for convenient uniform grid partition
#include <include/sfc_search.h>

#include "test/octree/gaussian.hpp"

#include <vector>


// -----------------------------
// Typedefs
// -----------------------------
using uint = unsigned int;
using LLU = long long unsigned;

template <int dim>
using Oct = ot::TreeNode<uint, dim>;

// -----------------------------
// Helper classes
// -----------------------------
struct SfcTableScope
{
  SfcTableScope(int dim) { _InitializeHcurve(dim); }
  ~SfcTableScope() { _DestroyHcurve(); }
};

template <class Class>
struct Constructors       // Callable object wrapping constructor overloads
{
  template <typename ... T>
  Class operator()(T && ... ts) const {
    return Class(std::forward<T>(ts)...);
  }
};


TEST_CASE("Insertion sort == Tree sort with 1000 normally-distributed octants")
{
  constexpr int DIM = 3;
  using Oct = Oct<DIM>;
  const SfcTableScope _(DIM);
  {
    const size_t n_octants = 1000;
    std::vector<Oct> octants_tsort = test::gaussian<uint, DIM>(0, n_octants, Constructors<Oct>{});
    std::vector<Oct> octants_insort = octants_tsort;

    // Insertion sort
    for (auto begin = octants_insort.begin(), end = octants_insort.end(),
        i = begin; i != end; ++i)
    {
      const size_t pos = ot::sfc_binary_search<uint, DIM>(
          *i, &*begin, 0, (i-begin), ot::RankType::inclusive);
      std::rotate(begin + pos, i, i + 1);
    }

    // TreeSort
    ot::SFC_Tree<uint, DIM>::locTreeSort(octants_tsort);

    REQUIRE(octants_insort == octants_tsort);
  }
}



