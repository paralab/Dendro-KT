//
// Created by masado on 4/26/22.
//

#include <mpi.h>
#include <iostream>
#include <vector>
#include <array>

#include <json.hpp>
#include "IO/hexlist/json_hexlist.h"
#include "test/octree/gaussian.hpp"

using uint = unsigned;
constexpr int DIM = 3;
using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;

int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);
  if (comm_rank != 0)
    return 0;

  m_uiMaxDepth = 10;
  const int unit_level = m_uiMaxDepth;

  const auto new_oct = [=](std::array<uint, DIM> coords, int lev)
  {
    /// lev = m_uiMaxDepth;  // override
    lev = lev * 2 / 3;
    const uint mask = (1u << m_uiMaxDepth) - (1u << m_uiMaxDepth - lev);
    for (int d = 0; d < DIM; ++d)
      coords[d] &= mask;
    return Oct(coords, lev);
  };

  using json = nlohmann::json;

  const int N = 100;
  const OctList octants = test::gaussian<uint, DIM>(0, N, new_oct);
  const json json_hexlist = io::JSON_Hexlist::from_octlist(octants, unit_level);
  OctList recovered_octants;
  json_hexlist.get<io::JSON_Hexlist>().to_octlist(recovered_octants, unit_level);
  const bool success = (recovered_octants == octants);

  const bool print = true;
  if (print)
  {
    std::cout << json_hexlist;
    std::cout << "\n\n";
  }

  fprintf(stderr, "Success? %s" NRM "\n", (success ? (GRN "true") : (RED "false")));

  MPI_Finalize();
  return 0;
}