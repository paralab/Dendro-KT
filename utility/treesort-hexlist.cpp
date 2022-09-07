
#include <include/tsort.h>
#include <IO/hexlist/json_hexlist.h>
#include <IO/vtk/include/json.hpp>

#include <iostream>
#include <fstream>

#include <mpi.h>

const static char expected_args[] = " input_file output_file";

int main(int argc, char * argv[])
{
  int err = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  if (comm_rank == 0)
  {
    if (argc >= 3)
    {
      using uint = unsigned int;
      constexpr int DIM = 3;
      _InitializeHcurve(DIM);
      std::ifstream input(argv[1]);
      nlohmann::json json;
      input >> json;
      input.close();
      std::vector<ot::TreeNode<uint, DIM>> octants =
          io::JSON_Hexlist(json).to_octlist<uint, DIM>(m_uiMaxDepth);
      ot::SFC_Tree<uint, DIM>::locTreeSort(octants);
      json = io::JSON_Hexlist::from_octlist(octants, m_uiMaxDepth);
      std::ofstream output(argv[2]);
      output << json;
      output.close();
      _DestroyHcurve();
    }
    else
    {
      std::cerr << "Usage: " << argv[0] << expected_args << "\n";
      err = 1;
    }
  }
  MPI_Finalize();
  return err;
}
