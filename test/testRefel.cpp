
#include <iostream>
#include <mpi.h>

#include "FEM/include/refel.h"

// ==============================
// main()
// ==============================
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0)
  {
    for (int degree = 1; degree <= 4; ++degree)
    {
      std::cout << std::endl;
      std::cout << "------------------";
      std::cout << " Degree = " << degree << " ";
      std::cout << "------------------" << std::endl;
      std::cout << std::endl;

      // RefElement constructor may print debug data.
      const int dimension = 2;
      RefElement ref_element(dimension, degree);

      std::cout << std::endl;
      std::cout << "------------------------------------------------" << std::endl;
      std::cout << std::endl;
    }
  }

  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}


