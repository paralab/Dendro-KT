//
// Created by masado on 4/20/22.
//   Based on the Doctest tutorial + MPI extension example
//

#include "doctest/extensions/doctest_mpi.h"

int my_function_to_test(MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
  {
    return 10;
  }
  return 11;
}

// Parallel test on 2 processes
MPI_TEST_CASE("test over two processes",2)
{
  int x = my_function_to_test(test_comm);

  MPI_CHECK( 0,  x==10 ); // CHECK for rank 0, that x==10
  MPI_CHECK( 1,  x==11 ); // CHECK for rank 1, that x==11
}


