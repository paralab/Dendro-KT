
#include "parUtils.h"

#include <petsc.h>

#include <vector>
#include <algorithm>
#include <random>

std::vector<int> uniform(int length, int comm_size);
std::vector<int> shuffled(int comm_size);

int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size,  comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  /// std::vector<int> destinations = shuffled(comm_size);
  std::vector<int> destinations = uniform(10, comm_size);

  std::vector<int> received = par::sendAll(destinations, destinations, comm);
  for (int x : received)
    assert(x == comm_rank);

  MPI_Barrier(comm);

  if (comm_rank == 0)
    printf("success\n");

  MPI_Finalize();
  return 0;
}


std::vector<int> uniform(int length, int comm_size)
{
  std::vector<int> v(length);
  std::mt19937_64 gen;
  std::uniform_int_distribution<int> dist(0, comm_size - 1);
  std::generate(v.begin(), v.end(), [&]() { return dist(gen); });
  return v;
}

std::vector<int> shuffled(int comm_size)
{
  std::vector<int> v(comm_size);
  std::iota(v.begin(), v.end(), 0);
  std::shuffle(v.begin(), v.end(), std::mt19937_64());
  return v;
}


