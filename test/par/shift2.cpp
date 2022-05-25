
#include <dendro.h>
#include "parUtils.h"

#include <petsc.h>

#include <vector>
#include <algorithm>
#include <random>

std::vector<int> uniform(int length, MPI_Comm comm);
std::vector<int> ragged(int init_length, int max_exchange, MPI_Comm comm);

template <typename ... Ts>
void ZPrintf(Ts && ... ts)
{
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  if (comm_rank == 0)
    printf(ts...);
}


template <typename T>
void shift(const std::vector<T> &input,
           std::vector<T> &output,
           MPI_Comm comm,
           const int ndofs = 1);


int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size,  comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  const int init_length = 10;
  const int length_var = 10;

  {
    ZPrintf("uniform to uniform (dof=1): ");
    const int ndofs = 1;
    const std::vector<int> input = uniform(init_length * ndofs, comm);
    std::vector<int> output = uniform(init_length * ndofs, comm);
    std::fill(output.begin(), output.end(), 0);
    shift(input, output, comm, ndofs);
    const bool equal = par::mpi_and(output == input, comm);
    ZPrintf("%s\n" NRM, (equal ? GRN "passed" : RED "failed"));
  }

  {
    ZPrintf("uniform to uniform (dof=3): ");
    const int ndofs = 3;
    const std::vector<int> input = uniform(init_length * ndofs, comm);
    std::vector<int> output = uniform(init_length * ndofs, comm);
    std::fill(output.begin(), output.end(), 0);
    shift(input, output, comm, ndofs);
    const bool equal = par::mpi_and(output == input, comm);
    ZPrintf("%s\n" NRM, (equal ? GRN "passed" : RED "failed"));
  }

  {
    ZPrintf("uniform to ragged: ");
    const std::vector<int> input = uniform(init_length, comm);
    std::vector<int> output = ragged(init_length, length_var, comm);
    std::fill(output.begin(), output.end(), 0);
    shift(input, output, comm);
    const bool monotone = par::mpi_and(
        std::is_sorted(output.begin(), output.end()), comm);
    ZPrintf("%s\n" NRM, (monotone ? GRN "passed" : RED "failed"));
  }

  {
    ZPrintf("ragged to uniform: ");
    const std::vector<int> input = ragged(init_length, length_var, comm);
    std::vector<int> output = uniform(init_length, comm);
    std::fill(output.begin(), output.end(), 0);
    shift(input, output, comm);
    const bool monotone = par::mpi_and(
        std::is_sorted(output.begin(), output.end()), comm);
    ZPrintf("%s\n" NRM, (monotone ? GRN "passed" : RED "failed"));
  }

  {
    ZPrintf("ragged to ragged: ");
    const std::vector<int> input = ragged(init_length, length_var, comm);
    std::vector<int> output = ragged(init_length, length_var, comm);
    std::fill(output.begin(), output.end(), 0);
    shift(input, output, comm);
    const bool monotone = par::mpi_and(
        std::is_sorted(output.begin(), output.end()), comm);
    ZPrintf("%s\n" NRM, (monotone ? GRN "passed" : RED "failed"));
  }

  MPI_Finalize();
  return 0;
}



// shift()
template <typename T>
void shift(const std::vector<T> &input,
           std::vector<T> &output,
           MPI_Comm comm,
           const int ndofs)
{
  const DendroIntL size_local[2] = { (DendroIntL) input.size() / ndofs,
                                     (DendroIntL) output.size() / ndofs };
  DendroIntL begin_global[2] = { 0, 0 };
  par::Mpi_Exscan(size_local, begin_global, 2, MPI_SUM, comm);
  par::shift(
      comm,
      input.data(), size_local[0], begin_global[0],
      output.data(), size_local[1], begin_global[1],
      ndofs);
}


// class StridedGenerator
template <class BaseGenerator>
class StridedGenerator
{
  private:
    const int m_size = 1;
    const int m_rank = 0;
    bool m_warm = false;
    BaseGenerator m_gen;

  public:
    using result_type = typename BaseGenerator::result_type;

    static result_type min() { return BaseGenerator::min(); }
    static result_type max() { return BaseGenerator::max(); }

    template <typename ... Ts>
    StridedGenerator(int size, int rank, Ts ... ts)
      : m_size(size), m_rank(rank), m_gen(ts...)
    {
      m_gen.discard(rank);
    }

    result_type operator()()
    {
      if (m_warm)
        m_gen.discard(m_size - 1);
      m_warm = true;
      return m_gen();
    }
};



// uniform()
std::vector<int> uniform(int length, MPI_Comm comm)
{
  int comm_size,  comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  std::vector<int> part(length);
  std::iota(part.begin(), part.end(), comm_rank * length);
  return part;
}


// ragged()
std::vector<int> ragged(int init_length, int max_exchange, MPI_Comm comm)
{
  assert(0 <= max_exchange);
  assert(max_exchange <= init_length);

  int comm_size,  comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  static const int seed = 42;
  static StridedGenerator<std::mt19937_64> gen(comm_size, comm_rank, seed);
  std::uniform_int_distribution<> distribution(0, max_exchange);
  int right_send = (comm_rank % 2 == 0 ? distribution(gen)
                  : comm_rank % 4 == 1 ? max_exchange
                  : 0);

  const int right = (comm_rank + 1 < comm_size ? comm_rank + 1 : 0);
  const int left = (0 + 1 <= comm_rank ? comm_rank - 1 : comm_size - 1);
  int left_recv = 0;
  par::Mpi_Sendrecv(
      &right_send, 1, right, 0,
      &left_recv, 1, left, 0,
      comm, MPI_STATUS_IGNORE);
  if (right == MPI_PROC_NULL)
    right_send = 0;

  const int length = init_length + left_recv - right_send;
  assert(length >= 0);

  const int empty_ranks = par::mpi_sum(int(length == 0), comm);
  if (comm_rank == 0 and empty_ranks)
    printf(" (%d/%d empty) ", empty_ranks, comm_size);

  int offset = 0;
  par::Mpi_Exscan(&length, &offset, 1, MPI_SUM, comm);

  std::vector<int> part(length);
  std::iota(part.begin(), part.end(), offset);
  return part;
}


