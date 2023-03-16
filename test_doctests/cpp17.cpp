
//
// Created by masado on 2023-03-16
//

#include <doctest/extensions/doctest_mpi.h>  // include doctest before dendro

template <typename...T>
auto fold_plus(T&&...t)
{
  return (t + ...);
}

MPI_TEST_CASE("fold expression", 1)
{
  CHECK(fold_plus(1, 2, 3, 4, 0.5) == 10.5);
}


