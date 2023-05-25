#define DOCTEST_CONFIG_IMPLEMENT

#include <doctest/extensions/doctest_mpi.h>
#include "petscsys.h"

int main(int argc, char** argv) {
  doctest::mpi_init_thread(argc,argv,MPI_THREAD_MULTIPLE);
  PetscInitialize(&argc, &argv, NULL, NULL);

  doctest::Context ctx;
  ctx.setOption("reporters", "MpiConsoleReporter");
  ctx.setOption("reporters", "MpiFileReporter");
  ctx.setOption("force-colors", true);
  ctx.applyCommandLine(argc, argv);

  int test_result = ctx.run();

  PetscFinalize();
  doctest::mpi_finalize();

  return test_result;
}
