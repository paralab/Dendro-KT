/**
 * @file trySorting.cpp
 * @author Masado Ishii, University of Utah, School of Computing
 * @created 2019-04-09
 * @description Test distTreeSort().
 */

#include "tsort.h"
#include "octUtils.h"
#include "treeNode.h"
#include "dendro.h"

#include <mpi.h>

#include <iostream>
#include <stdio.h>

struct Parameters
{
  unsigned int ptsPerProc;
  /// unsigned int maxPtsPerRegion;
  double loadFlexibility;
  /// unsigned int endL;
  /// unsigned int elementOrder;
};


// --------------------------------------------
template <unsigned int dim>
void test(const Parameters &pm, MPI_Comm comm);
// --------------------------------------------


//
// main()
//
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);

  int rProc, nProc;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  unsigned int dim;
  Parameters pm;

  // Set up accepted options.
  enum CmdOptions { progName, opDim, opPtsPerProc, opLoadFlexibility, NUM_CMD_OPTIONS };
  const char *cmdOptions[NUM_CMD_OPTIONS] = { argv[0], "dim", "ptsPerProc", "loadFlexibility"};
  const unsigned int firstOptional = opPtsPerProc;
  std::array<const char *, NUM_CMD_OPTIONS> argVals;
  argVals.fill("");
  for (unsigned int op = 0; op < argc; op++)
    argVals[op] = argv[op];

  // Check if we have the required arguments.
  if (argc < firstOptional)
  {
    if (!rProc)
    {
      std::cerr << "Usage: ";
      unsigned int op = 0;
      for (; op < firstOptional; op++)
        std::cerr << cmdOptions[op] << " ";
      for (; op < NUM_CMD_OPTIONS; op++)
        std::cerr << "[" << cmdOptions[op] << "] ";
      std::cerr << "\n";
    }
    exit(1);
  }

  // Parse arguments.
  dim = static_cast<unsigned int>(strtol(argVals[1], NULL, 0));
  pm.ptsPerProc = argc > opPtsPerProc ? strtol(argVals[opPtsPerProc], NULL, 0) : 100;
  /// pm.maxPtsPerRegion = argc > opMaxPtsPerRegion ? strtol(argVals[opMaxPtsPerRegion], NULL, 0) : 1;
  pm.loadFlexibility = argc > opLoadFlexibility ? strtol(argVals[opLoadFlexibility], NULL, 0) : 0.2;

  // Replay arguments.
  constexpr bool replayArguments = true;
  if (replayArguments && !rProc)
  {
    for (unsigned int op = 1; op < NUM_CMD_OPTIONS; op++)
      std::cout << cmdOptions[op] << "==" << argVals[op] << " \n";
    std::cout << "\n";
  }

  int synchronize;
  MPI_Bcast(&synchronize, 1, MPI_INT, 0, comm);

  _InitializeHcurve(dim);

  // Convert dimension argument to template parameter.
  switch(dim)
  {
    case 2: test<2>(pm, comm); break;
    case 3: test<3>(pm, comm); break;
    case 4: test<4>(pm, comm); break;
    default:
      if (!rProc)
        std::cerr << "Dimension " << dim << " not currently supported.\n";
  }

  _DestroyHcurve();

  MPI_Finalize();

  return 0;
}


//
// test()
//
template <unsigned int dim>
void test(const Parameters &pm, MPI_Comm comm)
{
  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const char * const functionName = "test";
  fprintf(stderr, "[%d] Running %s() with "
                  "dim==%u; "
                  "ptsPerProc==%u; "
                  "loadFlexibility==%f; "  "\n",
      rProc, functionName,
      dim,
      pm.ptsPerProc, pm.loadFlexibility);

  using T = unsigned int;

  std::vector<ot::TreeNode<T,dim>> points = ot::getPts<T,dim>(pm.ptsPerProc);

  ot::SFC_Tree<T,dim>::distTreeSort(points, pm.loadFlexibility, comm);

  fprintf(stderr, "[%d] Finished %s().\n", rProc, functionName);
}
