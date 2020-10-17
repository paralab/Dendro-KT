
#include "treeNode.h"
#include "tsort.h"
#include "genChannelPoints.h"

#include <random>
#include <vector>


namespace bench
{

  // depth must be at least 1.
  // depth must be at least lengthPower2.

  //  Example: dim=2
  //           depth=4
  //           lengthPower2=1  (2:1)
  //    ._____ _____ _____ _____._____ _____ _____ _____._____ _____ _____ _____._____ _____ _____ _____.
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  //    |     |     |           |           |           |           |           |           |     |     |
  //    |     |     |           |           |           |           |           |           |     |     |
  //    |- - -|- - -|           |           |           |           |           |           |- - -|- - -|
  //    |     |     |           |           |           |           |           |           |     |     |
  //    |     |     |           |           |           |           |           |           |     |     |
  //    +  -  -  -  -  -  -  -  +  -  -  -  -  -  -  -  +  -  -  -  -  -  -  -  +  -  -  -  -  -  -  -  +
  //    |     |     |           |           |           |           |           |           |     |     |
  //    |     |     |           |           |           |           |           |           |     |     |
  //    |- - -|- - -|           |           |           |           |           |           |- - -|- - -|
  //    |     |     |           |           |           |           |           |           |     |     |
  //    |     |     |           |           |           |           |           |           |     |     |
  //    -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  //    ^-----------------------^-----------------------^-----------------------^-----------------------^

  // If entire subdomain were refined to depth:
  //   2^depth * (2^(depth-lengthPower2))^(dim-1)
  //
  // Just outermost shell:
  //   2^depth * (2^(depth-lengthPower2))^(dim-1)  -  (2^depth - 2) * (2^(depth-lengthPower2) - 2)^(dim-1)
  //
  // Second-outermost shell:
  //   (2^depth - 2) * (2^(depth-lengthPower2) - 2)^(dim-1)  -  (2^depth - 4) * (2^(depth-lengthPower2) - 4)^(dim-1)
  //
  // General shell, L from depth down to (lengthPower2 + 2):
  //   (2^L - 2) * (2^(L-lengthPower2) - 2)^(dim-1)  -  (2^L - 4) * (2^(L-lengthPower2) - 4)^(dim-1)


  //
  // cellsInShell()
  //
  template <unsigned int dim>
  long long int cellsInShell(long long int longSideLength,
                             long long int shortSideLength)
  {
    long long int outer = longSideLength;
    long long int inner = longSideLength - 2;
    for (int d = 1; d < dim; ++d)
    {
      outer *= shortSideLength;
      inner *= (shortSideLength - 2);
    }
    return outer - inner;
  }


  //
  // estimateNumChannelPoints()
  //
  template <unsigned int dim>
  long long int estimateNumChannelPoints(int depth, int lengthPower2)
  {
    long long int numCells = cellsInShell<dim>( 1llu << depth,
                                                1llu << (depth - lengthPower2));

    for (int L = depth; L >= lengthPower2 + 2; --L)
      numCells += cellsInShell<dim>( (1llu << L) - 2,
                                     (1llu << (L - lengthPower2)) - 2);

    return numCells;
  }


  //
  // solveForDepth
  //
  template <unsigned int dim>
  int solveForDepth(long long int numPoints, int lengthPower2)
  {
    // Binary search to find depth such that
    //    numPoints <= estimateNumChannelPoints(depth).

    int minDepth = fmaxf(lengthPower2, 2);
    int maxDepth = minDepth;
    while (numPoints > estimateNumChannelPoints<dim>(maxDepth, lengthPower2))
      maxDepth *= 2;

    while (maxDepth - minDepth > 1)
    {
      const int midDepth = (minDepth + maxDepth) / 2;
      if (numPoints > estimateNumChannelPoints<dim>(midDepth, lengthPower2))
        minDepth = midDepth;
      else
        maxDepth = midDepth;
    }

    return maxDepth;
  }



  template <unsigned int dim>
  std::vector<ot::TreeNode<unsigned int, dim>> getChannelPoints(
      size_t ptsPerProc, int lengthPower2, MPI_Comm comm)
  {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    const long long int totalNumPts = npes * (long long int) ptsPerProc;
    const int depth = solveForDepth<dim>(totalNumPts, lengthPower2);
    const ibm::DomainDecider boxDecider = getBoxDecider<dim>(lengthPower2);

    //TODO use a distributed method to generate the initial grid in parallel
    //TODO compute how many points each processor needs to remove
    //TODO remove excess points.

    std::vector<ot::TreeNode<unsigned int, dim>> points;
    if (rank == 0)
      ot::SFC_Tree<unsigned int, dim>::locTreeConstructionWithFilter(boxDecider, false, points, 1, depth, 0, ot::TreeNode<unsigned int, dim>());

    ot::SFC_Tree<unsigned int, dim>::distTreeSort(points, 0.3, comm);

    return points;
  }






  // ---- Template instantiations ----

  template
  long long int estimateNumChannelPoints<2>(int depth, int lengthPower2);
  template
  long long int estimateNumChannelPoints<3>(int depth, int lengthPower2);
  template
  long long int estimateNumChannelPoints<4>(int depth, int lengthPower2);

  template
  int solveForDepth<2>(long long int numPoints, int lengthPower2);
  template
  int solveForDepth<3>(long long int numPoints, int lengthPower2);
  template
  int solveForDepth<4>(long long int numPoints, int lengthPower2);

  template
  std::vector<ot::TreeNode<unsigned int, 2>> getChannelPoints<2>(
      size_t ptsPerProc, int lengthPower2, MPI_Comm comm);
  template
  std::vector<ot::TreeNode<unsigned int, 3>> getChannelPoints<3>(
      size_t ptsPerProc, int lengthPower2, MPI_Comm comm);
  template
  std::vector<ot::TreeNode<unsigned int, 4>> getChannelPoints<4>(
      size_t ptsPerProc, int lengthPower2, MPI_Comm comm);
}
