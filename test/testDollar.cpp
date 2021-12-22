#include <vector>
#include <random>
#include <fstream>
#include <ostream>
#include <sstream>

#include <dollar.hpp>

#include "distTree.h"
#include "filterFunction.h"
#include "octUtils.h"
#include "tnUtils.h"
#include "treeNode.h"
#include "dollar_stat.h"

const int DIM = 2;
using uint = unsigned int;
using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;

OctList points(size_t n);
OctList points(DendroIntL nTotal, MPI_Comm comm);


// ---------------------------------------


//
// main()
//
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int commRank, commSize;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  const double sfc_tol = 0.1;

  OctList octList = points(1 * 1024 * 1024, comm);

  {$
    ot::SFC_Tree<uint, DIM>::locTreeSort(octList);
  }

  {$
    ot::SFC_Tree<uint, DIM>::distTreeSort(octList, sfc_tol, comm);
  }

  dollar::DollarStat dollar_stat(comm);

  /// if (commRank == 0)
  /// {
  ///   // Only root's data
  ///   std::ofstream file("chrome.json");
  ///   dollar::chrome(file);
  ///   dollar::csv(std::cout);

  ///   // Only root's data
  ///   std::cout << "\n";
  ///   dollar_stat.print(std::cout);
  /// }
  dollar::clear();

  // Collect mean, min, and max timings over all processes.
  dollar::DollarStat reduce_mean = dollar_stat.mpi_reduce_mean();
  dollar::DollarStat reduce_min = dollar_stat.mpi_reduce_min();
  dollar::DollarStat reduce_max = dollar_stat.mpi_reduce_max();
  if (commRank == 0)
  {
    std::ofstream file("mean_chrome.json");
    reduce_mean.chrome(file);

    std::cout << "\n" << "[Mean]\n";
    reduce_mean.tsv(std::cout);
    std::cout << "\n" << "[Min]\n";
    reduce_mean.tsv(std::cout);
    std::cout << "\n" << "[Max]\n";
    reduce_mean.tsv(std::cout);

    /// reduce_mean.print(std::cout);
    /// reduce_min.print(std::cout);
    /// reduce_max.print(std::cout);
  }

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}



//
// points()
//
OctList points(size_t n)
{
  OctList points;

  const double min = 0;
  const double max = 1.0 - (1.0 / (1u << m_uiMaxDepth));
  const auto clamp = [=](double x) {
    return x < min ? min : x > max ? max : x;
  };

  const auto toOctCoord = [=](double x) {
    return uint(x * (1u << m_uiMaxDepth));
  };

  std::normal_distribution<> normal{0.5, 0.2};
  std::random_device rd;
  std::mt19937_64 gen(rd());

  for (size_t i = 0; i < n; ++i)
  {
    Oct oct;
    oct.setLevel(m_uiMaxDepth);

    for (int d = 0; d < DIM; ++d)
      oct.setX(d, toOctCoord(clamp(normal(gen))));

    points.push_back(oct);
  }

  return points;
}


//
// points()  (distributed)
//
OctList points(DendroIntL nTotal, MPI_Comm comm)
{
  int commRank, commSize;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);
  return points(size_t(nTotal / commSize) + size_t(commRank < nTotal % commSize));
}


// -----------------------------------------

