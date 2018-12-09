/*
 * test_SFC_Tree.cpp
 *   Test the methods in SFC_Tree.h / SFC_Tree.cpp
 *
 *
 * Masado Ishii  --  UofU SoC, 2018-12-03
 */


#include "TreeNode.h"
#include "SFC_Tree.h"

#include "hcurvedata.h"

#include "genRand4DPoints.h"
#include <vector>

#include <assert.h>
#include <mpi.h>

//------------------------
// test_locTreeSort()
//------------------------
void test_locTreeSort()
{
  using T = unsigned int;
  const unsigned int dim = 4;
  using TreeNode = ot::TreeNode<T,dim>;

  _InitializeHcurve(dim);

  const T leafLevel = m_uiMaxDepth;

  //const int numPoints = 10000;
  const int numPoints = 1000;

  std::array<unsigned int, 1u<<dim> topOctCount_start, botOctCount_start,
                                    topOctCount_end, botOctCount_end;
  topOctCount_start.fill(0);
  botOctCount_start.fill(0);
  topOctCount_end.fill(0);
  botOctCount_end.fill(0);

  std::cout << "=============================\n";
  std::cout << "Begin Adding Points.\n";
  std::cout << "=============================\n";

  std::vector<TreeNode> points = genRand4DPoints<T,dim>(numPoints);

  for (const TreeNode &tn : points)
  {
    topOctCount_start[tn.getMortonIndex(0)]++;
    botOctCount_start[tn.getMortonIndex(leafLevel)]++;
  }

  for (int ii = 0; ii < TreeNode::numChildren; ii++)
  {
    printf("Top: s(%d)  \t    Bot: s(%d)\n",
        topOctCount_start[ii], botOctCount_start[ii]);
  }

  std::cout << "=============================\n";
  std::cout << "Begin Sort!\n";
  std::cout << "=============================\n";

  // Sort them with locTreeSort().
  ///std::vector<ot::TreeNode<T,dim>> sortedPoints;
  ///ot::SFC_Tree<T,dim>::locTreeSort(&(*points.begin()), &(*points.end()), sortedPoints, 0, leafLevel, 0);
  auto leafBuckets = ot::SFC_Tree<T,dim>::getEmptyBucketVector();
  ot::SFC_Tree<T,dim>::locTreeSort(&(*points.begin()), 0, points.size(), 0, leafLevel, 0, leafBuckets);

  std::vector<ot::TreeNode<T,dim>> &sortedPoints = points;

  std::cout << '\n';

  std::cout << "=============================\n";
  std::cout << "Sorted Order:\n";
  std::cout << "=============================\n";

  for (const TreeNode &tn : sortedPoints)
  {
    std::cout << tn << " \t " << tn.getBase32Hex().data() << '\n';
    topOctCount_end[tn.getMortonIndex(0)]++;
    botOctCount_end[tn.getMortonIndex(leafLevel)]++;
  }

  std::cout << '\n';

  std::cout << "Number of leaf buckets (leafLevel == " << leafLevel << "):  "
            << leafBuckets.size() << '\n';
  std::cout << "Buckets:\n";
  for (const ot::BucketInfo<unsigned int> &b : leafBuckets)
  {
    printf("{%4d %4u %4u %4u}\n", b.rot_id, b.lev, b.begin, b.end);
  }

  std::cout << "=============================\n";
  std::cout << "Verify Counts.:\n";
  std::cout << "=============================\n";

  bool success = true;
  for (int ii = 0; ii < TreeNode::numChildren; ii++)
  {
    bool locSuccess = (topOctCount_start[ii] == topOctCount_end[ii])
        && (botOctCount_start[ii] == botOctCount_end[ii]);
    printf("Top: s(%d) e(%d)   \t    Bot: s(%d) e(%d)  %c\n",
        topOctCount_start[ii], topOctCount_end[ii],
        botOctCount_start[ii], botOctCount_end[ii],
        (locSuccess ? ' ' : '*' ));
    success = success && locSuccess;
  }
  std::cout << "-----------------------------\n"
      << (success ? "Success: No losses." : "FAILURE: Lost some TreeNodes.")
      << '\n';
}
//------------------------


//------------------------
// test_distTreeSort()
//------------------------
void test_distTreeSort(MPI_Comm comm = MPI_COMM_WORLD)
{
  int nProc, rProc;
  MPI_Comm_size(comm, &nProc);
  MPI_Comm_rank(comm, &rProc);

  using T = unsigned int;
  const unsigned int dim = 2;
  using TreeNode = ot::TreeNode<T,dim>;

  const int numPoints = 23;

  _InitializeHcurve(dim);

  std::vector<TreeNode> points = genRand4DPoints<T,dim>(numPoints);

  // Sort!
  ///ot::SFC_Tree<T,dim>::distTreeSort(points, 0.125, MPI_COMM_WORLD);
  ot::SFC_Tree<T,dim>::distTreeSort(points, 0.0, comm);

  // 1. Verify that the points are locally sorted.
  int locallySorted = true;
  for (int ii = 1; ii < points.size(); ii++)
  {
    if (!(points[ii-1] <= points[ii]))
    {
      locallySorted = false;
      break;
    }
  }

  int allLocallySorted = false;
  MPI_Reduce(&locallySorted, &allLocallySorted, 1, MPI_INT, MPI_SUM, 0, comm);
  if (rProc == 0)
    printf("Local sorts: %s (%d succeeded)\n", (allLocallySorted == nProc ? "Success" : "SOME FAILED!"), allLocallySorted);

  
  // 2. Verify that the endpoints are globally sorted.
  std::vector<TreeNode> endpoints(2);
  endpoints[0] = points.front();
  endpoints[1] = points.back();

  /// std::cout << endpoints[0].getBase32Hex().data() << "\n";
  /// std::cout << endpoints[1].getBase32Hex().data() << "\n";

  std::vector<TreeNode> allEndpoints;
  if (rProc == 0)
    allEndpoints.resize(2*nProc);

  // Hack, because I don't want to finish verifying
  // the templated Mpi_datatype<> for TreeNode right now.
  MPI_Gather((unsigned char *) endpoints.data(), (int) 2*sizeof(TreeNode), MPI_UNSIGNED_CHAR,
      (unsigned char *) allEndpoints.data(), (int) 2*sizeof(TreeNode), MPI_UNSIGNED_CHAR,
      0, comm);

  if (rProc == 0)
  {
    int globallySorted = true;
    for (int ii = 2; ii < allEndpoints.size(); ii+=2)
    {
      /// std::cout << allEndpoints[ii-1].getBase32Hex().data() << "\n";
      /// std::cout << allEndpoints[ii].getBase32Hex().data() << "\n";
      if (!(allEndpoints[ii-1] <= allEndpoints[ii]))
      {
        globallySorted = false;
        break;
      }
    }

    printf("Global sort: %s\n", (globallySorted ? "Success" : "FAILED!"));
  }

}
//------------------------


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  //test_locTreeSort();

  test_distTreeSort(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}


