/*
 * testScatterMap.cpp
 *   Test consistency of scater/gather maps after dist_countCGNodes().
 *
 * Masado Ishii  --  UofU SoC, 2019-03-14
 */



#include "testAdaptiveExamples.h"

#include "treeNode.h"
#include "nsort.h"
#include "matvec.h"

#include "hcurvedata.h"

#include <vector>

#include <iostream>



template<typename X>
void distPrune(std::vector<X> &list, MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const int listSize = list.size();
  const int baseSeg = listSize / nProc;
  const int remainder = listSize - baseSeg * nProc;
  const int myStart = rProc * baseSeg + (rProc < remainder ? rProc : remainder);
  const int mySeg = baseSeg + (rProc < remainder ? 1 : 0);

  list.erase(list.begin(), list.begin() + myStart);
  list.resize(mySeg);
}

template<unsigned int dim, unsigned int endL, unsigned int order>
void testGatherMap(MPI_Comm comm);

template<unsigned int dim, unsigned int endL, unsigned int order>
void testMatvecSubtreeSizes(MPI_Comm comm);


//
// main()
//
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);

  int nProc, rProc;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  constexpr unsigned int dim = 2;
  const unsigned int endL = 4;
  const unsigned int order = 3;

  /// testGatherMap<dim,endL,order>(comm);

  testMatvecSubtreeSizes<dim,endL,order>(comm);

  MPI_Finalize();

  return 0;
}



//
// testGatherMap()
//
template<unsigned int dim, unsigned int endL, unsigned int order>
void testGatherMap(MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);
  double tol = 0.05;

  _InitializeHcurve(dim);

  unsigned int numPoints;
  Tree<dim> tree;
  NodeList<dim> nodeListExterior;
  /// NodeList<dim> nodeListInterior;

  ot::RankI numUniqueInteriorNodes;
  ot::RankI numUniqueExteriorNodes;
  ot::RankI numUniqueNodes;

  ot::ScatterMap scatterMap;
  ot::GatherMap gatherMap;

  // ----------------------------

  // Example3
  Example3<dim>::fill_tree(endL, tree);
  distPrune(tree, comm);
  ot::SFC_Tree<T,dim>::distTreeSort(tree, tol, comm);
  for (const ot::TreeNode<T,dim> &tn : tree)
  {
    /// ot::Element<T,dim>(tn).appendInteriorNodes(order, nodeListInterior);
    ot::Element<T,dim>(tn).appendExteriorNodes(order, nodeListExterior);
  }
  /// numUniqueInteriorNodes = nodeListInterior.size();
  numUniqueExteriorNodes = ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeListExterior, order, tree.data(), scatterMap, gatherMap, comm);
  /// ot::RankI globInterior = 0;
  /// par::Mpi_Allreduce(&numUniqueInteriorNodes, &globInterior, 1, MPI_SUM, comm);
  /// numUniqueInteriorNodes = globInterior;
  numUniqueNodes = /*numUniqueInteriorNodes +*/ numUniqueExteriorNodes;

  // ----------------------------

  // Send and receive some stuff, verify the ghost segments have allocated space
  // in order of increasing processor rank.

  // Allocate space for local data + ghost segments on either side.
  std::vector<int> dataArray(gatherMap.m_totalCount);
  int * const myDataBegin = dataArray.data() + gatherMap.m_locOffset;
  int * const myDataEnd = myDataBegin + gatherMap.m_locCount;

  std::vector<int> sendBuf(scatterMap.m_map.size());

  // Initialize values of our local data to rProc. Those that should not be sent are negative.
  for (int * myDataIter = myDataBegin; myDataIter < myDataEnd; myDataIter++)
    *myDataIter = -rProc;
  for (ot::RankI ii = 0; ii < sendBuf.size(); ii++)
    myDataBegin[scatterMap.m_map[ii]] = rProc;

  // Stage send data.
  for (ot::RankI ii = 0; ii < sendBuf.size(); ii++)
    sendBuf[ii] = myDataBegin[scatterMap.m_map[ii]];

  // Send/receive data.
  std::vector<MPI_Request> requestSend(scatterMap.m_sendProc.size());
  std::vector<MPI_Request> requestRecv(gatherMap.m_recvProc.size());
  MPI_Status status;

  for (int sIdx = 0; sIdx < scatterMap.m_sendProc.size(); sIdx++)
    par::Mpi_Isend(sendBuf.data() + scatterMap.m_sendOffsets[sIdx],   // Send.
        scatterMap.m_sendCounts[sIdx],
        scatterMap.m_sendProc[sIdx], 0, comm, &requestSend[sIdx]);

  for (int rIdx = 0; rIdx < gatherMap.m_recvProc.size(); rIdx++)
    par::Mpi_Irecv(dataArray.data() + gatherMap.m_recvOffsets[rIdx],  // Recv.
        gatherMap.m_recvCounts[rIdx],
        gatherMap.m_recvProc[rIdx], 0, comm, &requestRecv[rIdx]);

  for (int sIdx = 0; sIdx < scatterMap.m_sendProc.size(); sIdx++)     // Wait sends.
    MPI_Wait(&requestSend[sIdx], &status);
  for (int rIdx = 0; rIdx < gatherMap.m_recvProc.size(); rIdx++)      // Wait recvs.
    MPI_Wait(&requestRecv[rIdx], &status);

  // Check that everything got to the proper place.
  int success = true;
  int lastVal = dataArray[0];
  int ii;
  for (ii = 0; ii < dataArray.size(); ii++)
  {
    int val = dataArray[ii];

    /// fprintf(stderr, "%d(%d)  ", rProc, val);

    if (val < 0 && -val != rProc)
    {
      success = false;
      break;
    }
    if (val < 0)
      val = -val;
    if (val < lastVal)
    {
      success = false;
      break;
    }

    /// if (val != rProc)
    /// {
    ///   for (int k = 0; k < rProc; k++)
    ///     fprintf(stderr, "\t");
    ///   fprintf(stderr, "[%d](%d)\n", rProc, val);
    /// }
  }
  fprintf(stderr, "  [%d] >>Exiting loop<<  Success? %s\n", rProc, (success ? "Yes" : "NO, FAILED"));
  if (!success)
    fprintf(stderr, "[%d] Failed at dataArray[%d].\n", rProc, ii);

  // ----------------------------

  tree.clear();
  /// nodeListInterior.clear();
  nodeListExterior.clear();

  // ----------------------------

  _DestroyHcurve();
}


//
// testMatvecSubtreeSizes()
//
template<unsigned int dim, unsigned int endL, unsigned int order>
void testMatvecSubtreeSizes(MPI_Comm comm)
{
  using TNP = ot::TNPoint<T,dim>;
  using da = float;

  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);
  double tol = 0.05;

  _InitializeHcurve(dim);

  Tree<dim> tree;
  NodeList<dim> nodeList;
  ot::ScatterMap scatterMap;
  ot::GatherMap gatherMap;

  //
  // The method countSubtreeSizes() counts a theoretical upper bound
  // on the buffer capacity needed at each level in a way that should not depend
  // on having neighbor-owned nodes. The count should be the same before and after
  // scattering/gathering.
  //
  std::vector<ot::RankI> subtreeSizesBefore;
  std::vector<ot::RankI> subtreeSizesAfter;

  // Example3 tree.
  Example3<dim>::fill_tree(endL, tree);
  distPrune(tree, comm);
  ot::SFC_Tree<T,dim>::distTreeSort(tree, tol, comm);

  if (rProc == 0)
    std::cout << "The total number of points in the tree is "
        << Example3<dim>::num_points(endL, order) << ".\n";

  // Add exterior points and resolve ownership/hanging nodes.
  for (const ot::TreeNode<T,dim> &tn : tree)
    ot::Element<T,dim>(tn).appendExteriorNodes(order, nodeList);
  ot::SFC_NodeSort<T,dim>::dist_countCGNodes(nodeList, order, tree.data(), scatterMap, gatherMap, comm);

  // Add interior points (we definitely own them and they cannot be hanging).
  for (const ot::TreeNode<T,dim> &tn : tree)
    ot::Element<T,dim>(tn).appendInteriorNodes(order, nodeList);

  // Since we added new points to the list -> resize the gatherMap;
  // since we are going to change order during Matvec, -> remap the scatterMap.
  // We can't remap the scatterMap unless we have the shuffleMap (from countSubtreeSizes()).
  //     shuffleMap: [new_i] --> original_i.
  ot::GatherMap::resizeLocalCounts(gatherMap, (ot::RankI) nodeList.size(), rProc);
  std::vector<ot::RankI> shuffleMap(nodeList.size());
  for (ot::RankI ii = 0; ii < nodeList.size(); ii++)
    shuffleMap[ii] = ii;

  // ===Count Before===  Also, this initializes the shuffleMap.
  //                     Important: nodeList includes interior points.
  fem::SFC_Matvec<T,da,dim>::countSubtreeSizes(
      &(*nodeList.begin()), &(*shuffleMap.begin()),
      0, (ot::RankI) nodeList.size(),
      0, 0, order,
      subtreeSizesBefore);

  /// fem::SFC_Matvec<T,da,dim>::countSubtreeSizes(
  ///     &(*nodeList.begin()), &(*shuffleMap.begin()),
  ///     0, (ot::RankI) nodeList.size(),
  ///     0, 0, order,
  ///     subtreeSizesAfter);    // Basic check to see if it matches after sorting.

  // Now that we have the shuffleMap, we can remap the scatterMap.
  // To do it, compute the inverse of the shuffleMap (inefficiently), but just once.
  //     shuffleMap_inv: [original_i] --> new_i.
  std::vector<ot::RankI> shuffleMap_inv(nodeList.size());
  for (ot::RankI ii = 0; ii < nodeList.size(); ii++)
    shuffleMap_inv[shuffleMap[ii]] = ii;
  // Remap the scatterMap.
  for (ot::RankI ii = 0; ii < scatterMap.m_map.size(); ii++)
    scatterMap.m_map[ii] = shuffleMap_inv[scatterMap.m_map[ii]];

  //
  // Scatter/gather.
  //
  NodeList<dim> nodeListRecv(gatherMap.m_totalCount);
  NodeList<dim> sendBuf(scatterMap.m_map.size());

  // Stage send data.
  for (ot::RankI ii = 0; ii < sendBuf.size(); ii++)
    sendBuf[ii] = nodeList[scatterMap.m_map[ii]];
  // Note for user: When we do this for each iteration Matvec,
  // need to use like nodeListRecv[scaterMap.m_map[ii] + gatherMap.m_locOffset].

  // Send/receive data.
  std::vector<MPI_Request> requestSend(scatterMap.m_sendProc.size());
  std::vector<MPI_Request> requestRecv(gatherMap.m_recvProc.size());
  MPI_Status status;

  for (int sIdx = 0; sIdx < scatterMap.m_sendProc.size(); sIdx++)
    par::Mpi_Isend(sendBuf.data() + scatterMap.m_sendOffsets[sIdx],   // Send.
        scatterMap.m_sendCounts[sIdx],
        scatterMap.m_sendProc[sIdx], 0, comm, &requestSend[sIdx]);

  for (int rIdx = 0; rIdx < gatherMap.m_recvProc.size(); rIdx++)
    par::Mpi_Irecv(nodeListRecv.data() + gatherMap.m_recvOffsets[rIdx],  // Recv.
        gatherMap.m_recvCounts[rIdx],
        gatherMap.m_recvProc[rIdx], 0, comm, &requestRecv[rIdx]);

  for (int sIdx = 0; sIdx < scatterMap.m_sendProc.size(); sIdx++)     // Wait sends.
    MPI_Wait(&requestSend[sIdx], &status);
  for (int rIdx = 0; rIdx < gatherMap.m_recvProc.size(); rIdx++)      // Wait recvs.
    MPI_Wait(&requestRecv[rIdx], &status);

  // The original data must be copied to the middle of the active nodeListRecv list.
  memcpy(nodeListRecv.data() + gatherMap.m_locOffset, nodeList.data(), sizeof(TNP)*nodeList.size());

  // ===Count After===
  std::vector<ot::RankI> dummyShuffleMap(nodeListRecv.size());
  fem::SFC_Matvec<T,da,dim>::countSubtreeSizes(
      &(*nodeListRecv.begin()), &(*dummyShuffleMap.begin()),
      0, (ot::RankI) nodeListRecv.size(),
      0, 0, order,
      subtreeSizesAfter);


  //
  // Report results. Did the subtree size lists match or not?
  //
  ot::RankI subtreeDepthBefore_loc = subtreeSizesBefore.size(), subtreeDepthBefore_glob;
  ot::RankI subtreeDepthAfter_loc  = subtreeSizesAfter.size(),  subtreeDepthAfter_glob;
  ot::RankI subtreeDepthMax_glob;
  par::Mpi_Allreduce<ot::RankI>(&subtreeDepthBefore_loc, &subtreeDepthBefore_glob, 1, MPI_MAX, comm);
  par::Mpi_Allreduce<ot::RankI>(&subtreeDepthAfter_loc, &subtreeDepthAfter_glob, 1, MPI_MAX, comm);
  subtreeDepthMax_glob = (subtreeDepthBefore_glob >= subtreeDepthAfter_glob ? subtreeDepthBefore_glob : subtreeDepthAfter_glob);
  std::vector<ot::RankI> subtreeSizesBefore_glob;
  std::vector<ot::RankI> subtreeSizesAfter_glob;
  if (rProc == 0)
  {
    subtreeSizesBefore_glob.resize(nProc * subtreeDepthMax_glob, 0);
    subtreeSizesAfter_glob.resize(nProc * subtreeDepthMax_glob, 0);
  }
  subtreeSizesBefore.resize(subtreeDepthMax_glob, 0);
  subtreeSizesAfter.resize(subtreeDepthMax_glob, 0);
  par::Mpi_Gather<ot::RankI>(subtreeSizesBefore.data(), subtreeSizesBefore_glob.data(), subtreeDepthMax_glob, 0, comm);
  par::Mpi_Gather<ot::RankI>(subtreeSizesAfter.data(), subtreeSizesAfter_glob.data(), subtreeDepthMax_glob, 0, comm);

  if (rProc == 0)
  {
    // We will print out procs across columns,
    // depth down rows.
    // Every two rows will show before and after, then a blank line separator.
    std::vector<int> sumRows(subtreeDepthMax_glob, true);
    std::vector<int> sumColumns(nProc, true);
    int sumMatrix = true;

    std::cout << "Subtree sizes. Cols: processors. Rows: tree levels.\n";
    for (int col = 0; col < nProc; col++)
      std::cout << "\t" << col;
    std::cout << "\n";

    for (int row = 0; row < subtreeDepthMax_glob; row++)
    {
      for (int col = 0; col < nProc; col++)
      {
        bool cellMatch = (subtreeSizesBefore_glob[col * subtreeDepthMax_glob + row] ==
                          subtreeSizesAfter_glob[col * subtreeDepthMax_glob + row]);
        sumRows[row] &= cellMatch;
        sumColumns[col] &= cellMatch;
      }
      sumMatrix &= sumRows[row];

      std::cout << row;
      for (int col = 0; col < nProc; col++)
        std::cout << "\t" << subtreeSizesBefore_glob[col * subtreeDepthMax_glob + row];
      std::cout << "\t| " << (sumRows[row] ? "yes" : "NO!") << "\n";
      for (int col = 0; col < nProc; col++)
        std::cout << "\t" << subtreeSizesAfter_glob[col * subtreeDepthMax_glob + row];
      std::cout << "\t|\n\n";
    }

    for (int col = 0; col <= nProc; col++)
      std::cout << "----\t";
    std::cout << "\n";
    for (int col = 0; col < nProc; col++)
      std::cout << "\t" << (sumColumns[col] ? "yes" : "NO!");
    std::cout << "\n\n";

    std::cout << "Overall success?  " << (sumMatrix ? "yes" : "NO!") << "\n";
  }

  _DestroyHcurve();
}

