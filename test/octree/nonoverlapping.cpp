
#include <vector>
#include <array>

#include "multisphere.h"
#include "distTree.h"
#include "filterFunction.h"
#include "octUtils.h"
#include "tnUtils.h"
#include "treeNode.h"


using uint = unsigned int;
constexpr int DIM = 2;

// --------------------------------------------------------------------
using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;
using MortonLevel = std::pair<std::array<uint, DIM>, int>;
using MortonRange = std::pair<std::array<uint, DIM>, std::array<uint, DIM>>;

bool sizeAligned(const OctList &octList, MPI_Comm comm);
OctList localOverlappingOct(const OctList &octList, MPI_Comm comm);
bool equalCoverage(const OctList &octListA, const OctList &octListB, MPI_Comm comm);

size_t size(const OctList &octList);
MortonLevel mortonEncode(const Oct &oct);
MortonRange mortonRange(const Oct &oct);
DendroIntL mpi_sum(DendroIntL x, MPI_Comm comm);
bool mpi_and(bool x, MPI_Comm comm);

bool testOverlapper(const OctList &octList);


// --------------------------------------------------------------------

//
// main()
//
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int commRank, commSize;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  // Any octree.
  test::SphereSet<DIM> sphereSet;
  sphereSet.carveSphere(0.10, {0.15, 0.6});
  sphereSet.carveSphere(0.10, {0.7, 0.4});
  sphereSet.carveSphere(0.210, {0.45, 0.55});
  const int fineLevel = 7;
  const double sfc_tol = 0.1;
  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::minimalSubdomainDistTree(
          fineLevel, sphereSet, comm, sfc_tol);

  const auto printTest = [](bool success, const std::string &prefix, const std::string &condition) {
    fprintf(stdout, "%s%s %s%s%s\n", success ? GRN : RED,
        prefix.c_str(), success ? "" : "not ", condition.c_str(), NRM);
  };
  const auto printNeutral = [](bool success, const std::string &prefix, const std::string &condition) {
    fprintf(stdout, "%s %s%s\n",
        prefix.c_str(), success ? "" : "not ", condition.c_str());
  };

  // Test the octree.
  const OctList & octList = distTree.getTreePartFiltered();

  OctList balanceTest = octList;
  ot::SFC_Tree<uint, DIM>::locMinimalBalanced(balanceTest);

  OctList unstable = ot::SFC_Tree<uint, DIM>::unstableOctants(
      balanceTest, (commRank > 0), (commRank < commSize-1));

  const bool sorted = mpi_and(ot::isLocallySorted(balanceTest), comm);
  if (commRank == 0)
    printTest(sorted, "locMinimalBalaned() is", "sorted");

  const bool balanced = ot::is2to1Balanced(balanceTest, comm);
  if (commRank == 0)
    printNeutral(balanced, "locMinimalBalaned() is", "distBalanced");

  OctList distBalanceTest = octList;
  ot::SFC_Tree<uint, DIM>::distMinimalBalanced(distBalanceTest, sfc_tol, comm);
  const bool distBalanced = ot::is2to1Balanced(distBalanceTest, comm);
  if (commRank == 0)
    printTest(distBalanced, "distMinimalBalanced() is", "distBalanced");

  const bool balancedCoverTree = equalCoverage(octList, distBalanceTest, comm);
  if (commRank == 0)
    printTest(balancedCoverTree, "distMinimalBalanced()", "covers same as input");

  /// const bool testedOverlapper = mpi_and(testOverlapper(unstable), comm);
  /// if (commRank == 0)
  ///   printTest(testedOverlapper, "Overlaps", "agree");

  ot::quadTreeToGnuplot(octList, fineLevel, "input", comm);
  ot::quadTreeToGnuplot(balanceTest, fineLevel, "balanced", comm);
  ot::quadTreeToGnuplot(unstable, fineLevel, "unstable", comm);
  ot::quadTreeToGnuplot(distBalanceTest, fineLevel, "distBalanced", comm);

  /// const size_t overlapSize = size(localOverlappingOct(octList, comm));
  /// const size_t overlapSize = size(localOverlappingOct(balanceTest, comm));
  const size_t overlapSize = size(localOverlappingOct(distBalanceTest, comm));
  bool success = overlapSize == 0;

  // Report.
  fprintf(stdout, "[Rank %2d/%d] local overlaps: %s%lu%s\n",
      commRank, commSize,
      (success ? GRN : RED), overlapSize, NRM);
  bool globalSuccess = mpi_and(success, comm);
  if (commRank == 0)
    fprintf(stdout, "%s%s%s\n",
        (globalSuccess ? GRN : RED),
        (globalSuccess ? "Success" : "Fail"),
        NRM);

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}


// ----------------------------------------------------

// sizeAligned()
bool sizeAligned(const OctList &octList, MPI_Comm comm)
{
  bool partitionAligned = true;
  for (const Oct &tn : octList)
  {
    const int level = tn.getLevel();
    const uint len = 1u << (m_uiMaxDepth - level);
    const uint mask = len - 1;
    bool octAligned = true;
    for (int d = 0; d < DIM; ++d)
      octAligned &= ((tn.getX(d) & mask) == 0);
    partitionAligned &= octAligned;
  }
  return mpi_and(partitionAligned, comm);
}

// localOverlappingOct()
OctList localOverlappingOct(const OctList &octList, MPI_Comm comm)
{
  const bool testSizeAligned = sizeAligned(octList, comm);
  assert(testSizeAligned);

  OctList sorted = octList;
  std::sort(sorted.begin(), sorted.end(),
      [](const Oct &a, const Oct &b)
          { return mortonEncode(a) < mortonEncode(b); });

  const auto overlapping = [=](const Oct &a, const Oct &b)
      { MortonRange ra = mortonRange(a),  rb = mortonRange(b);
        return ra.first < rb.second
            && rb.first < ra.second; };

  OctList overlappingOct;

  OctList::const_iterator begin = sorted.begin();
  while (begin != sorted.end())
  {
    OctList::const_iterator end = std::next(begin);
    while (end != sorted.end() && overlapping(*begin, *end))
      ++end;
    if (end != std::next(begin))
      overlappingOct.insert(overlappingOct.end(), begin, end);
    begin = end;
  }

  return overlappingOct;
}


// mortonEncode()
MortonLevel mortonEncode(const Oct &oct)
{
  const int chunk_bits = 8 * sizeof(uint);
  const std::array<uint, DIM> coords = oct.getX();

  std::array<uint, DIM> msd;
  msd.fill(0);

  for (int big_bit = 0; big_bit < DIM * chunk_bits; ++big_bit)
  {
    const int in_axis = big_bit % DIM;
    const int in_bit = big_bit / DIM;

    const int out_chunk = (DIM - 1) - (big_bit / chunk_bits);
    const int out_bit = big_bit % chunk_bits;

    msd[out_chunk] |= ((coords[in_axis] >> in_bit) & 1u) << out_bit;
  }

  return MortonLevel{msd, oct.getLevel()};
}

// mortonRange()
MortonRange mortonRange(const Oct &oct)
{
  const MortonLevel code = mortonEncode(oct);
  const int level = code.second;
  const int abs_height = m_uiMaxDepth - level;
  // number of grid units in octant = pow(2, abs_height * DIM).

  const std::array<uint, DIM> begin = code.first;
  std::array<uint, DIM> end = begin;

  const int chunk_bits = 8 * sizeof(uint);
  const int bit = (abs_height * DIM) % chunk_bits;
  int chunk = (DIM - 1) - (abs_height * DIM) / chunk_bits;

  // Add
  end[chunk] += (1u << bit);
  while (chunk > 0 && !end[chunk])
    end[--chunk] += 1;

  return MortonRange{begin, end};
}

// size()
size_t size(const OctList &octList)
{
  return octList.size();
}

// mpi_sum()
DendroIntL mpi_sum(DendroIntL x, MPI_Comm comm)
{
  DendroIntL sum = 0;
  par::Mpi_Allreduce(&x, &sum, 1, MPI_SUM, comm);
  return sum;
}

// mpi_and()
bool mpi_and(bool x_, MPI_Comm comm)
{
  int x = x_, global = true;
  par::Mpi_Allreduce(&x, &global, 1, MPI_LAND, comm);
  return bool(global);
}




bool testOverlapper(const OctList &octList)
{
  // Neighbors of octList, for testing
  OctList neighbours;
  for (const Oct &oct : octList)
    oct.appendCoarseNeighbours(neighbours);
  ot::SFC_Tree<uint, DIM>::locTreeSort(neighbours);

  // Method 1: Get overlaps via O(n*m) search.
  std::vector<std::vector<Oct>> overlapsBrute;
  std::vector<Oct> catOverlapsBrute;
  for (const Oct &oct : octList)
  {
    std::vector<Oct> octOverlaps;
    for (const Oct &nbr : neighbours)
      if (nbr.isAncestorInclusive(oct) || oct.isAncestorInclusive(nbr))
      {
        octOverlaps.push_back(nbr);
        catOverlapsBrute.push_back(nbr);
      }
    overlapsBrute.push_back(octOverlaps);
  }

  // Method 2: Get overlaps via O(n+m) search.
  std::vector<std::vector<Oct>> overlapsLinear;
  std::vector<Oct> catOverlapsLinear;
  ot::Overlaps<uint, DIM> overlaps(neighbours, octList);
  std::vector<size_t> overlapIdxs;
  for (size_t i = 0; i < octList.size(); ++i)
  {
    std::vector<Oct> octOverlaps;
    overlaps.keyOverlaps(i, overlapIdxs);
    for (size_t nbrIdx : overlapIdxs)
      octOverlaps.push_back(neighbours[nbrIdx]);
    overlapsLinear.push_back(octOverlaps);
    catOverlapsLinear.insert(catOverlapsLinear.end(),
        octOverlaps.cbegin(), octOverlaps.cend());
  }

  printf("|overlapsBrute|==%lu  |overlapsLinear|==%lu\n",
      catOverlapsBrute.size(), catOverlapsLinear.size());

  // Compare Method 1 and Method 2.
  return overlapsBrute == overlapsLinear;
}


bool equalCoverage(const OctList &octListA_, const OctList &octListB_, MPI_Comm comm)
{
  OctList octListA = octListA_;
  ot::SFC_Tree<uint, DIM>::distCoalesceSiblings(octListA, comm);

  std::vector<int> dest = ot::SFC_Tree<uint, DIM>::treeNode2PartitionRank(
      octListB_,
      ot::SFC_Tree<uint, DIM>::allgatherSplitters(
          octListA.size() > 0, octListA.front(), comm) );
  OctList octListB = par::sendAll(octListB_, dest, comm);
  ot::SFC_Tree<uint, DIM>::locTreeSort(octListB);

  OctList octListUnion;
  octListUnion.insert(octListUnion.end(), octListA.cbegin(), octListA.cend());
  octListUnion.insert(octListUnion.end(), octListB.cbegin(), octListB.cend());
  ot::SFC_Tree<uint, DIM>::locTreeSort(octListUnion);
  ot::SFC_Tree<uint, DIM>::locRemoveDuplicates(octListUnion);

  std::vector<size_t> overlapIdx;
  std::vector<bool> missing(octListUnion.size(), false);

  bool unionInA = true;
  ot::Overlaps<uint, DIM> intersectA(octListA, octListUnion);
  for (size_t i = 0; i < octListUnion.size(); ++i)
  {
    intersectA.keyOverlaps(i, overlapIdx);
    const bool coversOct = overlapIdx.size() > 0
        && octListA[overlapIdx[0]].isAncestorInclusive(octListUnion[i]);
    unionInA &= coversOct;
    missing[i] = missing[i] || !coversOct;
  }

  bool unionInB = true;
  ot::Overlaps<uint, DIM> intersectB(octListB, octListUnion);
  for (size_t i = 0; i < octListUnion.size(); ++i)
  {
    intersectB.keyOverlaps(i, overlapIdx);
    const bool coversOct = overlapIdx.size() > 0
        && octListB[overlapIdx[0]].isAncestorInclusive(octListUnion[i]);
    unionInB &= coversOct;
    missing[i] = missing[i] || !coversOct;
  }

  OctList missingOcts;
  for (size_t i = 0; i < octListUnion.size(); ++i)
    if (missing[i])
      missingOcts.push_back(octListUnion[i]);

  return mpi_and(unionInA && unionInB, comm);
}

