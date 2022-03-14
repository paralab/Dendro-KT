

#include "dendro.h"

#include "pcoord.h"
#include "distTree.h"
#include "oda.h"

#include "sfcTreeLoop_matvec_io.h"

#include <mpi.h>
#include <array>
#include <iostream>
#include <sstream>
#include <set>

constexpr int DIM = 2;
using uint = unsigned int;
using RankI = ot::RankI;


ot::DistTree<uint, DIM> refineOnBoundary(const ot::DistTree<uint, DIM> &distTree);

//
// ConstMeshPointers  (wrapper)
//
template <unsigned int dim>
class ConstMeshPointers
{
  private:
    const ot::DistTree<unsigned int, dim> *m_distTree;
    const ot::DA<dim> *m_da;
    /// const unsigned m_stratum;

  public:
    // ConstMeshPointers constructor
    ConstMeshPointers(const ot::DistTree<unsigned int, dim> *distTree,
                      const ot::DA<dim> *da/*,
                      unsigned stratum = 0*/)
      :
        m_distTree(distTree),
        m_da(da)/*,
        m_stratum(stratum)*/
    {}

    // Copy constructor and copy assignment.
    ConstMeshPointers(const ConstMeshPointers &other) = default;
    ConstMeshPointers & operator=(const ConstMeshPointers &other) = default;

    // distTree()
    const ot::DistTree<unsigned int, dim> * distTree() const { return m_distTree; }

    // numElements()
    size_t numElements() const
    {
      return m_distTree->getTreePartFiltered(/*m_stratum*/).size();
    }

    // da()
    const ot::DA<dim> * da() const { return m_da; }

    /// unsigned stratum() const { return m_stratum; }
};


//
// Vector  (wrapper)
//
template <typename ValT>
class Vector
{
  private:
    std::vector<ValT> m_data;
    bool m_isGhosted;
    size_t m_ndofs = 1;
    unsigned long long m_globalNodeRankBegin = 0;

  public:
    // Vector constructor
    template <unsigned int dim>
    Vector(const ConstMeshPointers<dim> &mesh, bool isGhosted, size_t ndofs, const ValT *input = nullptr)
    : m_isGhosted(isGhosted), m_ndofs(ndofs)
    {
      mesh.da()->createVector(m_data, false, isGhosted, ndofs);
      if (input != nullptr)
        std::copy_n(input, m_data.size(), m_data.begin());
      m_globalNodeRankBegin = mesh.da()->getGlobalRankBegin();
    }

    // data()
    std::vector<ValT> & data() { return m_data; }
    const std::vector<ValT> & data() const { return m_data; }

    // isGhosted()
    bool isGhosted() const { return m_isGhosted; }

    // ndofs()
    size_t ndofs() const { return m_ndofs; }

    // ptr()
    ValT * ptr() { return m_data.data(); }
    const ValT * ptr() const { return m_data.data(); }

    // size()
    size_t size() const { return m_data.size(); }

    // globalRankBegin();
    unsigned long long globalRankBegin() const { return m_globalNodeRankBegin; }
};



//
// ElementLoopIn  (wrapper)
//
template <unsigned int dim, typename ValT>
class ElementLoopIn
{
  private:
    ot::MatvecBaseIn<dim, ValT> m_loop;

  public:
    ot::MatvecBaseIn<dim, ValT> & loop() { return m_loop; }

    ElementLoopIn(const ConstMeshPointers<dim> &mesh,
                  const Vector<ValT> &ghostedVec)
      :
        m_loop(mesh.da()->getTotalNodalSz(),
               ghostedVec.ndofs(),
               mesh.da()->getElementOrder(),
               false,
               0,
               mesh.da()->getTNCoords(),
               ghostedVec.ptr(),
               mesh.distTree()->getTreePartFiltered().data(),
               mesh.distTree()->getTreePartFiltered().size(),
               *mesh.da()->getTreePartFront(),
               *mesh.da()->getTreePartBack())
    { }
};





// main()
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();

  _InitializeHcurve(DIM);
  periodic::PCoord<uint, DIM>::periods({(1u<<m_uiMaxDepth)/2, periodic::NO_PERIOD});
  /// periodic::PCoord<uint, DIM>::periods({periodic::NO_PERIOD, periodic::NO_PERIOD});

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  const int fineLevel = 3;

  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::constructSubdomainDistTree(fineLevel, comm);

  ot::quadTreeToGnuplot(distTree.getTreePartFiltered(), fineLevel, "uniformPeriodic", comm);

  ot::DA<DIM> * da = new ot::DA<DIM>(distTree, comm, 1);

  {
    const RankI cellDims = (1u << fineLevel) / 2;
    const RankI vertexDims = (1u << fineLevel) + 1;
    const RankI expectedNodes = cellDims * vertexDims;  // 2D

    const RankI measuredNodes = da->getGlobalNodeSz();

    fprintf(stdout, "expectedNodes==%llu  %smeasuredNodes==%llu%s\n",
        expectedNodes,
        (expectedNodes == measuredNodes ? GRN : RED),
        measuredNodes,
        NRM);
  }

  // -----------------------------------------------------------

  ConstMeshPointers<DIM> mesh(&distTree, da);
  enum { ghosted = true };
  Vector<double> ghostedVec(mesh, ghosted, 1,
      std::vector<double>(mesh.da()->getTotalNodalSz(), 1).data());
  ElementLoopIn<DIM, double> loop(mesh, ghostedVec);
  const int nPe = mesh.da()->getNumNodesPerElement();

  std::vector<int> nodesPerElem;
  std::vector<double> sumNodesPerElem;
  while (!loop.loop().isFinished())
  {
    if (loop.loop().isPre()
        && loop.loop().subtreeInfo().isLeaf())
    {
      double sumNodes = 0;
      for (size_t nIdx = 0; nIdx < nPe; ++nIdx)
        sumNodes += loop.loop().subtreeInfo().readNodeValsIn()[nIdx];
      sumNodesPerElem.push_back(sumNodes);
      nodesPerElem.push_back(loop.loop().subtreeInfo().getNumNonhangingNodes());

      loop.loop().next();
    }
    else
      loop.loop().step();
  }

  const size_t loopedElements = nodesPerElem.size();
  fprintf(stdout, "elements==%lu  %slooped elements==%lu%s\n",
      mesh.numElements(),
      (loopedElements == mesh.numElements() ? GRN : RED),
      loopedElements,
      NRM);

  std::set<size_t> wrongElems;
  for (size_t ii = 0; ii < loopedElements; ++ii)
    if (sumNodesPerElem[ii] != nPe)
      wrongElems.insert(ii);
  if (wrongElems.size() > 0)
  {
    fprintf(stdout, (RED "%lu elements are missing nodes: " MAG), wrongElems.size());
    for (size_t ii : wrongElems)
      fprintf(stdout, "  %2lu(%.0f)", ii, sumNodesPerElem[ii]);
    fprintf(stdout, NRM "\n");
  }
  else
  {
    fprintf(stdout, GRN "All nodes accounted for.\n" NRM);
  }

  delete da;

  _DestroyHcurve();

  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}


// refineOnBoundary()
ot::DistTree<uint, DIM> refineOnBoundary(const ot::DistTree<uint, DIM> &distTree)
{
  const std::vector<ot::TreeNode<uint, DIM>> oldTree = distTree.getTreePartFiltered();
  const size_t oldSz = oldTree.size();
  std::vector<ot::OCT_FLAGS::Refine> refines(oldSz, ot::OCT_FLAGS::OCT_NO_CHANGE);
  for (size_t ii = 0; ii < oldSz; ++ii)
    if (oldTree[ii].getIsOnTreeBdry())
      refines[ii] = ot::OCT_FLAGS::OCT_REFINE;

  ot::DistTree<uint, DIM> newDistTree;
  ot::DistTree<uint, DIM> surrDistTree;
  ot::DistTree<uint, DIM>::distRemeshSubdomain(
      distTree, refines, newDistTree, surrDistTree, ot::SurrogateInByOut, 0.3);
  return newDistTree;
}

