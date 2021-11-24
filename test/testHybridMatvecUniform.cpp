// Dendro
#include "distTree.h"
#include "oda.h"
#include "subset.h"
#include "intergridTransfer.h"
#include "gmgMat.h"
#include "point.h"
#include "refel.h"
#include "tensor.h"
#include <set>

// Dendro examples
#include "poissonMat.h"
#include "poissonVec.h"

// PETSc
#include "petsc.h"

// AMat  (assembly-free matvec)
#include "../external/aMat/include/aMatBased.hpp"
#include "../external/aMat/include/aMatFree.hpp"

// stdlib
#include <iostream>

constexpr int dim = 2;
using uint = unsigned int;

template <typename NodeT>
std::ostream & print(const ot::DA<dim> *da,
                     const NodeT *vec,
                     std::ostream & out = std::cout);

template <typename C, unsigned int dim>
std::vector<bool> boundaryFlags(
    const ot::DistTree<C, dim> &distTree,
    const int stratum = 0);

template <typename NodeT>
NodeT * ptr(std::vector<NodeT> &vec) { return vec.data(); }

template <typename NodeT>
const NodeT * const_ptr(const std::vector<NodeT> &vec) { return vec.data(); }


namespace ot
{
  /// template <unsigned dim>  // hides ::dim.
  /// std::vector<size_t> createE2NMapping(
  ///     const GlobalSubset<dim> &sub);

  /// template <unsigned int dim,
  ///           typename DT, typename GI, typename LI>
  /// void allocAMatMaps(
  ///     const GlobalSubset<dim> &subset,
  ///     par::Maps<DT, GI, LI>* &meshMaps,
  ///     const DT *prescribedLocalBoundaryVals,
  ///     unsigned int dof);



  template <unsigned dim>  // hides ::dim.
  std::vector<size_t> createE2NMapping(
      const LocalSubset<dim> &sub);

  template <unsigned int dim,
            typename DT, typename GI, typename LI>
  void allocAMatMaps(
      const LocalSubset<dim> &subset,
      par::Maps<DT, GI, LI>* &meshMaps,
      const DT *prescribedLocalBoundaryVals,
      unsigned int dof);

  template <unsigned int dim,
           typename DT, typename GI, typename LI>
  void createAMat(
      const LocalSubset<dim> &subset,
      par::aMat<par::aMatFree<DT, GI, LI>, DT, GI, LI>* &aMat,
      par::Maps<DT, GI, LI>* &meshMaps);

  template <unsigned int dim,
            typename DT, typename LI, typename GI>
  void destroyAMat(
      const LocalSubset<dim> &subset,
      par::aMat<par::aMatFree<DT, GI, LI>, DT, GI, LI>* &aMat);


}


//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  _InitializeHcurve(dim);

  MPI_Comm comm = PETSC_COMM_WORLD;
  int mpiRank, mpiSize;
  MPI_Comm_rank(comm, &mpiRank);
  MPI_Comm_size(comm, &mpiSize);

  const int eleOrder = 1;
  const unsigned int ndofs = 1;
  const double sfc_tol = 0.3;

  using DTree_t = ot::DistTree<uint, dim>;
  using DA_t = ot::DA<dim>;

  // Static uniform refinement to depth coarseLev.
  const int coarseLev = 3;
  DTree_t dtree = DTree_t::constructSubdomainDistTree(
      coarseLev, comm, sfc_tol);

  // DA
  DA_t * da = new DA_t(dtree, comm, eleOrder, 100, sfc_tol);

  // Subset
  std::vector<bool> bdryFlags = boundaryFlags(dtree);
  ot::LocalSubset<dim> subsetExp(da, &dtree.getTreePartFiltered(), bdryFlags, true);
  ot::LocalSubset<dim> subsetImp(da, &dtree.getTreePartFiltered(), bdryFlags, false);

#if 0
  {
    // Sanity checks. Should all match due to partitioning from simpleDA.
    ot::GlobalSubset<dim> subsetAll(da, &dtree.getTreePartFiltered(), std::vector<bool>(bdryFlags.size(), true), true);

#define ASSERT_EQUAL_DA(method) assert(subsetAll.method() == da->method());
#define PRINT_DIFF(method) { if (subsetAll.method() != da->method()) {\
  fprintf(stderr, "%s:\t%u:%u (%d)\n",\
      #method,\
      (unsigned) subsetAll.method(),\
      (unsigned) da->method(),\
      int(subsetAll.method() - da->method()));\
  }}
  PRINT_DIFF( getLocalElementSz );              ASSERT_EQUAL_DA( getLocalElementSz );
  PRINT_DIFF( getLocalNodalSz );                ASSERT_EQUAL_DA( getLocalNodalSz );
  PRINT_DIFF( getLocalNodeBegin );              ASSERT_EQUAL_DA( getLocalNodeBegin );
  PRINT_DIFF( getLocalNodeEnd );                ASSERT_EQUAL_DA( getLocalNodeEnd );
  PRINT_DIFF( getPreNodalSz );                  ASSERT_EQUAL_DA( getPreNodalSz );
  PRINT_DIFF( getPostNodalSz );                 ASSERT_EQUAL_DA( getPostNodalSz );
  PRINT_DIFF( getTotalNodalSz );                ASSERT_EQUAL_DA( getTotalNodalSz );
  PRINT_DIFF( getGlobalNodeSz );                ASSERT_EQUAL_DA( getGlobalNodeSz );
  PRINT_DIFF( getGlobalNodeBegin );             ASSERT_EQUAL_DA( getGlobalNodeBegin );
  PRINT_DIFF( getGlobalElementSz );             ASSERT_EQUAL_DA( getGlobalElementSz );
  PRINT_DIFF( getGlobalElementBegin );          ASSERT_EQUAL_DA( getGlobalElementBegin );
#undef PRINT_DIFF
#undef ASSERT_EQUAL_DA
  }
#endif


  // poissonMat and poissonVec
  const double d_min=-0.5;
  const double d_max=0.5;
  Point<dim> pt_min(d_min,d_min,d_min);
  Point<dim> pt_max(d_max,d_max,d_max);
  PoissonEq::PoissonMat<dim> poissonMat(da, &dtree.getTreePartFiltered(), 1);
  poissonMat.setProblemDimensions(pt_min,pt_max);
  PoissonEq::PoissonVec<dim> poissonVec(da, &dtree.getTreePartFiltered(), 1);
  poissonVec.setProblemDimensions(pt_min,pt_max);

  // vectors
  std::vector<double> u, v_pure, v_hybrid;
  da->createVector(u, false, false, 1);
  da->createVector(v_pure, false, false, 1);
  da->createVector(v_hybrid, false, false, 1);
  std::fill(v_pure.begin(), v_pure.end(), 0);
  std::fill(v_hybrid.begin(), v_hybrid.end(), 0);

  // set u = x*x + y*y
  for (size_t i = 0; i < da->getLocalNodalSz(); ++i)
  {
    // unit cube coords
    double x[dim];
    ot::treeNode2Physical((da->getTNCoords() + da->getLocalNodeBegin())[i], eleOrder, x);

    // scale
    for (int d = 0; d < dim; ++d)
      x[d] *= pt_max.x(d) - pt_min.x(d);

    // shift
    for (int d = 0; d < dim; ++d)
      x[d] += pt_min.x(d);

    u[i] = x[0]*x[0] + x[1]*x[1];
  }


  // -----------------------------
  // aMat
  // -----------------------------
  typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>  aMatFree; // aMat type taking aMatBased as derived class
  typedef par::Maps<double,unsigned long,unsigned int> aMatMaps;

  //TODO subset: boundary, allocAMatMaps, createAMat, createVector, split/merge
  // TODO assemble elemental matrices for subset
  // TODO separate the boundary handling
  // future (not this test): find dependent and independent local element sets
  // future (not this test): explore streaming

  std::vector<double> dirichletZeros(da->getBoundaryNodeIndices().size() * ndofs, 0.0);

  // whole da
  aMatFree* stMatFree=NULL;
  aMatMaps* meshMaps=NULL;
  da->allocAMatMaps(meshMaps, dtree.getTreePartFiltered(), dirichletZeros.data(), ndofs);
  da->createAMat(stMatFree,meshMaps);
  stMatFree->set_matfree_type((par::MATFREE_TYPE)1);
  poissonMat.getAssembledAMat(stMatFree);
  stMatFree->finalize();

  stMatFree->matvec(ptr(v_pure), const_ptr(u), false);


  // subset explicit
  aMatMaps *meshMapsExp = nullptr;
  aMatFree *matFreeExp = nullptr;
  allocAMatMaps(subsetExp, meshMapsExp, dirichletZeros.data(), ndofs);
  createAMat(subsetExp, matFreeExp, meshMapsExp);
  matFreeExp->set_matfree_type((par::MATFREE_TYPE) 1);
  //TODO assemble
  matFreeExp->finalize();

  /// matFreeExp->matvec(ptr(v_subExp), const_ptr(u_subExp), false);


  print(da, v_pure.data(), std::cout) << "------------------------------------\n";

  delete da;
  _DestroyHcurve();
  PetscFinalize();

  return 0;
}


template <typename NodeT>
std::ostream & print(const ot::DA<dim> *da,
                     const NodeT *vec,
                     std::ostream & out)
{
  return ot::printNodes(
      da->getTNCoords() + da->getLocalNodeBegin(),
      da->getTNCoords() + da->getLocalNodeBegin() + da->getLocalNodalSz(),
      vec,
      da->getElementOrder(),
      out);
}


template <typename C, unsigned int dim>
std::vector<bool> boundaryFlags(
    const ot::DistTree<C, dim> &distTree,
    const int stratum)
{
  const std::vector<ot::TreeNode<C, dim>> &elements =
      distTree.getTreePartFiltered(stratum);
  const size_t size = elements.size();

  std::vector<bool> boundaryFlags(size, false);

  for (size_t ii = 0; ii < size; ++ii)
    boundaryFlags[ii] = elements[ii].getIsOnTreeBdry();

  return boundaryFlags;
}




namespace ot
{

  //
  // createE2NMapping()
  //
  template <unsigned dim>
  std::vector<size_t> createE2NMapping(
      const LocalSubset<dim> &sub)
  {
    const std::vector<TreeNode<unsigned, dim>> & octs = sub.relevantOctList();
    const std::vector<TreeNode<unsigned, dim>> & nodes = sub.relevantNodes();

    std::vector<unsigned> indices(nodes.size());
    std::iota(indices.begin(), indices.end(), 0);

    const int nPe = sub.da()->getNumNodesPerElement();
    const int eleOrder = sub.da()->getElementOrder();
    const int singleDof = 1;

    MatvecBaseIn<dim, unsigned> eleLoop(
        nodes.size(),
        singleDof,
        eleOrder,
        false, 0, 
        nodes.data(),
        indices.data(),
        octs.data(),
        octs.size());

    std::vector<size_t> e2nMapping;

    while (!eleLoop.isFinished())
    {
      // For each element...
      if (eleLoop.isPre() && eleLoop.subtreeInfo().isLeaf())
      {
        const unsigned *nodeIdxs = eleLoop.subtreeInfo().readNodeValsIn();

        // For each node on the element...
        for (unsigned ii = 0; ii < nPe; ++ii)
          e2nMapping.push_back(nodeIdxs[ii]);  // set the node id.
        eleLoop.next();
      }
      else
        eleLoop.step();
    }

    return e2nMapping;
  }



  //TODO subset amat: boundary, allocAMatMaps, createAMat, createVector, split/merge


  /** @note: Because indices are local rather than global,
   *         the resulting aMat cannot be assembled over petsc.
   */
  template <unsigned int dim,
            typename DT, typename GI, typename LI>
  void allocAMatMaps(
      const LocalSubset<dim> &subset,
      par::Maps<DT, GI, LI>* &meshMaps,
      const DT *prescribedLocalBoundaryVals,
      unsigned int dof)
  {
    using CoordType = unsigned int;
    if (subset.da()->isActive())
    {
      MPI_Comm selfComm = MPI_COMM_SELF;
      const std::vector<size_t> e2n_cg = createE2NMapping(subset);

      const LI cgSz = subset.getLocalNodalSz();
      const LI localElems = subset.getLocalElementSz();
      const LI nPe = subset.da()->getNumNodesPerElement();
      const LI dofPe = nPe * dof; // dof per elem
      const LI localNodes = subset.getLocalNodalSz();

      // dof_per_elem
      LI * dof_per_elem = new LI[localElems];
      for (LI i = 0; i < localElems; ++i)
        dof_per_elem[i] = dofPe;

      // dofmap_local ([element][element dof] --> vec dof).
      LI** dofmap_local = new LI*[localElems];
      for(LI i = 0; i < localElems; ++i)
        dofmap_local[i] = new LI[dof_per_elem[i]];

      for (LI ele = 0; ele < localElems; ++ele)
        for (LI node = 0; node < nPe; ++node)
          for (LI v = 0; v < dof; ++v)
            dofmap_local[ele][dof*node + v] = dof * e2n_cg[ele*nPe + node] + v;

      // There are no ghost nodes, only local.
      const LI valid_local_dof = dof * localNodes;

      // Create local_to_global_dofM
      // (extension of LocalToGlobalMap to dofs)
      // (except LocalToGlobalMap is the identity).
      std::vector<GI> local_to_global_dofM(cgSz * dof);
      std::iota(local_to_global_dofM.begin(),
                local_to_global_dofM.end(),
                0);
      const GI dof_start_global = 0;
      const GI dof_end_global   = dof * localNodes - 1;
      const GI total_dof_global = dof * localNodes;

      // Boundary data (using global dofs, which in this case are local dofs).
      const std::vector<size_t> &localBoundaryIndices = subset.getBoundaryNodeIndices();
      std::vector<GI> global_boundary_dofs(localBoundaryIndices.size() * dof);
      for (size_t ii = 0; ii < localBoundaryIndices.size(); ++ii)
        for (LI v = 0; v < dof; ++v)
          global_boundary_dofs[dof * ii + v] =
              dof * localBoundaryIndices[ii] + v;

      // Define output meshMaps.
      meshMaps = new par::Maps<DT, GI, LI>(selfComm);

      meshMaps->set_map(
          localElems,
          dofmap_local,
          dof_per_elem,
          valid_local_dof,
          local_to_global_dofM.data(),
          dof_start_global,
          dof_end_global,
          total_dof_global);

      meshMaps->set_bdr_map(
          global_boundary_dofs.data(),
          const_cast<DT *>( prescribedLocalBoundaryVals ),  // hack until AMat fixes const
          global_boundary_dofs.size());

      // cleanup.
      for(LI i = 0; i < localElems; ++i)
          delete [] dofmap_local[i];
      delete [] dofmap_local;
    }
  }


  // createAMat()
  template <unsigned int dim, typename DT, typename GI, typename LI>
  void createAMat(
      const LocalSubset<dim> &subset,
      par::aMat<par::aMatFree<DT, GI, LI>, DT, GI, LI>* &aMat,
      par::Maps<DT, GI, LI>* &meshMaps)
  {
    aMat = nullptr;
    if (subset.da()->isActive())
      aMat = new par::aMatFree<DT, GI, LI>(*meshMaps, par::BC_METH::BC_IMATRIX);
  }


  // destroyAMat()
  template <unsigned int dim, typename DT, typename LI, typename GI>
  void destroyAMat(
      const LocalSubset<dim> &subset,
      par::aMat<par::aMatFree<DT, GI, LI>, DT, GI, LI>* &aMat)
  {
    if (aMat != nullptr)
      delete aMat;
    aMat = nullptr;
  }








  /// template <unsigned int dim,
  ///           typename DT, typename GI, typename LI>
  /// void allocAMatMaps(
  ///     const GlobalSubset<dim> &subset,
  ///     par::Maps<DT, GI, LI>* &meshMaps,
  ///     const DT *prescribedLocalBoundaryVals,
  ///     unsigned int dof)
  /// {
  ///   using CoordType = unsigned int;
  ///   if (subset.da()->isActive())
  ///   {
  ///     MPI_Comm acomm = subset.da()->getCommActive();
  ///     const std::vector<size_t> e2n_cg = createE2NMapping(subset);

  ///     const LI cgSz = subset.getTotalNodalSz();
  ///     const LI localElems = subset.getLocalElementSz();
  ///     const LI nPe = subset.da()->getNumNodesPerElement();
  ///     const LI dofPe = nPe * dof; // dof per elem
  ///     const LI localNodes = subset.getLocalNodalSz();
  ///     const LI localBegin = subset.getLocalNodeBegin();
  ///     const LI localEnd = subset.getLocalNodeEnd();

  ///     // dof_per_elem
  ///     LI * dof_per_elem = new LI[localElems];
  ///     for (LI i = 0; i < localElems; ++i)
  ///       dof_per_elem[i] = dofPe;

  ///     // dofmap_local ([element][element dof] --> ghosted vec dof).
  ///     LI** dofmap_local = new LI*[localElems];
  ///     for(LI i = 0; i < localElems; ++i)
  ///       dofmap_local[i] = new LI[dof_per_elem[i]];

  ///     for (LI ele = 0; ele < localElems; ++ele)
  ///       for (LI node = 0; node < nPe; ++node)
  ///         for (LI v = 0; v < dof; ++v)
  ///           dofmap_local[ele][dof*node + v] = dof * e2n_cg[ele*nPe + node] + v;

  ///     // sm_ghost_id (set of ghost nodes incicent on local elements).
  ///     // Ultimately used to compute `valid_local_dof'.
  ///     //   (if node partition is consistent with element partition,
  ///     //   such that every ghost node is incident on at least one of
  ///     //   the owned elements, then  valid_local_dof == cgSz.
  ///     std::set<LI> sm_ghost_id;
  ///     std::copy_if( e2n_cg.begin(),   e2n_cg.end(),
  ///                   std::inserter(sm_ghost_id, sm_ghost_id.end()),
  ///                   [=](LI idx) { return idx < localBegin || idx >= localEnd; }
  ///         );
  ///     // valid_local_dof
  ///     const LI valid_local_dof = dof * (localNodes + sm_ghost_id.size());

  ///     // Create local_to_global_dofM
  ///     // (extension of LocalToGlobalMap to dofs)
  ///     const std::vector<RankI> & subset_local2global = subset.getNodeLocalToGlobalMap();
  ///     std::vector<GI> local_to_global_dofM(cgSz * dof);
  ///     for (size_t ghosted_nIdx = 0; ghosted_nIdx < cgSz; ++ghosted_nIdx)
  ///       for (LI v = 0; v < dof; ++v)
  ///         local_to_global_dofM[dof * ghosted_nIdx + v]
  ///           = dof * subset_local2global[ghosted_nIdx] + v;
  ///     const GI dof_start_global = subset_local2global[localBegin]*dof;
  ///     const GI dof_end_global   = subset_local2global[localEnd-1]*dof + (dof-1);
  ///     const GI total_dof_global = subset.getGlobalNodeSz() * dof;

  ///     // Boundary data
  ///     const std::vector<size_t> &localBoundaryIndices = subset.getBoundaryNodeIndices();
  ///     std::vector<GI> global_boundary_dofs(localBoundaryIndices.size() * dof);
  ///     for (size_t ii = 0; ii < localBoundaryIndices.size(); ++ii)
  ///       for (LI v = 0; v < dof; ++v)
  ///         global_boundary_dofs[dof * ii + v] =
  ///             dof * subset_local2global[localBoundaryIndices[ii]] + v;

  ///     // Define output meshMaps.
  ///     meshMaps = new par::Maps<DT, GI, LI>(acomm);

  ///     meshMaps->set_map(
  ///         localElems,
  ///         dofmap_local,
  ///         dof_per_elem,
  ///         valid_local_dof,
  ///         local_to_global_dofM.data(),
  ///         dof_start_global,
  ///         dof_end_global,
  ///         total_dof_global);

  ///     meshMaps->set_bdr_map(
  ///         global_boundary_dofs.data(),
  ///         const_cast<DT *>( prescribedLocalBoundaryVals ),  // hack until AMat fixes const
  ///         global_boundary_dofs.size());

  ///     // cleanup.
  ///     for(LI i = 0; i < localElems; ++i)
  ///         delete [] dofmap_local[i];
  ///     delete [] dofmap_local;
  ///   }
  /// }




}










