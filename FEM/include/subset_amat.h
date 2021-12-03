
/**
 * @author Masado Ishii
 * @date 2021-12-02
 */

#ifndef DENDRO_KT_SUBSET_AMAT_H
#define DENDRO_KT_SUBSET_AMAT_H

#ifdef BUILD_WITH_AMAT

#include <vector>
#include <numeric>

// Dendro
#include "subset.h"
#include "oda.h"
#include "treeNode.h"
#include "sfcTreeLoop_matvec_io.h"
#include "feMatrix.h"
#include "matRecord.h"

/// // AMat  (assembly-free matvec)
#include "../external/aMat/include/maps.hpp"
#include "../external/aMat/include/aMatFree.hpp"

#include <Eigen/Core>  // Eigen::Matrix

#include "mpi.h"

namespace ot
{
  // createE2NMapping()
  template <unsigned dim>  // hides ::dim.
  std::vector<size_t> createE2NMapping(
      const LocalSubset<dim> &sub);

  // allocAMatMaps()
  template <unsigned int dim,
            typename DT, typename GI, typename LI>
  void allocAMatMaps(
      const LocalSubset<dim> &subset,
      par::Maps<DT, GI, LI>* &meshMaps,
      const DT *prescribedLocalBoundaryVals,
      unsigned int dof);

  // createAMat()
  template <unsigned int dim,
           typename DT, typename GI, typename LI>
  void createAMat(
      const LocalSubset<dim> &subset,
      par::aMat<par::aMatFree<DT, GI, LI>, DT, GI, LI>* &aMat,
      par::Maps<DT, GI, LI>* &meshMaps);

  // destroyAMat()
  template <unsigned int dim,
            typename DT, typename LI, typename GI>
  void destroyAMat(
      const LocalSubset<dim> &subset,
      par::aMat<par::aMatFree<DT, GI, LI>, DT, GI, LI>* &aMat);
}

namespace fem
{
  // getAssembledAMat()
  template <typename feMatLeafT, unsigned int dim, typename AMATType>
  bool getAssembledAMat(
      const ot::LocalSubset<dim> &subset,
      feMatrix<feMatLeafT, dim> &fematrix,
      AMATType* J);

  using EigenMat = ::Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic>;

  template <typename feMatLeafT, unsigned int dim>
  std::vector<EigenMat> subsetEigens(
      const ot::LocalSubset<dim> &subset,
      feMatrix<feMatLeafT, dim> &fematrix);

  template <typename AMATType>
  bool getAssembledAMat(const std::vector<EigenMat> &e_mats, AMATType* J);
}


// ---------------------------------------------


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

    MatvecBaseIn<dim, unsigned, false> eleLoop(
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
      const std::vector<size_t> &ghostedBoundaryIndices = subset.getBoundaryNodeIndices();
      std::vector<GI> global_boundary_dofs(ghostedBoundaryIndices.size() * dof);
      for (size_t ii = 0; ii < ghostedBoundaryIndices.size(); ++ii)
        for (LI v = 0; v < dof; ++v)
          global_boundary_dofs[dof * ii + v] =
              dof * ghostedBoundaryIndices[ii] + v;

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
}

namespace fem
{
  template <typename feMatLeafT, unsigned int dim, typename AMATType>
  bool getAssembledAMat(
      const ot::LocalSubset<dim> &subset,
      feMatrix<feMatLeafT, dim> &fematrix,
      AMATType* J)
  {
    using DT = typename AMATType::DTType;
    using GI = typename AMATType::GIType;
    using LI = typename AMATType::LIType;

    if(subset.da()->isActive())
    {
      const size_t numLocElem = subset.getLocalElementSz();

      /// preMat();
      std::vector<unsigned char> elemNonzero(numLocElem, false);
      ot::MatCompactRows matCompactRows = fematrix.collectMatrixEntries(
          subset.relevantOctList(),
          subset.relevantNodes(),
          subset.getNodeLocalToGlobalMap(),  // identity
          elemNonzero.data());
      /// postMat();

      const unsigned ndofs = matCompactRows.getNdofs();
      const unsigned nPe = matCompactRows.getNpe();
      const std::vector<ot::MatCompactRows::ScalarT> & entryVals = matCompactRows.getColVals();

      EigenMat* eMat[2] = {nullptr, nullptr};
      eMat[0] = new EigenMat();
      eMat[0]->resize(ndofs*nPe, ndofs*nPe);
      const EigenMat * read_eMat[2] = {eMat[0], eMat[1]};

      unsigned aggRow = 0;
      for (unsigned int eid = 0; eid < numLocElem; ++eid)
      {
        if (elemNonzero[eid])
        {
          // Clear elemental matrix.
          for(unsigned int r = 0; r < (nPe*ndofs); ++r)
            for(unsigned int c = 0; c < (nPe*ndofs); ++c)
              (*(eMat[0]))(r,c) = 0;

          // Overwrite elemental matrix.
          for (unsigned int r = 0; r < (nPe*ndofs); ++r)
          {
            for (unsigned int c = 0; c < (nPe*ndofs); ++c)
              (*(eMat[0]))(r,c) = entryVals[aggRow * (nPe*ndofs) + c];
            aggRow++;
          }

          LI n_i[1]={0};
          LI n_j[1]={0};
          J->set_element_matrix(eid, n_i, n_j, read_eMat, 1u);
          // note that read_eMat[0] points to the same memory as eMat[0].
        }
      }

      delete eMat[0];
    }
    PetscFunctionReturn(0);
  }


  // subsetEigens()
  template <typename feMatLeafT, unsigned int dim>
  std::vector<EigenMat> subsetEigens(
      const ot::LocalSubset<dim> &subset,
      feMatrix<feMatLeafT, dim> &fematrix)
  {
    std::vector<EigenMat> e_mats;
    if(subset.da()->isActive())
    {
      const size_t numLocElem = subset.getLocalElementSz();

      std::vector<unsigned char> elemNonzero(numLocElem, false);
      ot::MatCompactRows matCompactRows = fematrix.collectMatrixEntries(
          subset.relevantOctList(),
          subset.relevantNodes(),
          subset.getNodeLocalToGlobalMap(),  // identity
          elemNonzero.data());

      const int dofPe = matCompactRows.getNdofs() * matCompactRows.getNpe();
      const std::vector<ot::MatCompactRows::ScalarT> & entryVals = matCompactRows.getColVals();

      e_mats.resize(numLocElem);
      size_t aggRow = 0;
      for (size_t eid = 0; eid < numLocElem; ++eid)
      {
        if (elemNonzero[eid])
        {
          e_mats[eid].resize(dofPe, dofPe);
          for (int r = 0; r < dofPe; ++r, ++aggRow)
            for (int c = 0; c < dofPe; ++c)
              e_mats[eid](r,c) = entryVals[aggRow * dofPe + c];
        }
      }
    }
    return e_mats;
  }

  template <typename AMATType>
  bool getAssembledAMat(const std::vector<EigenMat> &e_mats, AMATType* J)
  {
    for (size_t eid = 0; eid < e_mats.size(); ++eid)
      if (e_mats[eid].size() > 0)
        J->set_element_matrix(eid, e_mats[eid], 0, 0, 1);
    PetscFunctionReturn(0);
  }


}


#endif//BUILD_WITH_AMAT
#endif//DENDRO_KT_SUBSET_AMAT_H
