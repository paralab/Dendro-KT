/**
 * @file: sfcTreeLoop_matvec.h
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-10-24
 * @brief: Matvec-style iteration over node values, using SFC_TreeLoop.
 */


#ifndef DENDRO_KT_SFC_TREE_LOOP_MATVEC_H
#define DENDRO_KT_SFC_TREE_LOOP_MATVEC_H

#include "sfcTreeLoop.h"

#include "nsort.h"
#include "tsort.h"
#include "treeNode.h"
#include "mathUtils.h"
#include "binUtils.h"

/// #include "refel.h"
/// #include "tensor.h"


#include <vector>
#include <tuple>


namespace ot
{

  // For matvec, there can be at most one level of interpolation existing
  // at a time, so one leaf buffer and one parent buffer will suffice.
  // The same may or may not be true of intergrid transfer.
  // We'll cross that bridge later.
  // For now though, the matvec derived class/specialization can have its
  // own parent/leaf buffers that exist outside of the stack.
  // Because a parent buffer will still be shared by multiple children, it
  // needs to be initialized to 0s before the children are processed.
  // We can do that in the constructor and the bottomUpNodes.
  //

  // We actually don't need parent2Child or child2Parent as designated functions.
  // We can detect and interpolate hanging children during topDownNodes.
  // When requesting the element buffer, can always specify to possibly
  // interpolate all the nodes.

  template <unsigned int dim, typename NodeT>
  class MatvecBase : public SFC_TreeLoop<dim,
                                         Inputs<TreeNode<unsigned int, dim>, NodeT>,
                                         Outputs<NodeT>,
                                         DefaultSummary,
                                         MatvecBase<dim, NodeT>>
  {
    using FrameT = Frame<dim, Inputs<TreeNode<unsigned int, dim>, NodeT>, Outputs<NodeT>, DefaultSummary, MatvecBase>;

    public:
      static constexpr unsigned int NumChildren = 1u << dim;

      /// MatvecBase(input data);  // TODO calls parent constructor and then initializes.

      void topDownNodes(FrameT &parentFrame, ExtantCellFlagT *extantChildren);
      void bottomUpNodes(FrameT &parentFrame, ExtantCellFlagT extantChildren);
      void parent2Child(FrameT &parentFrame, FrameT &childFrame);
      void child2Parent(FrameT &parentFrame, FrameT &childFrame);

    protected:
      unsigned int m_ndofs;  //TODO needs to be initialized in constructor.
      unsigned int m_eleOrder;

      // Non-stack leaf buffer and parent-of-leaf buffer.
      std::vector<TreeNode<unsigned int,dim>> m_leafNodeCoords;
      /// std::vector<TreeNode<unsigned int,dim>> m_parentNodeCoords;
      std::vector<NodeT> m_leafNodeVals;  //TODO resize in constructor.
      std::vector<NodeT> m_parentNodeVals;  //TODO resize in constructor.
      std::vector<double> m_leafNodeCoordsFlat;  //TODO resize in constructor.

      InterpMatrices<dim, NodeT> m_interp_matrices;
  };

  /*
   *
   *
   *
   *
   *
   *
   */

  template <unsigned int dim, typename NodeT>
  void MatvecBase<dim, NodeT>::topDownNodes(FrameT &parentFrame, ExtantCellFlagT *extantChildren)
  {
    /**
     *  Copied from sfcTreeLoop.h:
     *
     *  topDownNodes()
     *  is responsible to
     *    1. Resize the child input buffers (SFC order) in the parent frame;
     *
     *    2. Duplicate elements of the parent input buffers to
     *       incident child input buffers (SFC order);
     *
     *    2.1. Initialize a summary object for each child (SFC order).
     *
     *    3. Indicate to SFC_TreeLoop which children to traverse,
     *       by accumulating into the extantChildren bit array (Morton order).
     *
     *  Restrictions
     *    - MAY NOT resize or write to parent input buffers.
     *    - MAY NOT resize or write to variably sized output buffers.
     *
     *  Utilities are provided to identify and iterate over incident children.
     */

    // =========================
    // Top-down Outline:
    // =========================
    // - First pass: Count (#nodes, finest node level) per child.
    //   - Note: A child is a leaf iff finest node level == subtree level.
    //   - Note: A child is a leaf with hanging nodes if #nodes < npe.
    //
    // - Allocate child input nodes (with at least npe per child).
    //
    // - For each child:
    //   - If child has hanging nodes, interpolate from parent.
    //     - Note: Any interpolated nonhanging nodes will be overwritten anyway.
    //
    // - Second pass: Duplicate parent nodes into children.
    //   - If a child is a leaf and #nonhanging nodes <= npe, copy into lex position.
    //   - Else copy nodes into same order as they appear in parent.
    // ========================================================================

    const unsigned npe = intPow(m_eleOrder+1, dim);
    const TreeNode<unsigned int,dim> & parSubtree = this->getCurrentSubtree();

    std::array<size_t, NumChildren> childNodeCounts;
    std::array<LevI, NumChildren> childFinestLevel;
    childNodeCounts.fill(0);
    childFinestLevel.fill(0);
    *extantChildren = 0u;

    const std::vector<TreeNode<unsigned int, dim>> &myNodes = parentFrame.template getMyInputHandle<0>();
    const size_t numInputNodes = parentFrame.mySummaryHandle.m_subtreeNodeCount;

    //
    // Initial pass over the input data.
    // Count #points per child, finest level, extant children.
    //
    for (const auto &nodeInstance : IterateNodesToChildren<dim>( this->getCurrentSubtree(),
                                                                 &(*myNodes.begin()),
                                                                 numInputNodes,
                                                                 this->getCurrentRotation() ))
    {
      const ChildI child_sfc = nodeInstance.getChild_sfc();

      const LevI nodeLevel = myNodes[nodeInstance.getPNodeIdx()].getLevel();
      if (childFinestLevel[child_sfc] < nodeLevel)
        childFinestLevel[child_sfc] = nodeLevel;
      childNodeCounts[child_sfc]++;

      extantChildren |= (1u << nodeInstance.getChild_m());
    }

    //
    // Update child summaries.
    //
    bool thereAreHangingNodes = false;
    DefaultSummary (&summaries)[NumChildren] = parentFrame.childSummaries;
    for (ChildI child_sfc = 0; child_sfc < NumChildren; child_sfc++)
    {
      summaries[child_sfc].m_subtreeFinestLevel = childFinestLevel[child_sfc];
      summaries[child_sfc].m_subtreeNodeCount = childNodeCounts[child_sfc];

      if (childNodeCounts[child_sfc] > 0 && childNodeCounts[child_sfc] < npe)
        thereAreHangingNodes = true;
    }
    //TODO need to add to DefaultSummary, bool isBoundary

    std::array<TreeNode<unsigned int, dim>, NumChildren> childSubtreesSFC;
    for (ChildI child_sfc = 0; child_sfc < NumChildren; child_sfc++)
    {
      const ChildI child_m = rotations[this->getCurrentRotation() * 2*NumChildren + child_sfc];
      childSubtreesSFC[child_sfc] = parSubtree.getChildMorton(child_m);
    }

    //
    // Resize child input buffers in the parent frame.
    //
    for (ChildI child_sfc = 0; child_sfc < NumChildren; child_sfc++)
    {
      size_t allocNodes = std::max(childNodeCounts[child_sfc], npe);  //TODO better max
      parentFrame.template getChildInput<1>(child_sfc).resize(m_ndofs * allocNodes);
      if (childFinestLevel[child_sfc] > parSubtree.getLevel() + 1)
        parentFrame.template getChildInput<0>(child_sfc).resize(allocNodes);
      else
      {
        std::vector<TreeNode<unsigned int, dim>> &childNodeCoords =
            parentFrame.template getChildInput<0>(child_sfc);
        childNodeCoords.clear();
        Element<unsigned int, dim>(childSubtreesSFC[child_sfc]).template
            appendNodes<TreeNode<unsigned int, dim>>(m_eleOrder, childNodeCoords);
      }
    }

    //
    // Perform any needed interpolations.
    //
    if (thereAreHangingNodes)
    {
      // Populate parent lexicographic buffer.
      const NodeT zero = 0;
      std::fill(m_parentNodeVals.begin(), m_parentNodeVals.end(), zero);
      for (size_t nIdx = 0; nIdx < numInputNodes; nIdx++)
      {
        if (myNodes[nIdx].getLevel() == parSubtree.getLevel())
        {
          const unsigned int nodeRank =
              TNPoint<unsigned int, dim>::get_lexNodeRank( parSubtree,
                                                           myNodes[nIdx],
                                                           m_eleOrder );
          assert(nodeRank < npe);
          std::copy_n( &parentFrame.template getMyInputHandle<1>()[m_ndofs * nIdx],
                       m_ndofs,
                       &m_parentNodeVals[m_ndofs * nodeRank] );
        }
      }

      for (ChildI child_sfc = 0; child_sfc < NumChildren; child_sfc++)
      {
        const ChildI child_m = rotations[this->getCurrentRotation() * 2*NumChildren + child_sfc];
        if (childNodeCounts > 0 && childNodeCounts[child_sfc] < npe)
        {
          // Has hanging nodes. Interpolate.
          // Non-hanging node values will be overwritten later, not to worry.
          constexpr bool transposeFalse = false;
          m_interp_matrices.template IKD_ParentChildInterpolation<transposeFalse>(
              &(*m_parentNodeVals.begin()),
              &(*parentFrame.template getChildInput<1>(child_sfc).begin()),
              m_ndofs,
              child_m);
        }
      }
    }

    childNodeCounts.fill(0);
    // Note: Re-uses the memory from childNodeCounts for mutable offsets.

    //
    // Copy input data to child buffers in parent frame.
    //
    for (const auto &nodeInstance : IterateNodesToChildren<dim>( this->getCurrentSubtree(),
                                                                 &(*myNodes.begin()),
                                                                 numInputNodes,
                                                                 this->getCurrentRotation() ))
    {
      const ChildI child_sfc = nodeInstance.getChild_sfc();
      const size_t nIdx = nodeInstance.getPNodeIdx();
      const size_t childOffset = childNodeCounts[child_sfc];

      if (childFinestLevel[child_sfc] > parSubtree.getLevel() + 1) // Nonleaf
      {
        // Node coordinates.
        parentFrame.template getChildInput<0>(child_sfc)[childOffset] = myNodes[nIdx];

        // Nodal values.
        std::copy_n( &parentFrame.template getMyInputHandle<1>()[m_ndofs * nIdx],  m_ndofs,
                     &parentFrame.template getChildInput<1>(child_sfc)[m_ndofs * childOffset]);

        childNodeCounts[child_sfc]++;
      }
      else   // Leaf
      {
        const unsigned int nodeRank = TNPoint<unsigned int, dim>::get_lexNodeRank(
                childSubtreesSFC[child_sfc],
                myNodes[nIdx],
                m_eleOrder );

        // Node coordinates.
        assert(parentFrame.template getChildInput<0>(child_sfc)[nodeRank] == myNodes[nIdx]);

        // Nodal values.
        std::copy_n( &parentFrame.template getMyInputHandle<1>()[m_ndofs * nIdx],  m_ndofs,
                     &parentFrame.template getChildInput<1>(child_sfc)[m_ndofs * nodeRank]);
      }
    }

    // Note: We have already set extantChildren during initial pass.
  }


  template <unsigned int dim, typename NodeT>
  void MatvecBase<dim, NodeT>::bottomUpNodes(FrameT &parentFrame, ExtantCellFlagT extantChildren)
  {
    /**
     *  Copied from sfcTreeLoop.h:
     *
     *  bottomUpNodes()
     *  is responsible to
     *    1. Resize the parent output buffers (handles to buffers are given);
     *
     *    2. Merge results from incident child output buffers (SFC order) into
     *       the parent output buffers.
     *
     *  The previously indicated extantChildren bit array (Morton order) will be supplied.
     *
     *  Utilities are provided to identify and iterate over incident children.
     */

    // =========================
    // Bottom-up Outline:
    // =========================
    // - Read from summary (#nodes, finest node level) per child.
    //   - Note: A child is a leaf iff finest node level == subtree level.
    //   - Note: A child is a leaf with hanging nodes if #nodes < npe.
    //
    // - Allocate parent output nodes and initialize to 0.
    //
    // - Pass through parent nodes. Accumulate nonhanging values from child output.
    //   - If a child is a leaf and #nonhanging nodes <= npe, find in lex position.
    //   - Else, find in same order as they appear in parent.
    //   - After receiving value from child, overwrite the child value with 0.
    //
    // - For each child:
    //   - If child has hanging nodes, interpolate-transpose in place in child buffer.
    //   - Pass through parent nodes.
    //         Accumulate into parent level nodes from child buffer lex position.
    // ========================================================================

    const unsigned npe = intPow(m_eleOrder+1, dim);
    const TreeNode<unsigned int,dim> & parSubtree = this->getCurrentSubtree();
    const NodeT zero = 0;

    std::array<size_t, NumChildren> childNodeCounts;
    std::array<size_t, NumChildren> childNodeOffsets;
    std::array<LevI, NumChildren> childFinestLevel;
    childNodeOffsets.fill(0);

    //
    // Retrieve child summaries.
    //
    bool thereAreHangingNodes = false;
    DefaultSummary (&summaries)[NumChildren] = parentFrame.childSummaries;
    size_t nodeCountAccum = 0;
    for (ChildI child_sfc = 0; child_sfc < NumChildren; child_sfc++)
    {
      childFinestLevel[child_sfc] = summaries[child_sfc].m_subtreeFinestLevel;
      childNodeCounts[child_sfc] = summaries[child_sfc].m_subtreeNodeCount;
      childNodeOffsets[child_sfc] = nodeCountAccum;
      nodeCountAccum += childNodeCounts[child_sfc];

      if (childNodeCounts[child_sfc] > 0 && childNodeCounts[child_sfc] < npe)
        thereAreHangingNodes = true;
    }

    const std::vector<TreeNode<unsigned int, dim>> &myNodes = parentFrame.template getMyInputHandle<0>();
    const size_t numParentNodes = parentFrame.mySummaryHandle.m_subtreeNodeCount;

    std::vector<NodeT> &myOutNodeValues = parentFrame.template getMyOutputHandle<0>();
    myOutNodeValues.clear();
    myOutNodeValues.resize(m_ndofs * numParentNodes, zero);

    std::array<TreeNode<unsigned int, dim>, NumChildren> childSubtreesSFC;
    for (ChildI child_sfc = 0; child_sfc < NumChildren; child_sfc++)
    {
      const ChildI child_m = rotations[this->getCurrentRotation() * 2*NumChildren + child_sfc];
      childSubtreesSFC[child_sfc] = parSubtree.getChildMorton(child_m);
    }

    //
    // Accumulate non-hanging node values from child buffers into parent frame.
    //
    for (const auto &nodeInstance : IterateNodesToChildren<dim>( parSubtree,
                                                                 &(*myNodes.begin()),
                                                                 numParentNodes,
                                                                 this->getCurrentRotation() ))
    {
      const ChildI child_sfc = nodeInstance.getChild_sfc();
      const size_t nIdx = nodeInstance.getPNodeIdx();
      const size_t childOffset = childNodeOffsets[child_sfc];

      if (childFinestLevel[child_sfc] > parSubtree.getLevel() + 1) // Nonleaf
      {
        // Nodal values.
        std::copy_n( &parentFrame.template getChildOutput<0>(child_sfc)[m_ndofs * childOffset],
                     m_ndofs, &myOutNodeValues[m_ndofs * nIdx]);

        childNodeOffsets[child_sfc]++;
      }
      else   // Leaf
      {
        const unsigned int nodeRank = TNPoint<unsigned int, dim>::get_lexNodeRank(
                childSubtreesSFC[child_sfc],
                myNodes[nIdx],
                m_eleOrder );

        // Nodal values.
        std::copy_n( &parentFrame.template getChildOutput<0>(child_sfc)[m_ndofs * nodeRank],
                     m_ndofs,  &myOutNodeValues[m_ndofs * nIdx]);

        // Zero out the values after they are transferred.
        // This is necessary so that later linear transforms are not contaminated.
        std::fill_n( &parentFrame.template getChildOutput<0>(child_sfc)[m_ndofs * nodeRank],
                     m_ndofs, zero );
      }
    }

    //
    // Perform any needed transpose-interpolations.
    //
    if (thereAreHangingNodes)
    {
      // Initialize intermediate parent lexicographic buffer.
      std::fill(m_parentNodeVals.begin(), m_parentNodeVals.end(), zero);

      // Use transpose of interpolation operator on each hanging child.
      for (ChildI child_sfc = 0; child_sfc < NumChildren; child_sfc++)
      {
        const ChildI child_m = rotations[this->getCurrentRotation() * 2*NumChildren + child_sfc];
        if (childNodeCounts > 0 && childNodeCounts[child_sfc] < npe)
        {
          // Has hanging nodes. Interpolation-transpose.
          constexpr bool transposeTrue = true;
          m_interp_matrices.template IKD_ParentChildInterpolation<transposeTrue>(
              &(*parentFrame.template getChildOutput<0>(child_sfc).begin()),
              &(*parentFrame.template getChildOutput<0>(child_sfc).begin()),
              m_ndofs,
              child_m);

          for (int nIdxDof = 0; nIdxDof < m_ndofs * npe; nIdxDof++)
            m_parentNodeVals[nIdxDof] += parentFrame.template getChildOutput<0>(child_sfc)[nIdxDof];
        }
      }

      // Accumulate from intermediate parent lex buffer to parent output.
      for (size_t nIdx = 0; nIdx < numParentNodes; nIdx++)
      {
        if (myNodes[nIdx].getLevel() == parSubtree.getLevel())
        {
          const unsigned int nodeRank =
              TNPoint<unsigned int, dim>::get_lexNodeRank( parSubtree,
                                                           myNodes[nIdx],
                                                           m_eleOrder );
          assert(nodeRank < npe);
          std::copy_n( &m_parentNodeVals[m_ndofs * nodeRank],
                       m_ndofs,  &myOutNodeValues[m_ndofs * nIdx]);
        }
      }
    }

  }


  template <unsigned int dim, typename NodeT>
  void MatvecBase<dim, NodeT>::parent2Child(FrameT &parentFrame, FrameT &childFrame)
  {

  }


  template <unsigned int dim, typename NodeT>
  void MatvecBase<dim, NodeT>::child2Parent(FrameT &parentFrame, FrameT &childFrame)
  {

  }


}//namespace ot


#endif//DENDRO_KT_SFC_TREE_LOOP_MATVEC_H
