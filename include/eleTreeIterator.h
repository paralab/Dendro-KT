/**
 * @file:eleTreeIterator.h
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-07-13
 * @brief: Stateful const iterator over implicit mesh, giving access to element nodes.
 */


#include "nsort.h"
#include "tsort.h"
#include "treeNode.h"
#include "mathUtils.h"
#include "binUtils.h"

#ifndef DENDRO_KT_ELE_TREE_ITERATOR_H
#define DENDRO_KT_ELE_TREE_ITERATOR_H

template <typename NodeT>
class ElementNodes
{
  protected:
    // TODO multiple pointers, by using template '...'
    const NodeT *nodePtr;
    const unsigned int &eleOrder;
    const unsigned int &nodesPerElement;

  public:
    const NodeT * getNodePtr() const { return nodePtr; }
    const unsigned int getEleOrder() const { return eleOrder; }
    const unsigned int getNodesPerElement() const { return nodesPerElement; }
};

    // Note: Tree addresses refer to the SFC ordering of elements. An address
    // is composed of `dim' components, where the first `lev' bits are relevant.
    // TreeNode, which has the same structure, is not used, to avoid confusion.
template <typename T, unsigned int dim>
struct TreeAddr
{
  unsigned int m_lev;
  std::array<T,dim> m_coords;

  // Add 1 at level l, following Morton interleaving of the coordinate bits.
  void step(unsigned int l)
  {
    unsigned int mask = 1u << (m_uiMaxDepth - l);
    bool carry = 1u;
    // Visit all the bits above l in succession (with Morton interleaving).
    while (mask)
    {
      for (int d = 0; d < dim; d++)
      {
        carry &= (m_coords[d] & mask);
        m_coords[d] ^= mask;
        if (!carry)
          return;
      }
      mask >>= 1;
    }
  }

  void step() { step(m_lev); }

  unsigned int getIndex(unsigned int level) const
  {
    // Repeat of TreeNode::getMortonIndex().
    const unsigned int shift = (m_uiMaxDepth - level);
    unsigned int index = 0u;
    for (int d = 0; d < dim; d++)
      index += ((m_coords[d] >> shift) & 1u) << d;
    return index;
  }

  unsigned int getIndex() { return getIndex(m_lev); }

};


/**
 * @brief Stateful--and heavyweight--const iterator over implicit mesh.
 * @tparam T Precision of integer coordinates of TreeNodes (element coords).
 * @tparam dim Number of components of each element coordinate.
 * @note For a full traversal, each node is copied roughly `lev' times,
 *     where `lev' is the level of the node.
 */
template <typename T, unsigned int dim, typename NodeT>
class EleTreeIterator
{
  protected:
    static constexpr unsigned int NumChildren = 1u << dim;

    // Static state - does not change during traversal.
    unsigned long m_numNodes;
    const ot::TreeNode<T,dim> * m_allNodeCoords;
    const NodeT * m_allNodeVals;
    unsigned int m_eleOrder;
    unsigned int m_nodesPerElement;
    TreeAddr<T,dim> m_beginTreeAddr;
    TreeAddr<T,dim> m_endTreeAddr;
    unsigned int m_L0;

    // Climbing gear - Includes level and coordinates during traversal/climbing.
    TreeAddr<T,dim> m_curTreeAddr;       // Level of climbing target is here.
    ot::TreeNode<T,dim> m_curSubtree;    // Level of climbing progress is here.

    // Climbing gear - Stacks for depth-first SFC traversal.
    std::vector<std::vector<ot::TreeNode<T,dim>>> m_siblingNodeCoords;
    std::vector<std::vector<NodeT>> m_siblingNodeVals;
    std::vector<std::array<unsigned int, NumChildren+1>> m_childTable;  // sfc-order splitters.
    //
    std::vector<ot::RotI> m_rot;
    std::vector<bool> m_isLastChild;

    // Leaf buffers.
    std::vector<ot::TreeNode<T,dim>> m_leafNodeCoords;
    std::vector<ot::TreeNode<T,dim>> m_parentNodeCoords;
    std::vector<NodeT> m_leafNodeVals;
    std::vector<NodeT> m_parentNodeVals;

    // Helper functions.
    void descendToLeafAddress();  // Descend to leaf at address pointed to by m_curTreeAddr.
    bool topDownNodes();          // Partition nodes for children of current node but don't change level.

  public:
    EleTreeIterator() = delete;
    EleTreeIterator( unsigned long numNodes,
                     const ot::TreeNode<T,dim> * allNodeCoords,
                     const NodeT * allNodeVals,
                     unsigned int eleOrder,
                     const ot::TreeNode<T,dim> &firstElement,
                     const ot::TreeNode<T,dim> &lastElement )

      : m_numNodes(numNodes),
        m_allNodeCoords(allNodeCoords),
        m_allNodeVals(allNodeVals),
        m_eleOrder(eleOrder),
        m_nodesPerElement(  intPow(eleOrder+1, dim) ),
        m_leafNodeCoords(   intPow(eleOrder+1, dim) ),
        m_parentNodeCoords( intPow(eleOrder+1, dim) ),
        m_leafNodeVals(     intPow(eleOrder+1, dim) ),
        m_parentNodeVals(   intPow(eleOrder+1, dim) ),
        m_curSubtree()
    {
      // Find L0, the deepest level for which all l at/less than L0
      //   firstElement.getMortonIndex(l) == lastElement.getMortonIndex(l).
      unsigned int L0 = 0;
      while (L0 < firstElement.getLevel() &&
             L0 < lastElement.getLevel() &&
             firstElement.getMortonIndex(L0+1) == lastElement.getMortonIndex(L0+1))
        L0++;

      m_L0 = L0;

      m_curSubtree = firstElement.getAncestor(L0);

      // Size stacks.
      m_childTable.resize(m_uiMaxDepth - L0 + 1);
      m_siblingNodeCoords.resize(m_uiMaxDepth - L0 + 1);
      m_siblingNodeVals.resize(m_uiMaxDepth - L0 + 1);
      m_rot.resize(m_uiMaxDepth - L0 + 1);
      m_isLastChild.resize(m_uiMaxDepth - L0 + 1);

      // The stacks are now accessed as stack[l - L0].

      // Before we can satisfy the traversal invariants by climbing,
      // need first and last addr to begin climbing.
      //

      // Start to find addresses for beginTreeAddr and endTreeAddr.
      // Find orientation of common ancestor to first and last element.
      ot::RotI ancestorRot = 0;
      ot::ChildI ancestorChildNum = 0;
      m_beginTreeAddr.m_coords = {0};
      for (unsigned int l = 1; l <= L0; l++)
      {
        ancestorChildNum = rotations[ancestorRot * 2*NumChildren +
                                     1*NumChildren +
                                     firstElement.getMortonIndex(l)];
        for (int d = 0; d < dim; d++)
          m_beginTreeAddr.m_coords[d] |= (bool) (ancestorChildNum  & (1u << d));

        ancestorRot = HILBERT_TABLE[ancestorRot*NumChildren + firstElement.getMortonIndex(l)];
      }
      // Result: ancestorRot and ancestorChildNum are initialized.

      // Deeper than this, firstElement and lastElement follow different paths.
      m_endTreeAddr.m_coords = m_beginTreeAddr.m_coords;
      m_beginTreeAddr.m_lev = firstElement.getLevel();
      m_endTreeAddr.m_lev = firstElement.getLevel();

      /// // Stash ancestor address in curTreeAddr.
      /// m_curTreeAddr.m_coords = m_beginTreeAddr.m_coords;
      /// m_curTreeAddr.m_lev = L0;

        // m_addrBegin <-- first element
      ot::RotI rot = ancestorRot;
      for (unsigned int l = L0 + 1; l <= firstElement.getLevel(); l++)
      {
        ot::ChildI child_sfc = rotations[rot * 2*NumChildren +
                                     1*NumChildren +
                                     firstElement.getMortonIndex(l)];
        for (int d = 0; d < dim; d++)
          m_beginTreeAddr.m_coords[d] |= (bool) (child_sfc & (1u << d));
        rot = HILBERT_TABLE[rot*NumChildren + firstElement.getMortonIndex(l)];
      }

        // m_addrEnd <-- last element + 1
      rot = ancestorRot;
      for (unsigned int l = L0 + 1; l <= lastElement.getLevel(); l++)
      {
        ot::ChildI child_sfc = rotations[rot * 2*NumChildren +
                                     1*NumChildren +
                                     lastElement.getMortonIndex(l)];
        for (int d = 0; d < dim; d++)
          m_endTreeAddr.m_coords[d] |= (bool) (child_sfc & (1u << d));
        rot = HILBERT_TABLE[rot*NumChildren + lastElement.getMortonIndex(l)];
      }
      m_endTreeAddr.step();  // Just computed lastElement, need to go one more.


      // Now that the endpoints are well-defined, descend to the first leaf.
      // Note we have so far initialized curTreeAddr to the common ancestor.
      //

      // Prepare climbing gear before descending to the first leaf.
      // At the level of the common ancestor, all nodes are associated
      // to a single element (the common ancestor) and none of the siblings.
      m_siblingNodeCoords[L0 - L0] =
          std::vector<ot::TreeNode<T,dim>>(allNodeCoords, allNodeCoords + numNodes);
      m_siblingNodeVals[L0 - L0] =
          std::vector<NodeT>(allNodeVals, allNodeVals + numNodes);
      m_isLastChild[L0 - L0] = true;
      m_rot[L0 - L0] = ancestorRot;
      for (ot::ChildI c = 0; c <= ancestorChildNum; c++)   // 0 |||.......|||||| numNodes
        m_childTable[L0 - L0][c] = 0;
      for (ot::ChildI c = ancestorChildNum+1; c < NumChildren+1; c++)
        m_childTable[L0 - L0][c] = numNodes;

      // Set m_curTreeAddr coordinates to the address of the first leaf, then descend to it.
      m_curTreeAddr = m_beginTreeAddr;
      descendToLeafAddress();
    }

    /** @brief Compare iterator positions. */
    bool operator==(const EleTreeIterator &other) const
    {
      //TODO
    }

    /** @brief Advance iterator position. */
    bool operator++()
    {
      //TODO
    }

    /** @brief Dereference iterator position, i.e. get current element values. */
    ElementNodes<NodeT> operator*()
    {
      //TODO
    }
};

template <typename T, unsigned int dim, typename NodeT>
bool EleTreeIterator<T, dim, NodeT>::topDownNodes()
{
  // The stacks only go as deep as m_uiMaxDepth, can't topDownNodes() further.
  if (m_curTreeAddr.m_lev >= m_uiMaxDepth)
    return true;

  unsigned int &curLev = m_curTreeAddr.m_lev;
  const ot::ChildI curChildNum = m_curTreeAddr.getIndex();
  const ot::RankI curBegin = m_childTable[curLev - m_L0][curChildNum];
  const ot::RankI curEnd = m_childTable[curLev - m_L0][curChildNum+1];

  const ot::TreeNode<T,dim> * sibNodeCoords = &(*m_siblingNodeCoords[curLev].begin());
  const NodeT               * sibNodeVals =   &(*m_siblingNodeVals[curLev].begin());

  // Check if this is a leaf element. If so, return true immediately.
  bool isLeaf = true;
  ot::RankI interrupt;
  for (interrupt = curBegin; interrupt < curEnd; interrupt++)
    if (sibNodeCoords[interrupt].getLevel() > curLev)
      break;
  isLeaf = (interrupt < curEnd);
  if (isLeaf)
    return true;

  // child_sfc = rot_inv[child_m]
  const ot::ChildI * const rot_inv = &rotations[m_rot[curLev - m_L0]*2*NumChildren + 1*NumChildren];

  const ot::Element<T,dim> curSubtree(m_curSubtree);

  using FType = typename ot::CellType<dim>::FlagType;
  FType firstIncidentChild_m, incidentSubspace, incidentSubspaceDim;

  std::array<unsigned int, NumChildren+1> nodeCounts;
  nodeCounts.fill(0);

  // Count the number of nodes contained by or incident on each child.
  for (ot::RankI nIdx = curBegin; nIdx < curEnd; nIdx++)
  {
    curSubtree.incidentChildren( sibNodeCoords[nIdx],
                                 firstIncidentChild_m,
                                 incidentSubspace,
                                 incidentSubspaceDim);

    binOp::TallBitMatrix<dim, FType> bitExpander =
        binOp::TallBitMatrix<dim, FType>::generateColumns(incidentSubspace);

    const ot::ChildI numIncidentChildren = 1u << incidentSubspaceDim;
    for (ot::ChildI c = 0; c < numIncidentChildren; c++)
    {
      ot::ChildI incidentChild_m = firstIncidentChild_m + bitExpander.expandBitstring(c);
      ot::ChildI incidentChild_sfc = rot_inv[incidentChild_m];
      nodeCounts[incidentChild_sfc]++;
    }
  }

  // Exclusive prefix sum to get child node splitters.
  ot::RankI accum = 0;
  for (unsigned int c = 0; c < NumChildren; c++)
  {
    accum += nodeCounts[c];
    nodeCounts[c] = accum - nodeCounts[c];
  }
  nodeCounts[NumChildren] = accum;
  std::array<unsigned int, NumChildren+1> &nodeOffsets = nodeCounts;
  // Result: nodeCounts now holds node offsets.

  // Push the offsets array onto the m_childTable stack.
  m_childTable[curLev+1 - m_L0] = nodeOffsets;

  // Iterate through the nodes again, but instead of counting, copy the nodes.
  m_siblingNodeCoords[curLev+1 - m_L0].resize(accum);
  m_siblingNodeVals[curLev+1 - m_L0].resize(accum);
  for (ot::RankI nIdx = curBegin; nIdx < curEnd; nIdx++)
  {
    curSubtree.incidentChildren( sibNodeCoords[nIdx],
                                 firstIncidentChild_m,
                                 incidentSubspace,
                                 incidentSubspaceDim);

    binOp::TallBitMatrix<dim, FType> bitExpander =
        binOp::TallBitMatrix<dim, FType>::generateColumns(incidentSubspace);

    const ot::ChildI numIncidentChildren = 1u << incidentSubspaceDim;
    for (ot::ChildI c = 0; c < numIncidentChildren; c++)
    {
      ot::ChildI incidentChild_m = firstIncidentChild_m + bitExpander.expandBitstring(c);
      ot::ChildI incidentChild_sfc = rot_inv[incidentChild_m];

      m_siblingNodeCoords[curLev+1 - m_L0][ nodeOffsets[incidentChild_sfc] ] = sibNodeCoords[nIdx];
      m_siblingNodeVals[curLev+1 - m_L0][   nodeOffsets[incidentChild_sfc] ] = sibNodeVals[nIdx];

      nodeOffsets[incidentChild_sfc]++;
    }
  }

  // Return 'was not already a leaf'.
  return false;
}


template <typename T, unsigned int dim, typename NodeT>
void EleTreeIterator<T, dim, NodeT>::descendToLeafAddress()
{
  // Responsible to set
  // - m_siblingNodeCoords[l+1 ...],
  //       m_siblingNodeVals[l+1 ...],
  //       m_childTable[l+1 ...]  by topDownNodes()
  //
  // - m_rot[l+1 ...] and m_isLastChild[l+1 ...] if descend.
  //
  // - m_curSubtree to a descendant of original m_curSubtree.
  //
  // - Leaf buffers, once reach a leaf.
  //   - m_leafNodeCoords
  //   - m_parentNodeCoords
  //   - m_leafNodeVals
  //   - m_parentNodeVals

  // - m_curTreeAddr is READONLY in this method.

  // Try to prepare child nodes for next level,
  // until there is no child because we reached leaf.
  while (!topDownNodes())
  {
    // Just had success initializing m_siblingNode{Coords,Vals}, m_childTable.

    const ot::LevI pLev = m_curSubtree.getLevel();
    const ot::RotI pRot = m_rot[pLev - m_L0];

    // child_m = rot_perm[child_sfc]
    const ot::ChildI * const rot_perm = &rotations[pRot*2*NumChildren + 0*NumChildren];
    const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*NumChildren];
    const ot::ChildI child_sfc = m_curTreeAddr.getIndex(pLev);
    const ot::ChildI child_m = rot_perm[child_sfc];

    m_rot[pLev+1 - m_L0] = HILBERT_TABLE[pRot*NumChildren + child_m];
    m_isLastChild[pLev+1 - m_L0] =
        m_isLastChild[pLev - m_L0] &&
        (m_curTreeAddr.getIndex(pLev+1) == m_endTreeAddr.getIndex(pLev+1) - 1);

    m_curSubtree = m_curSubtree.getChildMorton(child_m);
  }

  // topDownNodes() finally failed, which means the current level is a leaf.
  // (Or we hit m_uiMaxDepth).
  //TODO copy nodes in lexicographic order to leaf buffer,
  // optionally copy parent nodes, and interpolate if there are hanging nodes.
}

#endif//DENDRO_KT_ELE_TREE_ITERATOR_H
