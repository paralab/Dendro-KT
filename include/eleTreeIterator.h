/**
 * @file:eleTreeIterator.h
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-07-13
 * @brief: Stateful const iterator over implicit mesh, giving access to element nodes.
 */

#ifndef DENDRO_KT_ELE_TREE_ITERATOR_H
#define DENDRO_KT_ELE_TREE_ITERATOR_H

#include "nsort.h"
#include "tsort.h"
#include "treeNode.h"
#include "mathUtils.h"
#include "binUtils.h"

template <typename T, unsigned int dim, typename NodeT> class ElementLoop;
template <typename T, unsigned int dim, typename NodeT> class ELIterator;
template <typename T, unsigned int dim, typename NodeT> class ElementNodeBuffer;

template <typename T, unsigned int dim> struct TreeAddr;

/**
 * Mesh free element traversal to read or compute on the nodes of each element.
 *
 * Usage:
 * {
 *   ElementLoop elementLoop(numNodes, nodeCoords, order, firstElem, lastElem);
 *   elementLoop.initialize(inputNodeValues);
 *
 *   for (ELIterator it = elementLoop.begin(); it != elementLoop.end(); ++it)
 *   {
 *     ElementNodeBuffer & eleNodeBuf = *it;
 *
 *     // Read the values and coordinates of eleNodeBuf
 *     // using eleNodeBuf.getNodeCoords() and eleNodeBuf.getNodeValues();
 *
 *     // Optional:
 *     // Also write to node values through eleNodeBuf.getNodeValues(); then
 *     eleNodeBuf.submitElement();
 *
 *     // If you do not call eleNodeBuf.submitElement(), any changes made to
 *     // the element node values will be discarded. Also don't forget ++it;
 *   }
 *   // Note: ++it;   <--->  elementLoop.next();
 *   //       *it;    <--->  elementLoop.requestLeafBuffer();
 *
 *   // Optional:
 *   // If you wrote to the element nodes, get the results using
 *   elementLoop.finalize(outputNodeValues);
 * }
 *
 *
 * ElementLoop maintains its own copy of the node values. This internal
 * copy store is initialized through elementLoop.initialize();
 *
 * Element node values are delivered through a depth-first traversal,
 * with localized top-down duplication of nodes shared by neighboring elements.
 *
 * When changes to an element are submitted by the user and the iterator is advanced,
 * the new node values are propagated with partial bottom-up summation.
 * The summation for a subtree is not written until all elements in the subtree
 * have been visited. Therefore the temporary duplicates of a node
 * on a neighboring element are not affected from modifying the current element.
 *
 * After all elements have been visited, the top level internal copy
 * stores the summed results of all element computations. The results
 * can be transferred back to the user through elementLoop.finalize();
 *
 * ElementLoop represents a single in-progress traversal between initialize()
 * and finalize(). It is not recommended to store multiple ELIterator
 * objects which point to the same ElementLoop, because they will not be independent.
 */


/**
 * @class TreeAddr
 */
    // Note: Tree addresses refer to the SFC ordering of elements. An address
    // is composed of `dim' components, where the first `lev' bits are relevant.
    // TreeNode, which has the same structure, is not used, to avoid confusion.
template <typename T, unsigned int dim> struct TreeAddr;

template <typename T, unsigned int dim>
struct TreeAddr
{
  unsigned int m_lev;
  std::array<T,dim> m_coords;

  bool operator==(const TreeAddr &other);
  void step(unsigned int l);
  void step() { step(m_lev); }
  unsigned int getIndex(unsigned int level) const;
  unsigned int getIndex() { return getIndex(m_lev); }

  unsigned int static commonAncestorLevel(const TreeAddr &ta1, 
                                          const TreeAddr &ta2,
                                          bool &differentDomains);

  unsigned int static commonAncestorLevel(const TreeAddr &ta1, const TreeAddr &ta2);
};

template<typename T, unsigned int dim>
std::ostream & operator<<(std::ostream &out, const TreeAddr<T,dim> &self);


/**
 * @brief Stateful--and heavyweight--const iterator over implicit mesh.
 * @tparam T Precision of integer coordinates of TreeNodes (element coords).
 * @tparam dim Number of components of each element coordinate.
 * @note For a full traversal, each node is copied roughly `lev' times,
 *     where `lev' is the level of the node.
 */
template <typename T, unsigned int dim, typename NodeT>
class ElementLoop
{
  friend class ELIterator<T,dim,NodeT>;

  public:
    ElementLoop() = delete;
    ElementLoop( unsigned long numNodes,
                     const ot::TreeNode<T,dim> * allNodeCoords,
                     unsigned int eleOrder,
                     const ot::TreeNode<T,dim> &firstElement,
                     const ot::TreeNode<T,dim> &lastElement );

    void initialize(const NodeT *inputNodeVals);
    void finalize(NodeT * outputNodeVals);

    ElementNodeBuffer<T,dim,NodeT> requestLeafBuffer();
    void submitLeafBuffer();

    // Iteration using just ElementLoop: while(!isExhausted()) next();
    void next();
    bool isExhausted() { return m_curTreeAddr == m_endTreeAddr; }

    // Iteration using iterator pattern, range-based for loops, etc.
    ELIterator<T,dim,NodeT> current();
    ELIterator<T,dim,NodeT> begin();
    ELIterator<T,dim,NodeT> end();

    // Utilities to try to improve spatial locality.
    //TODO
    void dryRun();
    void reallocateStacks();

  /// protected:
  public:
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
    TreeAddr<T,dim> m_oldTreeAddr;
    TreeAddr<T,dim> m_curTreeAddr;       // Level of climbing target is here.
    ot::TreeNode<T,dim> m_curSubtree;    // Level of climbing progress is here.

    // Climbing gear - Stacks for depth-first SFC traversal.
    std::vector<std::vector<ot::TreeNode<T,dim>>> m_siblingNodeCoords;
    std::vector<std::vector<NodeT>> m_siblingNodeVals;
    std::vector<std::array<unsigned int, NumChildren+1>> m_childTable;  // sfc-order splitters.
    //
    std::vector<ot::RotI> m_rot;
    std::vector<bool> m_isLastChild;
    std::vector<bool> m_siblingsDirty;

    // Leaf buffers.
    std::vector<ot::TreeNode<T,dim>> m_leafNodeCoords;
    std::vector<ot::TreeNode<T,dim>> m_parentNodeCoords;
    std::vector<NodeT> m_leafNodeVals;
    std::vector<NodeT> m_parentNodeVals;

    // Helper functions.
    bool topDownNodes();    // Partition nodes for children of current node but don't change level.
    void bottomUpNodes();   // Sum up children nodes and store sum, but don't change level.
    void goToTreeAddr();    // Make curSubtree match curTreeAddr, using bottom up and top down.
};


/**
 * @class ELIterator
 * @brief Wrapper around ElementLoop::next() to make for-loops more familiar.
 */
template <typename T, unsigned int dim, typename NodeT>
class ELIterator
{
  friend class ElementLoop<T,dim,NodeT>;

  protected:
    TreeAddr<T,dim> m_pos;
    ElementLoop<T,dim,NodeT> &m_host_loop;

  public:

    /** @brief Compare iterator positions. */
    bool operator!=(const ELIterator &other) const;

    /** @brief Advance iterator position. Undefined if already at end(). */
    ELIterator & operator++();

    /** @brief Dereference iterator position, i.e. get current element values. */
    ElementNodeBuffer<T,dim,NodeT> operator*();
};



template <typename T, unsigned int dim, typename NodeT>
class ElementNodeBuffer
{
  protected:
    // TODO multiple pointers, by using template '...'
    NodeT *nodeValPtr;
    const ot::TreeNode<T,dim> *nodeCoordsPtr;
    const unsigned int &eleOrder;
    const unsigned int &nodesPerElement;
    ElementLoop<T,dim,NodeT> &m_host_loop;

  public:
    NodeT * getNodeBuffer() { return nodeValPtr; }
    const ot::TreeNode<T,dim> * getNodeCoords() const { return nodeCoordsPtr; }
    const unsigned int getEleOrder() const { return eleOrder; }
    const unsigned int getNodesPerElement() const { return nodesPerElement; }

    void submitElement() { m_host_loop.submitLeafBuffer(); }
};



// -------------- ElementLoop Definitions -----------------

//
// ElementLoop() (constructor)
//
template <typename T, unsigned int dim, typename NodeT>
ElementLoop<T,dim,NodeT>::ElementLoop( unsigned long numNodes,
                                       const ot::TreeNode<T,dim> * allNodeCoords,
                                       unsigned int eleOrder,
                                       const ot::TreeNode<T,dim> &firstElement,
                                       const ot::TreeNode<T,dim> &lastElement )

  : m_numNodes(numNodes),
    m_allNodeCoords(allNodeCoords),
    m_allNodeVals(nullptr),
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
  m_siblingsDirty.resize(m_uiMaxDepth - L0 + 1, false);

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
  m_endTreeAddr.m_lev = lastElement.getLevel();

  // Stash ancestor address in oldTreeAddr.
  m_oldTreeAddr.m_coords = m_beginTreeAddr.m_coords;
  m_oldTreeAddr.m_lev = L0;

    // m_addrBegin <-- first element
  ot::RotI rot = ancestorRot;
  T levBit = 1u << (m_uiMaxDepth - L0);
  for (unsigned int l = L0 + 1; l <= firstElement.getLevel(); l++)
  {
    ot::ChildI child_sfc = rotations[rot * 2*NumChildren +
                                 1*NumChildren +
                                 firstElement.getMortonIndex(l)];
    levBit >>= 1;
    for (int d = 0; d < dim; d++)
      m_beginTreeAddr.m_coords[d] |= levBit * (bool) (child_sfc & (1u << d));
    rot = HILBERT_TABLE[rot*NumChildren + firstElement.getMortonIndex(l)];
  }

    // m_addrEnd <-- last element + 1
  rot = ancestorRot;
  levBit = 1u << (m_uiMaxDepth - L0);
  for (unsigned int l = L0 + 1; l <= lastElement.getLevel(); l++)
  {
    ot::ChildI child_sfc = rotations[rot * 2*NumChildren +
                                 1*NumChildren +
                                 lastElement.getMortonIndex(l)];
    levBit >>= 1;
    for (int d = 0; d < dim; d++)
      m_endTreeAddr.m_coords[d] |= levBit * (bool) (child_sfc & (1u << d));
    rot = HILBERT_TABLE[rot*NumChildren + lastElement.getMortonIndex(l)];
  }
  m_endTreeAddr.step();  // Just computed lastElement, need to go one more.


  // Prepare climbing gear before descending to the first leaf.
  // At the level of the common ancestor, all nodes are associated
  // to a single element (the common ancestor) and none of the siblings.
  m_siblingNodeCoords[L0 - L0] =
      std::vector<ot::TreeNode<T,dim>>(allNodeCoords, allNodeCoords + numNodes);
  m_isLastChild[L0 - L0] = true;
  m_rot[L0 - L0] = ancestorRot;
  m_siblingsDirty[L0 - L0] = false;
  for (ot::ChildI c = 0; c <= ancestorChildNum; c++)   // 0 |||.......|||||| numNodes
    m_childTable[L0 - L0][c] = 0;
  for (ot::ChildI c = ancestorChildNum+1; c < NumChildren+1; c++)
    m_childTable[L0 - L0][c] = numNodes;

  // Set m_curTreeAddr (next target) coordinates to the address of the first leaf.
  m_curTreeAddr = m_beginTreeAddr;

  // Can't descend deeper than L0 yet because don't have the node values.
  /// goToTreeAddr();
}


//
// initialize()
//
template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::initialize(const NodeT *inputNodeVals)
{
  m_curTreeAddr = m_beginTreeAddr;
  m_oldTreeAddr.m_coords = m_beginTreeAddr.m_coords;
  m_oldTreeAddr.m_lev = m_L0;
  m_curSubtree = m_curSubtree.getAncestor(m_L0);

  m_siblingNodeVals[m_L0 - m_L0] =
      std::vector<NodeT>(inputNodeVals, inputNodeVals + m_numNodes);

  goToTreeAddr();
}


//
// finalize()
//
template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::finalize(NodeT * outputNodeVals)
{
  while (m_curSubtree.getLevel() > m_L0)
  {
    m_curSubtree = m_curSubtree.getParent();
    bottomUpNodes();
  }

  std::vector<NodeT> &topVals = m_siblingNodeVals[m_L0 - m_L0];
  std::copy_n(topVals.begin(), m_numNodes, outputNodeVals);
}


//
// topDownNodes()
//
template <typename T, unsigned int dim, typename NodeT>
bool ElementLoop<T, dim, NodeT>::topDownNodes()
{
  // The stacks only go as deep as m_uiMaxDepth, can't topDownNodes() further.
  const unsigned int curLev = m_curSubtree.getLevel();

  if (curLev >= m_uiMaxDepth)
    return true;

  const ot::ChildI curChildNum = m_curTreeAddr.getIndex(curLev);
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
  isLeaf = (interrupt == curEnd);
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

  // Declare that all the new children are clean.
  m_siblingsDirty[curLev+1 - m_L0] = false;

  // Iterate through the nodes again, but instead of counting, copy the nodes.
  if (m_siblingNodeCoords[curLev+1 - m_L0].size() < accum)
    m_siblingNodeCoords[curLev+1 - m_L0].resize(accum);
  if (m_siblingNodeVals[curLev+1 - m_L0].size() < accum)
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
void ElementLoop<T, dim, NodeT>::bottomUpNodes()
{
  //TODO
}


template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::goToTreeAddr()
{
  // On the way up, responsible to set
  // - m_siblingNodeVals[... l-1],
  //       m_siblingsDirty[... l-1]  by bottomUpNodes()

  // On the way down, responsible to set
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

  // - m_curTreeAddr coords are READONLY in this method, however need to find level of leaf.

  // Get m_curSubtree to be an ancestor of the target.
  unsigned int commonLev = TreeAddr<T,dim>::commonAncestorLevel(m_oldTreeAddr, m_curTreeAddr);
  if (m_curSubtree.getLevel() > commonLev)
  {
    // Ascend to siblingLevel.
    while (m_curSubtree.getLevel() > commonLev + 1)
    {
      m_curSubtree = m_curSubtree.getParent();
      bottomUpNodes();
    }

    // Because m_curSubtree originally had level >= commonLev+1,
    // we know that the stacks are initialized correctly down to commonLev,
    // and the sibling table itself is initialized correctly down to commonLev+1.
    // To make m_curSubtree an ancestor of the target, re-initialize everything
    // at level commonLev+1, except for the sibling table.
    const ot::RotI pRot = m_rot[commonLev - m_L0];

    // child_m = rot_perm[child_sfc]
    const ot::ChildI * const rot_perm = &rotations[pRot*2*NumChildren + 0*NumChildren];
    /// const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*NumChildren];
    const ot::ChildI child_sfc = m_curTreeAddr.getIndex(commonLev+1);
    const ot::ChildI child_m = rot_perm[child_sfc];
  
    m_rot[commonLev+1 - m_L0] = HILBERT_TABLE[pRot*NumChildren + child_m];
    m_isLastChild[commonLev+1 - m_L0] =
        m_isLastChild[commonLev - m_L0] &&
        (m_curTreeAddr.getIndex(commonLev+1) == m_endTreeAddr.getIndex(commonLev+1) - 1);
  
    m_curSubtree = m_curSubtree.getParent().getChildMorton(child_m);
  }
  // Otherwise, m_curSubtree is already an ancestor of the target.

  // Descend: Try to prepare child nodes for next level,
  // until there is no child because we reach a leaf.
  while (!topDownNodes())
  {
    // Just had success initializing m_siblingNode{Coords,Vals}, m_childTable.

    const ot::LevI pLev = m_curSubtree.getLevel();
    const ot::RotI pRot = m_rot[pLev - m_L0];

    // child_m = rot_perm[child_sfc]
    const ot::ChildI * const rot_perm = &rotations[pRot*2*NumChildren + 0*NumChildren];
    /// const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*NumChildren];
    const ot::ChildI child_sfc = m_curTreeAddr.getIndex(pLev+1);
    const ot::ChildI child_m = rot_perm[child_sfc];

    m_rot[pLev+1 - m_L0] = HILBERT_TABLE[pRot*NumChildren + child_m];
    m_isLastChild[pLev+1 - m_L0] =
        m_isLastChild[pLev - m_L0] &&
        (m_curTreeAddr.getIndex(pLev+1) == m_endTreeAddr.getIndex(pLev+1) - 1);

    m_curSubtree = m_curSubtree.getChildMorton(child_m);
  }

  // The stacks have reached the leaf. We now know the correct level of target.
  m_curTreeAddr.m_lev = m_curSubtree.getLevel();
}


//
// requestLeafBuffer()
//
template <typename T, unsigned int dim, typename NodeT>
ElementNodeBuffer<T,dim,NodeT> ElementLoop<T, dim, NodeT>::requestLeafBuffer()
{

  // topDownNodes() finally returned false, which means the current level is a leaf.
  // (Or we hit m_uiMaxDepth).
  //TODO copy nodes in lexicographic order to leaf buffer,
  // optionally copy parent nodes, and interpolate if there are hanging nodes.
}

template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::submitLeafBuffer()
{
  unsigned int curLev = m_curSubtree.getLevel();

  // optionally interpolate
  // TODO copy the nodes in lexicographic order from leaf buffer.

  m_siblingsDirty[curLev - m_L0] = true;
}


//
// next()
//
template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::next()
{
  if (m_curTreeAddr == m_endTreeAddr)
    return;  // Not what user expected, but better than infinite loop.

  m_oldTreeAddr = m_curTreeAddr;
  m_curTreeAddr.step();

  if (!(m_curTreeAddr == m_endTreeAddr))
  {
    goToTreeAddr();
  }
  else
  {
    m_curTreeAddr = m_oldTreeAddr;
    m_curTreeAddr.m_lev = m_L0;
    goToTreeAddr();  // Propagate bottom-up updates.

    m_curTreeAddr = m_endTreeAddr;  // Failsafe.
  }
}



// -------------- ELIterator Definitions -----------------

template <typename T, unsigned int dim, typename NodeT>
bool ELIterator<T,dim,NodeT>::operator!=(const ELIterator &other) const
{
  return !(m_pos == other.m_pos);
}

template <typename T, unsigned int dim, typename NodeT>
ELIterator<T,dim,NodeT> & ELIterator<T,dim,NodeT>::operator++()
{
  m_host_loop.next();
  m_pos = m_host_loop.m_curTreeAddr;
  return *this;
}

template <typename T, unsigned int dim, typename NodeT>
ElementNodeBuffer<T,dim,NodeT> ELIterator<T,dim,NodeT>::operator*()
{
  return m_host_loop.requestLeafBuffer();
}



// -------------- TreeAddr Definitions -----------------

template <typename T, unsigned int dim>
bool TreeAddr<T,dim>::operator==(const TreeAddr &other)
{
  return (m_lev == other.m_lev) && (m_coords == other.m_coords);
}

template <typename T, unsigned int dim>
std::ostream & operator<<(std::ostream &out, const TreeAddr<T,dim> &self)
{
  out << "{(" << self.m_lev << ")";
  for (int d = 0; d < dim; d++)
    out << ", " << self.m_coords[d];
  out << "}";
}

// Add 1 at level l, following Morton interleaving of the coordinate bits.
template <typename T, unsigned int dim>
void TreeAddr<T,dim>::step(unsigned int l)
{
  unsigned int mask = 1u << (m_uiMaxDepth - l);
  bool carry = 1u;
  // Visit all the bits above l in succession (with Morton interleaving).
  while (mask)
  {
    for (int d = 0; d < dim; d++)
    {
      carry &= bool(m_coords[d] & mask);
      m_coords[d] ^= mask;
      if (!carry)
        return;
    }
    mask <<= 1;
  }
}

template <typename T, unsigned int dim>
unsigned int TreeAddr<T,dim>::getIndex(unsigned int level) const
{
  // Repeat of TreeNode::getMortonIndex().
  const unsigned int shift = (m_uiMaxDepth - level);
  unsigned int index = 0u;
  for (int d = 0; d < dim; d++)
    index += ((m_coords[d] >> shift) & 1u) << d;
  return index;
}

/** brief Get commonAncestorLevel, detecting if ta1 and ta2 are not in same domain. */
template <typename T, unsigned int dim>
unsigned int TreeAddr<T,dim>::commonAncestorLevel(const TreeAddr &ta1, 
                                        const TreeAddr &ta2,
                                        bool &differentDomains)
{
  T coordsEq = 0u - 1u;
  for (int d = 0; d < dim; d++)
    coordsEq &= ~(ta1.m_coords[d] ^ ta2.m_coords[d]);

  differentDomains = !(coordsEq & (1u << m_uiMaxDepth));

  unsigned int lineageMaxDepth = ta1.m_lev;
  if (ta2.m_lev < lineageMaxDepth)
    lineageMaxDepth = ta2.m_lev;

  // Assuming both addresses are in the same domain,
  // coordsEq has '1' for each equal level, '0' for each unequal level.
  // Want to return level of deepest consecutive '1', starting from root.
  unsigned int lev = 0;
  while (lev < lineageMaxDepth && (coordsEq & (1u << (m_uiMaxDepth-lev-1))))
    lev++;
  return lev;
}

/** brief Get commonAncestorLevel assuming ta1 and ta2 are in same domain. */
template <typename T, unsigned int dim>
unsigned int TreeAddr<T,dim>::commonAncestorLevel(const TreeAddr &ta1,
                                        const TreeAddr &ta2)
{
  bool unusedFlag;
  return commonAncestorLevel(ta1, ta2, unusedFlag);
}




#endif//DENDRO_KT_ELE_TREE_ITERATOR_H
