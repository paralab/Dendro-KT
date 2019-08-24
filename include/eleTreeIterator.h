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

#include "refel.h"
#include "tensor.h"

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
 *     // the element node values will be discarded, and treated as 0 for
 *     // purpose of summing the results.
 *     // The (++it;) in the for-loop is also important to ensure propagation.
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
 * The summation for a subtree is written to a buffer that is separate from
 * the input node values. The temporary duplicates of input data are not
 * overwritten by output data sums. Furthermore, nonzero items can only be
 * present in the outbound buffer if they resulted in some way from a call to
 * submitElement().
 *
 * After all elements have been visited, the top level outbound buffer
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

  bool operator==(const TreeAddr &other) const;
  void step(unsigned int l);
  void step() { step(m_lev); }
  void clearBelowLev();
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
    std::vector<NodeT> m_ip_1D[2];   // Parent to child interpolation.
    std::vector<NodeT> m_ipT_1D[2];  // Child to parent.

    // Climbing gear - Includes level and coordinates during traversal/climbing.
    TreeAddr<T,dim> m_oldTreeAddr;
    TreeAddr<T,dim> m_curTreeAddr;       // Level of climbing target is here.
    ot::TreeNode<T,dim> m_curSubtree;    // Level of climbing progress is here.

    // Climbing gear - Stacks for depth-first SFC traversal.
    std::vector<std::vector<ot::TreeNode<T,dim>>> m_siblingNodeCoords;
    std::vector<std::vector<NodeT>> m_siblingNodeValsIn;
    std::vector<std::vector<NodeT>> m_siblingNodeValsOut;
    std::vector<std::array<unsigned int, NumChildren+1>> m_childTable;  // sfc-order splitters.
    //
    std::vector<ot::RotI> m_rot;
    std::vector<bool> m_isLastChild;
    std::vector<bool> m_siblingsDirty;

    // Leaf buffers.
    std::vector<ot::TreeNode<T,dim>> m_leafNodeCoords;
    /// std::vector<ot::TreeNode<T,dim>> m_parentNodeCoords;
    std::vector<NodeT> m_leafNodeVals;
    std::vector<NodeT> m_parentNodeVals;
    std::vector<double> m_leafNodeCoordsFlat;

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
    ELIterator() = delete;
    ELIterator(TreeAddr<T,dim> pos, ElementLoop<T,dim,NodeT> &host_loop);

    /** @brief Compare iterator positions. */
    bool operator!=(const ELIterator &other) const;

    /** @brief Advance iterator position. Undefined if already at end(). */
    ELIterator & operator++();

    /** @brief Dereference iterator position, i.e. get current element values. */
    ElementNodeBuffer<T,dim,NodeT> operator*();

    ot::TreeNode<T,dim> getElemTreeNode();
};



template <typename T, unsigned int dim, typename NodeT>
class ElementNodeBuffer
{
  protected:
    // TODO multiple pointers, by using template '...'
    NodeT *nodeValPtr;
    double *nodeCoordsPtr;
    const unsigned int &eleOrder;
    const unsigned int &nodesPerElement;
    const ot::TreeNode<T,dim> &elementTreeNode;
    ElementLoop<T,dim,NodeT> &m_host_loop;

  public:
    ElementNodeBuffer() = delete;
    ElementNodeBuffer( NodeT *i_nodeValPtr,
                       double *i_nodeCoordsPtr,
                       const unsigned int &i_eleOrder,
                       const unsigned int &i_nodesPerElement,
                       const ot::TreeNode<T,dim> &i_elementTreeNode,
                       ElementLoop<T,dim,NodeT> &i_m_host_loop )
      :
        nodeValPtr(i_nodeValPtr),
        nodeCoordsPtr(i_nodeCoordsPtr),
        eleOrder(i_eleOrder),
        nodesPerElement(i_nodesPerElement),
        elementTreeNode(i_elementTreeNode),
        m_host_loop(i_m_host_loop)
    {}

    NodeT * getNodeBuffer() { return nodeValPtr; }
    double * getNodeCoords() { return nodeCoordsPtr; }
    const unsigned int getEleOrder() const { return eleOrder; }
    const unsigned int getNodesPerElement() const { return nodesPerElement; }
    const ot::TreeNode<T,dim> & getElementTreeNode() const { return elementTreeNode; }

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
    /// m_parentNodeCoords( intPow(eleOrder+1, dim) ),
    m_leafNodeVals(     intPow(eleOrder+1, dim) ),
    m_parentNodeVals(   intPow(eleOrder+1, dim) ),
    m_leafNodeCoordsFlat( dim * intPow(eleOrder+1, dim) ),
    m_curSubtree()
{
  if (numNodes == 0 || &lastElement + 1 == &firstElement)  // Null
  {
    m_beginTreeAddr = m_curTreeAddr = m_oldTreeAddr = m_endTreeAddr = TreeAddr<T,dim>();
    m_curSubtree = ot::TreeNode<T,dim>();
    m_L0 = 0;
    m_numNodes = 0;

    m_childTable.resize(1);
    m_siblingNodeCoords.resize(1);
    m_siblingNodeValsIn.resize(1);
    m_siblingNodeValsOut.resize(1);
    m_rot.resize(1);
    m_isLastChild.resize(1);
    m_siblingsDirty.resize(1, false);

    m_isLastChild[0] = true;
    m_rot[0] = 0;
    m_siblingsDirty[0] = false;
    std::fill_n(m_childTable[0].begin(), NumChildren, 0);

    return;
  }

  // Fill interpolation matrices.
  {
    const unsigned int ipMatSz = (eleOrder+1)*(eleOrder+1);
    RefElement tempRefEl(dim, eleOrder);
    m_ip_1D[0] = std::vector<NodeT>(tempRefEl.getIMChild0(), tempRefEl.getIMChild0() + ipMatSz);
    m_ip_1D[1] = std::vector<NodeT>(tempRefEl.getIMChild1(), tempRefEl.getIMChild1() + ipMatSz);
    m_ipT_1D[0].resize(ipMatSz);
    m_ipT_1D[1].resize(ipMatSz);
    for (int ii = 0; ii < eleOrder+1; ii++)     // Transpose
      for (int jj = 0; jj < eleOrder+1; jj++)
      {
        m_ipT_1D[0][ii * (eleOrder+1) + jj] = m_ip_1D[0][jj * (eleOrder+1) + ii];
        m_ipT_1D[1][ii * (eleOrder+1) + jj] = m_ip_1D[1][jj * (eleOrder+1) + ii];
      }
  }

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
  m_siblingNodeValsIn.resize(m_uiMaxDepth - L0 + 1);
  m_siblingNodeValsOut.resize(m_uiMaxDepth - L0 + 1);
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
  for (int d = 0; d < dim; d++)
    m_beginTreeAddr.m_coords[d] = 0;
  for (unsigned int l = 1; l <= L0; l++)
  {
    ancestorChildNum = rotations[ancestorRot * 2*NumChildren +
                                 1*NumChildren +
                                 firstElement.getMortonIndex(l)];
    for (int d = 0; d < dim; d++)
      m_beginTreeAddr.m_coords[d] |= ((bool) (ancestorChildNum  & (1u << d))) << (m_uiMaxDepth - l);

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
  // However, the input may include nodes that are not incident on the
  // common ancestor, so we must filter those ones out.

  unsigned long numIncidentNodes = 0;
  const ot::Element<T, dim> subdomain(m_curSubtree.getAncestor(m_L0));
  m_siblingNodeCoords[L0 - L0].clear();
  for (unsigned long nIdx = 0; nIdx < numNodes; nIdx++)
  {
    if (subdomain.isIncident(allNodeCoords[nIdx]))
    {
      m_siblingNodeCoords[L0 - L0].push_back(allNodeCoords[nIdx]);
      numIncidentNodes++;
    }
  }

  m_isLastChild[L0 - L0] = true;
  m_rot[L0 - L0] = ancestorRot;
  m_siblingsDirty[L0 - L0] = false;
  for (ot::ChildI c = 0; c <= ancestorChildNum; c++)   // 0 |||.......|||||| numIncidentNodes
    m_childTable[L0 - L0][c] = 0;
  for (ot::ChildI c = ancestorChildNum+1; c < NumChildren+1; c++)
    m_childTable[L0 - L0][c] = numIncidentNodes;

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

  unsigned long numIncidentNodes = 0;
  const ot::Element<T, dim> subdomain(m_curSubtree.getAncestor(m_L0));
  m_siblingNodeValsIn[m_L0 - m_L0].clear();
  for (unsigned long nIdx = 0; nIdx < m_numNodes; nIdx++)
  {
    if (subdomain.isIncident(m_allNodeCoords[nIdx]))
    {
      m_siblingNodeValsIn[m_L0 - m_L0].push_back(inputNodeVals[nIdx]);
      numIncidentNodes++;
    }
  }
  m_siblingNodeValsOut[m_L0 - m_L0] =
      std::vector<NodeT>(numIncidentNodes, 0.0);

  goToTreeAddr();
}


//
// finalize()
//
template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::finalize(NodeT * outputNodeVals)
{
  if (!(m_curTreeAddr == m_endTreeAddr))
    while (m_curSubtree.getLevel() > m_L0)
    {
      m_curSubtree = m_curSubtree.getParent();
      bottomUpNodes();
    }

  std::vector<NodeT> &topVals = m_siblingNodeValsOut[m_L0 - m_L0];

  // Only copy back the subdomain-incident nodes.
  unsigned long numIncidentNodes = 0;
  const ot::Element<T, dim> subdomain(m_curSubtree.getAncestor(m_L0));
  for (unsigned long nIdx = 0; nIdx < m_numNodes; nIdx++)
  {
    if (subdomain.isIncident(m_allNodeCoords[nIdx]))
    {
      outputNodeVals[nIdx] = topVals[numIncidentNodes];
      numIncidentNodes++;
    }
    else
      outputNodeVals[nIdx] = 0.0;
  }
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

  const ot::ChildI curChildNum = m_curTreeAddr.getIndex(curLev); //TODO this may need to use curSubtree and then rotate by sfc.
  const ot::RankI curBegin = m_childTable[curLev - m_L0][curChildNum];
  const ot::RankI curEnd = m_childTable[curLev - m_L0][curChildNum+1];

  const ot::TreeNode<T,dim> * sibNodeCoords = &(*m_siblingNodeCoords[curLev - m_L0].begin());
  const NodeT               * sibNodeValsIn = &(*m_siblingNodeValsIn[curLev - m_L0].begin());
        NodeT               * sibNodeValsOut = &(*m_siblingNodeValsOut[curLev - m_L0].begin());

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
  m_siblingNodeCoords[curLev+1 - m_L0].resize(accum);
  m_siblingNodeValsIn[curLev+1 - m_L0].resize(accum);
  m_siblingNodeValsOut[curLev+1 - m_L0].clear();
  m_siblingNodeValsOut[curLev+1 - m_L0].resize(accum, 0.0);
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
      m_siblingNodeValsIn[curLev+1 - m_L0][   nodeOffsets[incidentChild_sfc] ] = sibNodeValsIn[nIdx];

      nodeOffsets[incidentChild_sfc]++;
    }
  }

  // Reset current level node out-values to 0, to prepare for
  // hanging node accumulation from our children.
  for (ot::RankI nIdx = curBegin; nIdx < curEnd; nIdx++)
    sibNodeValsOut[nIdx] = 0.0;

  // Return 'was not already a leaf'.
  return false;
}


template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::bottomUpNodes()
{
  const unsigned int curLev = m_curSubtree.getLevel();

  // child_sfc = rot_inv[child_m]
  const ot::ChildI * const rot_inv = &rotations[m_rot[curLev - m_L0]*2*NumChildren + 1*NumChildren];
  const ot::ChildI * const prot_inv =
      (curLev > m_L0 ?
        &rotations[m_rot[curLev-1 - m_L0]*2*NumChildren + 1*NumChildren]
      : &rotations[0*2*NumChildren + 1*NumChildren]);

  const ot::ChildI curChildNum_m = m_curSubtree.getMortonIndex(curLev);
  const ot::ChildI curChildNum_sfc = prot_inv[curChildNum_m];
  const ot::RankI curBegin = m_childTable[curLev - m_L0][curChildNum_sfc];
  const ot::RankI curEnd = m_childTable[curLev - m_L0][curChildNum_sfc+1];

  const ot::TreeNode<T,dim> * sibNodeCoords = &(*m_siblingNodeCoords[curLev - m_L0].begin());
  const NodeT               * sibNodeValsIn = &(*m_siblingNodeValsIn[curLev - m_L0].begin());
  NodeT                     * sibNodeValsOut = &(*m_siblingNodeValsOut[curLev - m_L0].begin());

  // Current level hanging node contributions were zeroed at the end of topDownNodes,
  // so that the buffer was ready for any hanging node accumulations.
  // Hanging node accumulations may have happened since then. Retain those
  // and also add accumulations from non-hanging nodes on children.

  if (curLev < m_uiMaxDepth && m_siblingsDirty[curLev+1 - m_L0])
  {
    m_siblingsDirty[curLev - m_L0] = true;

    // Summation from nodes shared across multiple children.
    //

    const ot::Element<T,dim> curSubtree(m_curSubtree);

    using FType = typename ot::CellType<dim>::FlagType;
    FType firstIncidentChild_m, incidentSubspace, incidentSubspaceDim;

    // Set of mutable pointers to child node offsets.
    std::array<unsigned int, NumChildren+1> nodeOffsets = m_childTable[curLev+1 - m_L0];

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

        sibNodeValsOut[nIdx] += m_siblingNodeValsOut[curLev+1 - m_L0][ nodeOffsets[incidentChild_sfc] ];

        nodeOffsets[incidentChild_sfc]++;  // Advance child pointer.
      }
    }
  }
}


template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::goToTreeAddr()
{
  // On the way up, responsible to set
  // - m_siblingNodeValsOut[... l-1],
  //       m_siblingsDirty[... l-1]  by bottomUpNodes()

  // On the way down, responsible to set
  // - m_siblingNodeCoords[l+1 ...],
  //       m_siblingNodeValsIn[l+1 ...],
  //       m_childTable[l+1 ...]  by topDownNodes()
  //
  // - m_rot[l+1 ...] and m_isLastChild[l+1 ...] if descend.
  //
  // - m_curSubtree to a descendant of original m_curSubtree.
  //
  // - Leaf buffers, once reach a leaf.
  //   - m_leafNodeCoords
  /// //   - m_parentNodeCoords
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
  m_curTreeAddr.clearBelowLev();
}


//
// requestLeafBuffer()
//
template <typename T, unsigned int dim, typename NodeT>
ElementNodeBuffer<T,dim,NodeT> ElementLoop<T, dim, NodeT>::requestLeafBuffer()
{
  // Assume m_curSubtree and m_curTreeAddr are pointing to a leaf.
  // Copy the nodes in lexicographic order to a leaf buffer.
  // If needed, interpolate missing values from the parent nodes.

  const unsigned int curLev = m_curSubtree.getLevel();

  const ot::ChildI curChildNum = m_curTreeAddr.getIndex(curLev);
  const ot::RankI curBegin = m_childTable[curLev - m_L0][curChildNum];
  const ot::RankI curEnd = m_childTable[curLev - m_L0][curChildNum+1];

  const ot::TreeNode<T,dim> * sibNodeCoords = &(*m_siblingNodeCoords[curLev - m_L0].begin());
  const NodeT               * sibNodeValsIn = &(*m_siblingNodeValsIn[curLev - m_L0].begin());

  const unsigned int npe = intPow(m_eleOrder+1, dim);

  // Generate leaf node coordinates as TreeNodes.
  // Copy and scale the node coordinates to a buffer of doubles.
  m_leafNodeCoords.clear();

  const double domainScale = 1.0 / double(1u << m_uiMaxDepth);
  const double elemSz = double(1u << m_uiMaxDepth - curLev) / double(1u << m_uiMaxDepth);
  std::array<unsigned int, dim> nodeIndices;
  nodeIndices.fill(0);

  double translation[dim];
  for (int d = 0; d < dim; d++)
    translation[d] = domainScale * m_curSubtree.getX(d);

  for (unsigned int n = 0; n < npe; n++)
  {
    for (int d = 0; d < dim; d++)
      m_leafNodeCoordsFlat[ n*dim + d ] = translation[d] + elemSz * nodeIndices[d] / m_eleOrder;

    incrementBaseB<unsigned int, dim>(nodeIndices, m_eleOrder+1);
  }

  // Limited to rounding precision of m_uiMaxDepth.
  /// ot::Element<T, dim>(m_curSubtree).template appendNodes<ot::TreeNode<T,dim>>(m_eleOrder, m_leafNodeCoords);

  // Diagnostics to tell if the nodes for the element are all present.
  std::vector<bool> leafEleFill(npe, false);
  unsigned int fillCheck = 0;
  assert(curEnd - curBegin <= npe);

  // Copy leaf points in lexicographic order.
  NodeT zero;  zero = 0.0;
  std::fill(m_leafNodeVals.begin(), m_leafNodeVals.end(), zero);
  for (ot::RankI nIdx = curBegin; nIdx < curEnd; nIdx++)
  {
    const unsigned int nodeRank = ot::TNPoint<T, dim>::get_lexNodeRank( m_curSubtree,
                                                                        sibNodeCoords[nIdx],
                                                                        m_eleOrder);
    assert(nodeRank < npe);  // If fails, check m_uiMaxDepth is deep enough for eleOrder.
    leafEleFill[nodeRank] = true;
    fillCheck += (nodeRank + 1);
    m_leafNodeVals[nodeRank] = sibNodeValsIn[nIdx];
  }

  const bool leafHasAllNodes = (fillCheck == npe*(npe+1));

  // Interpolate missing nodes (hanging nodes) from parent.
  if (!leafHasAllNodes)
  {
    const ot::ChildI parChildNum = m_curTreeAddr.getIndex(curLev-1);
    const ot::RankI parBegin = m_childTable[curLev-1 - m_L0][parChildNum];
    const ot::RankI parEnd = m_childTable[curLev-1 - m_L0][parChildNum+1];
    const NodeT * parNodeValsIn = &(*m_siblingNodeValsIn[curLev-1 - m_L0].begin());
    const ot::TreeNode<T,dim> * parNodeCoords = &(*m_siblingNodeCoords[curLev-1 - m_L0].begin());
    const ot::TreeNode<T, dim> parSubtree = m_curSubtree.getParent();

    // Copy node values from parent.
    std::fill(m_parentNodeVals.begin(), m_parentNodeVals.end(), zero);
    for (ot::RankI nIdx = parBegin; nIdx < parEnd; nIdx++)
    {
      if (parNodeCoords[nIdx].getLevel() != curLev-1)  // Only select parent-level nodes.
        continue;

      const unsigned int nodeRank = ot::TNPoint<T, dim>::get_lexNodeRank( parSubtree,
                                                                          parNodeCoords[nIdx],
                                                                          m_eleOrder );
      assert(nodeRank < npe);
      m_parentNodeVals[nodeRank] = parNodeValsIn[nIdx];
    }

    // Prepare to interpolate.

    // Line up 1D operators for each axis, based on childNum.
    const NodeT *ipAxis[dim];
    const unsigned int childNum_m = m_curSubtree.getMortonIndex();
    for (int d = 0; d < dim; d++)
      ipAxis[d] = m_ip_1D[bool(childNum_m & (1u << d))].data();

    // Double buffering of parent node coordinates during interpolation.
    const NodeT * imFrom[dim];
    NodeT * imTo[dim];
    std::vector<NodeT> imBufs[2];
    imBufs[0].resize(npe);
    imBufs[1].resize(npe);
    for (int d = 0; d < dim; d++)
    {
      imTo[d] = &(*imBufs[d % 2].begin());
      imFrom[d] = &(*imBufs[!(d % 2)].begin());
    }
    imFrom[0] = &(*m_parentNodeVals.begin());   // Overwrite pointer to first source.

    // Interpolate all element nodes.
    // (The ones we actually use should have valid values.)
    KroneckerProduct<dim, NodeT, true>(m_eleOrder+1, ipAxis, imFrom, imTo);
    // The results of the interpolation are stored in imTo[dim-1].

    for (unsigned int n = 0; n < npe; n++)
      if (!leafEleFill[n])
        m_leafNodeVals[n] = imTo[dim-1][n];
  }

  const ElementLoop * const_this = const_cast<const ElementLoop*>(this);

  return
    ElementNodeBuffer<T,dim,NodeT>
    {
      &(*this->m_leafNodeVals.begin()),
      &(*this->m_leafNodeCoordsFlat.begin()),
      const_this->m_eleOrder,
      const_this->m_nodesPerElement,
      const_this->m_curSubtree,
      *this,
    };
}

template <typename T, unsigned int dim, typename NodeT>
void ElementLoop<T, dim, NodeT>::submitLeafBuffer()
{
  // Assume m_curSubtree and m_curTreeAddr are pointing to a leaf.
  //
  // If there are missing nodes, do the transpose of interpolation,
  //   writing to the parent buffer.
  //
  // Also copy the non-missing nodes in lexicographic order from leaf buffer.

  const unsigned int curLev = m_curSubtree.getLevel();

  const ot::ChildI curChildNum = m_curTreeAddr.getIndex(curLev);
  const ot::RankI curBegin = m_childTable[curLev - m_L0][curChildNum];
  const ot::RankI curEnd = m_childTable[curLev - m_L0][curChildNum+1];

  const ot::TreeNode<T,dim> * sibNodeCoords = &(*m_siblingNodeCoords[curLev - m_L0].begin());
  NodeT                     * sibNodeValsOut = &(*m_siblingNodeValsOut[curLev - m_L0].begin());

  const unsigned int npe = intPow(m_eleOrder+1, dim);

  // Diagnostics to tell if the nodes for the element are all present.
  std::vector<bool> leafEleFill(npe, false);
  unsigned int fillCheck = 0;
  assert(curEnd - curBegin <= npe);

  // Copy leaf points in lexicographic order.
  NodeT zero;  zero = 0.0;
  for (ot::RankI nIdx = curBegin; nIdx < curEnd; nIdx++)
  {
    const unsigned int nodeRank = ot::TNPoint<T, dim>::get_lexNodeRank( m_curSubtree,
                                                                        sibNodeCoords[nIdx],
                                                                        m_eleOrder);
    assert(nodeRank < npe);  // If fails, check m_uiMaxDepth is deep enough for eleOrder.
    leafEleFill[nodeRank] = true;
    fillCheck += (nodeRank + 1);
    sibNodeValsOut[nIdx] = m_leafNodeVals[nodeRank];  // Reverse of requestLeafBuffer.
  }

  const bool leafHasAllNodes = (fillCheck == npe*(npe+1));

  // Uninterpolate hanging nodes back to parent.
  if (!leafHasAllNodes)
  {
    const ot::ChildI parChildNum = m_curTreeAddr.getIndex(curLev-1);
    const ot::RankI parBegin = m_childTable[curLev-1 - m_L0][parChildNum];
    const ot::RankI parEnd = m_childTable[curLev-1 - m_L0][parChildNum+1];
    NodeT *         parNodeValsOut = &(*m_siblingNodeValsOut[curLev-1 - m_L0].begin());
    const ot::TreeNode<T,dim> * parNodeCoords = &(*m_siblingNodeCoords[curLev-1 - m_L0].begin());
    const ot::TreeNode<T, dim> parSubtree = m_curSubtree.getParent();

    // Prepare to un-interpolate.

    for (unsigned int n = 0; n < npe; n++)
      if (leafEleFill[n])
        m_leafNodeVals[n] = 0.0;  // Only the hanging node values remain.

    // Line up 1D operators for each axis, based on childNum.
    const NodeT *ipTAxis[dim];
    const unsigned int childNum_m = m_curSubtree.getMortonIndex();
    for (int d = 0; d < dim; d++)
      ipTAxis[d] = m_ipT_1D[bool(childNum_m & (1u << d))].data();

    // Double buffering of nodes during interpolation.
    const NodeT * imFrom[dim];
    NodeT * imTo[dim];
    std::vector<NodeT> imBufs[2];
    imBufs[0].resize(npe);
    imBufs[1].resize(npe);
    for (int d = 0; d < dim; d++)
    {
      imTo[d] = &(*imBufs[d % 2].begin());
      imFrom[d] = &(*imBufs[!(d % 2)].begin());
    }
    imFrom[0] = &(*m_leafNodeVals.begin());   // Overwrite pointer to first source.

    // Un-Interpolate all element nodes.
    // (The non-hanging nodes have been zeroed out.)
    KroneckerProduct<dim, NodeT, true>(m_eleOrder+1, ipTAxis, imFrom, imTo);
    // The results of the interpolation are stored in imTo[dim-1].

    // Add contributions from hanging nodes to parent buffer.
    for (ot::RankI nIdx = parBegin; nIdx < parEnd; nIdx++)
    {
      if (parNodeCoords[nIdx].getLevel() != curLev-1)  // Only select parent-level nodes.
        continue;

      const unsigned int nodeRank = ot::TNPoint<T, dim>::get_lexNodeRank( parSubtree,
                                                                          parNodeCoords[nIdx],
                                                                          m_eleOrder );
      assert(nodeRank < npe);
      parNodeValsOut[nIdx] += imTo[dim-1][nodeRank];
    }
  }

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

    // Propagate bottom-up updates.
    while (m_curSubtree.getLevel() > m_L0)
    {
      m_curSubtree = m_curSubtree.getParent();
      bottomUpNodes();
    }

    m_curTreeAddr = m_endTreeAddr;  // Failsafe.
  }
}


template <typename T, unsigned int dim, typename NodeT>
ELIterator<T,dim,NodeT> ElementLoop<T, dim, NodeT>::current()
{
  return {m_curTreeAddr, *this};
}

template <typename T, unsigned int dim, typename NodeT>
ELIterator<T,dim,NodeT> ElementLoop<T, dim, NodeT>::begin()
{
  return {m_beginTreeAddr, *this};
}

template <typename T, unsigned int dim, typename NodeT>
ELIterator<T,dim,NodeT> ElementLoop<T, dim, NodeT>::end()
{
  return {m_endTreeAddr, *this};
}


// -------------- ELIterator Definitions -----------------

template <typename T, unsigned int dim, typename NodeT>
ELIterator<T,dim,NodeT>::ELIterator(
    TreeAddr<T,dim> pos,
    ElementLoop<T,dim,NodeT> &host_loop)

  : m_pos(pos),
    m_host_loop(host_loop)
{
}

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

template <typename T, unsigned int dim, typename NodeT>
ot::TreeNode<T,dim> ELIterator<T,dim,NodeT>::getElemTreeNode()
{
  return m_host_loop.m_curSubtree;
}


// -------------- TreeAddr Definitions -----------------

template <typename T, unsigned int dim>
bool TreeAddr<T,dim>::operator==(const TreeAddr &other) const
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
void TreeAddr<T,dim>::clearBelowLev()
{
  const T mask = (1u << (m_uiMaxDepth + 1)) - (1u << (m_uiMaxDepth - m_lev));
  for (int d = 0; d < dim; d++)
    m_coords[d] &= mask;
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
