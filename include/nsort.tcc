/**
 * @file:nsort.tcc
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-02-20
 */

#include <bitset>

namespace ot {

  // ==================== Begin: CellType ==================== //
 
  //
  // getExteriorOrientHigh2Low()
  //
  template <unsigned int OuterDim>
  std::array<CellType<OuterDim>, (1u<<OuterDim)-1>
  CellType<OuterDim>::getExteriorOrientHigh2Low()
  {
    std::array<CellType, (1u<<OuterDim)-1> orientations;
    CellType *dest = orientations.data();
    for (int fdim = OuterDim - 1; fdim >= 0; fdim--)
    {
      CellType *gpIter = dest;
      emitCombinations(0u, OuterDim, fdim, dest);
      while (gpIter < dest)
        (gpIter++)->set_dimFlag(fdim);
    }
    return orientations;
  }

  //
  // getExteriorOrientLow2High()
  //
  template <unsigned int OuterDim>
  std::array<CellType<OuterDim>, (1u<<OuterDim)-1>
  CellType<OuterDim>::getExteriorOrientLow2High()
  {
    std::array<CellType, (1u<<OuterDim)-1> orientations;
    CellType *dest = orientations.data();
    for (int fdim = 0; fdim < OuterDim; fdim++)
    {
      CellType *gpIter = dest;
      emitCombinations(0u, OuterDim, fdim, dest);
      while (gpIter < dest)
        (gpIter++)->set_dimFlag(fdim);
    }
    return orientations;
  }

  //
  // emitCombinations()
  //
  template <unsigned int OuterDim>
  void CellType<OuterDim>::emitCombinations(
      FlagType prefix, unsigned char lengthLeft, unsigned char onesLeft,
      CellType * &dest)
  {
    assert (onesLeft <= lengthLeft);

    if (onesLeft == 0)
      (dest++)->set_orientFlag(prefix | 0u);
    else if (onesLeft == lengthLeft)
      (dest++)->set_orientFlag(prefix | ((1u << lengthLeft) - 1u));
    else
    {
      emitCombinations(prefix, lengthLeft - 1, onesLeft, dest);
      emitCombinations(prefix | (1u << (lengthLeft - 1)), lengthLeft - 1, onesLeft - 1, dest);
    }
  }


  // ==================== End: CellType ==================== //


  // ============================ Begin: TNPoint === //

  /**
   * @brief Constructs a node at the extreme "lower-left" corner of the domain.
   */
  template <typename T, unsigned int dim>
  TNPoint<T,dim>::TNPoint() : TreeNode<T,dim>()
  { }

  /**
    @brief Constructs a point.
    @param coords The coordinates of the point.
    @param level The level of the point (i.e. level of the element that spawned it).
    @note Uses the "dummy" overload of TreeNode() so that the coordinates are copied as-is.
    */
  template <typename T, unsigned int dim>
  TNPoint<T,dim>::TNPoint (const std::array<T,dim> coords, unsigned int level) :
      TreeNode<T,dim>(coords, level)
  { }

  /**@brief Copy constructor */
  template <typename T, unsigned int dim>
  TNPoint<T,dim>::TNPoint (const TNPoint<T,dim> & other) :
      TreeNode<T,dim>(other),
      m_owner(other.m_owner), m_isCancellation(other.m_isCancellation)
  { }

  /** @brief Assignment operator. No checks for dim or maxD are performed. It's ok to change dim and maxD of the object using the assignment operator.*/
  template <typename T, unsigned int dim>
  TNPoint<T,dim> & TNPoint<T,dim>::operator = (TNPoint<T,dim> const  & other)
  {
    TreeNode<T,dim>::operator=(other);
    m_owner = other.m_owner;
    m_isCancellation = other.m_isCancellation;

    return *this;
  }


  template <typename T, unsigned int dim>
  bool TNPoint<T,dim>::operator== (TNPoint const &that) const
  {
    std::array<T,dim> coords1, coords2;
    return TreeNode<T,dim>::getLevel() == that.getLevel() && (TreeNode<T,dim>::getAnchor(coords1), coords1) == (that.getAnchor(coords2), coords2);
  }

  template <typename T, unsigned int dim>
  bool TNPoint<T,dim>::operator!= (TNPoint const &that) const
  {
    return ! operator==(that);
  }


  /**@brief Infer the type (dimension and orientation) of cell this point is interior to, from coordinates and level. */
  template <typename T, unsigned int dim>
  CellType<dim> TNPoint<T,dim>::get_cellType() const
  {
    return get_cellType(TreeNode<T,dim>::m_uiLevel);
  }

  template <typename T, unsigned int dim>
  CellType<dim> TNPoint<T,dim>::get_cellTypeOnParent() const
  {
    return get_cellType(TreeNode<T,dim>::m_uiLevel - 1);
  }

  template <typename T, unsigned int dim>
  CellType<dim> TNPoint<T,dim>::get_cellType(LevI lev) const
  {
    return TNPoint::get_cellType(*this, lev);
  }

  template <typename T, unsigned int dim>
  CellType<dim> TNPoint<T,dim>::get_cellType(const TreeNode<T, dim> &tnPoint, LevI lev)
  {
    const unsigned int len = 1u << (m_uiMaxDepth - lev);
    const unsigned int interiorMask = len - 1;

    unsigned char cellDim = 0u;
    unsigned char cellOrient = 0u;
    #pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
      bool axisIsInVolume = tnPoint.getX(d) & interiorMask;
      cellOrient |= (unsigned char) axisIsInVolume << d;
      cellDim += axisIsInVolume;
    }

    CellType<dim> cellType;
    cellType.set_dimFlag(cellDim);
    cellType.set_orientFlag(cellOrient);

    return cellType;
  }


  template <typename T, unsigned int dim>
  bool TNPoint<T,dim>::getIsCancellation() const
  {
    return m_isCancellation;
  }

  template <typename T, unsigned int dim>
  void TNPoint<T,dim>::setIsCancellation(bool isCancellation)
  {
    m_isCancellation = isCancellation;
  }



  template <typename T, unsigned int dim>
  bool TNPoint<T,dim>::isCrossing() const
  {
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - TreeNode::m_uiLevel);
    const unsigned int mask = (len << 1u) - 1u;
    for (int d = 0; d < dim; d++)
      if ((this->getX(d) & mask) == len)
        return true;
    return false;
  }

  template <typename T, unsigned int dim>
  unsigned int TNPoint<T,dim>::get_lexNodeRank(const TreeNode<T,dim> &hostCell, unsigned int polyOrder) const
  {
    return TNPoint<T, dim>::get_lexNodeRank(hostCell, *this, polyOrder);
  }


  // static Overload that accepts node (tnPoint) coordinates as a TreeNode.
  template <typename T, unsigned int dim>
  unsigned int TNPoint<T,dim>::get_lexNodeRank(const TreeNode<T,dim> &hostCell,
                                               const TreeNode<T, dim> &tnPoint,
                                               unsigned int polyOrder)
  {
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - hostCell.getLevel());

    unsigned int rank = 0;
    unsigned int stride = 1;
    #pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
      unsigned int index1D = TNPoint<T, dim>::get_nodeRank1D(hostCell, tnPoint, d, polyOrder);
      rank += index1D * stride;
      stride *= (polyOrder + 1);
    }

    return rank;
  }

  template <typename T, unsigned int dim>
  unsigned int TNPoint<T,dim>::get_nodeRank1D(const TreeNode<T, dim> &hostCell,
                                              const TreeNode<T, dim> &tnPoint,
                                              unsigned int d,
                                              unsigned int polyOrder)
  {
    const unsigned int len = 1u << (m_uiMaxDepth - hostCell.getLevel());

    // Round up here, since we round down when we generate the nodes.
    // The inequalities of integer division work out, as long as polyOrder < len.
    if (hostCell.range().upperEquals(d, tnPoint.coords().coord(d)))
      return polyOrder;
    else
      return polyOrder - (unsigned long) polyOrder * (hostCell.getX(d) + len - tnPoint.getX(d)) / len;
  }


  template <typename T, unsigned int dim>
  std::array<unsigned, dim> TNPoint<T, dim>::get_nodeRanks1D(
      const TreeNode<T, dim> &hostCell,
      const TreeNode<T, dim> &tnPoint,
      unsigned int polyOrder)
  {
    std::array<unsigned, dim> indices;
    for (int d = 0; d < dim; ++d)
      indices[d] = get_nodeRank1D(hostCell, tnPoint, d, polyOrder);
    return indices;
  }


  // Get's numerator and denominator of tnPoint node coordinates relative,
  // to containingSubtree, where 0/1 and 1/1 correspond to the subtree edges.
  template <typename T, unsigned int dim>
  void TNPoint<T,dim>::get_relNodeCoords(const TreeNode<T,dim> &containingSubtree,
                                         const TreeNode<T,dim> &tnPoint,
                                         unsigned int polyOrder,
                                         std::array<unsigned int, dim> &numerators,
                                         unsigned int &denominator)
  {
    const TreeNode<T,dim> hostCell = tnPoint.getAncestor(tnPoint.getLevel());
    const unsigned int levDiff = hostCell.getLevel() - containingSubtree.getLevel();
    const unsigned int hostShift = m_uiMaxDepth - hostCell.getLevel();

    denominator = polyOrder * (1u << levDiff);

    for (int d = 0; d < dim; d++)
      numerators[d] = ((hostCell.getX(d) - containingSubtree.getX(d)) >> hostShift) * polyOrder
          + get_nodeRank1D(hostCell, tnPoint, d, polyOrder);
  }



  // ============================ End: TNPoint ============================ //


  // ============================ Begin: Element ============================ //


  template <typename T, unsigned int dim>
  std::array<T, dim> Element<T, dim>::getNodeX(
      const std::array<unsigned, dim> &numerators, unsigned polyOrder) const
  {
    const unsigned int len = 1u << (m_uiMaxDepth - this->getLevel());
    std::array<T, dim> nodeCoords;
    for (int d = 0; d < dim; ++d)
      nodeCoords[d] = (unsigned long) len * numerators[d] / polyOrder  +  this->getX(d);
    return nodeCoords;
  }

  template <typename T, unsigned int dim>
  TNPoint<T, dim>    Element<T, dim>::getNode(
      const std::array<unsigned, dim> &numerators, unsigned polyOrder) const
  {
    return TNPoint<T, dim>(this->getNodeX(numerators, polyOrder), this->getLevel());
  }


  template <typename T, unsigned int dim>
  template <typename TN>
  void Element<T,dim>::appendNodes(unsigned int order, std::vector<TN> &nodeList) const
  {
    const unsigned int numNodes = intPow(order+1, dim);

    std::array<unsigned int, dim> nodeIndices;
    nodeIndices.fill(0);
    for (unsigned int node = 0; node < numNodes; node++)
    {
      nodeList.push_back(this->getNode(nodeIndices, order));
      incrementBaseB<unsigned int, dim>(nodeIndices, order+1);
    }
  }


  template <typename T, unsigned int dim>
  void Element<T,dim>::appendInteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const
  {
    // Basically the same thing as appendNodes (same dimension of volume, if nonempty),
    // just use (order-1) instead of (order+1), and shift indices by 1.
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - this->getLevel());

    const unsigned int numNodes = intPow(order-1, dim);

    std::array<unsigned int, dim> nodeIndices;
    nodeIndices.fill(0);
    for (unsigned int node = 0; node < numNodes; node++)
    {
      std::array<T,dim> nodeCoords;
      #pragma unroll(dim)
      for (int d = 0; d < dim; d++)
        nodeCoords[d] = len * (nodeIndices[d]+1) / order  +  this->getX(d);
      nodeList.push_back(TNPoint<T,dim>(nodeCoords, this->getLevel()));

      incrementBaseB<unsigned int, dim>(nodeIndices, order-1);
    }
  }


  template <typename T, unsigned int dim>
  std::array<double,dim> genPhysNodeCoords(const Element<T, dim> &element,
                                           const std::array<unsigned int, dim> &nodeIndices,
                                           const unsigned int order)
  {
    double physElemCoords[dim];
    double physSize;
    treeNode2Physical(element, physElemCoords, physSize);

    std::array<double, dim> physNodeCoordsOut;

    #pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
      physNodeCoordsOut[d] = (1.0 * nodeIndices[d] / order * physSize) + physElemCoords[d];
    }

    return physNodeCoordsOut;
  }

  template <typename T, unsigned int dim>
  std::array<T,dim> genNodeCoords(const Element<T, dim> &element,
                                  const std::array<unsigned int, dim> &nodeIndices,
                                  const unsigned int order)
  {
    const unsigned int len = 1u << (m_uiMaxDepth - element.getLevel());

    std::array<T,dim> nodeCoords;

    #pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
      nodeCoords[d] = len * nodeIndices[d] / order  +  element.getX(d);
    }

    return nodeCoords;
  }


  template <typename T, unsigned int dim>
  void Element<T,dim>::appendExteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList, const ::ibm::DomainDecider &domainDecider) const
  {
    // Duplicated the appendNodes() function, and then caused every interior node to be thrown away.
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - TreeNode::m_uiLevel);

    std::array<unsigned int, dim> nodeIndices;

    constexpr size_t numVertices = 1u << dim;
    std::bitset<numVertices> vertexBoundary;  // defaults to all false.

    // If this element has been flagged as boundary,
    // then evaluate vertexBoundary.
    if (this->getIsOnTreeBdry())
    {
      for (int v = 0; v < numVertices; ++v)
      {
        for (int d = 0; d < dim; ++d)
          nodeIndices[d] = bool(v & (1u << d)) * order;
        const std::array<double, dim> physNodeCoords = genPhysNodeCoords<T, dim>(*this, nodeIndices, order);
        vertexBoundary[v] = (domainDecider(physNodeCoords.data(), 0.0) == ibm::IN);
      }
    }

    nodeIndices.fill(0);
    const unsigned int numNodes = intPow(order+1, dim);
    for (unsigned int node = 0; node < numNodes; node++)
    {
      // For exterior, need at least one of the entries in nodeIndices to be 0 or order.
      if (std::count(nodeIndices.begin(), nodeIndices.end(), 0) == 0 &&
          std::count(nodeIndices.begin(), nodeIndices.end(), order) == 0)
      {
        nodeIndices[0] = order;   // Skip ahead to the lexicographically next boundary node.
        node += order - 1;
      }

      std::array<T,dim> nodeCoords = genNodeCoords<T,dim>(*this, nodeIndices, order);
      nodeList.push_back(TNPoint<T,dim>(nodeCoords, TreeNode::m_uiLevel));

      // Assign boundary flag (true/false) to the node.
      // This method does not cover all cases correctly,
      // but it does prevent faces from becoming _partially_ boundary faces.
      std::bitset<numVertices> vertexTestBoundary;
      vertexTestBoundary[0] = true;
      for (int d = 0; d < dim; ++d)
        if (0 == nodeIndices[d])
          vertexTestBoundary = vertexTestBoundary;
        else if (0 < nodeIndices[d] && nodeIndices[d] < order)
          vertexTestBoundary |= vertexTestBoundary << (1u << d);
        else if (nodeIndices[d] == order)
          vertexTestBoundary = vertexTestBoundary << (1u << d);

      // Tag a node if and only if all vertices on the face are boundary nodes.
      const bool isBoundaryNode = ((vertexBoundary & vertexTestBoundary) == vertexTestBoundary);
      nodeList.back().setIsOnTreeBdry(isBoundaryNode);

      incrementBaseB<unsigned int, dim>(nodeIndices, order+1);
    }
  }

  template <typename T, unsigned int dim>
  void Element<T,dim>::appendCancellationNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const
  {
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - this->getLevel());

    const unsigned int numNodes = intPow(order+1, dim);

    double physElemCoords[dim];
    double physSize;
    treeNode2Physical(*this, physElemCoords, physSize);

    std::array<unsigned int, dim> nodeIndices;

    if (this->getLevel() < m_uiMaxDepth)  // Don't need cancellations if no hanging elements.
    {
      int numOdd = 0;

      // Cancellations at odd external subnodes.
      const unsigned int numSubNodes = intPow(2*order + 1, dim);
      nodeIndices.fill(0);
      for (unsigned int subNode = 0; subNode < numSubNodes; ++subNode)
      {
        if (std::count(nodeIndices.begin(), nodeIndices.end(), 0) == 0 &&
            std::count(nodeIndices.begin(), nodeIndices.end(), 2*order) == 0)
        {
          nodeIndices[0] = 2*order;  // Skip
          subNode += 2*order - 1;
        }

        bool odd = false;
        for (int d = 0; d < dim; ++d)
          if (nodeIndices[d] % 2 == 1)
            odd = true;

        // Append cancellation nodes at odd locations.
        if (odd)
        {
          std::array<T, dim> nodeCoords;
          for (int d = 0; d < dim; ++d)
          {
            // Should be the same as if had children append.
            nodeCoords[d] = len / 2 * nodeIndices[d] / order + this->getX(d);
          }

          nodeList.push_back(TNPoint<T,dim>(nodeCoords, this->getLevel()+1));
          nodeList.back().setIsCancellation(true);

          numOdd++;
        }
        incrementBaseB<unsigned int, dim>(nodeIndices, 2*order+1);
      }
    }
  }


  template <typename T, unsigned int dim>
  void Element<T,dim>::appendKFaces(CellType<dim> kface,
      std::vector<TreeNode<T,dim>> &nodeList, std::vector<CellType<dim>> &kkfaces) const
  {
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - TreeNode::m_uiLevel);

    unsigned int fdim = kface.get_dim_flag();
    unsigned int orient = kface.get_orient_flag();

    const unsigned int numNodes = intPow(3, fdim);

    std::array<unsigned int, dim> nodeIndices;
    nodeIndices.fill(0);
    for (unsigned int node = 0; node < numNodes; node++)
    {
      unsigned char kkfaceDim = 0;
      unsigned int  kkfaceOrient = 0u;
      std::array<T,dim> nodeCoords = TreeNode::m_uiCoords;
      int vd = 0;
      for (int d = 0; d < dim; d++)
      {
        if (orient & (1u << d))
        {
          if (nodeIndices[vd] == 1)
          {
            kkfaceDim++;
            kkfaceOrient |= (1u << d);
          }
          else if (nodeIndices[vd] == 2)
            nodeCoords[d] += len;

          vd++;
        }
      }
      nodeList.push_back(TreeNode(nodeCoords, TreeNode::m_uiLevel));
      kkfaces.push_back(CellType<dim>());
      kkfaces.back().set_dimFlag(kkfaceDim);
      kkfaces.back().set_orientFlag(kkfaceOrient);

      incrementBaseB<unsigned int, dim>(nodeIndices, 3);
    }
  }



  template <typename T, unsigned int dim>
  std::array<unsigned, dim> Element<T,dim>::hanging2ParentIndicesBijection(
      const std::array<unsigned, dim> &indices, unsigned polyOrder) const
  {
    std::array<unsigned, dim> parentIndices;
    for (int d = 0; d < dim; ++d)
    {
      const unsigned p = polyOrder;
      const unsigned i = indices[d];

      const bool isLeft = !(this->getMortonIndex() & (1u << d));
      if (isLeft && i % 2 == 0)
        parentIndices[d] = i/2;
      else if (isLeft && i % 2 != 0)
        parentIndices[d] = p+1 - (p+1)/2 + (i-1)/2;
      else if (!isLeft && (p-i) % 2 == 0)
        parentIndices[d] = p - (p-i)/2;
      else
        parentIndices[d] = p-(p+1 - (p+1)/2 + ((p-i)-1)/2);
    }
    return parentIndices;
  }


  /**
   * @brief Identifies which virtual children are touching a point.
   * @param [in] pointCoords Coordinates of the point incident on 0 or more children.
   * @param [out] incidenceOffset The Morton child # of the first incident child.
   * @param [out] incidenceSubspace A bit string of axes, with a '1'
   *                for each incident child that is adjacent to the first incident child.
   * @param [out] incidenceSubspaceDim The number of set ones in incidenceSubspace.
   *                The number of incident children is pow(2, incidenceSubspaceDim).
   * @note Use with TallBitMatrix to easily iterate over the child numbers of incident children.
   */
  template <typename T, unsigned int dim>
  void Element<T,dim>::incidentChildren(
      const ot::TreeNode<T,dim> &pt,
      typename ot::CellType<dim>::FlagType &incidenceOffset,
      typename ot::CellType<dim>::FlagType &incidenceSubspace,
      typename ot::CellType<dim>::FlagType &incidenceSubspaceDim) const
  {
    periodic::PRange<T, dim> lowerRange = this->getChildMorton(0).range();
    periodic::PRange<T, dim> upperRange = this->getChildMorton((1u << dim) - 1).range();

    incidenceOffset = 0;
    incidenceSubspace = 0;
    incidenceSubspaceDim = 0;
    for (int d = 0; d < dim; ++d)
    {
      const bool child0 = lowerRange.closedContains(d, pt.coords().coord(d));
      const bool child1 = upperRange.closedContains(d, pt.coords().coord(d));

      if (child0 && child1)
      {
        incidenceSubspace |= (1u << d);
        incidenceSubspaceDim++;
      }
      else if (child1)
      {
        incidenceOffset |= (1u << d);
      }
      // if just child0, then no offset needed.
    }
  }


  template <typename T, unsigned int dim>
  bool Element<T,dim>::isIncident(const ot::TreeNode<T,dim> &pointCoords) const
  {
    const unsigned int elemSize = (1u << m_uiMaxDepth - this->getLevel());
    unsigned int nbrId = 0;
    for (int d = 0; d < dim; d++)
      if (this->getX(d) == pointCoords.getX(d))
        nbrId += (1u << d);
      else if (this->getX(d) < pointCoords.getX(d)
                            && pointCoords.getX(d) <= this->getX(d) + elemSize)
        ;
      else
        return false;

    return true;
  }


  // ============================ End: Element ============================ //


  // ============================ Begin: SFC_NodeSort ============================ //

  template <typename T, unsigned int dim>
  GatherMap SFC_NodeSort<T,dim>::scatter2gather(const ScatterMap &sm, RankI localCount, MPI_Comm comm)
  {
    int nProc, rProc;
    MPI_Comm_rank(comm, &rProc);
    MPI_Comm_size(comm, &nProc);

    std::vector<RankI> fullSendCounts(nProc);
    auto scountIter = sm.m_sendCounts.cbegin();
    auto sprocIter = sm.m_sendProc.cbegin();
    while (scountIter < sm.m_sendCounts.cend())
      fullSendCounts[*(sprocIter++)] = *(scountIter++);

    // All to all to exchange counts. Receivers need to learn who they receive from.
    std::vector<RankI> fullRecvCounts(nProc);
    par::Mpi_Alltoall<RankI>(fullSendCounts.data(), fullRecvCounts.data(), 1, comm);

    // Compact the receive counts into the GatherMap struct.
    GatherMap gm;
    RankI accum = 0;
    for (int proc = 0; proc < nProc; proc++)
    {
      if (fullRecvCounts[proc] > 0)
      {
        gm.m_recvProc.push_back(proc);
        gm.m_recvCounts.push_back(fullRecvCounts[proc]);
        gm.m_recvOffsets.push_back(accum);
        accum += fullRecvCounts[proc];
      }
      else if (proc == rProc)
      {
        gm.m_locOffset = accum;             // Allocate space for our local nodes.
        accum += localCount;
      }
    }
    gm.m_totalCount = accum;
    gm.m_locCount = localCount;

    return gm;
  }


  template <typename T, unsigned int dim>
  template <typename da>
  void SFC_NodeSort<T,dim>::ghostExchange(da *data, da *sendBuf, const ScatterMap &sm, const GatherMap &gm, MPI_Comm comm)
  {
    const RankI sendSize = sm.m_map.size();
    const da * const mydata = data + gm.m_locOffset;

    // Stage send data.
    for (ot::RankI ii = 0; ii < sendSize; ii++)
      sendBuf[ii] = mydata[sm.m_map[ii]];

    // Send/receive data.
    std::vector<MPI_Request> requestSend(sm.m_sendProc.size());
    std::vector<MPI_Request> requestRecv(gm.m_recvProc.size());
    MPI_Status status;

    for (int sIdx = 0; sIdx < sm.m_sendProc.size(); sIdx++)
      par::Mpi_Isend(sendBuf+ sm.m_sendOffsets[sIdx],   // Send.
          sm.m_sendCounts[sIdx],
          sm.m_sendProc[sIdx], 0, comm, &requestSend[sIdx]);

    for (int rIdx = 0; rIdx < gm.m_recvProc.size(); rIdx++)
      par::Mpi_Irecv(data + gm.m_recvOffsets[rIdx],  // Recv.
          gm.m_recvCounts[rIdx],
          gm.m_recvProc[rIdx], 0, comm, &requestRecv[rIdx]);

    for (int sIdx = 0; sIdx < sm.m_sendProc.size(); sIdx++)     // Wait sends.
      MPI_Wait(&requestSend[sIdx], &status);
    for (int rIdx = 0; rIdx < gm.m_recvProc.size(); rIdx++)      // Wait recvs.
      MPI_Wait(&requestRecv[rIdx], &status);
  }

 
  template <typename T, unsigned int dim>
  template <typename da>
  void SFC_NodeSort<T,dim>::ghostReverse(da *data, da *sendBuf, const ScatterMap &sm, const GatherMap &gm, MPI_Comm comm)
  {
    // In this function we do the reverse of ghostExchange().

    // 'data' is the outVec from matvec().
    // Assume that the unused positions in outVec were initialized to 0
    // and are still 0. Otherwise we will send, receive, and accum garbage.

    std::vector<MPI_Request> requestSend(gm.m_recvProc.size());
    std::vector<MPI_Request> requestRecv(sm.m_sendProc.size());
    MPI_Status status;

    // 1. We have contributions to remote nodes.
    //    They are already staged in the ghost layers.
    //    Send back to owners via the 'gather map'.
    for (int rIdx = 0; rIdx < gm.m_recvProc.size(); rIdx++)
      par::Mpi_Isend(data + gm.m_recvOffsets[rIdx],
          gm.m_recvCounts[rIdx],
          gm.m_recvProc[rIdx], 0, comm, &requestSend[rIdx]);

    for (int rIdx = 0; rIdx < gm.m_recvProc.size(); rIdx++)
      par::Mpi_Irecv(data + gm.m_recvOffsets[rIdx],  // Recv.
          gm.m_recvCounts[rIdx],
          gm.m_recvProc[rIdx], 0, comm, &requestRecv[rIdx]);


    // 2. Owners receive back contributions from non-owners.
    //    Contributions are received into the send buffer space
    //    via the counts/offsets of the 'scatter map'.
    for (int sIdx = 0; sIdx < sm.m_sendProc.size(); sIdx++)
      par::Mpi_Irecv(sendBuf+ sm.m_sendOffsets[sIdx],
          sm.m_sendCounts[sIdx],
          sm.m_sendProc[sIdx], 0, comm, &requestRecv[sIdx]);

    // Wait for sends and recvs.
    for (int rIdx = 0; rIdx < gm.m_recvProc.size(); rIdx++)      // Wait sends.
      MPI_Wait(&requestSend[rIdx], &status);
    for (int sIdx = 0; sIdx < sm.m_sendProc.size(); sIdx++)     // Wait recvs.
      MPI_Wait(&requestRecv[sIdx], &status);


    // 3. Owners locally accumulate the contributions from the send buffer space
    //    into the proper node positions via the map of the 'scatter map'.
    const RankI sendSize = sm.m_map.size();
    const da * const mydata = data + gm.m_locOffset;
    for (ot::RankI ii = 0; ii < sendSize; ii++)
      mydata[sm.m_map[ii]] += sendBuf[ii];
  }

  // ============================ End: SFC_NodeSort ============================ //

}//namespace ot
