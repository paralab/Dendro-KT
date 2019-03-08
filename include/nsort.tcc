/**
 * @file:nsort.tcc
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-02-20
 */

namespace ot {

  // ============================ Begin: TNPoint === //

  /**
   * @brief Constructs a node at the extreme "lower-left" corner of the domain.
   */
  template <typename T, unsigned int dim>
  TNPoint<T,dim>::TNPoint() : TreeNode<T,dim>(), m_isSelected(Maybe)
  { }

  /**
    @brief Constructs a point.
    @param coords The coordinates of the point.
    @param level The level of the point (i.e. level of the element that spawned it).
    @note Uses the "dummy" overload of TreeNode() so that the coordinates are copied as-is.
    */
  template <typename T, unsigned int dim>
  TNPoint<T,dim>::TNPoint (const std::array<T,dim> coords, unsigned int level) :
      TreeNode<T,dim>(0, coords, level), m_isSelected(Maybe)
  { }

  /**@brief Copy constructor */
  template <typename T, unsigned int dim>
  TNPoint<T,dim>::TNPoint (const TNPoint<T,dim> & other) :
      TreeNode<T,dim>(other),
      m_isSelected(other.m_isSelected), m_numInstances(other.m_numInstances), m_owner(other.m_owner)
  { }

  /**
    @brief Constructs an octant (without checks).
    @param dummy : not used yet.
    @param coords The coordinates of the point.
    @param level The level of the point (i.e. level of the element that spawned it).
  */
  template <typename T, unsigned int dim>
  TNPoint<T,dim>::TNPoint (const int dummy, const std::array<T,dim> coords, unsigned int level) :
      TreeNode<T,dim>(dummy, coords, level), m_isSelected(Maybe)
  { }

  /** @brief Assignment operator. No checks for dim or maxD are performed. It's ok to change dim and maxD of the object using the assignment operator.*/
  template <typename T, unsigned int dim>
  TNPoint<T,dim> & TNPoint<T,dim>::operator = (TNPoint<T,dim> const  & other)
  {
    TreeNode<T,dim>::operator=(other);
    m_isSelected = other.m_isSelected;
    m_numInstances = other.m_numInstances;
    m_owner = other.m_owner;
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


  template <typename T, unsigned int dim>
  unsigned char TNPoint<T,dim>::get_firstIncidentHyperplane(unsigned int hlev) const
  {
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - hlev);
    const unsigned int interiorMask = len - 1;

    for (int d = 0; d < dim; d++)
    {
      bool axisIsInVolume = TreeNode::m_uiCoords[d] & interiorMask;
      if (!axisIsInVolume)   // Then the point is on the hyperplane face normal to this axis.
        return d;
    }
    return dim;
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
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - lev);
    const unsigned int interiorMask = len - 1;

    unsigned char cellDim = 0u;
    unsigned char cellOrient = 0u;
    #pragma unroll(dim)
    for (int d = 0; d < dim; d++)
    {
      bool axisIsInVolume = TreeNode::m_uiCoords[d] & interiorMask;
      cellOrient |= (unsigned char) axisIsInVolume << d;
      cellDim += axisIsInVolume;
    }

    CellType<dim> cellType;
    cellType.set_dimFlag(cellDim);
    cellType.set_orientFlag(cellOrient);

    return cellType;
  }

  template <typename T, unsigned int dim>
  TreeNode<T,dim> TNPoint<T,dim>::getFinestOpenContainer() const
  {
    assert((!TreeNode<T,dim>::isOnDomainBoundary()));  // When we bucket we need to set aside boundary points first.

    //
    // A node is on a boundary at level lev iff any coordinates have only zeros
    // for all bits strictly deeper than lev. Take the contrapositive:
    // A node is NOT on a boundary at level lev iff all coordinates have at least
    // one nonzero bit strictly deeper than lev. We need the deepest such level.
    // To get the deepest such level, find the deepest nonzero bit in each
    // coordinate, and take the coarsest of finest levels.
    //
    using TreeNode = TreeNode<T,dim>;
    unsigned int coarsestFinestHeight = 0;
    for (int d = 0; d < dim; d++)
    {
      unsigned int finestHeightDim = binOp::lowestOnePos(TreeNode::m_uiCoords[d]);
      // Maximum height is minimum depth.
      if (finestHeightDim > coarsestFinestHeight)
        coarsestFinestHeight = finestHeightDim;
    }
    unsigned int level = m_uiMaxDepth - coarsestFinestHeight; // Convert to depth.
    level--;                           // Nonzero bit needs to be strictly deeper.

    // Use the non-dummy overload so that the coordinates get clipped.
    return TreeNode(TreeNode::m_uiCoords, level);
  }

  template <typename T, unsigned int dim>
  TreeNode<T,dim> TNPoint<T,dim>::getCell() const
  {
    using TreeNode = TreeNode<T,dim>;
    return TreeNode(TreeNode::m_uiCoords, TreeNode::m_uiLevel);  // Truncates coordinates to cell anchor.
  }

  template <typename T, unsigned int dim>
  void TNPoint<T,dim>::appendAllBaseNodes(std::vector<TNPoint> &nodeList)
  {
    // The base nodes are obtained by adding or subtracting `len' in every
    // normal axis (self-boundary axis) and scaling x2 the off-anchor offset in every
    // tangent axis (self-interior axis). Changes are only made for the
    // parent-interior axes.
    // In other words, scaling x2 the offset from every vertex of the cellTypeOnParent.
    // Note: After scaling, may have rounding artifacts!
    // Must compare within a +/- distance of the least significant bit.
    // Will at least end up on the same interface/hyperplane, and should
    // be very close by in SFC order.

    using TreeNode = TreeNode<T,dim>;
    const unsigned int parentLen = 1u << (m_uiMaxDepth - (TreeNode::m_uiLevel - 1));
    const unsigned int interiorMask = parentLen - 1;

    std::array<T,dim> anchor;
    getCell().getParent().getAnchor(anchor);

    std::array<unsigned char, dim> interiorAxes;
    unsigned char celldim = 0;
    for (int d = 0; d < dim; d++)
    {
      if (TreeNode::m_uiCoords[d] & interiorMask)
        interiorAxes[celldim++] = d;
    }

    for (unsigned char vId = 0; vId < (1u << celldim); vId++)
    {
      TNPoint<T,dim> base(*this);
      base.setLevel(TreeNode::m_uiLevel - 1);
      for (int dIdx = 0; dIdx < celldim; dIdx++)
      {
        int d = interiorAxes[dIdx];
        T vtxCoord = anchor[d] + (vId & (1u << dIdx) ? parentLen : 0);
        base.setX(d, ((base.getX(d) - vtxCoord) << 1) + vtxCoord);
      }

      nodeList.push_back(base);
    }
  }

  // ============================ End: TNPoint ============================ //


  // ============================ Begin: Element ============================ //

  template <typename T, unsigned int dim>
  void Element<T,dim>::appendNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const
  {
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - TreeNode::m_uiLevel);

    const unsigned int numNodes = intPow(order+1, dim);

    std::array<unsigned int, dim> nodeIndices;
    nodeIndices.fill(0);
    for (unsigned int node = 0; node < numNodes; node++)
    {
      std::array<T,dim> nodeCoords;
      #pragma unroll(dim)
      for (int d = 0; d < dim; d++)
        nodeCoords[d] = len * nodeIndices[d] / order  +  TreeNode::m_uiCoords[d];
      nodeList.push_back(TNPoint<T,dim>(nodeCoords, TreeNode::m_uiLevel));

      incrementBaseB<unsigned int, dim>(nodeIndices, order+1);
    }
  }


  template <typename T, unsigned int dim>
  void Element<T,dim>::appendInteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const
  {
    // Basically the same thing as appendNodes (same dimension of volume, if nonempty),
    // just use (order-1) instead of (order+1), and shift indices by 1.
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - TreeNode::m_uiLevel);

    const unsigned int numNodes = intPow(order-1, dim);

    std::array<unsigned int, dim> nodeIndices;
    nodeIndices.fill(0);
    for (unsigned int node = 0; node < numNodes; node++)
    {
      std::array<T,dim> nodeCoords;
      #pragma unroll(dim)
      for (int d = 0; d < dim; d++)
        nodeCoords[d] = len * (nodeIndices[d]+1) / order  +  TreeNode::m_uiCoords[d];
      nodeList.push_back(TNPoint<T,dim>(nodeCoords, TreeNode::m_uiLevel));

      incrementBaseB<unsigned int, dim>(nodeIndices, order-1);
    }
  }

  template <typename T, unsigned int dim>
  void Element<T,dim>::appendExteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const
  {
    // Duplicated the appendNodes() function, and then caused every interior node to be thrown away.
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - TreeNode::m_uiLevel);

    const unsigned int numNodes = intPow(order+1, dim);

    std::array<unsigned int, dim> nodeIndices;
    nodeIndices.fill(0);
    for (unsigned int node = 0; node < numNodes; node++)
    {
      // For exterior, need at least one of the entries in nodeIndices to be 0 or order.
      if (std::count(nodeIndices.begin(), nodeIndices.end(), 0) == 0 &&
          std::count(nodeIndices.begin(), nodeIndices.end(), order) == 0)
      {
        nodeIndices[0] = order;   // Skip ahead to the lexicographically next boundary node.
        node += order - 1;
      }

      std::array<T,dim> nodeCoords;
      #pragma unroll(dim)
      for (int d = 0; d < dim; d++)
        nodeCoords[d] = len * nodeIndices[d] / order  +  TreeNode::m_uiCoords[d];
      nodeList.push_back(TNPoint<T,dim>(nodeCoords, TreeNode::m_uiLevel));

      incrementBaseB<unsigned int, dim>(nodeIndices, order+1);
    }
  }

  // ============================ End: Element ============================ //



  // ============================ Begin: SFC_NodeSort ============================ //

  //
  // SFC_NodeSort::dist_countCGNodes()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::dist_countCGNodes(std::vector<TNPoint<T,dim>> &points, unsigned int order, const TreeNode<T,dim> *treePartStart, MPI_Comm comm)
  {
    // TODO TODO TODO

    using TNP = TNPoint<T,dim>;

    // TODO Ownership: The local sort is not stable. We ought to scan through the recv'd nodes again to find out which proc-ranks are represented.

    // 1. Call countCGNodes(classify=false) to agglomerate node instances -> node representatives.
    // 2. For every representative, determine which processor boundaries the node is incident on,
    //    by generating neighbor-keys and sorting them against the proc-bdry splitters.
    // 3. Each representative is copied once per incident proc-boundary.
    // 4. Send and receive the boundary layer nodes.
    // 5. Finally, sort and count nodes with countCGNodes(classify=true).

    int nProc, rProc;
    MPI_Comm_rank(comm, &rProc);
    MPI_Comm_size(comm, &nProc);

    /// fprintf(stderr, "\n");

    if (nProc == 1)
      return countCGNodes(&(*points.begin()), &(*points.end()), order, true);

    if (points.size() == 0)
      return 0;

    // First local pass: Don't classify, just sort and count instances.
    countCGNodes(&(*points.begin()), &(*points.end()), order, false);

    /// //TODO remove
    /// for (const TNP &pt : points)
    ///   fprintf(stderr, "[%d] aftrSort  numIns==%d, l==%u, coords==%s\n", rProc, pt.get_numInstances(), pt.getLevel(), pt.getBase32Hex().data());
    /// fprintf(stderr, "[[%d]]\n\n", rProc);

    // Compact node list (remove literal duplicates).
    RankI numUniquePoints = 0;
    for (const TNP &pt : points)
    {
      if (pt.get_numInstances() > 0)
        points[numUniquePoints++] = pt;
    }
    points.resize(numUniquePoints);


    //
    // Preliminary sharing information. Prepare to test for globally hanging nodes.
    // For each node (unique) node, find out if it's a proc-boundary node, and who shares.
    //
    struct BdryNodeInfo { RankI ptIdx; int numProcNb; };
    std::vector<BdryNodeInfo> bdryNodeInfo;
    std::vector<int> shareLists;

    std::vector<RankI> sendCounts(nProc, 0);
    std::vector<RankI> recvCounts(nProc, 0);

    std::vector<RankI> sendOffsets;
    std::vector<RankI> recvOffsets;
    std::vector<int> sendProc;
    std::vector<int> recvProc;
    std::vector<RankI> shareMap;

    // Get neighbour information.
    std::vector<TreeNode<T,dim>> splitters = dist_bcastSplitters(treePartStart, comm);
    assert((splitters.size() == nProc));
    const RankI numPoints  = points.size();
    for (RankI ptIdx = 0; ptIdx < numPoints; ptIdx++)
    {
      if (points[ptIdx].get_numInstances() > 0)  // Should always be true if we already compacted.
      {
        numUniquePoints++;

        // Concatenates list of (unique) proc neighbours to shareLists.
        int numProcNb = getProcNeighbours(points[ptIdx], splitters.data(), nProc, shareLists);
        //TODO test the assertion that all listed proc-neighbours in a sublist are unique.

        // Remove ourselves from neighbour list.
        if (std::remove(shareLists.end() - numProcNb, shareLists.end(), rProc) < shareLists.end())
        {
          shareLists.erase(shareLists.end() - 1);
          numProcNb--;
        }

        if (numProcNb > 0)  // Shared, i.e. is a proc-boundary node.
        {
          bdryNodeInfo.push_back({ptIdx, numProcNb});    // Record which point and how many proc-neighbours.
          for (int nb = numProcNb; nb >= 1; nb--)
          {
            sendCounts[*(shareLists.end() - nb)]++;
          }
        }
      }
    }


    // Create the preliminary send buffer, ``share buffer.''
    // We compute the preliminary send buffer only once, so don't make a separate scatter map for it.
    std::vector<TNP> shareBuffer;
    int sendTotal = 0;
    sendOffsets.resize(nProc);
    for (int proc = 0; proc < nProc; proc++)
    {
      sendOffsets[proc] = sendTotal;
      sendTotal += sendCounts[proc];
    }
    shareBuffer.resize(sendTotal);

    // Determine the receive counts, via Alltoall of send counts.
    par::Mpi_Alltoall(sendCounts.data(), recvCounts.data(), 1, comm); 

    // Copy outbound data into the share buffer.
    // Note: Advances sendOffsets, so will need to recompute sendOffsets later.
    const int *shareListPtr = &(*shareLists.begin());
    for (const BdryNodeInfo &nodeInfo : bdryNodeInfo)
    {
      for (int ii = 0; ii < nodeInfo.numProcNb; ii++)
      {
        int proc = *(shareListPtr++);
        points[nodeInfo.ptIdx].set_owner(rProc);   // Reflected in both copies, else default -1.
        shareBuffer[sendOffsets[proc]++] = points[nodeInfo.ptIdx];
      }
    }

    // Compact lists of neighbouring processors and (re)compute share offsets.
    // Note: This should only be done after we are finished with shareLists.
    int numSendProc = 0;
    int numRecvProc = 0;
    sendTotal = 0;
    int recvTotal = 0;
    sendOffsets.clear();
    for (int proc = 0; proc < nProc; proc++)
    {
      if (sendCounts[proc] > 0)   // Discard empty counts, which includes ourselves.
      {
        sendCounts[numSendProc++] = sendCounts[proc];  // Compacting.
        sendOffsets.push_back(sendTotal);
        sendProc.push_back(proc);
        sendTotal += sendCounts[proc];
      }
      if (recvCounts[proc] > 0)   // Discard empty counts, which includes ourselves.
      {
        recvCounts[numRecvProc++] = recvCounts[proc];  // Compacting.
        recvOffsets.push_back(recvTotal);
        recvProc.push_back(proc);
        recvTotal += recvCounts[proc];
      }

    }
    sendCounts.resize(numSendProc);
    recvCounts.resize(numRecvProc);

    // Preliminary receive will be into end of existing node list.
    points.resize(numUniquePoints + recvTotal);

    /// //TODO remove
    /// for (const TNP &pt : shareBuffer)
    ///   fprintf(stderr, "[%d] shBuf leastRank: o==%d, l==%u, coords==%s\n", rProc, pt.get_owner(), pt.getLevel(), pt.getBase32Hex().data());
    /// fprintf(stderr, "[[%d]]\n\n", rProc);


    //
    // Send and receive. Sends and receives may not be symmetric.
    //
    std::vector<MPI_Request> requestSend(numSendProc);
    std::vector<MPI_Request> requestRecv(numRecvProc);
    MPI_Status status;

    // Send data.
    for (int sIdx = 0; sIdx < numSendProc; sIdx++)
      par::Mpi_Isend(shareBuffer.data() + sendOffsets[sIdx], sendCounts[sIdx], sendProc[sIdx], 0, comm, &requestSend[sIdx]);

    // Receive data.
    for (int rIdx = 0; rIdx < numRecvProc; rIdx++)
      par::Mpi_Irecv(points.data() + numUniquePoints + recvOffsets[rIdx], recvCounts[rIdx], recvProc[rIdx], 0, comm, &requestRecv[rIdx]);

    // Wait for sends and receives.
    for (int sIdx = 0; sIdx < numSendProc; sIdx++)
      MPI_Wait(&requestSend[sIdx], &status);
    for (int rIdx = 0; rIdx < numRecvProc; rIdx++)
      MPI_Wait(&requestRecv[rIdx], &status);


    // Second local pass, classifying nodes as hanging or non-hanging.
    countCGNodes(&(*points.begin()), &(*points.end()), order, true);

    /// //TODO remove
    /// for (const TNP &pt : points)
    ///   fprintf(stderr, "[%d] leastRank: o==%d, l==%u, coords==%s\n", rProc, pt.get_owner(), pt.getLevel(), pt.getBase32Hex().data());
    /// fprintf(stderr, "[[%d]]\n", rProc);


    //TODO For a more sophisticated version, could sort just the new points (fewer),
    //     then merge (faster) two sorted lists before applying the resolver.


    //
    // Compute the scattermap.
    //
    // > For every non-hanging ("Yes") boundary node, determine ownership.
    // > For every hanging ("No") boundary node, insert all "base nodes" in parent cell to node list,
    //   marking them as neither "Yes" nor "No," but as a third tag, "Base".
    // > Sort the union of nodes according to the final desired ordering,
    //   so that duplicates are grouped together.
    //
    //   > Note: The number of possible base nodes equals the number of children
    //     who share the node, i.e. the number of duplicates from a single processor.
    //     If we had not collapsed duplicates before sending, we could have simply
    //     replaced sets of hanging nodes by their respective bases, rather than
    //     appending them. (We'd still need to re-sort though.)
    //
    // > For every non-hanging node,
    //   if it is a boundary node or a base of a boundary node, and if we own it,
    //   then insert the node with all borrowers into the scattermap.
    //   Note that borrowers might be listed more than once as we read the node list.
    //
    // > Compact the node list.
    //

    // Current ownership policy: Least-rank-processor (without the Base tag).

    // TODO 2019-03-07  Today we decided that a cell-decomposition approach would
    //   be cleaner, since don't have to deal with rounding problems.

    //TODO

    // Can compute # of owned nodes without the scattermap. At least we can
    // test the counting methods.

    RankI numOwnedPoints = 0;
    typename std::vector<TNP>::iterator ptIter = points.begin();
    while (ptIter < points.end())
    {
      // Skip hanging nodes.
      if (ptIter->get_isSelected() != TNP::Yes)
      {
        ptIter++;
        continue;
      }

      // A node marked 'Yes' is the first instance.
      // Examine all instances and choose the one with least rank.
      typename std::vector<TNP>::iterator leastRank = ptIter;
      while (ptIter < points.end() && *ptIter == *leastRank)  // Equality compares only coordinates and level.
      {
        ptIter->set_isSelected(TNP::No);
        if (ptIter->get_owner() < leastRank->get_owner())
          leastRank = ptIter;
        ptIter++;
      }

      /// //TODO remove
      /// fprintf(stderr, "[%d] leastRank: o==%d, l==%u, coords==%s\n", rProc, leastRank->get_owner(), leastRank->getLevel(), leastRank->getBase32Hex().data());

      // If the chosen node is ours, select it and increment count.
      if (leastRank->get_owner() == -1 || leastRank->get_owner() == rProc)
      {
        leastRank->set_isSelected(TNP::Yes);
        numOwnedPoints++;
      }
    }
    

    // With ownership, modify the local number, MPI_Allreduce to get global number.

    RankI numCGNodes = 0;
    par::Mpi_Allreduce(&numOwnedPoints, &numCGNodes, 1, MPI_SUM, comm);

    return numCGNodes;
  }


  //
  // SFC_NodeSort::countCGNodes()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::countCGNodes(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order, bool classify)
  {
    using TNP = TNPoint<T,dim>;
    constexpr char numChildren = TreeNode<T,dim>::numChildren;
    RankI totalUniquePoints = 0;
    RankI numDomBdryPoints = filterDomainBoundary(start, end);

    if (!(end > start))
      return 0;

    // Sort the domain boundary points. Root-1 level requires special handling.
    //
    std::array<RankI, numChildren+1> rootSplitters;
    RankI unused_ancStart, unused_ancEnd;
    SFC_Tree<T,dim>::template SFC_bucketing_impl<KeyFunIdentity_Pt<TNP>, TNP, TNP>(
        end-numDomBdryPoints, 0, numDomBdryPoints, 0, 0,
        KeyFunIdentity_Pt<TNP>(), false, true,
        rootSplitters,
        unused_ancStart, unused_ancEnd);
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      if (rootSplitters[child_sfc+1] - rootSplitters[child_sfc+0] <= 1)
        continue;

      locTreeSortAsPoints(
          end-numDomBdryPoints, rootSplitters[child_sfc+0], rootSplitters[child_sfc+1],
          1, m_uiMaxDepth, 0);    // Re-use the 0th rotation for each 'root'.
    }

    // Counting/resolving/marking task.
    if (classify)
    {
      // Count the domain boundary points.
      //
      for (TNP *bdryIter = end - numDomBdryPoints; bdryIter < end; bdryIter++)
        bdryIter->set_isSelected(TNP::No);
      RankI numUniqBdryPoints = 0;
      TNP *bdryIter = end - numDomBdryPoints;
      TNP *firstCoarsest, *unused_firstFinest;
      unsigned int unused_numDups;
      while (bdryIter < end)
      {
        scanForDuplicates(bdryIter, end, firstCoarsest, unused_firstFinest, bdryIter, unused_numDups);
        firstCoarsest->set_isSelected(TNP::Yes);
        numUniqBdryPoints++;
      }

      totalUniquePoints += numUniqBdryPoints;


      // Bottom-up counting interior points
      //
      if (order <= 2)
        totalUniquePoints += countCGNodes_impl(resolveInterface_lowOrder, start, end - numDomBdryPoints, 1, 0, order);
      else
        totalUniquePoints += countCGNodes_impl(resolveInterface_highOrder, start, end - numDomBdryPoints, 1, 0, order);
    }
    // Sorting/instancing task.
    else
    {
      countInstances(end - numDomBdryPoints, end, order);
      countCGNodes_impl(countInstances, start, end - numDomBdryPoints, 1, 0, order);
    }

    return totalUniquePoints;
  }


  //
  // SFC_NodeSort::filterDomainBoundary()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::filterDomainBoundary(TNPoint<T,dim> *start, TNPoint<T,dim> *end)
  {
    using TNP = TNPoint<T,dim>;

    // Counting phase.
    RankI numDomBdryPoints = 0;
    std::queue<TNP *> segSplitters;
    std::vector<TNP> bdryPts;
    for (TNP *iter = start; iter < end; iter++)
    {
      if (iter->isOnDomainBoundary())
      {
        numDomBdryPoints++;
        segSplitters.push(iter);
        bdryPts.push_back(*iter);
      }
    }

    // Movement phase.
    TNP *writepoint;
    if (!segSplitters.empty())
      writepoint = segSplitters.front();
    while (!segSplitters.empty())
    {
      const TNP * const readstart = segSplitters.front() + 1;
      segSplitters.pop();
      const TNP * const readstop = (!segSplitters.empty() ? segSplitters.front() : end);
      // Shift the next segment of non-boundary points.
      for (const TNP *readpoint = readstart; readpoint < readstop; readpoint++)
        *(writepoint++) = *readpoint;
    }

    for (const TNP &bdryPt : bdryPts)
    {
      *(writepoint++) = bdryPt;
    }

    return numDomBdryPoints;
  }


  //
  // SFC_NodeSort::locTreeSortAsPoints()
  //
  template<typename T, unsigned int dim>
  void
  SFC_NodeSort<T,dim>:: locTreeSortAsPoints(TNPoint<T,dim> *points,
                            RankI begin, RankI end,
                            LevI sLev,
                            LevI eLev,
                            RotI pRot)
  {
    using TNP = TNPoint<T,dim>;
    constexpr char numChildren = TreeNode<T,dim>::numChildren;
    constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

    // Lookup tables to apply rotations.
    const ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
    const RotI * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

    if (end <= begin) { return; }

    // Reorder the buckets on sLev (current level).
    std::array<RankI, numChildren+1> tempSplitters;
    RankI unused_ancStart, unused_ancEnd;
    SFC_Tree<T,dim>::template SFC_bucketing_impl<KeyFunIdentity_Pt<TNP>, TNP, TNP>(
        points, begin, end, sLev, pRot,
        KeyFunIdentity_Pt<TNP>(), false, true,
        tempSplitters,
        unused_ancStart, unused_ancEnd);

    if (sLev < eLev)  // This means eLev is further from the root level than sLev.
    {
      // Recurse.
      // Use the splitters to specify ranges for the next level of recursion.
      for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
      {
        // Check for empty or singleton bucket.
        if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] <= 1)
          continue;

        ChildI child = rot_perm[child_sfc];
        RotI cRot = orientLookup[child];

        // Check for identical coordinates.
        bool allIdentical = true;
        std::array<T,dim> first_coords;
        points[tempSplitters[child_sfc+0]].getAnchor(first_coords);
        for (RankI srchIdentical = tempSplitters[child_sfc+0] + 1;
            srchIdentical < tempSplitters[child_sfc+1];
            srchIdentical++)
        {
          std::array<T,dim> other_coords;
          points[srchIdentical].getAnchor(other_coords);
          if (other_coords != first_coords)
          {
            allIdentical = false;
            break;
          }
        }

        if (!allIdentical)
        {
          locTreeSortAsPoints(points,
              tempSplitters[child_sfc+0], tempSplitters[child_sfc+1],
              sLev+1, eLev,
              cRot);
        }
        else
        {
          // Separate points by level. Assume at most 2 levels present,
          // due to 2:1 balancing. The first point determines which level is preceeding.
          const RankI segStart = tempSplitters[child_sfc+0];
          const RankI segSize = tempSplitters[child_sfc+1] - segStart;
          LevI firstLevel = points[segStart].getLevel();
          RankI ptIdx = 0;
          RankI ptOffsets[2] = {0, 0};
          while (ptIdx < segSize)
            if (points[segStart + (ptIdx++)].getLevel() == firstLevel)
              ptOffsets[1]++;

          if (ptOffsets[1] != 1 && ptOffsets[1] != segSize)
          {
            RankI ptEnds[2] = {ptOffsets[1], segSize};

            TNP buffer[2];
            ptOffsets[0]++;  // First point already in final position.
            buffer[0] = points[segStart + ptOffsets[0]];
            buffer[1] = points[segStart + ptOffsets[1]];
            unsigned char bufferSize = 2;

            while (bufferSize > 0)
            {
              TNP &bufferTop = buffer[bufferSize-1];
              unsigned char destBucket = !(bufferTop.getLevel() == firstLevel);

              points[segStart + ptOffsets[destBucket]]= bufferTop;
              ptOffsets[destBucket]++;

              if (ptOffsets[destBucket] < ptEnds[destBucket])
                bufferTop = points[segStart + ptOffsets[destBucket]];
              else
                bufferSize--;
            }
          }
        }
      }
    }
  }// end function()


  //
  // SFC_NodeSort::scanForDuplicates()
  //
  template <typename T, unsigned int dim>
  void SFC_NodeSort<T,dim>::scanForDuplicates(
      TNPoint<T,dim> *start, TNPoint<T,dim> *end,
      TNPoint<T,dim> * &firstCoarsest, TNPoint<T,dim> * &firstFinest,
      TNPoint<T,dim> * &next, unsigned int &numDups)
  {
    std::array<T,dim> first_coords, other_coords;
    start->getAnchor(first_coords);
    next = start + 1;
    firstCoarsest = start;
    firstFinest = start;
    unsigned char numInstances = start->get_numInstances();
    numDups = 1;  // Something other than 0.
    while (next < end && (next->getAnchor(other_coords), other_coords) == first_coords)
    {
      numInstances += next->get_numInstances();
      if (numDups && next->getLevel() != firstCoarsest->getLevel())
        numDups = 0;
      if (next->getLevel() < firstCoarsest->getLevel())
        firstCoarsest = next;
      if (next->getLevel() > firstFinest->getLevel())
        firstFinest = next;
      next++;
    }
    if (numDups)
      numDups = numInstances;
  }


  template <typename T, unsigned int dim>
  struct ParentOfContainer
  {
    // PointType == TNPoint<T,dim>    KeyType == TreeNode<T,dim>
    TreeNode<T,dim> operator()(const TNPoint<T,dim> &pt) { return pt.getFinestOpenContainer(); }
  };


  //
  // SFC_NodeSort::countCGNodes_impl()
  //
  template <typename T, unsigned int dim>
  template<typename ResolverT>
  RankI SFC_NodeSort<T,dim>::countCGNodes_impl(
      ResolverT &resolveInterface,
      TNPoint<T,dim> *start, TNPoint<T,dim> *end,
      LevI sLev, RotI pRot,
      unsigned int order)
  {
    // Algorithm:
    //
    // Bucket points using the keyfun ParentOfContainer, and ancestors after sibilings.
    //   Points on our own interface end up in our `ancestor' bucket. We'll reconsider these later.
    //   Other points fall into the interiors of our children.
    // Recursively, for each child:
    //   Look up sfc rotation for child.
    //   Call countCGNodes_impl() on that child.
    //     The interfaces within all children are now resolved.
    // Reconsider our own interface.
    //   `Bucket' points on our interface, not by child, but by which hyperplane they reside on.
    //     Apply the test greedily, so that intersections have a consistent hyperplane choice.
    //   For each hyperplane:
    //     Sort the points (as points) in the SFC ordering, using ourselves as parent.
    //     Call resolveInterface() on the points in the hyperplane.

    using TNP = TNPoint<T,dim>;
    using TreeNode = TreeNode<T,dim>;
    constexpr char numChildren = TreeNode::numChildren;
    constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

    // Lookup tables to apply rotations.
    const ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
    const RotI * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

    if (!(start < end))
      return 0;

    RankI numUniqPoints = 0;

    // Reorder the buckets on sLev (current level).
    std::array<RankI, numChildren+1> tempSplitters;
    RankI ancStart, ancEnd;
    SFC_Tree<T,dim>::template SFC_bucketing_impl<ParentOfContainer<T,dim>, TNP, TreeNode>(
        start, 0, end-start, sLev, pRot,
        ParentOfContainer<T,dim>(), true, false,
        tempSplitters,
        ancStart, ancEnd);

    // Recurse.
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      // Check for empty or singleton bucket.
      if (tempSplitters[child_sfc+1] - tempSplitters[child_sfc+0] <= 1)
        continue;

      ChildI child = rot_perm[child_sfc];
      RotI cRot = orientLookup[child];

      numUniqPoints += countCGNodes_impl<ResolverT>(
          resolveInterface,
          start + tempSplitters[child_sfc+0], start + tempSplitters[child_sfc+1],
          sLev+1, cRot,
          order);
    }

    // Process own interface. (In this case hlev == sLev).
    std::array<RankI, dim+1> hSplitters;
    bucketByHyperplane(start + ancStart, start + ancEnd, sLev, hSplitters);
    for (int d = 0; d < dim; d++)
    {
      locTreeSortAsPoints(
          start + ancStart, hSplitters[d], hSplitters[d+1],
          sLev, m_uiMaxDepth, pRot);

      // The actual counting happens here.
      numUniqPoints += resolveInterface(start + ancStart + hSplitters[d], start + ancStart + hSplitters[d+1], order);
    }

    return numUniqPoints;
  }


  //
  // SFC_NodeSort::bucketByHyperplane()
  //
  template <typename T, unsigned int dim>
  void SFC_NodeSort<T,dim>::bucketByHyperplane(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int hlev, std::array<RankI,dim+1> &hSplitters)
  {
    // Compute offsets before moving points.
    std::array<RankI, dim> hCounts, hOffsets;
    hCounts.fill(0);
    for (TNPoint<T,dim> *pIter = start; pIter < end; pIter++)
      hCounts[pIter->get_firstIncidentHyperplane(hlev)]++;
    RankI accum = 0;
    for (int d = 0; d < dim; d++)
    {
      hOffsets[d] = accum;
      hSplitters[d] = accum;
      accum += hCounts[d];
    }
    hSplitters[dim] = accum;

    // Move points with a full size buffer.
    std::vector<TNPoint<T,dim>> buffer(end - start);
    for (TNPoint<T,dim> *pIter = start; pIter < end; pIter++)
      buffer[hOffsets[pIter->get_firstIncidentHyperplane(hlev)]++] = *pIter;
    for (auto &&pt : buffer)
      *(start++) = pt;
  }


  //
  // SFC_NodeSort::resolveInterface_lowOrder()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::resolveInterface_lowOrder(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order)
  {
    //
    // The low-order counting method is based on counting number of points per spatial location.
    // If there are points from two levels, then points from the higher level are non-hanging;
    // pick one of them as the representative (isSelected=Yes). The points from the lower
    // level are hanging and can be discarded (isSelected=No). If there are points all
    // from the same level, they are non-hanging iff we count pow(2,dim-cdim) instances, where
    // cdim is the dimension of the cell to which the points are interior. For example, in
    // a 4D domain, vertex-interior nodes (cdim==0) should come in sets of 16 instances per location,
    // while 3face-interior nodes (aka octant-interior nodes, cdim==3), should come in pairs
    // of instances per location.
    //
    // Given a spatial location, it is assumed that either all the instances generated for that location
    // are present on this processor, or none of them are.
    //
    using TNP = TNPoint<T,dim>;

    for (TNP *ptIter = start; ptIter < end; ptIter++)
      ptIter->set_isSelected(TNP::No);

    RankI totalCount = 0;
    TNP *ptIter = start;
    TNP *firstCoarsest, *unused_firstFinest;
    unsigned int numDups;
    while (ptIter < end)
    {
      scanForDuplicates(ptIter, end, firstCoarsest, unused_firstFinest, ptIter, numDups);
      if (!numDups)   // Signifies mixed levels. We have a winner.
      {
        firstCoarsest->set_isSelected(TNP::Yes);
        totalCount++;
      }
      else            // All same level and cell type. Test whether hanging or not.
      {
        unsigned char cdim = firstCoarsest->get_cellType().get_dim_flag();
        unsigned int expectedDups = 1u << (dim - cdim);
        if (numDups == expectedDups)
        {
          firstCoarsest->set_isSelected(TNP::Yes);
          totalCount++;
        }
      }
    }

    return totalCount;
  }


  //
  // SFC_NodeSort::resolveInterface_highOrder()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::resolveInterface_highOrder(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order)
  {
    //
    // The high-order counting method (order>=3) cannot assume that a winning node
    // will co-occur with finer level nodes in the same exact coordinate location.
    //
    // Instead, we have to consider open k-faces (k-cells), which generally
    // contain multiple nodal locations. We say an open k-cell is present when
    // an incident node of the same level is present. If an open k-cell is present,
    // it is non-hanging iff its parent open k'-cell is not present.
    //
    // # Locality property.
    // This algorithm relies on the `non-reentrant' property of SFC's visitng cells.
    // A corollary of non-reentrancy to K-cells (embedding dimension) is that,
    // once the SFC has entered a k-face of some level and orientation, it must
    // finish traversing the entire k-face before entering another k-face of
    // the same dimension and orientaton and of *coarser or equal level*.
    // The SFC is allowed to enter another k-face of finer level, but once
    // it has reached the finest level in a region, it cannot enter
    // a k-face of finer level. Therefore, considering a finer k-face in
    // a possibly mixed-level k-cell, the SFC length of the k-face is bounded
    // by a function of order and dimension only.
    //
    // # Overlap property.
    // This algorithm further relies on the order being 3 or higher.
    // Given an open k-cell C that is present, there are two cases.
    // - Case A: The parent P of the k-cell C has dimension k (C is `noncrossing').
    // - Case B: The parent P of the k-cell C has dimension k' > k (C is `crossing').
    // In Case A (C is noncrossing), if P is present, then at least one k-cell interior
    // node from P falls within the interior of C.
    // In Case B (C is crossing), then C is shared by k'-cell siblings under P;
    // if P is present, then, for each (k'-cell) child of P, at least one
    // k'-cell interior node from P falls within that child. Furthermore,
    // one of the k'-cell siblings under P shares the same K-cell anchor
    // as C.
    // --> This means we can detect the parent's node while traversing
    //     the child k-face, by checking levels of either k- or k'-cells.
    //
    // # Algorithm.
    // Combining the properties of SFC locality and high-order parent/child overlap,
    // we can determine for each k'-cell whether it is hanging or not, during a single
    // pass of the interface nodes by using a constant-sized buffer. (The buffer
    // size is a function of dimension and order, which for the current purpose
    // are constants).
    //
    // - Keep track of what K-cell we are traversing. Exiting the K-cell at any
    //   dimension means we must reset the rest of the state.
    //
    // - Keep track of which two distinct levels could be present on the k-faces
    //   of the current K-cell. There can be at most two distinct levels.
    //
    // - Maintain a table with one row per cell type (cell dimension, orientation).
    //   Each row is either uninitialized, or maintains the following:
    //   - the coarsest level of point hitherto witnessed for that cell type;
    //   - a list of pending points whose selection depends on the final coarseness of the k-face.
    //
    // - Iterate through all unique locations in the interface:
    //   - If the next point is contained in the current K-cell,
    //     - Find the point's native cell type, and update the coarseness of that cell type.
    //     - If the row becomes coarser while nodes are pending,
    //       - Deselect all pending nodes and remove them.
    //
    //     - If the two distinct levels in the K-cell are not yet known,
    //       and the new point does not differ in level from any seen before,
    //       - Append the point to a set of `unprocessed nodes'.
    //
    //     - Elif the two distinct levels in the K-cell are not yet known,
    //       and the new point differs in level from any seen before,
    //       - Put the new point on hold. Update the two distinct levels.
    //         Process all the unprocessed nodes (as defined next), then process the new point.
    //
    //     - Else, process the point:
    //       - If Lev(pt) == coarserLevel,
    //         - Set pt.isSelected := Yes.
    //       - Elif table[pt.parentCellType()].coarsestLevel == coarserLevel,
    //         - Set pt.isSelected := No.
    //       - Else
    //         - Append table.[pt.parentCellType()].pendingNodes <- pt
    //
    //   - Else, we are about to exit the current K-cell:
    //     - For all rows, select all pending points and remove them.
    //     - Reset the K-cell-wide state: Cell identity (anchor) and clear distinct levels.
    //     - Proceed with new point.
    //
    // - When we reach the end of the interface, select all remaining pending nodes.
    //

    using TNP = TNPoint<T,dim>;

    // Helper struct.
    struct kFaceStatus
    {
      void resetCoarseness()
      {
        m_isInitialized = false;
      }

      /** @brief Updates coarsenss to coarser or equal level. */
      void updateCoarseness(LevI lev)
      {
        if (!m_isInitialized || lev < m_curCoarseness)
        {
          m_curCoarseness = lev;
          m_isInitialized = true;
        }
      }

      RankI selectAndRemovePending()
      {
        RankI numSelected = m_pendingNodes.size();
        for (TNP *pt : m_pendingNodes)
          pt->set_isSelected(TNP::Yes);
        m_pendingNodes.clear();
        return numSelected;
      }

      void deselectAndRemovePending()
      {
        for (TNP *pt : m_pendingNodes)
        {
          pt->set_isSelected(TNP::No);
        }
        m_pendingNodes.clear();
      }

      // Data members.
      bool m_isInitialized = false;
      LevI m_curCoarseness;
      std::vector<TNP *> m_pendingNodes;
    };

    // Initialize table. //TODO make this table re-usable across function calls.
    int numLevelsWitnessed = 0;
    TreeNode<T,dim> currentKCell;
    LevI finerLevel;       // finerLevel and coarserLevel have meaning when
    LevI coarserLevel;     //   numLevelsWitnessed becomes 2, else see currentKCell.
    std::vector<kFaceStatus> statusTbl(1u << dim);   // Enough for all possible orientation indices.
    std::vector<TNP *> unprocessedNodes;

    RankI totalCount = 0;

    while (start < end)
    {
      // Get the next unique location (and cell type, using level as proxy).
      TNP *next = start + 1;
      while (next < end && (*next == *start))  // Compares both coordinates and level.
        (next++)->set_isSelected(TNP::No);
      start->set_isSelected(TNP::No);

      const unsigned char nCellType = start->get_cellType().get_orient_flag();
      kFaceStatus &nRow = statusTbl[nCellType];

      // First initialization of state for new K-cell.
      if (numLevelsWitnessed == 0 || !currentKCell.isAncestor(start->getDFD()))
      {
        // If only saw a single level, they are all non-hanging.
        totalCount += unprocessedNodes.size();
        for (TNP * &pt : unprocessedNodes)
          pt->set_isSelected(TNP::Yes);
        unprocessedNodes.clear();

        // We probably have some pending nodes to clean up.
        for (kFaceStatus &row : statusTbl)
        {
          totalCount += row.selectAndRemovePending();
          row.resetCoarseness();
        }

        // Initialize state to traverse new K-cell.
        currentKCell = start->getCell();
        numLevelsWitnessed = 1;
      }

      // Second initialization of state for new K-cell.
      if (numLevelsWitnessed == 1)
      {
        if (start->getLevel() < currentKCell.getLevel())
        {
          coarserLevel = start->getLevel();
          finerLevel = currentKCell.getLevel();
          numLevelsWitnessed = 2;
        }
        else if (start->getLevel() > currentKCell.getLevel())
        {
          coarserLevel = currentKCell.getLevel();
          finerLevel = start->getLevel();
          numLevelsWitnessed = 2;
          currentKCell = start->getCell();
        }
      }

      // Read from current node to update nRow.
      nRow.updateCoarseness(start->getLevel());
      if (numLevelsWitnessed == 2 && nRow.m_curCoarseness == coarserLevel)
        nRow.deselectAndRemovePending();

      // Enqueue node for later interaction with pRow.
      unprocessedNodes.push_back(start);

      // Process all recently added nodes.
      if (numLevelsWitnessed == 2)
      {
        for (TNP * &pt : unprocessedNodes)
        {
          const unsigned char pCellType = pt->get_cellTypeOnParent().get_orient_flag();
          kFaceStatus &pRow = statusTbl[pCellType];

          if (pt->getLevel() == coarserLevel)
          {
            pt->set_isSelected(TNP::Yes);
            totalCount++;
          }
          else if (pRow.m_isInitialized && pRow.m_curCoarseness == coarserLevel)
          {
            pt->set_isSelected(TNP::No);
          }
          else
          {
            pRow.m_pendingNodes.push_back(pt);
          }
        }
        unprocessedNodes.clear();
      }

      start = next;
    }

    // Finally, flush remaining unprocessed or pending nodes in table.
    totalCount += unprocessedNodes.size();
    for (TNP * &pt : unprocessedNodes)
      pt->set_isSelected(TNP::Yes);
    unprocessedNodes.clear();

    for (kFaceStatus &row : statusTbl)
      totalCount += row.selectAndRemovePending();

    return totalCount;
  }


  //
  // SFC_NodeSort::countInstances()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::countInstances(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int unused_order)
  {
    using TNP = TNPoint<T,dim>;
    while (start < end)
    {
      // Get the next unique pair (location, level).
      unsigned char delta_numInstances = 0;
      TNP *next = start + 1;
      while (next < end && (*next == *start))  // Compares both coordinates and level.
      {
        delta_numInstances += next->get_numInstances();
        next->set_numInstances(0);
        next++;
      }
      start->incrementNumInstances(delta_numInstances);

      start = next;
    }

    return 0;
  }


  //
  // SFC_NodeSort::dist_bcastSplitters()
  //
  template <typename T, unsigned int dim>
  std::vector<TreeNode<T,dim>> SFC_NodeSort<T,dim>::dist_bcastSplitters(const TreeNode<T,dim> *start, MPI_Comm comm)
  {
    int nProc, rProc;
    MPI_Comm_rank(comm, &rProc);
    MPI_Comm_size(comm, &nProc);

    using TreeNode = TreeNode<T,dim>;
    std::vector<TreeNode> splitters(nProc);
    splitters[rProc] = *start;

    for (int turn = 0; turn < nProc; turn++)
      par::Mpi_Bcast<TreeNode>(&splitters[turn], 1, turn, comm);

    return splitters;
  }

  //
  // SFC_NodeSort::getProcNeighbours()
  //
  template <typename T, unsigned int dim>
  int SFC_NodeSort<T,dim>::getProcNeighbours(TNPoint<T,dim> pt,
      const TreeNode<T,dim> *splitters, int numSplitters,
      std::vector<int> &procNbList)
  {
    std::vector<TreeNode<T,dim>> keyList(intPow(3,dim));    // Allocate then shrink.
    keyList.clear();
    pt.getDFD().appendAllNeighboursAsPoints(keyList);  // Includes domain boundary points.
    
    int procNbListSizeOld = procNbList.size();
    SFC_Tree<T,dim>::getContainingBlocks(keyList.data(), 0, (int) keyList.size(), splitters, numSplitters, procNbList);
    int procNbListSize = procNbList.size();

    return procNbListSize - procNbListSizeOld;
  }


  //
  // SFC_NodeSort:: resolveInterface_scattermapStruct
  //     operator()   (resolver)
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::resolveInterface_scattermapStruct::
      operator() (TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order)
  {
    //TODO
    return 0;
  }

  //
  // SFC_NodeSort:: resolveInterface_scattermapStruct
  //     computeScattermap()
  //
  template <typename T, unsigned int dim>
  void SFC_NodeSort<T,dim>::resolveInterface_scattermapStruct::
      computeScattermap(std::vector<RankI> &outScattermap, std::vector<RankI> &outSendCounts, std::vector<RankI> &outSendOffsets)
  {
    //TODO
  }





  // ============================ End: SFC_NodeSort ============================ //

}//namespace ot
