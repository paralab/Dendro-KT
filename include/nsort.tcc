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
      TreeNode<T,dim>(other), m_isSelected(other.m_isSelected)
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
    using TreeNode = TreeNode<T,dim>;
    const unsigned int len = 1u << (m_uiMaxDepth - TreeNode::m_uiLevel);
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

  // ============================ End: TNPoint ============================ //


  // ============================ Begin: SFC_NodeSort ============================ //

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

  // ============================ End: TNPoint ============================ //



  // ============================ Begin: SFC_NodeSort ============================ //


  //
  // SFC_NodeSort::countCGNodes()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::countCGNodes(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order)
  {
    using TNP = TNPoint<T,dim>;
    constexpr char numChildren = TreeNode<T,dim>::numChildren;
    RankI totalUniquePoints = 0;
    RankI numDomBdryPoints = filterDomainBoundary(start, end);


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
    numDups = 1;  // Something other than 0.
    while (next < end && (next->getAnchor(other_coords), other_coords) == first_coords)
    {
      if (numDups && next->getLevel() != firstCoarsest->getLevel())
        numDups = 0;
      if (next->getLevel() < firstCoarsest->getLevel())
        firstCoarsest = next;
      if (next->getLevel() > firstFinest->getLevel())
        firstFinest = next;
      next++;
    }
    if (numDups)
      numDups = next - start;
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
      ResolverT resolveInterface,
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
    // Instead, we have to consider open k'-faces (k'-cells), which generally
    // contain multiple nodal locations. We say an open k'-cell is present when
    // an incident node of the same level is present. If an open k'cell is present,
    // it is non-hanging iff its parent open k'-cell is not present.
    //
    // # Locality property.
    // This algorithm relies on the `non-reentrant' property of SFC's visitng cells.
    // A corollary of non-reentrancy to K-cells (embedding dimension) is that,
    // once the SFC has entered a k'-face of some level and orientation, it must
    // finish traversing the entire k'-face before entering another k'-face of
    // the same dimension and orientaton and of *coarser or equal level*.
    // The SFC is allowed to enter another k'-face of finer level, but once
    // it has reached the finest level in a region, it cannot enter
    // a k'-face of finer level. Therefore, considering a finer k'-face in
    // a possibly mixed-level k'-cell, the SFC length of the k'-face is bounded
    // by a function of order and dimension only.
    //
    // # Overlap property.
    // This algorithm further relies on the order being 3 or higher. Given an
    // open k'-cell that is present, if its parent k'-cell is also present,
    // then at least one k'-cell-interior node from the parent level must
    // fall within the interior of the child k'-cell. This means we can detect
    // the parent's node while traversing the child k'-face.
    //
    // # Algorithm.
    // Combining the properties of SFC locality and high-order parent/child overlap,
    // we can determine for each k'-cell whether it is hanging or not, during a single
    // pass of the interface nodes by using a constant-sized buffer. (The buffer
    // size is a function of dimension and order, which for the current purpose
    // are constants).
    //
    // - Maintain a table with one row per cell type (cell dimension, orientation).
    //   Each row has a current cell level and identity (represented as a TreeNode
    //   in conjunction with the row number==orientation), a current selection status
    //   (Yes/No/Maybe), and a list of pointers to pending ("Maybe") nodes.
    //   - If the row has status "Yes" or "No" then there are no pending nodes.
    //
    // - Iterate through all unique locations in the interface. For each one,
    //   get the cell type of the point and use it to look up the appropriate row
    //   in the table. Take one of the following branches:
    //   - If the point is contained in the row's current cell identity:
    //     - If the point has coarser level than the row cell:
    //       - Set: pt.isSelected=Yes
    //       - If (row.isSelected == Maybe)  //(can be Maybe or No, but not Yes)
    //         - Set all pending points to: isSelected=No, and remove them.
    //         - Set: row.isSelected=No
    //     - Elif the point has finer level than the row cell:
    //       - Set: pt.isSelected=No
    //       - If (row.isSelected == Maybe)  //(can really only be Maybe, not No or Yes)
    //         - Set all pending points to: isSelected=Yes, and remove them.
    //         - Set: row cell identity to pt.cell
    //         - Set: row.isSelected=No
    //     - Else, the point has same level as row cell:
    //       - If (row.isSelected != Maybe):
    //         - Set: pt.isSelected=(row.isSelected)
    //       - Else, (row.isSelected == Maybe):
    //         - Append pointer to pt into row.
    //
    //   - Else, the point is NOT contained in the row's current cell identity:
    //     - If (row.isSelected == Maybe):
    //       - Set all pending points to: isSelected=Yes, and remove them.
    //     - Set: row identity to pt.cell
    //     - Set: row.isSelected=Maybe
    //     - Append pointer to pt into row.
    //
    // - When we reach the end of the interface, set all remaining pending nodes
    //   to: isSelected=Yes.
    //

    using TNP = TNPoint<T,dim>;

    // Helper struct.
    struct kFaceStatus
    {
      void initialize(const TreeNode<T,dim> &cellIdentity, typename TNP::IsSelected status)
      {
        m_cellIdentity = cellIdentity;
        m_isSelected = status;
        m_isInitialized = true;
      }

      void selectAndRemovePending()
      {
        for (TNP *pt : m_pendingNodes)
          pt->set_isSelected(TNP::Yes);
        m_pendingNodes.clear();
      }

      void deselectAndRemovePending()
      {
        for (TNP *pt : m_pendingNodes)
          pt->set_isSelected(TNP::No);
        m_pendingNodes.clear();
      }

      bool m_isInitialized = false;
      TreeNode<T,dim> m_cellIdentity;
      typename TNP::IsSelected m_isSelected;
      std::vector<TNP *> m_pendingNodes;
    };

    // Initialize table. //TODO make this table re-usable across function calls.
    std::vector<kFaceStatus> statusTbl(1u << dim);   // Enough for all possible orientation indices.

    while (start < end)
    {
      // Get the next unique location.
      TNP *firstCoarsest, *firstFinest, *next;
      unsigned int numDups;
      scanForDuplicates(start, end, firstCoarsest, firstFinest, next, numDups);
      for (TNP *ptIter = start; ptIter < next; ptIter++)
        ptIter->set_isSelected(TNP::No);
      /// firstCoarsest->set_isSelected(TNP::Maybe);

      // Look up appropriate row.
      unsigned char cOrient = firstCoarsest->get_cellType().get_orient_flag();
      kFaceStatus &row = statusTbl[cOrient];

      if (!row.m_isInitialized)
        row.initialize(firstFinest->getCell(), (!numDups ? TNP::No : TNP::Maybe));

      // TODO we actually do need the points to be separated by level so that
      // we get the cell type right.


      start = next;
    }

    return 0;
  }

  // ============================ End: SFC_NodeSort ============================ //

}//namespace ot
