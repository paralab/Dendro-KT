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
    TNP *firstCoarsest;
    unsigned int unused_numDups;
    while (bdryIter < end)
    {
      scanForDuplicates(bdryIter, end, firstCoarsest, bdryIter, unused_numDups);
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
          /* We could arrange them by level, but don't have to. */
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
      TNPoint<T,dim> * &firstCoarsest, TNPoint<T,dim> * &next, unsigned int &numDups)
  {
    std::array<T,dim> first_coords, other_coords;
    start->getAnchor(first_coords);
    next = start + 1;
    firstCoarsest = start;
    numDups = 1;  // Something other than 0.
    while (next < end && (next->getAnchor(other_coords), other_coords) == first_coords)
    {
      if (numDups && next->getLevel() != firstCoarsest->getLevel())
        numDups = 0;
      if (next->getLevel() < firstCoarsest->getLevel())
        firstCoarsest = next;
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
    TNP *firstCoarsest;
    unsigned int numDups;
    while (ptIter < end)
    {
      scanForDuplicates(ptIter, end, firstCoarsest, ptIter, numDups);
      if (!numDups)   // Signifies mixed levels. We have a winner.
      {
        /// std::cout << "Mixed Levels!\n";
        firstCoarsest->set_isSelected(TNP::Yes);
        totalCount++;
      }
      else            // All same level and cell type. Test whether hanging or not.
      {
        /// std::cout << "<----\n";
        unsigned char cdim = firstCoarsest->get_cellType().get_dim_flag();
        unsigned int expectedDups = 1u << (dim - cdim);
        /// std::cout << firstCoarsest->getBase32Hex(5).data()
        ///     << "(" << firstCoarsest->getFinestOpenContainer().getBase32Hex().data() << ")"
        ///     << ":  \t";
        /// std::cout << (int) cdim << "\t";
        /// std::cout << std::bitset<dim>(firstCoarsest->get_cellType().get_orient_flag()).to_string() << "\t";
        /// std::cout << "numDups==" << numDups << "\t";
        if (numDups == expectedDups)
        {
          firstCoarsest->set_isSelected(TNP::Yes);
          totalCount++;
          /// std::cout << "Not hanging!\n";
          /// std::cout << "\n";
        }
        else
        {
          /// std::cout << "Hanging!\n";
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
    //TODO
    return 0;
  }

  // ============================ End: SFC_NodeSort ============================ //

}//namespace ot
