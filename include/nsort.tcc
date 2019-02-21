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
    assert(!isTouchingDomainBoundary());  // When we bucket we need to set aside boundary points first.

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
  // SFC_NodeSort::countCGNodes_lowOrder()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::countCGNodes_lowOrder(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order)
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
    RankI totalCount = 0u;
    //TODO

    return totalCount;
  }


  //
  // SFC_NodeSort::countCGNodes_highOrder()
  //
  template <typename T, unsigned int dim>
  RankI SFC_NodeSort<T,dim>::countCGNodes_highOrder(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order)
  {
    //TODO
    return 0;
  }

  // ============================ End: SFC_NodeSort ============================ //

}//namespace ot
