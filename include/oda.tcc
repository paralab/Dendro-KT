/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando.
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class.
 * @date 04/04/2019
 **/

namespace ot
{

    namespace util
    {
      template <typename C, unsigned int dim>
      void constructRegularGrid(MPI_Comm comm, unsigned int grainSz, double sfc_tol, std::vector<ot::TreeNode<C,dim>> &outTree)
      {
        int nProc, rProc;
        MPI_Comm_size(comm, &nProc);
        MPI_Comm_rank(comm, &rProc);

        if (grainSz == 0)
          grainSz = 1;

        // numElements == pow(2, dim*endL); --> endL = roundUp(log(numElements)/dim);
        const unsigned int endL = (binOp::binLength(nProc*grainSz - 1) + dim - 1) / dim;
        const unsigned int numElem1D = 1u << endL;
        const unsigned int globNumElem = 1u << (endL*dim);
        const unsigned int len = 1u << (m_uiMaxDepth - endL);

        // To make a distributed regular grid (regular implies balanced),
        // follow the lexicographic order and then use distTreeSort().
        const unsigned int locEleCount = globNumElem / nProc + (rProc < globNumElem % nProc ? 1 : 0);
        const unsigned int locEleRank = globNumElem / nProc * rProc + (rProc < globNumElem % nProc ? rProc : globNumElem % nProc);

        std::array<C,dim> eleMultiIdx;
        unsigned int r = locEleRank, q = 0;
        eleMultiIdx[0] = 1;
        for (int d = 1; d < dim; d++)                // Build up strides.
          eleMultiIdx[d] = eleMultiIdx[d-1] * numElem1D;

        for (int d = 0; d < dim; d++)                // Compute start coords.
        {
          q = r / eleMultiIdx[dim-1 - d];
          r = r % eleMultiIdx[dim-1 - d];
          eleMultiIdx[dim-1 - d] = q;
        }

        // Create part of the tree in lexicographic order.
        outTree.resize(locEleCount);
        for (unsigned int ii = 0; ii < locEleCount; ii++)
        {
          std::array<C,dim> eleCoords;
          for (int d = 0; d < dim; d++)
            eleCoords[d] = len * eleMultiIdx[d];
          outTree[ii] = ot::TreeNode<C,dim>(1, eleCoords, endL);

          incrementBaseB<C,dim>(eleMultiIdx, numElem1D);    // Lexicographic advancement.
        }

        SFC_Tree<C,dim>::distTreeSort(outTree, sfc_tol, comm);
      }

    }//namespace ot::util



    template <unsigned int dim>
    DendroIntL constructRegularSubdomainDA(DA<dim> &newSubDA,
                                     unsigned int level,
                                     std::array<unsigned int, dim> extentPowers,
                                     unsigned int eleOrder,
                                     MPI_Comm comm,
                                     double sfc_tol)
    {
      using C = typename DA<dim>::C;

      newSubDA.m_refel = RefElement(dim, eleOrder);

      // =========================
      // Outline
      // =========================
      // * Generate tree in Morton order.
      // * Partition tree using sfc_tol
      // * Store front/back tree elements, discard tree.
      //
      // * Generate all the nodes in Morton order and assign correct extant cell flag.
      // * Partition the nodes to agree with tree splitters.
      //
      // * Compute scatter/gather maps (other overload of construct()).
      // =========================

      constexpr unsigned int NUM_CHILDREN = 1u << dim;
      constexpr unsigned int rotOffset = 2*NUM_CHILDREN;  // num columns in rotations[].

      int nProc, rProc;
      MPI_Comm_size(comm, &nProc);
      MPI_Comm_rank(comm, &rProc);

      for (int d = 0; d < dim; d++)
      {
        if (level < extentPowers[d])
        {
          fprintf(stderr, "WARNING [constructRegularSubdomainDA()]: "
                          "level(%u) must be >= extentPowers(%u). "
                          "Increasing level to %u to compensate.\n",
                          level, extentPowers[d], extentPowers[d]);
          level = extentPowers[d];
        }
      }

      // TODO we can easily accept the subdomain box coordinates as parameters.

      // Establish the subdomain box.
      const unsigned int elemSz = 1u << (m_uiMaxDepth - level);
      DendroIntL totalNumElements = 1;
      std::array<C,dim> subdomainBBMins;  // uiCoords.
      std::array<C,dim> subdomainBBMaxs;  //
      #pragma unroll(dim)
      for (int d = 0; d < dim; d++)
      {
        subdomainBBMins[d] = 0u;
        subdomainBBMaxs[d] = elemSz << extentPowers[d];
        totalNumElements *= (1u << extentPowers[d]);
      }

      // Need to find treePartFront and treePartBack.
      ot::TreeNode<C,dim> treePartFront, treePartBack;
      DendroIntL locNumActiveElements = totalNumElements;

      /// if (nProc > 1)
      /// {
      ///   // In the distributed case, need to simulate distTreePartition.

      ///   // Search parameters.
      ///   const DendroIntL idealSplitterRankPrev = double(totalNumElements) * (rProc) / nProc;
      ///   const DendroIntL idealSplitterRank     = double(totalNumElements) * (rProc+1) / nProc;
      ///   const DendroIntL accptSplitterRankMin =  idealSplitterRankPrev * sfc_tol + idealSplitterRank * (1-sfc_tol);
      ///   const DendroIntL accptSplitterRankMax =  idealSplitterRankPrev * -sfc_tol + idealSplitterRank * (1+sfc_tol);

      ///   // Search for our end splitter.
      ///   ot::TreeNode<C,dim> subtree;
      ///   ot::TreeNode<C,dim> afterSubtree;  // If there is a split, the original value won't matter.
      ///   ot::RotI pRot = 0;
      ///   ot::RotI afterPRot = 0;
      ///   DendroIntL parentPreRank = 0;
      ///   DendroIntL parentPostRank = totalNumElements;

      ///   while (!(accptSplitterRankMin <= parentPostRank
      ///                                 && parentPostRank <= accptSplitterRankMax)
      ///       && subtree.getLevel() < m_uiMaxDepth)
      ///   {
      ///     // Lookup tables to apply rotations.
      ///     const ot::ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*NUM_CHILDREN];
      ///     const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*NUM_CHILDREN];

      ///     ot::TreeNode<C,dim> childSubtree;
      ///     ot::ChildI child_sfc, child_m;
      ///     DendroIntL childPreRank = parentPreRank;
      ///     DendroIntL childPostRank;

      ///     // Get child subtrees and compute child ranks.
      ///     for (child_sfc = 0; child_sfc < NUM_CHILDREN; child_sfc++)
      ///     {
      ///        child_m = rot_perm[child_sfc];
      ///        childSubtree = subtree.getChildMorton(child_m);

      ///        DendroIntL overlapNumElements = 1u;
      ///        for (int d = 0; d < dim; d++)
      ///        {
      ///          C x, y;

      ///          /// C overlapMax = min(childSubtree.maxX(d), subdomainBBMaxs[d]);
      ///          x = childSubtree.maxX(d);
      ///          y = subdomainBBMaxs[d];
      ///          C overlapMax = (x <= y ? x : y);

      ///          /// C overlapMin = max(childSubtree.minX(d), subdomainBBMins[d]);
      ///          x = childSubtree.minX(d);
      ///          y = subdomainBBMins[d];
      ///          C overlapMin = (x >= y ? x : y);

      ///          overlapNumElements *= (overlapMax - overlapMin) >> (m_uiMaxDepth - level);
      ///        }

      ///        childPostRank = childPreRank + overlapNumElements;

      ///        if (childPreRank < idealSplitterRank
      ///                        && idealSplitterRank <= childPostRank)
      ///          break;
      ///        else
      ///          childPreRank = childPostRank;
      ///     }

      ///     // Descend to the child that contains the ideal splitter rank.

      ///     // Before erasing the parent, do bookkeeping for element after new subtree.
      ///     if (child_sfc < NUM_CHILDREN)
      ///     {
      ///       afterSubtree = subtree.getChildMorton(rot_perm[child_sfc+1]);
      ///       afterPRot = orientLookup[ rot_perm[child_sfc+1] ];
      ///     }
      ///     else
      ///     {
      ///       afterSubtree = afterSubtree.getChildMorton(rotations[afterPRot*rotOffset + 0]);
      ///       afterPRot = HILBERT_TABLE[afterPRot*NUM_CHILDREN + rotations[afterPRot*rotOffset + 0]];
      ///     }

      ///     subtree = childSubtree;
      ///     pRot = orientLookup[child_m];
      ///     parentPreRank = childPreRank;
      ///     parentPostRank = childPostRank;
      ///   }

      ///   // Share splitter ranks to determine who owns what part of the tree.
      ///   parentPreRank = 0;
      ///   treePartBack = subtree;

      ///   // treePartFront of next proc and treePartBack of us.
      ///   SFC_Tree<C,dim>::firstDescendant(afterSubtree, afterPRot, level);
      ///   SFC_Tree<C,dim>::lastDescendant(treePartBack, pRot, level);

      ///   MPI_Request reqRank, reqFrontTN;
      ///   MPI_Status status;
      ///   if (rProc < nProc - 1)
      ///   {
      ///     par::Mpi_Isend(&parentPostRank, 1, rProc+1, 24, comm, &reqRank);
      ///     par::Mpi_Isend(&afterSubtree, 1, rProc+1, 13, comm, &reqFrontTN);
      ///   }
      ///   if (rProc > 0)
      ///   {
      ///     par::Mpi_Recv(&parentPreRank, 1, rProc-1, 24, comm, &status);
      ///     par::Mpi_Recv(&treePartFront, 1, rProc-1, 13, comm, &status);
      ///   }
      ///   if (rProc < nProc - 1)
      ///   {
      ///     MPI_Wait(&reqRank, &status);
      ///     MPI_Wait(&reqFrontTN, &status);
      ///   }

      ///   locNumActiveElements = parentPostRank - parentPreRank;
      /// }
      /// else
      /// {
      ///   ot::RotI tpFrontRot = 0;
      ///   ot::RotI tpBackRot = 0;
      ///   SFC_Tree<C,dim>::firstDescendant(treePartFront, tpFrontRot, level);
      ///   SFC_Tree<C,dim>::lastDescendant(treePartBack, tpBackRot, level);
      /// }

      // ------------------------------------------


      /// // Simpler way for debugging.

      // Generate elements in lexicographic order.
      DendroIntL genPart = totalNumElements / nProc +
                           (rProc < totalNumElements % nProc);
      DendroIntL genStart = (totalNumElements / nProc) * rProc +
                            (rProc < totalNumElements % nProc ? rProc : totalNumElements % nProc);
      std::array<C,dim> genLimits;
      std::array<C,dim> genStrides;
      std::array<C,dim> genIdx;
      for (int d = 0; d < dim; d++)
        genLimits[d] = 1u << extentPowers[d];
      genStrides[0] = 1;
      for (int d = 1; d < dim; d++)
        genStrides[d] = genStrides[d-1] * genLimits[d-1];
      DendroIntL remainder = genStart;
      for (int d = dim-1; d >= 0; d--)
      {
        genIdx[d] = remainder / genStrides[d];
        remainder -= genIdx[d] * genStrides[d];
      }

      /// fprintf(stderr, "[%d] Generate starting at (%llu, %llu, %llu)\n",
      ///     rProc, genIdx[0], genIdx[1], genIdx[2]);

      std::vector<ot::TreeNode<C, dim>> treePart;
      ot::TreeNode<C, dim> elem;
      elem.setLevel(level);
      for (DendroIntL ii = 0; ii < genPart; ii++)
      {
        for (int d = 0; d < dim; d++)
          elem.setX(d, genIdx[d] * elemSz);

        treePart.emplace_back(elem);

        incrementFor<C,dim>(genIdx, genLimits);
      }

      /// fprintf(stderr, "[%d] Ended generating at (%llu, %llu, %llu)\n",
      ///     rProc, genIdx[0], genIdx[1], genIdx[2]);

      bool isActive0 = (treePart.size() > 0);
      MPI_Comm activeComm0;
      MPI_Comm_split(comm, (treePart.size() ? 1 : MPI_UNDEFINED), rProc, &activeComm0);

      /// for (DendroIntL ii = 0; ii < treePart.size(); ii++)
      ///   fprintf(stderr, "%*slev==%u, coords==%s\n", 30*rProc, "",
      ///       treePart[ii].getLevel(), ot::dbgCoordStr(treePart[ii], 2).c_str());

      if (isActive0)
        SFC_Tree<C, dim>::distTreeSort(treePart, sfc_tol, activeComm0);

      locNumActiveElements = treePart.size();
      treePartFront = treePart.front();
      treePartBack = treePart.back();

      // ------------------------------------------


      /// DendroIntL locNumEle = locNumActiveElements;
      /// DendroIntL globNumEle = 0;
      /// par::Mpi_Allreduce(&locNumEle, &globNumEle, 1, MPI_SUM, comm);
      /// fprintf(stderr, "[%d] during construction: globNumEle==%llu\n", rProc, globNumEle);


      // Now we know how many elements would go to each proc,
      // and what is the front and back of the local entire partition.

      // Next, generate points.

      // A processor is 'active' if it has elements, otherwise 'inactive'.
      bool isActive = (locNumActiveElements > 0);
      MPI_Comm activeComm;
      MPI_Comm_split(comm, (isActive ? 1 : MPI_UNDEFINED), rProc, &activeComm);

      std::vector<ot::TNPoint<C,dim>> nodeList;
      std::vector<ot::TNPoint<C,dim>> nodeListBuffer;

      DendroIntL locTraversedEle = 0;
      DendroIntL globTraversedEle = 0;

      if (isActive)
      {
        // Generate local points.
        // Do this by traversing the element tree and adding points owned by each element.

        // Invariant: A TreeNode on the stack contains a descendant subtree
        //            that corresponds to a locally owned element.

        /// fprintf(stderr, "%*s [%d] Front==(%d)%s  Back==(%d)%s\n",
        ///     15*rProc, "", rProc,
        ///     treePartFront.getLevel(), treePartFront.getBase32Hex().data(),
        ///     treePartBack.getLevel(), treePartBack.getBase32Hex().data());

        std::vector<ot::TreeNode<C,dim>> treeStack;
        std::vector<ot::RotI> rotStack;
        treeStack.emplace_back();      // Root.
        rotStack.push_back(0);         //

        while (treeStack.size())
        {
          // Pop the next subtree from the stack.
          const ot::TreeNode<C,dim> subtree = treeStack.back();
          const ot::RotI pRot = rotStack.back();
          treeStack.pop_back();
          rotStack.pop_back();

          /// fprintf(stderr, "%*s [%d] next subtree: (%d)%s\n",
          ///     15*rProc, "", rProc, subtree.getLevel(), subtree.getBase32Hex().data());

          //
          // Leaf: A local element. Add its points.
          //
          if (subtree.getLevel() == level)
          {
            locTraversedEle++;

            const ot::Element<C,dim> element(subtree);
            DendroIntL nodeListOldSz = nodeList.size();

            // Neighbour flags of interior points label the host cell as neighbour 0.
            // Because the neighbourhood has dimension 0, no other cells are tested.
            element.appendInteriorNodes(eleOrder, nodeList);
            for (TNPoint<C,dim> *nodePtr = &nodeList[nodeListOldSz];
                nodePtr < &(*nodeList.end()); nodePtr++)
            {
              nodePtr->resetExtantCellFlagNoNeighbours();
              nodePtr->addNeighbourExtantCellFlag(0);
            }

            // For neighbour flags of exterior points, test each axis for boundary.
            // On an axis that is 'internal', none of the 1-side neighbours are marked.
            nodeListBuffer.clear();
            element.appendExteriorNodes(eleOrder, nodeListBuffer);
            for (TNPoint<C,dim> &node : nodeListBuffer)
            {
              node.resetExtantCellFlagAllNeighbours();

              // Exclude a node from nodeList if it is 'upperEdge' on an axis,
              // unless it is boundary on the same axis.
              bool excludeNode = false;
              for (int d = 0; d < dim; d++)
              {
                bool isBoundaryOnAxis = false;

                // Check for subdomain boundaries.
                if (node.getX(d) == subdomainBBMins[d])
                {
                  node.excludeSideExtantCellFlag(d, 0);
                  isBoundaryOnAxis = true;
                }
                else if (node.getX(d) == subdomainBBMaxs[d])
                {
                  node.excludeSideExtantCellFlag(d, 1);
                  isBoundaryOnAxis = true;
                }

                // Check for upper edge of the cell (owned by another cell).
                if (node.getX(d) == element.maxX(d) && !isBoundaryOnAxis)
                  excludeNode = true;

                // Check for interiorness.
                else if (element.minX(d) < node.getX(d)
                                        && node.getX(d) < element.maxX(d))
                  node.excludeSideExtantCellFlag(d, 1);
              }

              if (!excludeNode)
                nodeList.emplace_back(node);
            }
          }


          //
          // Nonleaf: Ancestor of a local element. Push its children onto stack.
          //
          else
          {
            const bool ancestorOfFront = subtree.isAncestor(treePartFront);
            const bool ancestorOfBack = subtree.isAncestor(treePartBack);

            const ot::ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*NUM_CHILDREN];
            const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*NUM_CHILDREN];

            // Push children onto stack in reverse sfc order.
            // There can be 0, 1, or 2 thresholds of entering/leaving local partition.
            bool keepNextChild = !ancestorOfBack;
            for (ot::ChildI child_rev = 0; child_rev < NUM_CHILDREN; child_rev++)
            {
              const ot::ChildI child_sfc = NUM_CHILDREN-1 - child_rev;
              const ot::ChildI child_m = rot_perm[child_sfc];
              const ot::RotI cRot = orientLookup[child_m];
              ot::TreeNode<C,dim> child = subtree.getChildMorton(child_m);

              bool intersectsSubdomainBB = true;
              for (int d = 0; d < dim; d++)
              {
                if (subdomainBBMaxs[d] <= child.minX(d) ||
                    child.maxX(d) <= subdomainBBMins[d])
                {
                  intersectsSubdomainBB = false;
                  break;
                }
              }


              if (child.isAncestorInclusive(treePartBack))
                keepNextChild = true;

              if (keepNextChild && intersectsSubdomainBB)
              {
                treeStack.emplace_back(child);
                rotStack.push_back(cRot);
              }

              if (child.isAncestorInclusive(treePartFront))
                keepNextChild = false;
            }
          }
        }

      }

      /// par::Mpi_Allreduce(&locTraversedEle, &globTraversedEle, 1, MPI_SUM, comm);
      /// fprintf(stderr, "[%d] during construction: globTraversedEle==%llu\n", rProc, globTraversedEle);

      /// DendroIntL locNumNodes = nodeList.size();
      /// DendroIntL globNumNodes = 0;
      /// par::Mpi_Allreduce(&locNumNodes, &globNumNodes, 1, MPI_SUM, comm);
      /// fprintf(stderr, "[%d] during construction: globNumNodes==%llu\n", rProc, globNumNodes);

      // Finish constructing.
      newSubDA.construct(nodeList, eleOrder, &treePartFront, &treePartBack, isActive, comm, activeComm);

      return locNumActiveElements;
    }



    template <unsigned int dim>
    template <typename T>
    DA<dim>::DA(std::function<void(const T *, T *)> func, unsigned int dofSz, MPI_Comm comm, unsigned int order, double interp_tol, unsigned int grainSz, double sfc_tol)
        : m_refel{dim, order}
    {
      std::vector<unsigned int> varIndex(dofSz);
      for (unsigned int ii = 0; ii < dofSz; ii++)
        varIndex[ii] = ii;

      // Get a complete tree sufficiently granular to represent func with accuracy interp_tol.
      std::vector<ot::TreeNode<C,dim>> completeTree;
      function2Octree<C,dim>(func, dofSz, &(*varIndex.cbegin()), dofSz, completeTree, m_uiMaxDepth, interp_tol, sfc_tol, order, comm);

      // Make the tree balanced, using completeTree as a minimal set of TreeNodes.
      // Calling distTreeBalancing() on a complete tree with ptsPerElement==1
      // should do exactly what we want.
      std::vector<ot::TreeNode<C,dim>> balancedTree;
      ot::SFC_Tree<C,dim>::distTreeBalancing(completeTree, balancedTree, 1, sfc_tol, comm);

      ot::DistTree<C,dim> distTree(balancedTree);   // Uses default domain decider.

      // Create ODA based on balancedTree.
      construct(distTree, comm, order, grainSz, sfc_tol);
    }

    template <unsigned int dim>
    DA<dim>::DA(MPI_Comm comm, unsigned int order, unsigned int grainSz, double sfc_tol)
        : m_refel{dim, order}
    {
        // Ignore interp_tol and just pick a uniform refinement level to satisfy grainSz.
        std::vector<ot::TreeNode<C,dim>> tree;
        util::constructRegularGrid<C,dim>(comm, grainSz, sfc_tol, tree);

        ot::DistTree<C,dim> distTree(tree);   // Uses default domain decider.

        construct(distTree, comm, order, grainSz, sfc_tol);
    }

    template <unsigned int dim>
    template <typename T>
    int DA<dim>::createVector(T*& local, bool isElemental, bool isGhosted, unsigned int dof) const
    {

        if(!(m_uiIsActive))
        {
            local=NULL;

        }else {

            if(isElemental)
            {
                if(isGhosted)
                    local=new T[dof*m_uiTotalElementSz];
                else
                    local=new T[dof*m_uiLocalElementSz];

            }else {

                if(isGhosted)
                    local=new T[dof*m_uiTotalNodalSz];
                else
                    local=new T[dof*m_uiLocalNodalSz];
            }
        }

        return 0;


    }

    template <unsigned int dim>
    template<typename T>
    int DA<dim>::createVector(std::vector<T>& local, bool isElemental, bool isGhosted, unsigned int dof) const
    {
        if(!(m_uiIsActive))
        {
            local.clear();

        }else {

            if(isElemental)
            {
                if(isGhosted)
                    local.resize(dof*m_uiTotalElementSz);
                else
                    local.resize(dof*m_uiLocalElementSz);

            }else {

                if(isGhosted)
                    local.resize(dof*m_uiTotalNodalSz);
                else
                    local.resize(dof*m_uiLocalNodalSz);
            }
        }

        return 0;
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::destroyVector(T*& local) const
    {
        delete [] local;
        local=NULL;
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::destroyVector(std::vector<T>& local) const
    {
        local.clear();
    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::nodalVecToGhostedNodal(const T* in, T*& out,bool isAllocated,unsigned int dof) const
    {

        if(!(m_uiIsActive))
            return;

        if(!isAllocated)
            createVector<T>(out,false,true,dof);

        // Assumes layout [abc][abc][...], so just need single shift.
        std::copy(in, in + dof*m_uiLocalNodalSz, out + dof*m_uiLocalNodeBegin);
    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::ghostedNodalToNodalVec(const T* gVec,T*& local,bool isAllocated,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return;

        if(!isAllocated)
            createVector(local,false,false,dof);

        // Assumes layout [abc][abc][...], so just need single shift.
        const T* srcStart = gVec + dof*m_uiLocalNodeBegin;
        std::copy(srcStart, srcStart + dof*m_uiLocalNodalSz, local);
    }



    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostBegin(T* vec,unsigned int dof)
    {
        // Send to downstream, recv from upstream.

        if (m_uiGlobalNpes==1)
            return;

        if (m_uiIsActive)
        {
          // send recv buffers.
          T* dnstB = NULL;
          T* upstB = NULL;

          // 1. Prepare asynchronous exchange context.
          const unsigned int nUpstProcs = m_gm.m_recvProc.size();
          const unsigned int nDnstProcs = m_sm.m_sendProc.size();
          m_uiMPIContexts.emplace_back(vec, typeid(T).hash_code(), nUpstProcs, nDnstProcs);
          AsyncExchangeContex &ctx = m_uiMPIContexts.back();

          const unsigned int upstBSz = m_uiTotalNodalSz - m_uiLocalNodalSz;
          const unsigned int dnstBSz = m_sm.m_map.size();

          // 2. Initiate recvs. Since vec is collated [abc abc], can receive into vec.
          if (upstBSz)
          {
            /// ctx.allocateRecvBuffer(sizeof(T)*upstBSz*dof);
            /// upstB = (T*) ctx.getRecvBuffer();
            upstB = vec;
            MPI_Request *reql = ctx.getUpstRequestList();

            for (unsigned int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
            {
              T *upstProcStart = upstB + dof*m_gm.m_recvOffsets[upstIdx];
              unsigned int upstCount = dof*m_gm.m_recvCounts[upstIdx];
              unsigned int upstProc = m_gm.m_recvProc[upstIdx];
              par::Mpi_Irecv(upstProcStart, upstCount, upstProc, m_uiCommTag, m_uiActiveComm, &reql[upstIdx]);
            }
          }

          // 3. Send data.
          if (dnstBSz)
          {
            ctx.allocateSendBuffer(sizeof(T)*dnstBSz*dof);
            dnstB = (T*) ctx.getSendBuffer();
            MPI_Request *reql = ctx.getDnstRequestList();

            // 3a. Stage the send data.
            for (unsigned int k = 0; k < dnstBSz; k++)
            {
              const T *nodeSrc = vec + dof * (m_sm.m_map[k] + m_uiLocalNodeBegin);
              std::copy(nodeSrc, nodeSrc + dof, dnstB + dof * k);
            }

            // 3b. Fire the sends.
            for (unsigned int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
            {
              T *dnstProcStart = dnstB + dof * m_sm.m_sendOffsets[dnstIdx];
              unsigned int dnstCount = dof*m_sm.m_sendCounts[dnstIdx];
              unsigned int dnstProc = m_sm.m_sendProc[dnstIdx];
              par::Mpi_Isend(dnstProcStart, dnstCount, dnstProc, m_uiCommTag, m_uiActiveComm, &reql[dnstIdx]);
            }
          }
        }

        m_uiCommTag++;   // inactive procs also advance tag.
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostEnd(T *vec,unsigned int dof)
    {
        if (m_uiGlobalNpes==1)
            return;

        if (!m_uiIsActive)
            return;

        // 1. Find asynchronous exchange context.
        MPI_Request *reql;
        MPI_Status status;
        auto const ctxPtr = std::find_if(
              m_uiMPIContexts.begin(), m_uiMPIContexts.end(),
              [vec](const AsyncExchangeContex &c){ return ((T*) c.getBuffer()) == vec; });

        // Weak type safety check.
        assert(ctxPtr->getBufferType() == typeid(T).hash_code());

        const unsigned int nUpstProcs = m_gm.m_recvProc.size();
        const unsigned int nDnstProcs = m_sm.m_sendProc.size();

        // 2. Wait on recvs and sends.
        reql = ctxPtr->getUpstRequestList();
        for (int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
          MPI_Wait(&reql[upstIdx], &status);

        reql = ctxPtr->getDnstRequestList();
        for (int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
          MPI_Wait(&reql[dnstIdx], &status);

        // 3. Release the asynchronous exchange context.
        /// ctxPtr->deAllocateRecvBuffer();
        ctxPtr->deAllocateSendBuffer();
        m_uiMPIContexts.erase(ctxPtr);
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::writeToGhostsBegin(T *vec, unsigned int dof)
    {
        // The same as readFromGhosts, but roles reversed:
        // Send to upstream, recv from downstream.

        if (m_uiGlobalNpes==1)
            return;

        if (m_uiIsActive)
        {
          // send recv buffers.
          T* dnstB = NULL;
          T* upstB = NULL;

          // 1. Prepare asynchronous exchange context.
          const unsigned int nUpstProcs = m_gm.m_recvProc.size();
          const unsigned int nDnstProcs = m_sm.m_sendProc.size();
          m_uiMPIContexts.emplace_back(vec, typeid(T).hash_code(), nUpstProcs, nDnstProcs);
          AsyncExchangeContex &ctx = m_uiMPIContexts.back();

          const unsigned int upstBSz = m_uiTotalNodalSz - m_uiLocalNodalSz;
          const unsigned int dnstBSz = m_sm.m_map.size();

          // 2. Initiate receives. (De-staging done in writeToGhostEnd().)
          if (dnstBSz)
          {
            ctx.allocateSendBuffer(sizeof(T)*dnstBSz*dof);  // Re-use the send buffer for receiving.
            dnstB = (T*) ctx.getSendBuffer();
            MPI_Request *reql = ctx.getDnstRequestList();

            for (unsigned int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
            {
              T *dnstProcStart = dnstB + dof * m_sm.m_sendOffsets[dnstIdx];
              unsigned int dnstCount = dof*m_sm.m_sendCounts[dnstIdx];
              unsigned int dnstProc = m_sm.m_sendProc[dnstIdx];
              par::Mpi_Irecv(dnstProcStart, dnstCount, dnstProc, m_uiCommTag, m_uiActiveComm, &reql[dnstIdx]);
            }
          }

          // 3. Send data. Since vec is collated [abc abc], ghosts can be sent directly from vec.
          if (upstBSz)
          {
            /// ctx.allocateRecvBuffer(sizeof(T)*upstBSz*dof);
            /// upstB = (T*) ctx.getRecvBuffer();
            upstB = vec;
            MPI_Request *reql = ctx.getUpstRequestList();

            for (unsigned int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
            {
              T *upstProcStart = upstB + dof*m_gm.m_recvOffsets[upstIdx];
              unsigned int upstCount = dof*m_gm.m_recvCounts[upstIdx];
              unsigned int upstProc = m_gm.m_recvProc[upstIdx];
              par::Mpi_Isend(upstProcStart, upstCount, upstProc, m_uiCommTag, m_uiActiveComm, &reql[upstIdx]);
            }
          }
        }

        m_uiCommTag++;   // inactive procs also advance tag.
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::writeToGhostsEnd(T *vec, unsigned int dof)
    {
        // The same as readFromGhosts, but roles reversed:
        // Send to upstream, recv from downstream.

        if (m_uiGlobalNpes==1)
            return;

        if (!m_uiIsActive)
            return;

        T* dnstB = NULL;

        // 1. Find asynchronous exchange context.
        MPI_Request *reql;
        MPI_Status status;
        auto const ctxPtr = std::find_if(
              m_uiMPIContexts.begin(), m_uiMPIContexts.end(),
              [vec](const AsyncExchangeContex &c){ return ((T*) c.getBuffer()) == vec; });

        // Weak type safety check.
        assert(ctxPtr->getBufferType() == typeid(T).hash_code());

        dnstB = (T*) ctxPtr->getSendBuffer();

        const unsigned int nUpstProcs = m_gm.m_recvProc.size();
        const unsigned int nDnstProcs = m_sm.m_sendProc.size();
        const unsigned int dnstBSz = m_sm.m_map.size();

        // 2. Wait on recvs.
        reql = ctxPtr->getDnstRequestList();
        for (int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
          MPI_Wait(&reql[dnstIdx], &status);

        // 3. "De-stage" the received downstream data.
        for (unsigned int k = 0; k < dnstBSz; k++)
        {
          // Instead of simply copying from the downstream data, we need to accumulate it.
          const T *nodeSrc = dnstB + dof * k;
          /// std::copy(nodeSrc, nodeSrc + dof, vec + dof * m_sm.m_map[k]);
          for (unsigned int v = 0; v < dof; v++)
            vec[dof * (m_sm.m_map[k] + m_uiLocalNodeBegin) + v] += nodeSrc[v];
        }

        // 4. Wait on sends.
        reql = ctxPtr->getUpstRequestList();
        for (int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
          MPI_Wait(&reql[upstIdx], &status);

        // 5. Release the asynchronous exchange context.
        /// ctxPtr->deAllocateRecvBuffer();
        ctxPtr->deAllocateSendBuffer();
        m_uiMPIContexts.erase(ctxPtr);
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByFunction(T* local,std::function<void(const T *, T*)>func,bool isElemental, bool isGhosted, unsigned int dof) const
    {
        // todo @massado can you please write this part.

        // TODO allow multiple dimensionality of input function. Until then, workaround:
        constexpr unsigned int edim = (dim < 3 ? dim : 3);
        T fCoords[3] = {0, 0, 0};
        std::array<C,dim> tnCoords;

        const T scale = 1.0 / (1u << m_uiMaxDepth);

        //TODO initialize the ghost segments too? I will for now...

        if (!isElemental)
        {
            const unsigned int nodalSz = (isGhosted ? m_uiTotalNodalSz : m_uiLocalNodalSz);
            // Assumes interleaved variables, [abc][abc].
            for (unsigned int k = 0; k < nodalSz; k++)
            {
                m_tnCoords[k].getAnchor(tnCoords);
                #pragma unroll(edim)
                for (int d = 0; d < edim; d++)
                  fCoords[d] = scale * tnCoords[d];

                func(fCoords, &local[dof*k]);
            }
        }
        else
        {
          //TODO
        }
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByScalar(T* local,const T* value,bool isElemental, bool isGhosted, unsigned int dof, unsigned int initDof) const
    {

        unsigned int arrSz;

        if(!isElemental)
        {

            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }

            for(unsigned int node=0;node<arrSz;node++)
                for(unsigned int var = 0; var < initDof; var++)
                    local[dof*node + var] = value[var];

        }else{
            // TODO, haven't considered elemental case.

            if(isGhosted) {
                arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

            for(unsigned int var=0;var<dof;var++)
            {
                for(unsigned int ele=0;ele<arrSz;ele++)
                    local[ (var*arrSz) + ele]=value[var];
            }

        }


    }

    template <unsigned int dim>
    template<typename T>
    T* DA<dim>::getVecPointerToDof(T* in ,unsigned int dofInex, bool isElemental,bool isGhosted, unsigned int dof) const
    {

        if(!(m_uiIsActive))
            return NULL;

        unsigned int arrSz;

        /// if(!isElemental)
        /// {
        ///     if(isGhosted) {
        ///         arrSz = m_uiTotalNodalSz;
        ///     }else{
        ///         arrSz=m_uiLocalNodalSz;
        ///     }


        /// }else{

        ///     if(isGhosted) {
        ///         arrSz = m_uiTotalElementSz;
        ///     }else{
        ///         arrSz=m_uiLocalElementSz;
        ///     }

        /// }

        return in + dofInex;
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::vecTopvtu(T* local, const char * fPrefix,char** nodalVarNames,bool isElemental,bool isGhosted,unsigned int dof)
    {
      //TODO

    }


    template <unsigned int dim>
    template<typename T>
    void DA<dim>::copyVector(T* dest,const T* source,bool isElemental,bool isGhosted) const
    {
        if(!(m_uiIsActive))
            return ;

        unsigned int arrSz;

        if(!isElemental)
        {
            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }


        }else{

            if(isGhosted) {
                arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

        }


        std::memcpy(dest,source,sizeof(T)*arrSz);

    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::copyVectors(T* dest,const T* source,bool isElemental,bool isGhosted,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return ;

        unsigned int arrSz;

        if(!isElemental)
        {
            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }


        }else{

            if(isGhosted) {
                arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

        }


        std::memcpy(dest,source,sizeof(T)*arrSz*dof);
    }





#ifdef BUILD_WITH_PETSC


    template <unsigned int dim>
    template<typename T>
    void DA<dim>::petscSetVectorByFunction(Vec& local,std::function<void(const T *, T*)>func,bool isElemental, bool isGhosted, unsigned int dof) const
    {

        PetscScalar * arry=NULL;
        VecGetArray(local,&arry);

        setVectorByFunction(arry,func,isElemental,isGhosted,dof);

        VecRestoreArray(local,&arry);


    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::petscSetVectorByScalar(Vec& local,const T* value,bool isElemental, bool isGhosted, unsigned int dof) const
    {

        PetscScalar * arry=NULL;
        VecGetArray(local,&arry);

        setVectorByScalar(arry,value,isElemental,isGhosted,dof);

        VecRestoreArray(local,&arry);

    }

#endif



}



