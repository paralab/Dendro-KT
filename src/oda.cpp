/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando.
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class.
 * @date 04/04/2019
 **/

#include "oda.h"
#include "meshLoop.h"
#include "sfcTreeLoop_matvec_io.h"

#include <algorithm>
#include <set>

#define OCT_NO_CHANGE 0u
#define OCT_SPLIT 1u
#define OCT_COARSE 2u

namespace ot
{
    template <unsigned int dim>
    DA<dim>::DA(unsigned int order) : m_refel{dim,order} {
        // Does nothing except set order!
        m_uiTotalNodalSz = 0;
        m_uiLocalNodalSz = 0;
        m_uiLocalElementSz = 0;
        /// m_uiTotalElementSz = 0;  // Ghosted elements not computed automatically
        m_uiPreNodeBegin = 0;
        m_uiPreNodeEnd = 0;
        m_uiLocalNodeBegin = 0;
        m_uiLocalNodeEnd = 0;
        m_uiPostNodeBegin = 0;
        m_uiPostNodeEnd = 0;
        m_uiCommTag = 0;
        m_uiGlobalNodeSz = 0;
        m_uiGlobalElementSz = 0;
        m_uiElementOrder = order;
        m_uiNpE = 0;
        m_uiActiveNpes = 0;
        m_uiGlobalNpes = 0;
        m_uiRankActive = 0;
        m_uiRankGlobal = 0;
    }


    /// /**@brief: Constructor for the DA data structures
    ///   * @param [in] inTree : input octree, need to be 2:1 balanced unique sorted octree.
    ///   * @param [in] comm: MPI global communicator for mesh generation.
    ///   * @param [in] order: order of the element.
    ///  * */
    /// template <unsigned int dim>
    /// DA<dim>::DA(std::vector<ot::TreeNode<C,dim>> &inTree, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
    ///     : m_refel{dim, order}
    /// {
    ///     ot::DistTree<C, dim> distTree(inTree, comm);   // Uses default domain decider.
    ///     inTree = distTree.getTreePartFiltered();       // Give back a copy of the in tree.
    ///     construct(distTree, comm, order, grainSz, sfc_tol);
    ///     //TODO (need straightforward interface for tree/DistTree)
    ///     //     Without a change to the interface, we can avoid copying
    ///     //     if we give back the DistTree instead, and let the user
    ///     //     get a const ref to the tree partition.
    /// }


    /// /**@brief: Constructor for the DA data structures
    ///   * @param [in] inTree : input octree, need to be 2:1 balanced unique sorted octree.
    ///   * @param [in] comm: MPI global communicator for mesh generation.
    ///   * @param [in] order: order of the element.
    ///  * */
    /// template <unsigned int dim>
    /// DA<dim>::DA(const std::vector<ot::TreeNode<C,dim>> &inTree, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
    ///     : m_refel{dim, order}
    /// {
    ///     std::vector<ot::TreeNode<C,dim>> inTreeCopy = inTree;  // Use a copy of the in tree.
    ///     ot::DistTree<C, dim> distTree(inTreeCopy, comm);       // Uses default domain decider.
    ///     construct(distTree, comm, order, grainSz, sfc_tol);
    ///     //TODO (need straightforward interface for tree/DistTree)
    ///     //     Without a change to the interface, we can avoid copying
    ///     //     if we give back the DistTree instead, and let the user
    ///     //     get a const ref to the tree partition.
    /// }


    /**@brief: Constructor for the DA data structures
      * @param [in] inDistTree : input octree that is already filtered,
      *                          need to be 2:1 balanced unique sorted octree.
      *                          Will NOT be emptied during construction of DA.
      * @param [in] comm: MPI global communicator for mesh generation.
      * @param [in] order: order of the element.
      * @note If you have a custom domain decider function, use this overload.
     * */
    template <unsigned int dim>
    DA<dim>::DA(const ot::DistTree<C,dim> &inDistTree, int stratum, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
        : m_refel{dim, order}
    {
      constructStratum(inDistTree, stratum, comm, order, grainSz, sfc_tol);
    }

    /**@brief: Constructor for the DA data structures
      * @param [in] inDistTree : input octree that is already filtered,
      *                          need to be 2:1 balanced unique sorted octree.
      *                          Will NOT be automatically emptied during construction of DA.
      * @param [in] comm: MPI global communicator for mesh generation.
      * @param [in] order: order of the element.
      * @note If you have a custom domain decider function, use this overload.
     * */

    template <unsigned int dim>
    DA<dim>::DA(const ot::DistTree<C,dim> &inDistTree, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
        : DA(inDistTree, 0, comm, order, grainSz, sfc_tol)
    { }

    // Construct multiple DA for multigrid.
    template <unsigned int dim>
    void DA<dim>::multiLevelDA(std::vector<DA> &outDAPerStratum, const DistTree<C, dim> &inDistTree, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
    {
      const int numStrata = inDistTree.getNumStrata();
      outDAPerStratum.clear();
      outDAPerStratum.reserve(numStrata);
      for (int l = 0; l < numStrata; ++l)
        outDAPerStratum.emplace_back(inDistTree, l, comm, order, grainSz, sfc_tol);
    }


    /**
     * @param distTree contains a vector of TreeNode (will be drained),
     *        and a domain decider function.
     */
    template <unsigned int dim>
    /// void DA<dim>::construct(const ot::TreeNode<C,dim> *inTree, size_t nEle, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
    void DA<dim>::construct(const ot::DistTree<C, dim> &distTree, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
    {
      constructStratum(distTree, 0, comm, order, grainSz, sfc_tol);
    }


    //
    // getNodeElementOwnership()
    //
    template <unsigned int dim>
    std::vector<DendroIntL> getNodeElementOwnership(
        DendroIntL globElementBegin,
        const std::vector<TreeNode<unsigned int, dim>> &octList,
        const std::vector<TreeNode<unsigned int, dim>> &ghostedNodeList,
        const unsigned int eleOrder,
        const DA<dim> &ghostExchange)  //TODO factor part as class GhostExchange
    {
      DOLLAR("getNodeElementOwnership()");
      using OwnershipT = DendroIntL;
      using DirtyT = char;

      OwnershipT globElementId = globElementBegin;  // enumerate elements in loop.

      const unsigned int nPe = intPow(eleOrder+1, dim);

      MatvecBaseOut<dim, OwnershipT, false> elementLoop(
          ghostedNodeList.size(),
          1,
          eleOrder,
          false,
          0,
          ghostedNodeList.data(),
          octList.data(),
          octList.size(),
          (octList.size() ? octList.front() : dummyOctant<dim>()),
          (octList.size() ? octList.back() : dummyOctant<dim>())
          );

      std::vector<OwnershipT> leafBuffer(nPe, 0);
      std::vector<DirtyT> leafDirty(nPe, 0);
      while (!elementLoop.isFinished())
      {
        if (elementLoop.isPre() && elementLoop.subtreeInfo().isLeaf())
        {
          for (size_t nIdx = 0; nIdx < nPe; ++nIdx)
            if (elementLoop.subtreeInfo().readNodeNonhangingIn()[nIdx])
            {
              leafBuffer[nIdx] = globElementId;
              leafDirty[nIdx] = true;
            }
            else
            {
              leafBuffer[nIdx] = 0;
              leafDirty[nIdx] = true;
            }

          elementLoop.subtreeInfo().overwriteNodeValsOut(leafBuffer.data(), leafDirty.data());
          elementLoop.next();
          globElementId++;
        }
        else
          elementLoop.step();
      }

      std::vector<OwnershipT> ghostedOwners(ghostedNodeList.size(), 0);
      std::vector<DirtyT> ghostedDirty(ghostedNodeList.size(), 0);

      const size_t writtenSz = elementLoop.finalize(ghostedOwners.data(), ghostedDirty.data());

      ghostExchange.writeToGhostsBegin(ghostedOwners.data(), 1, ghostedDirty.data());
      ghostExchange.writeToGhostsEnd(ghostedOwners.data(), 1, false, ghostedDirty.data()); // overwrite mode
      ghostExchange.readFromGhostBegin(ghostedOwners.data(), 1);
      ghostExchange.readFromGhostEnd(ghostedOwners.data(), 1);

      return ghostedOwners;
    }

    template <typename C, unsigned dim>
    void sortUniqXPreferCoarser(std::vector<TNPoint<C, dim>> &points,
                                std::vector<TreeNode<C, dim>> &elems,
                                std::vector<TNPoint<C, dim>> &tmp_p,
                                std::vector<TreeNode<C, dim>> &tmp_e
                                );

    template <typename C, unsigned dim>
    TNPoint<C, dim> hangingBijection(const TreeNode<C, dim> &elem,
                                     const TNPoint<C, dim> tnpoint,
                                     unsigned eleOrder);

    template <typename C, unsigned dim>
    void distPartitionEdges(std::vector<TNPoint<C, dim>> &nodesInOut,
                            std::vector<TreeNode<C, dim>> &elemsInOut,
                            double sfc_tol,
                            MPI_Comm comm);

    template <typename TN, typename ForEachIndexBody>
    size_t forEachInNodeGroup(const TN *nodeArray,
                              size_t groupBegin,
                              size_t arrayEnd,
                              const ForEachIndexBody &forEachIndexBody)
    {
      size_t index = groupBegin;
      while (index < arrayEnd && nodeArray[index].getX() == nodeArray[groupBegin].getX())
      {
        forEachIndexBody( index );
        index++;
      }
      return index;
    }


    /**
     * @param distTree contains a vector of TreeNode (will be drained),
     *        and a domain decider function.
     */
    template <unsigned int dim>
    /// void DA<dim>::construct(const ot::TreeNode<C,dim> *inTree, size_t nEle, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
    void DA<dim>::constructStratum(const ot::DistTree<C, dim> &distTree, int stratum, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol)
    {
      DOLLAR("DA::constructStratum()");
      // TODO remove grainSz parameter from ODA, which must respect the tree!

      int nProc, rProc;
      MPI_Comm_size(comm, &nProc);
      MPI_Comm_rank(comm, &rProc);

      const size_t nActiveEle = distTree.getFilteredTreePartSz(stratum);
      m_uiLocalElementSz = nActiveEle;

      // A processor is 'active' if it has elements, otherwise 'inactive'.
      bool isActive = (nActiveEle > 0);
      const bool allActive = par::mpi_and(isActive, comm);
      MPI_Comm activeComm = comm;
      if (not allActive)
        MPI_Comm_split(comm, (isActive ? 1 : MPI_UNDEFINED), rProc, &activeComm);

      m_stratum = stratum;

      m_dist_tree = &distTree;
      m_dist_tree_lifetime = distTree.live_ptr();

      TreeNode<C, dim> treePartFront;
      TreeNode<C, dim> treePartBack;

      std::vector<TreeNode<C, dim>> myTNCoords;

      ScatterMap scatterMap;
      GatherMap gatherMap;
      gatherMap.m_locOffset = 0;
      gatherMap.m_locCount = 0;
      gatherMap.m_totalCount = 0;

      //
      // Create edges from host elements to nodes.
      // When we sort edges we do it based on the node coordinate.
      // Later the edges will determine which ranks share a node.
      //
      // Using an Edge(element->node) rather than Edge(rank->node)
      // makes it possible to propagate hanging node dependencies
      // as Edge(element->parent_node). The element is needed
      // to compute the parent_node.
      //
      // -------------------------------------------------------------------
      // Pseudocode:
      // -------------------------------------------------------------------
      //   Vector<Pair<Node, Elem>> nodeEdges <-- exteriorNodes(localElems);
      //   Vector<Pair<Node, Elem>> cancelEdges <-- cancellationNodes(localElems);
      //   Vector<Pair<Node, Elem>> combinedEdges <-- concat(nodeEdges, cancelEdges);
      //
      //   combinedEdges <-- distributedTreeSort(combinedEdges, BY_NODES);
      //   edgeGroups <-- locGroupByCoordinate(combinedEdges, BY_NODES);
      //
      //   Vector<Pair<Node, Elem>> newEdges;
      //   for each (group in edgeGroups):
      //     if (cancellation is present):  // hanging node
      //       for each ((hanging_node, child_elem) in group):
      //         parent_node <-- hangingBijection(child_elem, hanging_node);
      //         newEdges.push_back((parent_node, child_elem));
      //     else   // nonhanging node
      //       for each ((nonhanging_node, elem) in group):
      //         newEdges.push_back((nonhanging_node, elem));
      //
      //   newEdges <-- distributedTreeSort(newEdges, BY_NODES);
      //   edgeGroups <-- locGroupByCoordinate(newEdges, BY_NODES);
      //
      //   for each (group in edgeGroups):
      //     ownerRank <-- mapElementToRank(coarsest element in group.elems);
      //     borrowerRanks <-- map(mapElementToRank, group.elems) - {ownerRank};
      //
      //     new_message(to:ownerRank, group.node, "borrowers=", borrowerRanks);
      //     new_message(to:borrowerRanks, group.node, "owner=", ownerRank);
      //
      //   locAndGhostNodes, borrowers, owners <-- send_messages();
      //
      //   ownedNodes <-- filterOwnedNodes(locAndGhostNodes);
      //   ghostNodes <-- locAndGhostNodes - ownedNodes;
      //   scatterMap <-- createScatterMap(borrowedNodes, borrowers);
      //   gatherMap <-- createGatherMap(ghostNodes, owners);
      // -------------------------------------------------------------------
      //

      if (isActive)
      {
        int nProcActive, rProcActive;
        MPI_Comm_size(activeComm, &nProcActive);
        MPI_Comm_rank(activeComm, &rProcActive);

        // Splitters for distributed exchanges.
        treePartFront = distTree.getTreePartFront(stratum);
        treePartBack = distTree.getTreePartBack(stratum);

        const std::vector<TreeNode<C, dim>> &inTreeFiltered = distTree.getTreePartFiltered(stratum);
        // ^ includes marked boundary elements from distTree.filterTree().

        std::vector<TNPoint<C, dim>> combinedNodes;
        std::vector<TreeNode<C, dim>> combinedElems;
        {DOLLAR("emit.combinedNodes");
          // Generate nodes from the tree. First, element-exterior nodes.
          std::vector<TNPoint<C,dim>> exteriorNodeList;
          std::vector<TNPoint<C,dim>> cancelNodeList;
          std::vector<TreeNode<C,dim>> exteriorNodeElements;
          std::vector<TreeNode<C,dim>> cancelNodeElements;
          const size_t exteriorNpe = intPow(order+1, dim) - intPow(order-1, dim);
          const size_t cancelNpe = intPow(2*order+1, dim) - intPow(2*order-1, dim) - exteriorNpe;
          const size_t inElements = inTreeFiltered.size();
          exteriorNodeList.reserve(inElements * exteriorNpe);
          cancelNodeList.reserve(inElements * cancelNpe);
          exteriorNodeElements.reserve(inElements * exteriorNpe);
          cancelNodeElements.reserve(inElements * cancelNpe);
          {DOLLAR("emit.separate");
          for (const TreeNode<C, dim> &elem : inTreeFiltered)
          {
              size_t countNewNodes1 = exteriorNodeList.size();
              size_t countNewNodes2 = cancelNodeList.size();

              Element<C, dim>(elem).appendExteriorNodes(order, exteriorNodeList, distTree.getDomainDecider());
              Element<C, dim>(elem).appendCancellationNodes(order, cancelNodeList);

              countNewNodes1 = exteriorNodeList.size() - countNewNodes1;
              countNewNodes2 = cancelNodeList.size() - countNewNodes2;

              exteriorNodeElements.insert(exteriorNodeElements.end(), countNewNodes1, elem);
              cancelNodeElements.insert(cancelNodeElements.end(), countNewNodes2, elem);
          }
          }
          // Also appends cancellation nodes where potential hanging nodes could be.
          // Only tests domainDecider if the element has been flagged as a boundary element.

          // Compact local exterior node list and local cancellation node list.
          {DOLLAR("sortUniqXPreferCoarser");
            std::vector<TNPoint<C,dim>> tmpList;
            std::vector<TreeNode<C,dim>> tmpElemList;
            sortUniqXPreferCoarser(exteriorNodeList, exteriorNodeElements, tmpList, tmpElemList);
            sortUniqXPreferCoarser(cancelNodeList, cancelNodeElements, tmpList, tmpElemList);
          }

          // Create a combined list of edges to be sorted.
          combinedNodes.reserve(exteriorNodeList.size() + cancelNodeList.size());
          combinedElems.reserve(exteriorNodeElements.size() + cancelNodeElements.size());
          {DOLLAR("combine");
          for (size_t ii = 0; ii < exteriorNodeList.size(); ++ii)
          {
            const TNPoint<C, dim> pt = exteriorNodeList[ii];
            const TreeNode<C, dim> elem = exteriorNodeElements[ii];
            combinedNodes.push_back(pt);
            combinedElems.push_back(elem);
          }
          for (size_t ii = 0; ii < cancelNodeList.size(); ++ii)
          {
            const TNPoint<C, dim> pt = cancelNodeList[ii];
            const TreeNode<C, dim> elem = cancelNodeElements[ii];
            combinedNodes.push_back(pt);
            combinedElems.push_back(elem);
          }
          }
        }

        {DOLLAR("distPartitionEdges(combined)");
        if (nProcActive > 1)
          distPartitionEdges(combinedNodes, combinedElems, sfc_tol, activeComm);
        }
        {DOLLAR("locTreeSortMaxDepth(combined)");
        SFC_Tree<C, dim>::locTreeSortMaxDepth(combinedNodes, combinedElems);
        }


        //
        // Convert edges of hanging nodes and re-sort.
        //
        std::vector<TNPoint<C, dim>> convertedNodes;
        std::vector<TreeNode<C, dim>> convertedElems;
        {DOLLAR("convert.hanging");
          size_t nNonCancellation = 0;
          for (const TNPoint<C, dim> &tnp : combinedNodes)
            nNonCancellation += !tnp.getIsCancellation();
          convertedNodes.reserve(nNonCancellation);
          convertedElems.reserve(nNonCancellation);

          const std::vector<TNPoint<C, dim>> &nodes = combinedNodes;
          const std::vector<TreeNode<C, dim>> &elems = combinedElems;
          const size_t numEdges = nodes.size();
          size_t nextEdgeId;
          for (size_t edgeId = 0; edgeId < numEdges; edgeId = nextEdgeId)
          {
            // Scan the edge group for a cancelled node.
            bool isCancelled = false;
            bool isOrdinary = false;
            nextEdgeId = forEachInNodeGroup(&nodes[0], edgeId, nodes.size(), [&](size_t ii) {
                isCancelled |= (nodes[ii].getIsCancellation());
                isOrdinary |= !(nodes[ii].getIsCancellation());
            });

            // Emit identical edges if nonhanging,
            // or edges with parent nodes if hanging.
            if (isCancelled && isOrdinary)
            {
              for (size_t ii = edgeId; ii < nextEdgeId; ++ii)
              {
                if (!nodes[ii].getIsCancellation())
                {
                  TNPoint<C, dim> parentNode = hangingBijection(elems[ii], nodes[ii], order);
                  parentNode.setIsCancellation(false);
                  convertedNodes.push_back(parentNode);
                  convertedElems.push_back(elems[ii]);
                }
              }
            }
            else if (isOrdinary)
            {
              for (size_t ii = edgeId; ii < nextEdgeId; ++ii)
              {
                convertedNodes.push_back(nodes[ii]);
                convertedElems.push_back(elems[ii]);
              }
            }
            else
            {
              // Discard pure cancellation nodes.
            }
          }
        }

        combinedNodes.clear();
        combinedElems.clear();

        {DOLLAR("distPartitionEdges(converted)");
        if (nProcActive > 1)
          distPartitionEdges(convertedNodes, convertedElems, sfc_tol, activeComm);
        }
        {DOLLAR("locTreeSortMaxDepth(converted)");
        SFC_Tree<C, dim>::locTreeSortMaxDepth(convertedNodes, convertedElems);
        }

        assert((convertedNodes.size() == convertedElems.size()));

        // Map elements of edges to the ranks that own those elements.
        // These are the ranks that share or depend on corresponding nodes.
        const std::vector<TreeNode<C, dim>> activeFrontSplitters
            = SFC_Tree<C, dim>::dist_bcastSplitters(&treePartFront, activeComm);
        const std::vector<int> sharingRanks
            = SFC_Tree<C, dim>::treeNode2PartitionRank(convertedElems, activeFrontSplitters);


        //
        // Assign owners to nodes.
        // Inform each owner about the borrowers, and borrowers about the owner.
        //
        std::vector<TNPoint<C, dim>> ownShareNodes;
        std::vector<TreeNode<C, dim>> ownShareElems;
        std::vector<int> ownShareDestRank;
        combinedNodes.clear();
        combinedElems.clear();
        std::swap(ownShareNodes, combinedNodes);
        std::swap(ownShareElems, combinedElems);
        ownShareDestRank.reserve(2 * convertedNodes.size());

        {DOLLAR("assign.owners");
          std::set<int> ranksOfNode;
          size_t nextEdgeId = 0;
          for (size_t edgeId = 0; edgeId < convertedNodes.size(); edgeId = nextEdgeId)
          {
            size_t bestRepId = edgeId;
            ranksOfNode.clear();
            nextEdgeId = forEachInNodeGroup(&convertedNodes[0], edgeId, convertedNodes.size(),
            [&](size_t ii) {
                if (convertedElems[bestRepId].getLevel() > convertedElems[ii].getLevel()
                    || convertedElems[bestRepId].getLevel() == convertedElems[ii].getLevel()
                       && sharingRanks[bestRepId] > sharingRanks[ii])
                  bestRepId = ii;

                ranksOfNode.insert(sharingRanks[ii]);
            });

            // We will send the node (with owner tag) to all borrowers).
            // We will also send the node (with sharers tagged) to the owner.
            // Only the owner will see itself tagged.
            const int owner = sharingRanks[bestRepId];
            TNPoint<C, dim> node = convertedNodes[bestRepId];
            node.set_owner(owner);
            const TreeNode<C, dim> elem = convertedElems[bestRepId];

            assert(ranksOfNode.find(owner) != ranksOfNode.end());

            for (int borrower : ranksOfNode)
              if (borrower != owner)
              {
                ownShareDestRank.push_back(borrower);
                ownShareNodes.push_back(node);
                ownShareElems.push_back(elem);
              }
            for (int sharer : ranksOfNode)
            {
              ownShareDestRank.push_back(owner);
              node.set_owner(sharer);
              ownShareNodes.push_back(node);
              ownShareElems.push_back(elem);
            }
          }
        }
        convertedNodes.clear();
        convertedElems.clear();

        {DOLLAR("sendAll(ownShare)");
        ownShareNodes = par::sendAll(ownShareNodes, ownShareDestRank, activeComm);
        ownShareElems = par::sendAll(ownShareElems, ownShareDestRank, activeComm);
        ownShareDestRank.clear();
        }
        {DOLLAR("locTreeSortMaxDepth(ownShare)");
        SFC_Tree<C, dim>::locTreeSortMaxDepth(ownShareNodes, ownShareElems);
        }


        // The information for scattermap and gathermap is jumbled together.
        // Sort them out.
        std::vector<TNPoint<C, dim>> ownedAndScatteredNodes;
        std::vector<TreeNode<C, dim>> ownedAndScatteredElems;
        std::vector<std::vector<TreeNode<C, dim>>> gatherSets(nProcActive);
        std::vector<std::vector<RankI>> scatterSets(nProcActive);
        {DOLLAR("scatter.gather.stage");
          size_t nextId;
          for (size_t edgeId = 0; edgeId < ownShareNodes.size(); edgeId = nextId)
          {
            bool isOwned = false;
            nextId = forEachInNodeGroup(&ownShareNodes[0], edgeId, ownShareNodes.size(), [&](size_t ii) {
              isOwned |= (ownShareNodes[ii].get_owner() == rProcActive);
            });
            if (isOwned)
            {
              // Contribute to owned nodes and scattermap.
              for (size_t instanceId = edgeId; instanceId < nextId; ++instanceId)
              {
                ownedAndScatteredNodes.push_back(ownShareNodes[instanceId]);
                ownedAndScatteredElems.push_back(ownShareElems[instanceId]);
              }
            }
            else
            {
              // Contribute to gathermap.
              gatherSets[ownShareNodes[edgeId].get_owner()].push_back(ownShareNodes[edgeId]);
              assert(nextId == edgeId + 1);
            }
          }
        }

        // Sort by winning element, which has also determined the owning rank.
        // Ensure consistent global ordering of nodes, regardless of partitioning,
        // which is an assumption needed by distShiftNodes().
        //
        // The global ordering is:
        //   node1 < node2 iff element(node1) < element(node2) OR
        //                     element(node1) == element(node2) AND
        //                        (isElementExterior(node1) AND isElementInterior(node2) OR
        //                         isElementExterior(node1) AND isElementExterior(node2) AND SFC(node1) < SFC(node2) OR
        //                         isElementInterior(node1) AND isElementInterior(node2) AND lex(node1) < lex(node2))
        // For this ordering, the element-interior nodes need
        // to be next to element-exterior nodes for the same element.
        // 
        // Also element sort must be stable so that forEachInNodeGroup() works.
        {DOLLAR("locTreeSort(by.element)");
        SFC_Tree<C, dim>::locTreeSort(ownedAndScatteredElems,
                                      ownedAndScatteredNodes);
        }

        // Merge exterior nodes and interior nodes
        // (both are now sorted by element)
        // and separate owned-node-indicators from scattering-indicators.
        {DOLLAR("merge.interior.nodes");
          std::vector<TNPoint<C,dim>> interiorNodeList;

          // Append element-by-element.
          // elem : ownedAndScatteredNodes[i] -> ownedAndScatteredElems[i]
          // elem : Element(inTreeFiltered[j]).interiorNodes[k] -> inTreeFiltered[j]
          size_t elemIntI = 0, elemExtJ = 0;
          while (elemIntI < inTreeFiltered.size())
          {
            const TreeNode<C, dim> elemKey = inTreeFiltered[elemIntI];

            interiorNodeList.clear();
            ot::Element<C,dim>(elemKey).appendInteriorNodes(order, interiorNodeList);
            for (const auto &pt : interiorNodeList)  // convert from TNPoint to TreeNode
              myTNCoords.push_back(pt);
            elemIntI++;

            while (elemExtJ < ownedAndScatteredElems.size()
                && ownedAndScatteredElems[elemExtJ] == elemKey)
            {
              const size_t localRank = myTNCoords.size();
              const size_t nextId = forEachInNodeGroup(
                  &ownedAndScatteredNodes[0],
                  elemExtJ, ownedAndScatteredNodes.size(),
                  [&](size_t ii) {
                    const int sharer = ownedAndScatteredNodes[ii].get_owner();
                    if (sharer == rProcActive)
                      myTNCoords.push_back(ownedAndScatteredNodes[ii]);  // own nodes
                    else
                      scatterSets[sharer].push_back(localRank);  // scattermap
                  });
              elemExtJ = nextId;
            }
          }
        }

        // Note that if we want to re-order myTNCoords
        // then the scatterSets must be mapped to the new indices.

        // Compute scatterMap and gatherMap using scatterSets and gatherSets.
        {DOLLAR("scatter.gather.create");
        RankI smapSendOffset = 0;
        for (int r = 0; r < scatterSets.size(); ++r)
        {
          if (scatterSets[r].size() > 0)
          {
            scatterMap.m_map.insert(scatterMap.m_map.end(), scatterSets[r].cbegin(), scatterSets[r].cend());
            scatterMap.m_sendCounts.push_back(scatterSets[r].size());
            scatterMap.m_sendOffsets.push_back(smapSendOffset);
            smapSendOffset += scatterSets[r].size();
            scatterMap.m_sendProc.push_back(r);
          }
        }

        RankI gmapRecvOffset = 0;
        for (int r = 0; r < rProcActive; ++r)
        {
          if (gatherSets[r].size() > 0)
          {
            gatherMap.m_recvProc.push_back(r);
            gatherMap.m_recvCounts.push_back(gatherSets[r].size());
            gatherMap.m_recvOffsets.push_back(gmapRecvOffset);
            gmapRecvOffset += gatherSets[r].size();
          }
        }
        gatherMap.m_locCount = myTNCoords.size();
        gatherMap.m_locOffset = gmapRecvOffset;
        gmapRecvOffset += myTNCoords.size();
        for (int r = rProcActive+1; r < gatherSets.size(); ++r)
        {
          if (gatherSets[r].size() > 0)
          {
            gatherMap.m_recvProc.push_back(r);
            gatherMap.m_recvCounts.push_back(gatherSets[r].size());
            gatherMap.m_recvOffsets.push_back(gmapRecvOffset);
            gmapRecvOffset += gatherSets[r].size();
          }
        }
        gatherMap.m_totalCount = gmapRecvOffset;
      }
      }

      // Finish assigning object attributes.
      {DOLLAR("DA::_constructInner()");
      this->_constructInner(myTNCoords, scatterMap, gatherMap, order, &treePartFront, &treePartBack, isActive, comm, activeComm);
      }

      m_totalSendSz = computeTotalSendSz(m_sm);
      m_totalRecvSz = totalRecvSz(m_gm);
      m_numDestNeighbors = m_sm.m_sendProc.size();
      m_numSrcNeighbors = m_gm.m_recvProc.size();

      // TODO for cleaner code, factor the scattermap/gatthermap as GhostExchange
      // for now, use the DA interface for ghost exchange.
      const DA<dim> &ghostExchange = *this;
      m_ghostedNodeOwnerElements = getNodeElementOwnership(
          m_uiGlobalElementBegin,
          distTree.getTreePartFiltered(stratum),
          m_tnCoords,
          order,
          ghostExchange); //need ghost maps

      // Active comm is not destroyed here because they are used in formation of DA.
      // This is finally destroyed in the destructor of DA.
    }


    template <typename C, unsigned dim>
    void sortUniqXPreferCoarser(std::vector<TNPoint<C, dim>> &points,
                                std::vector<TreeNode<C, dim>> &elems,
                                std::vector<TNPoint<C, dim>> &tmp_p,
                                std::vector<TreeNode<C, dim>> &tmp_e
                                )
    {
      SFC_Tree<C, dim>::locTreeSortMaxDepth(points, elems);
      tmp_p.clear();
      tmp_e.clear();
      if (points.size() > 0)
      {
        tmp_p.reserve(points.size());
        tmp_e.reserve(elems.size());

        tmp_p.push_back(points[0]);
        tmp_e.push_back(elems[0]);
        for (size_t ii = 1; ii < points.size(); ++ii)
          if (points[ii].getX() == tmp_p.back().getX())
          {
            if (tmp_p.back().getLevel() > points[ii].getLevel())
            {
              tmp_p.back() = points[ii];
              tmp_e.back() = elems[ii];
            }
          }
          else
          {
            tmp_p.push_back(points[ii]);
            tmp_e.push_back(elems[ii]);
          }
      }
      points.clear();
      elems.clear();
      std::swap(points, tmp_p);
      std::swap(elems, tmp_e);
    }


    template <typename C, unsigned dim>
    TNPoint<C, dim> hangingBijection(const TreeNode<C, dim> &elem,
                                     const TNPoint<C, dim> tnpoint,
                                     unsigned eleOrder)
    {
      const std::array<unsigned, dim> childIndices
        = TNPoint<C, dim>::get_nodeRanks1D(elem, tnpoint, eleOrder);

      const std::array<unsigned, dim> parentIndices
        = Element<C, dim>(elem).hanging2ParentIndicesBijection(
            childIndices, eleOrder);

      for (int d = 0; d < dim; ++d)
        assert((0 <= parentIndices[d] && parentIndices[d] <= eleOrder));

      const TNPoint<C, dim> parentPoint
        = Element<C, dim>(elem.getParent()).getNode(parentIndices, eleOrder);

      TNPoint<C, dim> parentPointKeepProperties = tnpoint;
      parentPointKeepProperties.setX(parentPoint.getX());
      parentPointKeepProperties.setLevel(parentPoint.getLevel());

      return parentPointKeepProperties;
    }


    template <typename C, unsigned dim>
    void distPartitionEdges(std::vector<TNPoint<C, dim>> &nodesInOut,
                            std::vector<TreeNode<C, dim>> &elemsInOut,
                            double sfc_tol,
                            MPI_Comm comm)
    {
      // distTreePartition only works on TreeNode inside the unit cube, not TNPoint.
      // Create a key for each tnpoint.
      std::vector<TreeNode<C, dim>> keys;
      for (const auto &pt : nodesInOut)
      {
        const TreeNode<C, dim> key(clampCoords<C, dim>(pt.getX(), m_uiMaxDepth), m_uiMaxDepth);
        keys.push_back(key);
      }

      distTreePartition_kway(comm, keys, nodesInOut, elemsInOut, sfc_tol);
    }





    //
    // construct() - given the partition of owned points.
    //
    template <unsigned int dim>
    void DA<dim>::_constructInner(const std::vector<TreeNode<C,dim>> &ownedNodes,
                            const ScatterMap &sm,
                            const GatherMap &gm,
                            unsigned int eleOrder,
                            const TreeNode<C,dim> *treePartFront,
                            const TreeNode<C,dim> *treePartBack,
                            bool isActive,
                            MPI_Comm globalComm,
                            MPI_Comm activeComm)
    {
      if (eleOrder != m_refel.getOrder())
        m_refel = RefElement(dim, eleOrder);

      m_uiElementOrder = eleOrder;
      m_uiNpE = intPow(eleOrder + 1, dim);

      int nProc, rProc;

      m_uiGlobalComm = globalComm;

      MPI_Comm_size(m_uiGlobalComm, &nProc);
      MPI_Comm_rank(m_uiGlobalComm, &rProc);
      m_uiGlobalNpes = nProc;
      m_uiRankGlobal = rProc;

      m_uiActiveComm = activeComm;
      m_uiIsActive = isActive;


      if (m_uiIsActive)
      {
        MPI_Comm_size(m_uiActiveComm, &nProc);
        MPI_Comm_rank(m_uiActiveComm, &rProc);
        m_uiActiveNpes = nProc;
        m_uiRankActive = rProc;

        m_uiCommTag = 0;

        m_treePartFront = *treePartFront;
        m_treePartBack = *treePartBack;

        m_uiLocalNodalSz = ownedNodes.size();

        // Gather/scatter maps.
        m_sm = sm;
        m_gm = gm;

        // Export from gm: dividers between local and ghost segments.
        m_uiTotalNodalSz   = m_gm.m_totalCount;
        m_uiPreNodeBegin   = 0;
        m_uiPreNodeEnd     = m_gm.m_locOffset;
        m_uiLocalNodeBegin = m_gm.m_locOffset;
        m_uiLocalNodeEnd   = m_gm.m_locOffset + m_gm.m_locCount;
        m_uiPostNodeBegin  = m_gm.m_locOffset + m_gm.m_locCount;;
        m_uiPostNodeEnd    = m_gm.m_totalCount;

        // Note: We will offset the starting address whenever we copy with scattermap.
        // Otherwise we should build-in the offset to the scattermap here.
      }
      else
      {
        m_uiLocalNodalSz = 0;
        m_uiTotalNodalSz   = 0;
        m_uiPreNodeBegin   = 0;
        m_uiPreNodeEnd     = 0;
        m_uiLocalNodeBegin = 0;
        m_uiLocalNodeEnd   = 0;
        m_uiPostNodeBegin  = 0;
        m_uiPostNodeEnd    = 0;
      }


      // Find offset into the global array.  All ranks take part.
      DendroIntL locSz = m_uiLocalNodalSz;
      par::Mpi_Allreduce(&locSz, &m_uiGlobalNodeSz, 1, MPI_SUM, m_uiGlobalComm);
      par::Mpi_Scan(&locSz, &m_uiGlobalRankBegin, 1, MPI_SUM, m_uiGlobalComm);
      m_uiGlobalRankBegin -= locSz;

      DendroIntL elementCount = m_uiLocalElementSz;
      par::Mpi_Allreduce(&elementCount, &m_uiGlobalElementSz, 1, MPI_SUM, m_uiGlobalComm);
      par::Mpi_Scan(&elementCount, &m_uiGlobalElementBegin, 1, MPI_SUM, m_uiGlobalComm);
      m_uiGlobalElementBegin -= elementCount;

      if (m_uiIsActive)
      {
        // Create vector of node coordinates, with ghost segments allocated.
        m_tnCoords.resize(m_uiTotalNodalSz);
        for (size_t ii = 0; ii < m_uiLocalNodalSz; ii++)
          m_tnCoords[m_uiLocalNodeBegin + ii] = ownedNodes[ii];

        // Fill ghost segments of node coordinates vector.
        std::vector<ot::TreeNode<C,dim>> tmpSendBuf(m_sm.m_map.size());
        ot::SFC_NodeSort<C,dim>::template ghostExchange<ot::TreeNode<C,dim>>(
            &(*m_tnCoords.begin()), &(*tmpSendBuf.begin()), m_sm, m_gm, m_uiActiveComm);
        //TODO transfer ghostExchange into this class, then use new method.

        // Compute global ids of all nodes, including local and ghosted.
        m_uiLocalToGlobalNodalMap.resize(m_uiTotalNodalSz, 0);
        for (size_t ii = 0; ii < m_uiLocalNodalSz; ii++)
          m_uiLocalToGlobalNodalMap[m_uiLocalNodeBegin + ii] = m_uiGlobalRankBegin + ii;
        std::vector<ot::RankI> tmpSendGlobId(m_sm.m_map.size());
        ot::SFC_NodeSort<C,dim>::template ghostExchange<ot::RankI>(
            &(*m_uiLocalToGlobalNodalMap.begin()), &(*tmpSendGlobId.begin()), m_sm, m_gm, m_uiActiveComm);
        //TODO transfer ghostExchange into this class, then use new method.

        // Identify the (local ids of) domain boundary nodes in local vector.
        // To use the ids in the ghosted vector you need to shift by m_uiLocalNodeBegin.
        m_uiBdyNodeIds.clear();
        for (size_t ii = 0; ii < m_uiLocalNodalSz; ii++)
        {
          if (m_tnCoords[ii + m_uiLocalNodeBegin].getIsOnTreeBdry())
            m_uiBdyNodeIds.push_back(ii);
        }
      }
    }


    template <unsigned int dim>
    DA<dim>::~DA()
    {
      m_uiMPIContexts.clear();
      if (m_uiActiveComm != MPI_COMM_NULL and m_uiActiveComm != MPI_COMM_WORLD
          and m_uiActiveComm != m_uiGlobalComm)
        MPI_Comm_free(&m_uiActiveComm);
      // DA does not own the global comm it is constructed with.
    }


    template <unsigned int dim>
    const std::vector<int> & DA<dim>::elements_per_node() const
    {
      if (not m_elements_per_node.initialized())
      {
        std::vector<int> element_count(this->getTotalNodalSz(), 0);

        const TreeNode<C, dim> *tn_list = this->dist_tree()->getTreePartFiltered().data();
        const TreeNode<C, dim> *node_list = this->getTNCoords();
        const size_t tn_list_sz = this->dist_tree()->getTreePartFiltered().size();
        const size_t node_list_sz = this->getTotalNodalSz();
        const int single_dof = 1;
        const int degree = this->getElementOrder();
        const int one = 1;

        MatvecBaseOut<dim, int, true> loop(
            node_list_sz, single_dof, degree, false, 0, node_list, tn_list, tn_list_sz,
            dummyOctant<dim>(), dummyOctant<dim>());
        while (not loop.isFinished())
        {
          if (loop.isPre() and loop.isLeaf())
          {
            loop.subtreeInfo().overwriteNodeValsOutScalar(&one);
            loop.next();
          }
          else
            loop.step();
        }
        loop.finalize(element_count.data());

        this->writeToGhostsBegin(element_count.data());
        this->writeToGhostsEnd(element_count.data());
        this->readFromGhostBegin(element_count.data());
        this->readFromGhostEnd(element_count.data());

        m_elements_per_node = element_count;
      }

      return m_elements_per_node.get();
    }



    template <unsigned int dim>
    void DA<dim>::computeTreeNodeOwnerProc(const TreeNode<C, dim> * pNodes, unsigned int n, int* ownerRanks) const
    {
      std::vector<int> active2global;

      std::vector<TreeNode<C, dim>> fsplitters =
        SFC_Tree<C, dim>::dist_bcastSplitters(
            (this->isActive() ? this->getTreePartFront() : nullptr),
            this->m_uiGlobalComm,
            this->m_uiActiveComm,
            this->isActive(),
            active2global);

      // Mutable copy of the TreeNode points, for sorting.
      std::vector<TreeNode<C, dim>> pNodeVec;
      pNodeVec.reserve(n);
      pNodeVec.insert(pNodeVec.end(), pNodes, pNodes + n);
      for (TreeNode<C, dim> &pNode : pNodeVec)
        pNode.setLevel(m_uiMaxDepth);  // Enforce they are points.

      // Keep track of positions in input array so we can report ranks.
      std::vector<size_t> inpos(n);
      std::iota(inpos.begin(), inpos.end(), 0);

      // Make points sorted. Use companion sort so that positions match.
      SFC_Tree<C, dim>::locTreeSort(pNodeVec, inpos);

      // Tree Traversal
      {
        MeshLoopInterface_Sorted<C, dim, true, true, false> lpSplitters(fsplitters);
        MeshLoopInterface_Sorted<C, dim, true, true, false> lpPoints(pNodeVec);

        int splitterCount = 0;
        while (!lpSplitters.isFinished())
        {
          const MeshLoopFrame<C, dim> &subtreeSplitters = lpSplitters.getTopConst();
          const MeshLoopFrame<C, dim> &subtreePoints = lpPoints.getTopConst();

          assert((subtreeSplitters.getLev() == subtreePoints.getLev()));
          assert((subtreeSplitters.getPRot() == subtreePoints.getPRot()));

          int splittersInSubtree = subtreeSplitters.getTotalCount();

          // Case 0: The item subtree is empty.
          //     --> Advance the bucket by the number of contained splitters.
          // Case 1: There are no splitters in the subtree.
          //     --> add all items to current bucket.
          // Case 2: The splitter subtree is a leaf.
          //     --> No items can be deeper than the current subtree.
          //         Advance the bucket and add the items to it.
          //         (Since we use front splitters here).
          // Case 3a: The splitter subtree is a nonempty nonleaf, and the item subtree is a leaf.
          //     --> add the current item to all buckets split by the splitters.
          // Case 3b: The splitter subtree is a nonempty nonleaf, and the item subtree is not a leaf.
          //     --> descend.

          // Case 0
          if (subtreePoints.isEmpty())
          {
            splitterCount += subtreeSplitters.getTotalCount();
            lpSplitters.next();
            lpPoints.next();
          }

          // Cases 1 & 2
          else if (subtreeSplitters.isEmpty() || subtreeSplitters.isLeaf())
          {
            if (!subtreeSplitters.isEmpty() && subtreeSplitters.isLeaf())  // Case 2
            {
              ++splitterCount;
            }

            for (size_t cIdx = subtreePoints.getBeginIdx(); cIdx < subtreePoints.getEndIdx(); ++cIdx)
            {
              ownerRanks[inpos[cIdx]] = active2global[splitterCount-1];
            }

            lpSplitters.next();
            lpPoints.next();
          }

          // Case 3
          else
          {
            // Case 3a
            if (!subtreePoints.isEmpty() && subtreePoints.isLeaf() && splittersInSubtree > 0)
            {
              throw std::logic_error("A point spans multiple partitions!");
            }

            lpSplitters.step();
            lpPoints.step();
          }

        }
      } // end tree traversal
    }







    // all the petsc functionalities goes below.
    #ifdef BUILD_WITH_PETSC

    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscCreateVector(Vec &local, bool isElemental, bool isGhosted, unsigned int dof) const
    {
        size_t sz=0;
        PetscErrorCode status = 0;
        if(!m_uiIsActive)
        {
            local=NULL;

        }else {

            if(isElemental)
            {
                if(isGhosted)
                {
                    throw std::logic_error("Ghosted elemental size not automatically computed.");
                    /// sz=dof*m_uiTotalElementSz;
                }
                else
                    sz=dof*m_uiLocalElementSz;

            }else {

                if(isGhosted)
                    sz=dof*m_uiTotalNodalSz;
                else
                    sz=dof*m_uiLocalNodalSz;
            }
            MPI_Comm activeComm = this->getCommActive();
            VecCreate(activeComm,&local);
            status=VecSetSizes(local,sz,PETSC_DECIDE);

            if (this->getNpesAll() > 1) {
                VecSetType(local,VECMPI);
            } else {
                VecSetType(local,VECSEQ);
            }

        }


        return status;


    }

    template <unsigned int dim>
    PetscErrorCode DA<dim>::createMatrix(Mat &M, MatType mtype, unsigned int dof) const
    {DOLLAR("createMatrix")

        if(m_uiIsActive)
        {
            const size_t lSz = dof * this->getLocalNodalSz();
            const unsigned int dofsPerElem = dof * this->getNumNodesPerElement();
            const std::vector<int> &epn_ghosted = this->elements_per_node();
            std::vector<PetscInt> nnz_bound;
            nnz_bound.reserve(lSz);
            for (size_t i = this->getLocalNodeBegin(),
                end = i + this->getLocalNodalSz(); i < end; ++i)
            {
              const int relevant_cells = epn_ghosted[i];
              nnz_bound.insert(nnz_bound.end(), dof, dofsPerElem * relevant_cells);
            }

            const unsigned int npesAll=m_uiGlobalNpes;
            const unsigned int eleOrder=m_uiElementOrder;

            MPI_Comm activeComm=m_uiActiveComm;

            PetscBool isAij, isAijSeq, isAijPrl, isSuperLU, isSuperLU_Dist;
            PetscStrcmp(mtype,MATAIJ,&isAij);
            PetscStrcmp(mtype,MATSEQAIJ,&isAijSeq);
            PetscStrcmp(mtype,MATMPIAIJ,&isAijPrl);
            isSuperLU = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU,&isSuperLU);
            isSuperLU_Dist = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU_DIST,&isSuperLU_Dist);

            MatCreate(activeComm, &M);
            MatSetSizes(M, lSz,lSz, PETSC_DETERMINE, PETSC_DETERMINE);
            MatSetType(M,mtype);


            if(isAij || isAijSeq || isAijPrl || isSuperLU || isSuperLU_Dist) {
                if(npesAll > 1) {
                    MatMPIAIJSetPreallocation(M, int{}, nnz_bound.data(), int{} , nnz_bound.data());
                }else {
                    MatSeqAIJSetPreallocation(M, int{}, nnz_bound.data());
                }
            }

        }



        return 0;
    }


    template <unsigned int dim>
    void DA<dim>::petscVecTopvtu(const Vec& local, const char * fPrefix,char** nodalVarNames,bool isElemental,bool isGhosted,unsigned int dof) 
    {
        const PetscScalar *arry=NULL;
        VecGetArrayRead(local,&arry);

        vecTopvtu(arry,fPrefix,nodalVarNames,isElemental,isGhosted,dof);

        VecRestoreArrayRead(local,&arry);
    }



    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscDestroyVec(Vec & vec) const
    {
            VecDestroy(&vec);
            vec=NULL;
            return 0;
    }
    
    
    
#endif

template class DA<2u>;
template class DA<3u>;
template class DA<4u>;

}



