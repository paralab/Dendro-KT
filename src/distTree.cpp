
#include "distTree.h"
#include "meshLoop.h"
#include "tsort.h"
#include "nsort.h"

namespace ot
{

  // A class to manage bucket sort.
  // Allows sending items to more than one bucket.
  // Useful if the cost of traversing the original, or computing bucket ids,
  // is expensive or inconvenient.
  template <typename IdxT = size_t, typename BdxT = int>
  class BucketMultiplexer
  {
      struct SrcDestPair
      {
        IdxT idx;
        BdxT bucket;
      };

    private:
      std::vector<IdxT> m_bucketCounts;
      std::vector<SrcDestPair> m_srcDestList;

    public:
      BucketMultiplexer() = delete;
      BucketMultiplexer(BdxT numBuckets, IdxT listReserveSz = 0)
        : m_bucketCounts(numBuckets, 0)
      {
        m_srcDestList.reserve(listReserveSz);
      }


      // Definitions

      // addToBucket()
      inline void addToBucket(IdxT index, BdxT bucket)
      {
        SrcDestPair srcDestPair;
        srcDestPair.idx = index;
        srcDestPair.bucket = bucket;

        m_srcDestList.emplace_back(srcDestPair);
        m_bucketCounts[bucket]++;
      }

      // getTotalItems()
      inline IdxT getTotalItems() const
      {
        return (IdxT) m_srcDestList.size();
      }

      // getBucketCounts()
      std::vector<IdxT> getBucketCounts() const
      {
        return m_bucketCounts;
      }

      // getBucketOffsets()
      std::vector<IdxT> getBucketOffsets() const
      {
        std::vector<IdxT> bucketOffsets(1, 0);
        for (IdxT c : m_bucketCounts)
          bucketOffsets.push_back(bucketOffsets.back() + c);
        bucketOffsets.pop_back();

        return bucketOffsets;
      }

      // transferCopies()
      template <typename X, typename Y>
      inline void transferCopies(Y * dest, const X * src) const
      {
        std::vector<IdxT> bucketOffsets = getBucketOffsets();
        for (const SrcDestPair &sd : m_srcDestList)
          dest[bucketOffsets[sd.bucket]++] = src[sd.idx];
      }
  };


  //
  // distRemeshSubdomainViaWhole()
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::distRemeshSubdomainViaWhole( const DistTree &inTree,
                                              const std::vector<OCT_FLAGS::Refine> &refnFlags,
                                              DistTree &_outTree,
                                              DistTree &_surrogateTree,
                                              RemeshPartition remeshPartition,
                                              double loadFlexibility)
  {
    DOLLAR("DistTree::distRemeshSubdomainViaWhole()");
    MPI_Comm comm = inTree.m_comm;

    const std::vector<TreeNode<T, dim>> &inTreeVec = inTree.getTreePartFiltered();
    std::vector<TreeNode<T, dim>> outTreeVec;

    SFC_Tree<T, dim>::distRemeshWholeDomain(
        inTreeVec, refnFlags, outTreeVec, loadFlexibility, comm);

    // Filter and repartition based on filtered octlist,
    // before fixing the octlist into a DistTree.
    DistTree<T, dim>::filterOctList(inTree.getDomainDecider(), outTreeVec);
    SFC_Tree<T, dim>::distTreeSort(outTreeVec, loadFlexibility, comm);
    SFC_Tree<T, dim>::distCoalesceSiblings(outTreeVec, comm);

    std::vector<TreeNode<T, dim>> surrogateTreeVec =
        SFC_Tree<T, dim>::getSurrogateGrid(
            remeshPartition, inTreeVec, outTreeVec, comm);

    DistTree outTree(outTreeVec, comm);
    DistTree surrogateTree(surrogateTreeVec, comm, NoCoalesce);

    outTree.filterTree(inTree.getDomainDecider());
    surrogateTree.filterTree(inTree.getDomainDecider());

    _outTree = outTree;
    _surrogateTree = surrogateTree;
  }


  //
  // distRemeshSubdomain()
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::distRemeshSubdomain( const DistTree &inTree,
                                              const std::vector<OCT_FLAGS::Refine> &refnFlags,
                                              DistTree &_outTree,
                                              DistTree &_surrogateTree,
                                              RemeshPartition remeshPartition,
                                              double loadFlexibility)
  {
    // validate args
    DENDRO_KT_ASSERT_SORTED_UNIQ(inTree.getTreePartFiltered(), inTree.m_comm);

    DOLLAR("DistTree::distRemeshSubdomain()");
    MPI_Comm comm = inTree.m_comm;

    const std::vector<TreeNode<T, dim>> &inTreeVec = inTree.getTreePartFiltered();
    std::vector<TreeNode<T, dim>> outTreeVec;

    SFC_Tree<T, dim>::distRemeshSubdomain(
        inTreeVec, refnFlags, outTreeVec, loadFlexibility, comm);

    // Filter and repartition based on filtered octlist,
    // before fixing the octlist into a DistTree.
    DistTree<T, dim>::filterOctList(inTree.getDomainDecider(), outTreeVec);
    SFC_Tree<T, dim>::distTreeSort(outTreeVec, loadFlexibility, comm);
    SFC_Tree<T, dim>::distCoalesceSiblings(outTreeVec, comm);

    std::vector<TreeNode<T, dim>> surrogateTreeVec =
        SFC_Tree<T, dim>::getSurrogateGrid(
            remeshPartition, inTreeVec, outTreeVec, comm);

    DistTree outTree(outTreeVec, comm);
    DistTree surrogateTree(surrogateTreeVec, comm, NoCoalesce);

    outTree.filterTree(inTree.getDomainDecider());
    surrogateTree.filterTree(inTree.getDomainDecider());

    _outTree = outTree;
    _surrogateTree = surrogateTree;
  }


  // distRefine
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::distRefine(
      const DistTree &inTree,
      std::vector<int> &&delta_level,  // Consumed. To reuse, clear() and resize().
      DistTree &outTree,
      double sfc_tol,
      bool repartition)
  {
    DOLLAR("DistTree::distRefine()");
    MPI_Comm comm = inTree.getComm();

    // future: Repartition using weights based on delta_level.

    std::vector<TreeNode<T, dim>> newOctList =
        SFC_Tree<T, dim>::locRefine(inTree.getTreePartFiltered(), std::move(delta_level));

    DistTree<T, dim>::filterOctList(inTree.getDomainDecider(), newOctList);
    SFC_Tree<T, dim>::distTreeSort(newOctList, sfc_tol, comm);//future: distTreePartition(), if stable

#ifndef USE_2TO1_GLOBAL_SORT
    SFC_Tree<T, dim>::distMinimalBalanced(newOctList, sfc_tol, comm);
#else
    SFC_Tree<T, dim>::distMinimalBalancedGlobalSort(newOctList, sfc_tol, comm);
#endif
    DistTree<T, dim>::filterOctList(inTree.getDomainDecider(), newOctList);
    if (repartition)
      SFC_Tree<T, dim>::distTreeSort(newOctList, sfc_tol, comm);//future: distTreePartition(), if stable

    SFC_Tree<T, dim>::distCoalesceSiblings(newOctList, comm);

    outTree = DistTree<T, dim>(newOctList, comm);
    outTree.filterTree(inTree.getDomainDecider());
  }


  template <typename T, unsigned int dim>
  DistTree<T, dim> DistTree<T, dim>::repartitioned(const double sfc_tol) &&
  {
    DOLLAR("repartitioned()");
    std::vector<TreeNode<T, dim>> &octList = this->get_m_treePartFiltered();
    SFC_Tree<T, dim>::distTreeSort(octList, sfc_tol, this->getComm());//future: distTreePartition(), if stable
    SFC_Tree<T, dim>::distCoalesceSiblings(octList, this->getComm());

    DistTree outTree = DistTree<T, dim>(octList, this->getComm());
    outTree.filterTree(this->getDomainDecider());

    return outTree;
  }


  //
  // insertRefinedGrid()
  //   - refnFlags must be same length as treePart in finest grid of distTree.
  //   - refnFlags must not contain OCT_FLAGS::OCT_COARSEN.
  //   - All strata in distTree and surrDistTree are moved coarser by one.
  //   - A new grid is defined in stratum 0.
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::insertRefinedGrid(DistTree &distTree,
                                           DistTree &surrDistTree,
                                           const std::vector<OCT_FLAGS::Refine> &refnFlags,
                                           GridAlignment gridAlignment,
                                           double loadFlexibility)
  {
    bool hasCoarseningFlag = false;
    for (auto &f : refnFlags)
      if (f == OCT_FLAGS::OCT_COARSEN)
        hasCoarseningFlag = true;
    if (hasCoarseningFlag)
      std::cerr << "Coarsening flags are not allowed in insertRefinedGrid().\n";

    MPI_Comm comm = distTree.m_comm;

    if (surrDistTree.getNumStrata() == 0)
    {
      std::vector<TreeNode<T, dim>> emptyOctList;
      surrDistTree = DistTree(emptyOctList, comm, NoCoalesce);
      surrDistTree.filterTree(distTree.getDomainDecider());
    }

    const int finestStratum = 0;

    const std::vector<TreeNode<T, dim>> &inTreeVec = distTree.getTreePartFiltered(finestStratum);
    std::vector<TreeNode<T, dim>> outTreeVec;
    std::vector<TreeNode<T, dim>> surrogateTreeVec;

    if (gridAlignment == GridAlignment::CoarseByFine)
    {
      SFC_Tree<T, dim>::distRemeshWholeDomain(
          inTreeVec, refnFlags,
          outTreeVec,
          loadFlexibility,
          comm);
      surrogateTreeVec = SFC_Tree<T, dim>::getSurrogateGrid(
          RemeshPartition::SurrogateInByOut, inTreeVec, outTreeVec, comm);
    }
    else
    {
      SFC_Tree<T, dim>::distRemeshWholeDomain(
          inTreeVec, refnFlags,
          outTreeVec,
          loadFlexibility,
          comm);
      surrogateTreeVec = SFC_Tree<T, dim>::getSurrogateGrid(
          RemeshPartition::SurrogateOutByIn, inTreeVec, outTreeVec, comm);
    }

    distTree.insertStratum(finestStratum, outTreeVec);
    distTree.filterStratum(finestStratum);

    if (gridAlignment == GridAlignment::CoarseByFine)
    {
      surrDistTree.insertStratum(finestStratum + 1, surrogateTreeVec);
      // surrogateTreeVec elements pre-existed, don't need to filter.
    }
    else
    {
      surrDistTree.insertStratum(finestStratum, surrogateTreeVec);
      // surrogateTreeVec elements were produced with new fine grid.
      surrDistTree.filterStratum(finestStratum);
    }
  }


  //
  // defineCoarsenedGrid()
  //   - refnFlags must be same length as treePart in coarsest grid of distTree.
  //   - refnFlags must not contain OCT_FLAGS::OCT_REFINE.
  //   - A new grid is defined in (coarsestStratum + 1).
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::defineCoarsenedGrid(DistTree &distTree,
                                             DistTree &surrDistTree,
                                             const std::vector<OCT_FLAGS::Refine> &refnFlags,
                                             GridAlignment gridAlignment,
                                             double loadFlexibility)
  {
    bool hasRefiningFlag = false;
    for (auto &f : refnFlags)
      if (f == OCT_FLAGS::OCT_REFINE)
        hasRefiningFlag = true;
    if (hasRefiningFlag)
      std::cerr << "Refining flags are not allowed in defineCoarsenedGrid().\n";

    MPI_Comm comm = distTree.m_comm;

    if (surrDistTree.getNumStrata() == 0)
    {
      std::vector<TreeNode<T, dim>> emptyOctList;
      surrDistTree = DistTree(emptyOctList, comm, NoCoalesce);
      surrDistTree.filterTree(distTree.getDomainDecider());
    }

    const int oldCoarsestStratum = distTree.getNumStrata() - 1;
    const int newCoarsestStratum = distTree.getNumStrata();

    const std::vector<TreeNode<T, dim>> &inTreeVec = distTree.getTreePartFiltered(oldCoarsestStratum);
    std::vector<TreeNode<T, dim>> outTreeVec;
    std::vector<TreeNode<T, dim>> surrogateTreeVec;

    if (gridAlignment == GridAlignment::FineByCoarse)
    {
      SFC_Tree<T, dim>::distRemeshWholeDomain(
          inTreeVec, refnFlags,
          outTreeVec,
          loadFlexibility,
          comm);
      surrogateTreeVec = SFC_Tree<T, dim>::getSurrogateGrid(
          RemeshPartition::SurrogateInByOut, inTreeVec, outTreeVec, comm);
    }
    else
    {
      SFC_Tree<T, dim>::distRemeshWholeDomain(
          inTreeVec, refnFlags,
          outTreeVec,
          loadFlexibility,
          comm);
      surrogateTreeVec = SFC_Tree<T, dim>::getSurrogateGrid(
          RemeshPartition::SurrogateOutByIn, inTreeVec, outTreeVec, comm);
    }

    distTree.insertStratum(newCoarsestStratum, outTreeVec);
    distTree.filterStratum(newCoarsestStratum);

    if (gridAlignment == GridAlignment::CoarseByFine)
    {
      surrDistTree.insertStratum(newCoarsestStratum, surrogateTreeVec);
      // surrogateTreeVec elements were produced with new coarse grid.
      surrDistTree.filterStratum(newCoarsestStratum);
    }
    else
    {
      surrDistTree.insertStratum(oldCoarsestStratum, surrogateTreeVec);
      // surrogateTreeVec elements pre-existed, don't need to filter.
    }
  }


  //
  // generateGridHierarchyUp()
  //
  template <typename T, unsigned int dim>
  DistTree<T, dim> DistTree<T, dim>::generateGridHierarchyUp(bool isFixedNumStrata,
                                                             unsigned int lev,
                                                             double loadFlexibility)
  {
    DistTree<T, dim> surrogateCoarseByFine(NoCoalesce);

    // Determine the number of grids in the desired grid hierarchy.
    int targetNumStrata = 1;
    {
      int nProc, rProc;
      MPI_Comm_size(m_comm, &nProc);
      MPI_Comm_rank(m_comm, &rProc);
      MPI_Comm activeComm = m_comm;
      int nProcActive = nProc;
      int rProcActive = rProc;

      if (isFixedNumStrata) // interpret lev as desired number of strata.
        targetNumStrata = lev;
      else
      {  // interpret lev as the desired finest level in the coarsest grid.
        LevI observedMaxDepth_loc = 0, observedMaxDepth_glob = 0;
        for (const TreeNode<T, dim> & tn : m_gridStrata[0])
          if (observedMaxDepth_loc < tn.getLevel())
            observedMaxDepth_loc = tn.getLevel();

        par::Mpi_Allreduce<LevI>(&observedMaxDepth_loc, &observedMaxDepth_glob, 1, MPI_MAX, m_comm);

        targetNumStrata = 1 + (observedMaxDepth_glob - lev);
      }
    }

    // Successively coarsen.
    std::vector<OCT_FLAGS::Refine> refineFlags;
    while (this->getNumStrata() < targetNumStrata)
    {
      // Try to coarsen everything.
      refineFlags.clear();
      refineFlags.resize(
          this->getTreePartFiltered(this->getNumStrata() - 1).size(),
          OCT_FLAGS::OCT_COARSEN);

      // Add a new grid to this distTree and the surrogate.
      DistTree<T, dim>::defineCoarsenedGrid(*this,
                                            surrogateCoarseByFine,
                                            refineFlags,
                                            GridAlignment::CoarseByFine,
                                            loadFlexibility);
    }

    return surrogateCoarseByFine;
  }


  // generateGridHierarchyDown()
  template <typename T, unsigned int dim>
  DistTree<T, dim>  DistTree<T, dim>::generateGridHierarchyDown(unsigned int numStrata,
                                                                double loadFlexibility)
  {
    int nProc, rProc;
    MPI_Comm_size(m_comm, &nProc);
    MPI_Comm_rank(m_comm, &rProc);

    // Determine the number of grids in the grid hierarchy. (m_numStrata)
    {
      LevI observedMaxDepth_loc = 0, observedMaxDepth_glob = 0;
      for (const TreeNode<T, dim> & tn : m_gridStrata[0])
        if (observedMaxDepth_loc < tn.getLevel())
          observedMaxDepth_loc = tn.getLevel();

      par::Mpi_Allreduce<LevI>(&observedMaxDepth_loc, &observedMaxDepth_glob, 1, MPI_MAX, m_comm);

      const unsigned int strataLimit = 1 + m_uiMaxDepth - observedMaxDepth_glob;

      if (numStrata > strataLimit)
      {
        std::cerr << "Warning: generateGridHierarchyDown() cannot generate all requested "
                  << numStrata << " grids.\n"
                  "Conditional refinement is currently unsupported. "
                  "(Enforcing m_uiMaxDepth would require conditional refinement).\n"
                  "Only " << strataLimit << " grids are generated.\n";

        m_numStrata = strataLimit;
      }
      else
        m_numStrata = numStrata;
    }

    // Goal: Copy over global properties from primary DT to surrogate DT,
    //   but initialize all local properties on every level to empty.
    DistTree surrogateDT(*this);
    //
    surrogateDT.m_numStrata = m_numStrata;
    //
    for (int vl = 0; vl < surrogateDT.m_gridStrata.size(); vl++)
    {
      surrogateDT.m_gridStrata[vl].clear();
      surrogateDT.m_tpFrontStrata[vl] = TreeNode<T, dim>{};
      surrogateDT.m_tpBackStrata[vl] = TreeNode<T, dim>{};
      surrogateDT.m_originalTreePartSz[vl] = 0;
      surrogateDT.m_filteredTreePartSz[vl] = 0;
    }

    // The only grid we know about is in layer [0].
    // It becomes the coarsest, in layer [m_numStrata-1].
    std::swap(m_gridStrata[0], m_gridStrata[m_numStrata-1]);
    std::swap(m_tpFrontStrata[0], m_tpFrontStrata[m_numStrata-1]);
    std::swap(m_tpBackStrata[0], m_tpBackStrata[m_numStrata-1]);
    std::swap(m_originalTreePartSz[0], m_originalTreePartSz[m_numStrata-1]);
    std::swap(m_filteredTreePartSz[0], m_filteredTreePartSz[m_numStrata-1]);

    for (int coarseStratum = m_numStrata-1; coarseStratum >= 1; coarseStratum--)
    {
      // Identify coarse and fine strata.
      int fineStratum = coarseStratum - 1;
      std::vector<TreeNode<T, dim>> &coarseGrid = m_gridStrata[coarseStratum];
      std::vector<TreeNode<T, dim>> &fineGrid = m_gridStrata[fineStratum];
      std::vector<TreeNode<T, dim>> &surrogateCoarseGrid = surrogateDT.m_gridStrata[coarseStratum];

      fineGrid.clear();
      surrogateCoarseGrid.clear();

      // Generate fine grid elements.
      // For each element in the coarse grid, add all children to fine grid.
      MeshLoopInterface_Sorted<T, dim, false, true, false> lpCoarse(coarseGrid);
      while (!lpCoarse.isFinished())
      {
        const MeshLoopFrame<T, dim> &subtreeCoarse = lpCoarse.getTopConst();

        if (subtreeCoarse.isLeaf())
        {
          const TreeNode<T, dim> &parent = coarseGrid[subtreeCoarse.getBeginIdx()];
          for (int cm = 0; cm < (1u << dim); cm++)
          {
            // Children added in Morton order.
            // Could add them in SFC order, but no need if they will be sorted later.
            fineGrid.emplace_back(parent.getChildMorton(cm));
          }
        }

        lpCoarse.step();
      }

      // Re partition and sort the fine grid.
      SFC_Tree<T, dim>::distTreeSort(fineGrid, loadFlexibility, m_comm);

      // Enforce intergrid criterion, distCoalesceSiblings().
      SFC_Tree<T, dim>::distCoalesceSiblings(fineGrid, m_comm);

      // Initialize fine grid meta data.
      if (!m_hasBeenFiltered)
        m_originalTreePartSz[fineStratum] = fineGrid.size();
      m_filteredTreePartSz[fineStratum] = fineGrid.size();

      m_tpFrontStrata[fineStratum] = (fineGrid.size() ? fineGrid.front() : TreeNode<T, dim>{});
      m_tpBackStrata[fineStratum] = (fineGrid.size() ? fineGrid.back() : TreeNode<T, dim>{});

      // Create the surrogate coarse grid by duplicating the
      // coarse grid but partitioning it by the fine grid splitters.
      surrogateCoarseGrid = SFC_Tree<T, dim>::getSurrogateGrid(coarseGrid, fineGrid, m_comm);

      if (!surrogateDT.m_hasBeenFiltered)
        surrogateDT.m_originalTreePartSz[coarseStratum] = surrogateCoarseGrid.size();
      surrogateDT.m_filteredTreePartSz[coarseStratum] = surrogateCoarseGrid.size();

      if (surrogateCoarseGrid.size())
      {
        surrogateDT.m_tpFrontStrata[coarseStratum] = surrogateCoarseGrid.front();
        surrogateDT.m_tpBackStrata[coarseStratum] = surrogateCoarseGrid.back();
      }
    }

    return surrogateDT;
  }







  template <typename T, unsigned int dim>
  void addMortonDescendants(unsigned int finestLevel,
                            const TreeNode<T, dim> &anc,
                            std::vector<TreeNode<T, dim>> &list)
  {
    if (anc.getLevel() == finestLevel)
      list.emplace_back(anc);
    else
      for (int c = 0; c < (1u << dim); ++c)
        addMortonDescendants(finestLevel, anc.getChildMorton(c), list);
  }



  template <typename T, unsigned int dim>
  DistTree<T, dim>  DistTree<T, dim>::constructSubdomainDistTree(
          unsigned int finestLevel,
          MPI_Comm comm,
          double sfc_tol)
  {
    int rProc, nProc;
    MPI_Comm_size(comm, &nProc);
    MPI_Comm_rank(comm, &rProc);

    std::vector<TreeNode<T, dim>> treePart;
    if (rProc == 0)
      treePart.emplace_back(); // Root

    unsigned int level = 0;
    const unsigned int jump = 3;

    while (level < finestLevel)
    {
      // Extend deeper.
      std::vector<TreeNode<T, dim>> finerTreePart;
      unsigned int nextLevel = fmin(finestLevel, level + jump);
      for (const TreeNode<T, dim> &tn : treePart)
        addMortonDescendants(nextLevel, tn, finerTreePart);

      // Re-partition.
      SFC_Tree<T, dim>::distTreeSort(finerTreePart, sfc_tol, comm);
      SFC_Tree<T, dim>::distRemoveDuplicates(
          finerTreePart, sfc_tol, SFC_Tree<T, dim>::RM_DUPS_ONLY, comm);
      SFC_Tree<T, dim>::distCoalesceSiblings(finerTreePart, comm);

      std::swap(treePart, finerTreePart);
      level = nextLevel;
    }

    DistTree<T, dim> dtree(treePart, comm);
    return dtree;
  }

  template <typename T, unsigned int dim>
  DistTree<T, dim>  DistTree<T, dim>::constructSubdomainDistTree(
          unsigned int finestLevel,
          const ::ibm::DomainDecider &domainDecider,
          MPI_Comm comm,
          double sfc_tol)
  {
    int rProc, nProc;
    MPI_Comm_size(comm, &nProc);
    MPI_Comm_rank(comm, &rProc);

    std::vector<TreeNode<T, dim>> treePart;
    if (rProc == 0)
      treePart.emplace_back(); // Root

    unsigned int level = 0;
    const unsigned int jump = 3;

    while (level < finestLevel)
    {
      // Extend deeper.
      std::vector<TreeNode<T, dim>> finerTreePart;
      unsigned int nextLevel = fmin(finestLevel, level + jump);
      for (const TreeNode<T, dim> &tn : treePart)
        addMortonDescendants(nextLevel, tn, finerTreePart);

      // Re-select.
      DendroIntL numKept = 0;

      double phycd[dim];
      double physz;

      for (DendroIntL i = 0; i < finerTreePart.size(); ++i)
        if ((treeNode2Physical(finerTreePart[i], phycd, physz), domainDecider(phycd, physz) != ibm::IN) && (numKept++ < i))
          finerTreePart[numKept-1] = finerTreePart[i];
      finerTreePart.resize(numKept);

      // Re-partition.
      SFC_Tree<T, dim>::distTreeSort(finerTreePart, sfc_tol, comm);
      SFC_Tree<T, dim>::distRemoveDuplicates(
          finerTreePart, sfc_tol, SFC_Tree<T, dim>::RM_DUPS_ONLY, comm);
      SFC_Tree<T, dim>::distCoalesceSiblings(finerTreePart, comm);

      std::swap(treePart, finerTreePart);
      level = nextLevel;
    }

    DistTree<T, dim> dtree(treePart, comm);
    dtree.filterTree(domainDecider);  // Associate decider with dtree.

    return dtree;
  }



  template <typename T, unsigned int dim>
  DistTree<T, dim>  DistTree<T, dim>::minimalSubdomainDistTree(
          unsigned int finestLevel,
          const ::ibm::DomainDecider &domainDecider,
          MPI_Comm comm,
          double sfc_tol)
  {
    int rProc, nProc;
    MPI_Comm_size(comm, &nProc);
    MPI_Comm_rank(comm, &rProc);

    using Oct = TreeNode<T, dim>;
    using OctList = std::vector<Oct>;

    /// std::vector<TreeNode<T, dim>> treePart;
    OctList treeFinal, treeIntercepted, finerIntercepted;
    OctList octDescendants;

    if (rProc == 0)
      treeIntercepted.emplace_back(); // Root

    unsigned int level = 0;
    const unsigned int jump = 1;

    while (level < finestLevel)
    {
      // Extend deeper.
      finerIntercepted.clear();
      unsigned int nextLevel = fmin(finestLevel, level + jump);
      for (const Oct &tn : treeIntercepted)
      {
        octDescendants.clear();
        addMortonDescendants(nextLevel, tn, octDescendants);

        double phycd[dim];
        double physz;
        for (const Oct &descendant : octDescendants)
        {
          treeNode2Physical(descendant, phycd, physz);
          const ibm::Partition partition = domainDecider(phycd, physz);

          if (partition == ibm::OUT)
            treeFinal.push_back(descendant);
          else if (partition == ibm::INTERCEPTED)
            finerIntercepted.push_back(descendant);
        }
      }

      // Re-partition.
      SFC_Tree<T, dim>::distTreeSort(finerIntercepted, sfc_tol, comm);

      std::swap(treeIntercepted, finerIntercepted);
      level = nextLevel;
    }

    // Union, re-partition.
    treeFinal.insert(treeFinal.end(), treeIntercepted.begin(), treeIntercepted.end());
    treeIntercepted.clear();
    treeIntercepted.shrink_to_fit();
    SFC_Tree<T, dim>::distTreeSort(treeFinal, sfc_tol, comm);
#ifndef USE_2TO1_GLOBAL_SORT
    SFC_Tree<T, dim>::distMinimalBalanced(treeFinal, sfc_tol, comm);
#else
    SFC_Tree<T, dim>::distMinimalBalancedGlobalSort(treeFinal, sfc_tol, comm);
#endif
    SFC_Tree<T, dim>::distTreeSort(treeFinal, sfc_tol, comm);
    SFC_Tree<T, dim>::distCoalesceSiblings(treeFinal, comm);

    DistTree<T, dim> dtree(treeFinal, comm);
    dtree.filterTree(domainDecider);  // Associate decider with dtree.

    return dtree;
  }


  template <typename T, unsigned int dim>
  DistTree<T, dim>  DistTree<T, dim>::minimalSubdomainDistTreeGrain(
          size_t grainMin,
          const ::ibm::DomainDecider &domainDecider,
          MPI_Comm comm,
          double sfc_tol)
  {
    int rProc, nProc;
    MPI_Comm_size(comm, &nProc);
    MPI_Comm_rank(comm, &rProc);

    using Oct = TreeNode<T, dim>;
    using OctList = std::vector<Oct>;

    OctList tree;
    OctList nextLevel;

    if (rProc == 0)
      tree.emplace_back(); // Root
    filterOctList(domainDecider, tree);

    int saturated = bool(tree.size() > grainMin);
    { int saturatedGlobal;
      par::Mpi_Allreduce(&saturated, &saturatedGlobal, 1, MPI_LAND, comm);
      saturated = saturatedGlobal;
    }

    while (!saturated)
    {
      nextLevel.clear();
      for (const Oct &oct : tree)
        if (oct.getIsOnTreeBdry())
          addMortonDescendants(oct.getLevel() + 1, oct, nextLevel);
      tree.insert(tree.end(), nextLevel.begin(), nextLevel.end());
      SFC_Tree<T, dim>::locTreeSort(tree);
      SFC_Tree<T, dim>::locRemoveDuplicates(tree);
      SFC_Tree<T, dim>::distTreeSort(tree, sfc_tol, comm);  //future: distTreePartition(), once stable
#ifndef USE_2TO1_GLOBAL_SORT
      SFC_Tree<T, dim>::distMinimalBalanced(tree, sfc_tol, comm);
#else
      SFC_Tree<T, dim>::distMinimalBalancedGlobalSort(tree, sfc_tol, comm);
#endif
      SFC_Tree<T, dim>::distTreeSort(tree, sfc_tol, comm);  //future: distTreePartition(), once stable
      SFC_Tree<T, dim>::distCoalesceSiblings(tree, comm);
      filterOctList(domainDecider, tree);

      saturated = bool(tree.size() > grainMin);
      { int saturatedGlobal;
        par::Mpi_Allreduce(&saturated, &saturatedGlobal, 1, MPI_LAND, comm);
        saturated = saturatedGlobal;
      }
    }

    DistTree<T, dim> dtree(tree, comm);
    dtree.filterTree(domainDecider);  // Associate decider with dtree.

    return dtree;
  }



  // Explicit instantiations
  template class DistTree<unsigned int, 2u>;
  template class DistTree<unsigned int, 3u>;
  template class DistTree<unsigned int, 4u>;

}
