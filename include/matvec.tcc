/**
 * @file:matvec.tcc
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-03-15
 */



namespace fem {

  template <typename T, typename da, unsigned int dim>
  ot::RankI
  SFC_Matvec<T,da,dim>:: countSubtreeSizes(ot::TNPoint<T,dim> *points,
                            ot::RankI begin, ot::RankI end,
                            ot::LevI sLev,
                            ot::LevI eLev,
                            ot::RotI pRot,
                            int order,
                            std::vector<ot::RankI> &outSubtreeSizes)
  {
    //
    // The goal is to find an upper bound for the subtree size
    // at each level along any path down the tree.
    //
    // A bucket is a non-leaf if it contains (within its interior) a point of finer level.
    // A bucket is a leaf if it is not a non-leaf.
    //
    // A leaf bucket contributes a <= nodesPerElement to its own level and to its parent.
    // A non-leaf bucket contributes <= SUM{child contributions} to its own level and to its parent.
    //

    using TNP = ot::TNPoint<T,dim>;
    constexpr unsigned int numChildren = 1u<<dim;
    const int nPe = intPow(order+1, dim);
    /// const int nPe_ext = nPe - intPow(order-1, dim);

    // Re-usable buffer for quick-and-dirty bucketing.
    static struct { ot::RankI topBegin; TNP *buf; } longBuffer = {0, nullptr};
    bool deleteBuffer = false;
    if (longBuffer.buf == nullptr)
    {
      deleteBuffer = true;
      longBuffer.topBegin = begin;
      longBuffer.buf = new TNP[end - begin];
    }

    // Move all interface points and boundary points to the beginning.
    // This way we count and bucket only (parent-level) internal nodes,
    // which suffices to compute our upper bound.
    ot::RankI numChildExteriorPts = 0;
    for (ot::RankI ptIter = begin; ptIter < end; ptIter++)
      if (points[ptIter].get_CellType(sLev).get_dim_flag() < dim)
        numChildExteriorPts++;

    {
      TNP * const ourBuf = &longBuffer[begin - longBuffer.topBegin];
      ot::RankI lidChildExteriorPts = 0;
      ot::RankI lidChildInteriorPts = numChildExteriorPts;
      for (ot::RankI ptIter = begin; ptIter < end; ptIter++)
        if (points[ptIter].get_CellType(sLev).get_dim_flag() < dim)
          ourBuf[lidChildExteriorPts++] = points[ptIter];
        else
          ourBuf[lidChildInteriorPts++] = points[ptIter];

      memcpy(points, ourBuf, sizeof(TNP) * (end - begin));
    }


    // Determine whether the current bucket is a leaf or nonleaf.
    // (Theoretically we should be able to tell from just the first point.)
    bool bucketIsLeaf = true;
    for (ot::RankI ptIter = numChildExteriorPts;
        ptIter < end && (bucketIsLeaf = (points[ptIter].getLevel() < sLev));
        ptIter++);

    ot::RankI myContribution = 0;

    // Add contributions.
    if (bucketIsLeaf)
    {
      myContribution = nPe;
    }
    else
    {
      // Regular bucketing, except we know there will be no ancestor points.
      std::array<ot::RankI, 1+numChildren> tempSplitters;
      ot::RankI unused_ancS, unused_ancE;
      ot::SFC_Tree<T,dim>::template SFC_bucketing_impl
          <ot::KeyFunIdentity_Pt<TNP>, TNP, TNP>(
          points, numChildExteriorPts, end, sLev, pRot,
          ot::KeyFunIdentity_Pt<TNP>(), false, false,
          tempSplitters, unused_ancS, unused_ancE);

      // Shortcuts to lookup SFC rotations.
      constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].
      const ot::ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
      const ot::RotI * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

      if (sLev > 0)   // Recurse normally.
      {
        for (ot::ChildI child_sfc = 0; child_sfc < numChildren; child_sfc++)
        {
          ot::ChildI child = rot_perm[child_sfc];
          ot::RotI cRot = orientLookup[child];

          myContribution += countSubtreeSizes(
              points, tempSplitters[child_sfc], tempSplitters[child_sfc+1],
              sLev+1, eLev, cRot,
              order, outSubtreeSizes);
        }
      }
      else            // Recurse with special handling for children of level 0.
      {
        for (ot::ChildI child_sfc = 0; child_sfc < numChildren; child_sfc++)
        {
          ot::ChildI child = rot_perm[child_sfc];
          ot::RotI cRot = orientLookup[child];

          myContribution += countSubtreeSizes(
              points, tempSplitters[child_sfc], tempSplitters[child_sfc+1],
              sLev+1, eLev, pRot,
              order, outSubtreeSizes);
        }
      }
    }

    // Clean up the re-usable buffer if at top of recursion.
    if (deleteBuffer == true)
    {
      delete [] longBuffer;
      longBuffer.topBegin = 0;
      longBuffer.buf = nullptr;
    }

    // Set/return contribution.
    outSubtreeSizes.resize(sLev+1, 0);
    if (myContribution > outSubtreeSizes[sLev])
      outSubtreeSizes[sLev] = myContribution;

    return myContribution;
  }// end function()


}//namespace fem

