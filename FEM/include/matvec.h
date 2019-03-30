/**
 * @author: Milinda Fernando
 * School of Computing, University of Utah
 * @brief: Contains utility function to traverse the k-tree in the SFC order, 
 * These utility functions will be used to implement the mesh free matvec. 
 * 
 * 
*/

#ifndef DENDRO_KT_TRAVERSE_H
#define DENDRO_KT_TRAVERSE_H

#include "tsort.h"    // RankI, ChildI, LevI, RotI
#include "nsort.h"    // TNPoint

#include<iostream>
#include<functional>


namespace fem
{
  using RankI = ot::RankI;
  using ChildI = ot::ChildI;
  using LevI = ot::LevI;
  using RotI = ot::RotI;

    // Declaring the matvec at the top.
    template<typename T,typename TN, typename RE,  unsigned int dim>
    void matvec(const T* vecIn, T* vecOut, const TN* coords, unsigned int sz, const TN &partFront, const TN &partBack, std::function<void(const T*, T*, TN* coords)> eleOp, const RE* refElement);

    template<typename T,typename TN, typename RE,  unsigned int dim>
    void matvec_rec(const T* vecIn, T* vecOut, const TN* coords, TN subtreeRoot, RotI pRot, unsigned int sz, const TN &partFront, const TN &partBack, std::function<void(const T*, T*, TN* coords)> eleOp, const RE* refElement, const T* pVecIn, const TN* pCoords, unsigned int pSz);

    /**
     * @brief: top_down bucket function
     * @param [in] coords: input points
     * @param [out] coords_dup: input points bucketed/duplicated for all children.
     * @param [in] vec: input vector 
     * @param [out] vec_dup: input vector bucketed/duplicated for all children.
     * @param [in] sz: number of points (in points)
     * @param [out] offsets: bucket offsets in bucketed arrays, length would be 1u<<dim
     * @param [out] counts: bucket counts
     * @param [in] refElement: Reference element
     * @param [out] scattermap: bucketing scatter map from parent to child: scattermap[node_i * (1u<<dim) + destIdx] == destChild_sfc; A child of -1 (aka max unsigned value) means empty.
     * @return: Return true when we hit a leaf element. 
     */
    template<typename T,typename TN, typename RE, unsigned int dim>
    bool top_down(const TN* coords, std::vector<TN> &coords_dup, const T* vec, std::vector<T> &vec_dup, unsigned int sz, unsigned int* offsets, unsigned int* counts, const RE* refElement, std::vector<ChildI> &scattermap, const TN subtreeRoot, RotI pRot)
    {
      /**
       * @author Masado Ishii
       * @author Milinda Fernando
       */

      // Read-only inputs:                         coords,   vec,    refElement.
      // Pre-allocated outputs:                    offsets,     counts.
      // Internally allocated (TODO pre-allocate): coords_dup,  vec_dup,   scattermap.

      // This method performs bucketing with duplication on closed subtrees.
      // The (closed) interfaces between multiple children are duplicated to those children.

      // The bucketing scattermap is a (linearized) 2D array,
      //     {send input node i to which children?} == {scattermap[i][0] .. scattermap[i][nDup-1]}
      // where nDup is a power of 2 between 1 and 1u<<dim, depending on the kface dimension.
      // In general this information is jagged. However, to simplify linearization,
      // for now we will inflate the number of columns in each row to 1u<<dim (an upper bound).

      LevI pLev = subtreeRoot.getLevel();
      const unsigned int numChildren=1u<<dim;
      constexpr unsigned int rotOffset = 2*(1u<<dim);  // num columns in rotations[].
      const ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren]; // child_m = rot_perm[child_sfc]
      const ChildI * const rot_inv =  &rotations[pRot*rotOffset + 1*numChildren]; // child_sfc = rot_inv[child_m]

      // 0. Check if this is a leaf element. If so, return true immediately.
      bool isLeaf = true;
      for (RankI ii = 0; ii < sz; ii++)
        if (isLeaf && coords[ii].getLevel() > subtreeRoot.getLevel())
          isLeaf = false;
      if (isLeaf)
        return true;

      std::fill(counts, counts + (1u<<dim), 0);
      std::fill(offsets, offsets + (1u<<dim), 0);

      // 1. (Allocate and) compute the bucketing scattermap.  Also compute counts[].
      scattermap.resize(sz * (1u<<dim));
      std::fill(scattermap.begin(), scattermap.end(), (ChildI) -1);
      for (RankI ii = 0; ii < sz; ii++)
      {
        // TODO these type casts are ugly and they might even induce more copying than necessary.
        std::array<typename TN::coordType,dim> ptCoords;
        ot::TNPoint<typename TN::coordType,dim> pt(1, (coords[ii].getAnchor(ptCoords), ptCoords), coords[ii].getLevel());

        ChildI baseBucket_m = (pt.getMortonIndex(pLev) ^ subtreeRoot.getMortonIndex(pLev))  | pt.getMortonIndex(pLev + 1);  // One of the duplicates.
        ot::CellType<dim> paCellt = pt.get_cellType(pLev);
        ot::CellType<dim> chCellt = pt.get_cellType(pLev+1);

        // Note that dupDim is the number of set bits in dupOrient.
        typename ot::CellType<dim>::FlagType dupOrient = paCellt.get_orient_flag() & ~chCellt.get_orient_flag();
        typename ot::CellType<dim>::FlagType dupDim =    paCellt.get_dim_flag()    -  chCellt.get_dim_flag();

        baseBucket_m = ~dupOrient & baseBucket_m;  // The least Morton-child among all duplicates.

        // Decompose the set bits in dupOrient.
        std::array<ChildI, dim> strides;
        /// strides.fill(0);
        for (int d = 0, dup_d = 0; d < dim; d++)
        {
          strides[dup_d] = (1u<<d);
          dup_d += (bool) (dupOrient & (1u<<d));   // Only advance dup_d if bit(d) is set.
        }

        // Compute the child numbers of duplicates by modulating baseBucket_m.
        // Also add to counts[].
        for (ChildI destChIdx = 0; destChIdx < (1u<<dupDim); destChIdx++)
        {
          ChildI child_m = baseBucket_m;
          for (int dup_d = 0; dup_d < dupDim; dup_d++)
            child_m += strides[dup_d] * (bool) (destChIdx & (1u<<dup_d));

          ChildI child_sfc = rot_inv[child_m];
          scattermap[ii * (1u<<dim) + destChIdx] = child_sfc;

          counts[child_sfc]++;
        }
      }

      // 2. Compute offsets[].
      //    Note: offsets[] and counts[] are preallocated outside this function.
      RankI accum = 0;
      for (ChildI ch = 0; ch < (1u<<dim); ch++)
      {
        offsets[ch] = accum;
        accum += counts[ch];
      }

      // 3. (Allocate and) copy the outputs. Destroys offsets[].
      if (coords_dup.size() < accum)
        coords_dup.resize(accum);
      if (vec_dup.size() < accum)
        vec_dup.resize(accum);

      for (RankI ii = 0; ii < sz; ii++)
      {
        ChildI child_sfc;
        ChildI destChIdx = 0;
        while (destChIdx < (1u<<dim) && (child_sfc = scattermap[ii * (1u<<dim) + destChIdx]) != -1)
        {
          coords_dup[offsets[child_sfc]] = coords[ii];
          vec_dup[offsets[child_sfc]] = vec[ii];
          offsets[child_sfc]++;
          destChIdx++;
        }
      }

      // 4. Recompute offsets[].
      accum = 0;
      for (ChildI ch = 0; ch < (1u<<dim); ch++)
      {
        offsets[ch] = accum;
        accum += counts[ch];
      }

      return false;   // Non-leaf.
    }

    /**
     * 
     * @brief: bottom_up bucket function
     * @param [in] in: input points
     * @param [in] sz: number of points (in points)
     * @param [out] out: bucktet points with duplications on the element boundary. 
     * @param [out] offsets: offsets for bucketed array size would be 1u<<dim
     * @param [out] counts: bucket counts
     * @param [in] refElement: Reference element
     * @return: Return true when we hit the root element. 
     */
    template<typename T,typename TN, typename RE, unsigned int dim>
    bool bottom_up(const TN* coords_in, TN* coords_out, const T* vec_in, T* vec_out, unsigned int sz, const unsigned int* offsets, const unsigned int* counts,const RE* refElement,unsigned int* gathermap)
    {
      //TODO fill in stub
      return false;
    }

    /**
     * @brief : mesh-free matvec
     * @param [in] vecIn: input vector (local vector)
     * @param [out] vecOut: output vector (local vector) 
     * @param [in] coords: coordinate points for the partition
     * @param [in] sz: number of points
     * @param [in] partFront: front TreeNode in local segment of tree partition.
     * @param [in] partBack: back TreeNode in local segment of tree partition.
     * @param [in] eleOp: Elemental operator (i.e. elemental matvec)
     * @param [in] refElement: reference element.
     */
    template<typename T,typename TN, typename RE,  unsigned int dim>
    void matvec(const T* vecIn, T* vecOut, const TN* coords, unsigned int sz, const TN &partFront, const TN &partBack, std::function<void(const T*, T*, TN* coords)> eleOp, const RE* refElement)
    {
      // Top level of recursion.
      TN treeRoot;  // Default constructor constructs root cell.
      matvec_rec<T,TN,RE,dim>(vecIn, vecOut, coords, treeRoot, 0, sz, partFront, partBack, eleOp, refElement, nullptr, nullptr, 0);
    }

    // Recursive implementation.
    template<typename T,typename TN, typename RE,  unsigned int dim>
    void matvec_rec(const T* vecIn, T* vecOut, const TN* coords, TN subtreeRoot, RotI pRot, unsigned int sz, const TN &partFront, const TN &partBack, std::function<void(const T*, T*, TN* coords)> eleOp, const RE* refElement, const T* pVecIn, const TN* pCoords, unsigned int pSz)
    {
        if (sz == 0)
          return;

        const LevI pLev = subtreeRoot.getLevel();
        const int nElePoints = intPow(refElement->getOrder() + 1, dim);

        // 1. initialize the output vector to zero. 
        for(unsigned int i=0;i<sz;i++)
            vecOut[i] = (T)0;

        const unsigned int numChildren=1u<<dim;

        //todo: if possible move these memory allocations out of the function. 
        // One option would be to do a dummy matvec (traversal) and figure out exact memory needed and use pre allocation for
        // subsequent matvecs. 

        unsigned int * offset = new unsigned int[numChildren];
        unsigned int * counts = new unsigned int[numChildren];
        
        // change this to the correct pointer. 
        /// TN* coords_out=NULL;
        /// T* vec_out = NULL;
        /// T* smap = NULL;
        T* gmap = NULL;

        // Hack to get away with no-reallocation without actually doing pre-allocation.
        // Arrangement in memory may be unpredictable, but at least we don't have to reallocate.
        struct InternalBuffers
        {
          std::vector<TN> coords_dup;
          std::vector<T> vec_in_dup;
          std::vector<T> vec_out_contrib;
          std::vector<ChildI> smap;
        };
        static std::vector<InternalBuffers> ibufs;
        static std::vector<T> parentEleBuffer, leafEleBuffer;
        static std::vector<bool> parentEleFill, leafEleFill;

        if (pLev == 0)
        {
            // Initialize static buffers.
            ibufs.resize(m_uiMaxDepth+1);
            parentEleBuffer.resize(nElePoints);
            parentEleFill.resize(nElePoints);
            leafEleBuffer.resize(nElePoints);
            leafEleFill.resize(nElePoints);
        }

        // For now, this may increase the size of coords_dup and vec_in_dup.
        // We can get the proper size for vec_out_contrib from the result.
        bool isLeaf = top_down<T,TN,RE,dim>(coords, ibufs[pLev].coords_dup, vecIn, ibufs[pLev].vec_in_dup, sz, offset, counts, refElement, ibufs[pLev].smap, subtreeRoot, pRot);
        
        if(!isLeaf)
        {
            ibufs[pLev].vec_out_contrib.resize(ibufs[pLev].vec_in_dup.size());

            constexpr unsigned int rotOffset = 2*(1u<<dim);  // num columns in rotations[].
            const ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren]; // child_m = rot_perm[child_sfc]
            const ChildI * const rot_inv =  &rotations[pRot*rotOffset + 1*numChildren]; // child_sfc = rot_inv[child_m]
            const RotI * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

            // As descend subtrees in sfc order, check their intersection with the local partition:
            // Partially contained, disjointly contained, or disjointly uncontained.
            bool chBeforePart = true;
            bool chAfterPart = false;
            const bool willMeetFront = subtreeRoot.isAncestor(partFront);
            const bool willMeetBack = subtreeRoot.isAncestor(partBack);

            // input points counts[i] > nPe assert();
            for(unsigned int child_sfc = 0; child_sfc < numChildren; child_sfc++)
            {
                ChildI child_m = rot_perm[child_sfc];
                RotI   cRot = orientLookup[child_m];
                TN tnChild = subtreeRoot.getChildMorton(child_m);

                chBeforePart &= (bool) (willMeetFront && !(tnChild == partFront || tnChild.isAncestor(partFront)));

                if (!chBeforePart && !chAfterPart)
                    matvec_rec<T,TN,RE,dim>(&(*ibufs[pLev].vec_in_dup.cbegin())      + offset[child_sfc],
                                            &(*ibufs[pLev].vec_out_contrib.begin()) + offset[child_sfc],
                                            &(*ibufs[pLev].coords_dup.cbegin())       + offset[child_sfc],
                                            tnChild, cRot,
                                            counts[child_sfc], partFront, partBack,
                                            eleOp, refElement,
                                            vecIn, coords, sz);

                chAfterPart |= (bool) (willMeetBack && (tnChild == partBack || tnChild.isAncestor(partBack)));
            }
        
        }else
        {
            /// // DEBUG print the leaft element.
            /// fprintf(stderr, "Leaf: (%u) \t%s\n", pLev, subtreeRoot.getBase32Hex(m_uiMaxDepth).data());

            const int polyOrder = refElement->getOrder();

            // Put leaf points in order and check if leaf has all the nodes it needs.
            std::fill(leafEleFill.begin(), leafEleFill.end(), false);
            for (int ii = 0; ii < sz; ii++)
            {
                // TODO these type casts are ugly and they might even induce more copying than necessary.
                std::array<typename TN::coordType,dim> ptCoords;
                ot::TNPoint<typename TN::coordType,dim> pt(1, (coords[ii].getAnchor(ptCoords), ptCoords), coords[ii].getLevel());

                unsigned int nodeRank = pt.get_lexNodeRank(subtreeRoot, polyOrder);
                assert((nodeRank < intPow(polyOrder+1, dim)));

                leafEleFill[nodeRank] = true;
                leafEleBuffer[nodeRank] = vecIn[ii];
            }

            bool leafHasAllNodes = true;
            for (int nr = 0; leafHasAllNodes && nr < intPow(polyOrder+1, dim); nr++)
                leafHasAllNodes = leafEleFill[nr];

            // If not, copy parent nodes and interpolate.
            if (!leafHasAllNodes)
            {
                if (pVecIn == nullptr || pCoords == nullptr || pSz == 0)
                {
                    fprintf(stderr, "Error: Tried to interpolate parent->child, but parent has no nodes!\n");
                    assert(false);
                }

                // Copy parent nodes.
                TN subtreeParent = subtreeRoot.getParent();
                std::fill(parentEleFill.begin(), parentEleFill.end(), false);
                for (int ii = 0; ii < pSz; ii++)
                {
                    if (pCoords[ii].getLevel() != pLev-1)
                      continue;

                    // TODO these type casts are ugly and they might even induce more copying than necessary.
                    std::array<typename TN::coordType,dim> ptCoords;
                    ot::TNPoint<typename TN::coordType,dim> pt(1, (pCoords[ii].getAnchor(ptCoords), ptCoords), pCoords[ii].getLevel());

                    unsigned int nodeRank = pt.get_lexNodeRank(subtreeParent, polyOrder);
                    assert((nodeRank < intPow(polyOrder+1, dim)));
                    /// fprintf(stderr, "parent nodeRank==%u\n", nodeRank);

                    parentEleFill[nodeRank] = true;
                    parentEleBuffer[nodeRank] = pVecIn[ii];
                }

                // Interpolate.

                // TODO capture inside a DEBUG guard.
                  // For now just check a necessary condition:
                  // If a node on leaf is missing, it's corresponding node on parent should be available.
                bool parentHasNeededNodes = true;
                for (int nr = 0; parentHasNeededNodes && nr < intPow(polyOrder+1, dim); nr++)
                    parentHasNeededNodes = leafEleFill[nr] || parentEleFill[nr];

                if (!parentHasNeededNodes)
                {
                    fprintf(stderr, "Error: Tried to interpolate parent->child, but parent is missing needed nodes!\n");
                    assert(false);
                }

                // TODO Perform interpolation using refElement.
            }

            // TODO call eleOp function
            // Note you might need to identify the parent nodes for parent to child interpolation. 
            // (use reference element class for interpolation)

        }

        // call for bottom up;
        //TODO
        /// bool isRoot= bottom_up<T,TN,RE,dim>(coords,coords_out.data(),vecOut,vec_out.data(),offset,counts,refElement,gmap);
        
        delete [] offset;
        delete [] counts;

        if (pLev == 0)
        {
            /// ibufs.clear();
        }
    }
   

} // end of namespace fem


#endif
