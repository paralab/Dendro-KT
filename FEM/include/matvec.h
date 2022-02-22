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

#include "sfcTreeLoop_matvec.h"

#include<iostream>
#include<functional>


namespace fem
{
  using RankI = ot::RankI;
  using ChildI = ot::ChildI;
  using LevI = ot::LevI;
  using RotI = ot::RotI;

  // TODO support moving/accumulating tuples with dof>1

  /**
   * @tparam da: Type of scalar components of data.
   * @param coords: Flattened array of coordinate tuples, [xyz][xyz][...]
   */
  template <typename da>
  /// using EleOpT = std::function<void(const da *in, da *out, unsigned int ndofs, double *coords, double scale)>;
  using EleOpT = std::function<void(const da *in, da *out, unsigned int ndofs, const double *coords, double scale, bool isElementBoundary)>;

    /**
     * @brief : mesh-free matvec
     * @param [in] vecIn: input vector (local vector)
     * @param [out] vecOut: output vector (local vector) 
     * @param [in] ndofs: number of components at each node, 3 for XYZ XYZ
     * @param [in] coords: coordinate points for the partition
     * @param [in] sz: number of points
     * @param [in] partFront: front TreeNode in local segment of tree partition.
     * @param [in] partBack: back TreeNode in local segment of tree partition.
     * @param [in] eleOp: Elemental operator (i.e. elemental matvec)
     * @param [in] refElement: reference element.
     */
    template <typename T, typename TN, typename RE>
    void matvec(const T* vecIn, T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN *treePartPtr, size_t treePartSz, const TN &partFront, const TN &partBack, EleOpT<T> eleOp, double scale, const RE* refElement)
    /// void matvec_sfctreeloop(const T* vecIn, T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN &partFront, const TN &partBack, EleOpT<T> eleOp, double scale, const RE* refElement)
    {
      // Initialize output vector to 0.
      std::fill(vecOut, vecOut + ndofs*sz, 0);

      using C = typename TN::coordType;  // If not unsigned int, error.
      constexpr unsigned int dim = ot::coordDim((TN*){});
      const unsigned int eleOrder = refElement->getOrder();
      const unsigned int npe = intPow(eleOrder+1, dim);

      ot::MatvecBase<dim, T> treeloop(sz, ndofs, eleOrder, coords, vecIn, treePartPtr, treePartSz, partFront, partBack);
      std::vector<T> leafResult(ndofs*npe, 0.0);

      while (!treeloop.isFinished())
      {
        if (treeloop.isPre() && treeloop.subtreeInfo().isLeaf())
        {

#ifdef DENDRO_KT_MATVEC_BENCH_H
          bench::t_elemental.start();
#endif

          const double * nodeCoordsFlat = treeloop.subtreeInfo().getNodeCoords();
          const T * nodeValsFlat = treeloop.subtreeInfo().readNodeValsIn();

          eleOp(nodeValsFlat, &(*leafResult.begin()), ndofs, nodeCoordsFlat, scale, treeloop.subtreeInfo().isElementBoundary());

          treeloop.subtreeInfo().overwriteNodeValsOut(&(*leafResult.begin()));

#ifdef DENDRO_KT_MATVEC_BENCH_H
          bench::t_elemental.stop();
#endif

          treeloop.next();
        }
        else
          treeloop.step();
      }

      size_t writtenSz = treeloop.finalize(vecOut);

      if (sz > 0 && writtenSz == 0)
        std::cerr << "Warning: matvec() did not write any data! Loop misconfigured?\n";
    }



} // end of namespace fem


#endif
