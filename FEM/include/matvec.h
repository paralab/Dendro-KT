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
#include "sfcTreeLoop_matvec_io.h"
#include "tnUtils.h"

#include<iostream>
#include<unordered_set>
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

  template <typename da, typename TN>
  using EleOpTForVertices = std::function<void( da *out, unsigned int ndofs, const TN& leafOctant, const TN* nodeCoords, const int numNodes, const std::unordered_set<int>& vertexRanks,  const unsigned int eleOrder )>;

  template <typename da, typename TN>
  using EleOpTForMiddleNode = std::function<void( const da* in, da *out, unsigned int ndofs, const TN& leafOctant, const TN* nodeCoords, const int numNodes, const std::unordered_set<int>& vertexRanks, const unsigned int eleOrder )>;

  template <typename da, unsigned int dim>
  /// using EleOpT = std::function<void(const da *in, da *out, unsigned int ndofs, double *coords, double scale)>;
  using EleOpTWithNodeConf = std::function<void(const da *in, da *out, unsigned int ndofs, const std::bitset< intPow( 3, dim ) >& nodeConf, const double *coords, double scale, bool isElementBoundary, const int eleOrder)>;


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

    template <typename T, typename TN, typename RE, unsigned int dim>
    void matvec(const T* vecIn, T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN *treePartPtr, size_t treePartSz, const TN &partFront, const TN &partBack, EleOpTWithNodeConf<T, dim> eleOp, double scale, const RE* refElement, int version)
    /// void matvec_sfctreeloop(const T* vecIn, T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN &partFront, const TN &partBack, EleOpT<T> eleOp, double scale, const RE* refElement)
    {

      if( version == 0 ) {
        matvec( vecIn, vecOut, ndofs, coords, sz, treePartPtr, treePartSz, partFront, partBack, eleOp, scale, refElement );
      }
      else if( version == 1 ) {

        // Initialize output vector to 0.
        std::fill(vecOut, vecOut + ndofs*sz, 0);

        using C = typename TN::coordType;  // If not unsigned int, error.
        // constexpr unsigned int dim = ot::coordDim((TN*){});
        const unsigned int eleOrder = refElement->getOrder();
        const unsigned int npe = intPow(eleOrder+1, dim);

        ot::MatvecBase<dim, T> treeloop(sz, ndofs, eleOrder, coords, vecIn, treePartPtr, treePartSz, partFront, partBack);
        std::vector<T> leafResult(ndofs*npe, 0.0);

        std::unordered_set<int> vertexAndMiddleNodeSet;
        refElement->populateSecondOrderVertexSet( vertexAndMiddleNodeSet );

        if( dim == 2 )
          vertexAndMiddleNodeSet.insert( 4 );
        else if( dim == 3 )
          vertexAndMiddleNodeSet.insert( 13 );

        while (!treeloop.isFinished())
        {
          if (treeloop.isPre() && treeloop.subtreeInfo().isLeaf( version ))
          {

  #ifdef DENDRO_KT_MATVEC_BENCH_H
            bench::t_elemental.start();
  #endif

            const TN* nodeCoords = treeloop.subtreeInfo().readNodeCoordsIn();
            const int numNodes = treeloop.subtreeInfo().getNumNodesIn();
            const TN& currTree = treeloop.subtreeInfo().getCurrentSubtree();

            leafResult.resize( ndofs*numNodes );
            std::fill( leafResult.begin(), leafResult.end(), 0.0 );

            std::bitset<intPow( 3, dim )> nodeConf;

            for( int idx = 0; idx < numNodes; idx++ ) {
              const unsigned int nodeRank = ot::TNPoint<unsigned int, dim>::get_lexNodeRank( currTree,
                                                           nodeCoords[idx],
                                                           eleOrder + 1 );

              if( vertexAndMiddleNodeSet.find( nodeRank ) == vertexAndMiddleNodeSet.end() ) {

                nodeConf |= 1 << nodeRank;
              
              }

            }

            const double * nodeCoordsFlat = treeloop.subtreeInfo().getNodeCoords();
            const T * nodeValsFlat = treeloop.subtreeInfo().readNodeValsIn();

            eleOpWithNodeConf(nodeValsFlat, &(*leafResult.begin()), ndofs, nodeConf, nodeCoordsFlat, scale, treeloop.subtreeInfo().isElementBoundary());

            treeloop.subtreeInfo().overwriteNodeValsOut(&(*leafResult.begin()));

  #ifdef DENDRO_KT_MATVEC_BENCH_H
            bench::t_elemental.stop();
  #endif

            treeloop.next( version );
          }
          else
            treeloop.step( version );
        }

        size_t writtenSz = treeloop.finalize(vecOut);

        if (sz > 0 && writtenSz == 0)
          std::cerr << "Warning: matvec() did not write any data! Loop misconfigured?\n";
      }
      else {
        throw std::invalid_argument( "Only 0 or 1 allowed for version number" );
      }
    }



    template <typename T, typename TN, typename RE>
    void matvecForVertexNode( T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN *treePartPtr, size_t treePartSz, const TN &partFront, const TN &partBack, EleOpTForVertices<T, TN> eleOp, double scale, const RE* refElement)
    /// void matvec_sfctreeloop(const T* vecIn, T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN &partFront, const TN &partBack, EleOpT<T> eleOp, double scale, const RE* refElement)
    {

      assert( ndofs == 1 );

      // Initialize output vector to 0.
      std::fill(vecOut, vecOut + ndofs*sz, 0);

      using C = typename TN::coordType;  // If not unsigned int, error.
      constexpr unsigned int dim = ot::coordDim((TN*){});
      const unsigned int eleOrder = refElement->getOrder();
      const unsigned int npe = intPow(eleOrder+1, dim);

      ot::MatvecBaseOut<dim, T, true> treeloop(sz, ndofs, eleOrder, false, 0, coords, treePartPtr, treePartSz, partFront, partBack);
     
      std::vector<T> leafResult(ndofs*npe, 0);

      std::unordered_set<int> vertexSet;
      refElement->populateSecondOrderVertexSet( vertexSet );

      while (!treeloop.isFinished())
      {
        if (treeloop.isPre() && treeloop.subtreeInfo().isLeaf())
        {

#ifdef DENDRO_KT_MATVEC_BENCH_H
          bench::t_elemental.start();
#endif

          const TN* nodeCoords = treeloop.subtreeInfo().readNodeCoordsIn();

          const int numNodes = treeloop.subtreeInfo().getNumNodesIn();

          const TN& currTree = treeloop.subtreeInfo().getCurrentSubtree();

          // BaseT::getCurrentFrame().template getMyInputHandle<0>(),
          //                    BaseT::getCurrentSubtree(),

          eleOp(&(*leafResult.begin()), ndofs, currTree, nodeCoords, numNodes, vertexSet, eleOrder);

          // const unsigned int nodeRank = TNPoint<unsigned int, dim>::get_lexNodeRank(
          //       childSubtreesSFC[child_sfc],
          //       myNodes[nIdx],
          //       m_eleOrder );

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

    template <typename T, typename TN, typename RE>
    void matvecForMiddleNode(const T* vecIn, T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN *treePartPtr, size_t treePartSz, const TN &partFront, const TN &partBack, EleOpTForMiddleNode<T, TN> eleOp, double scale, const RE* refElement)
    /// void matvec_sfctreeloop(const T* vecIn, T* vecOut, unsigned int ndofs, const TN *coords, unsigned int sz, const TN &partFront, const TN &partBack, EleOpT<T> eleOp, double scale, const RE* refElement)
    {

      assert( ndofs == 1 );

      using C = typename TN::coordType;  // If not unsigned int, error.
      constexpr unsigned int dim = ot::coordDim((TN*){});
      const unsigned int eleOrder = refElement->getOrder();
      const unsigned int npe = intPow(eleOrder+1, dim);

      ot::MatvecBase<dim, T> treeloop(sz, ndofs, eleOrder, coords, vecIn, treePartPtr, treePartSz, partFront, partBack);
     
      std::vector<T> leafResult(ndofs*npe, 0);

      std::unordered_set<int> vertexSet;
      refElement->populateSecondOrderVertexSet( vertexSet );

      while (!treeloop.isFinished())
      {
        if (treeloop.isPre() && treeloop.subtreeInfo().isLeaf())
        {

#ifdef DENDRO_KT_MATVEC_BENCH_H
          bench::t_elemental.start();
#endif

          const TN* nodeCoords = treeloop.subtreeInfo().readNodeCoordsIn();

          const TN& currTree = treeloop.subtreeInfo().getCurrentSubtree();

          const T * nodeValsFlat = treeloop.subtreeInfo().readNodeValsIn();

          const int numNodes = treeloop.subtreeInfo().getNumNodesIn();

          eleOp(nodeValsFlat, &(*leafResult.begin()), ndofs, currTree, nodeCoords, numNodes, vertexSet, eleOrder);

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
