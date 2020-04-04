/**
 * @author: Masado Ishii
 * School of Computing, University of Utah
 * 
*/

#ifndef DENDRO_KT_INTERGRID_TRANSFER_H
#define DENDRO_KT_INTERGRID_TRANSFER_H

#include "tsort.h"    // RankI, ChildI, LevI, RotI
#include "nsort.h"    // TNPoint
#include "sfcTreeLoop_matvec_io.h"
#include "refel.h"

#include<iostream>
#include<functional>

namespace fem
{
  // MeshFreeInputContext
  template <typename DofT, typename TN>
  struct MeshFreeInputContext
  {
    const DofT *vecIn;
    const TN *coords;
    unsigned int sz;
    const TN &partFront;
    const TN &partBack;
  };

  // MeshFreeOutputContext
  template <typename DofT, typename TN>
  struct MeshFreeOutputContext
  {
    DofT * const vecOut;
    const TN *coords;
    unsigned int sz;
    const TN &partFront;
    const TN &partBack;
  };


  /**
   * intergridTransfer()
   */
  template <typename DofT, typename TN>
  void intergridTransfer(MeshFreeInputContext<DofT, TN> in,
                         MeshFreeOutputContext<DofT, TN> out,
                         unsigned int ndofs,
                         /// EleOpT<DofT> eleOp,
                         /// double scale,
                         const RefElement *refElement)
  {
    // Initialize output vector to 0.
    std::fill(out.vecOut, out.vecOut + ndofs * out.sz, 0);

    using C = typename TN::coordType;    // If not unsigned int, error.
    constexpr unsigned int dim = TN::coordDim;
    const unsigned int eleOrder = refElement->getOrder();
    const unsigned int npe = intPow(eleOrder+1, dim);

    const bool visitEmpty = true;
    ot::MatvecBaseIn<dim, DofT> treeLoopIn(in.sz, ndofs, eleOrder, visitEmpty, in.coords, in.vecIn, in.partFront, in.partBack);
    ot::MatvecBaseOut<dim, DofT> treeLoopOut(out.sz, ndofs, eleOrder, visitEmpty, out.coords, out.partFront, out.partBack);

    while (!treeLoopOut.isFinished())
    {
      assert(treeLoopIn.getCurrentSubtree() == treeLoopOut.getCurrentSubtree());

      if (!treeLoopIn.isPre() && !treeLoopOut.isPre())
      {
        treeLoopIn.next();
        treeLoopOut.next();
      }
      else if (treeLoopIn.isPre() && treeLoopOut.isPre())
      {
        // At least one tree not at leaf, need to descend.
        // If one is but not the other, the interpolations are handled by tree loop.
        if (!treeLoopIn.subtreeInfo().isLeafOrLower() ||
            !treeLoopOut.subtreeInfo().isLeafOrLower())
        {
          treeLoopIn.step();
          treeLoopOut.step();
        }

        // Both leafs, can directly transfer.
        else
        {
          treeLoopOut.subtreeInfo().overwriteNodeValsOut( treeLoopIn.subtreeInfo().readNodeValsIn() );
        }
      }
      else
      {
        std::cerr << "Error: intergridTransfer() traversals out of sync." << std::endl;
        assert(false);
      }
    }

    size_t writtenSz = treeLoopOut.finalize(out.vecOut);

    if (out.sz > 0 && writtenSz == 0)
      std::cerr << "Warning: intergridTransfer did not write any data! Loop misconfigured?\n";
  }
}

#endif//DENDRO_KT_INTERGRID_TRANSFER_H
