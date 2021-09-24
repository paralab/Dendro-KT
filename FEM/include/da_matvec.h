
#ifndef DENDRO_KT_DA_MATVEC_H
#define DENDRO_KT_DA_MATVEC_H

#include "oda.h"
#include "subset.h"
#include "matvec.h"


namespace fem
{
  // Shorthand for matvec()
  template<unsigned int dim, typename T, typename TN>
  inline void da_matvec(
      const ot::DA<dim> *da,
      const std::vector<TN> *octList,
      const T* vecIn,
      T* vecOut,
      unsigned int ndofs,
      EleOpT<T> eleOp,
      double scale)
  {
    const TN *coords = da->getTNCoords();
    const size_t sz = da->getTotalNodalSz();

    const TN *treePartPtr = &(*octList->begin());
    const size_t treePartSz = octList->size();

#ifndef BUILD_WITH_AMAT
    matvec(
        vecIn, vecOut, ndofs,
        coords, sz,
        treePartPtr, treePartSz,
        eleOp,
        scale,
        da->getReferenceElement());
#else
    const std::vector<bool> &explicitFlags = da->explicitFlags();

    if (explicitFlags.size() == 0)
    {
      // Regular traversal-based matvec, no need to separate for amat.
      matvec(
          vecIn, vecOut, ndofs,
          coords, sz,
          treePartPtr, treePartSz,
          eleOp,
          scale,
          da->getReferenceElement());
    }
    else
    {
      std::fill_n(vecOut, sz, 0);  // init output

      //TODO TODO these things should not be recomputed every matvec!
      //
      const std::vector<TN> oct_exp =
          ot::filter_where(*octList, explicitFlags, true);

      const std::vector<TN> oct_imp =
          ot::filter_where(*octList, explicitFlags, false);

      const std::vector<size_t> ghostedIdx_exp =
          ot::index_nodes_where_element(*da, *octList, explicitFlags, true);

      const std::vector<size_t> ghostedIdx_imp =
          ot::index_nodes_where_element(*da, *octList, explicitFlags, false);

      const std::vector<TN> coords_exp = ot::gather(coords, ghostedIdx_exp);
      const std::vector<TN> coords_imp = ot::gather(coords, ghostedIdx_imp);

      // Do implicit (traversal-based) matvec.
      const std::vector<T> vecIn_imp = ot::gather_ndofs(vecIn, ghostedIdx_imp, ndofs);
      std::vector<T> vecOut_imp(vecIn_imp.size());

      matvec(
          vecIn_imp.data(), vecOut_imp.data(), ndofs,
          coords_imp.data(), coords_imp.size(),
          oct_imp.data(), oct_imp.size(),
          eleOp,
          scale,
          da->getReferenceElement());

      const auto overwrite =
          [](const T &new_v, const T &old_v) { return new_v; };

      // Add results of implicit matvec to total output.
      ot::scatter_ndofs(
          vecOut_imp.data(),
          overwrite,
          vecOut, ghostedIdx_imp, ndofs);

      // I believe things so far

      // matvec amat based  //TODO new map


      //
      //TODO TODO


      //TODO so I need
      //  - subsetNodes: Takes DA(nodes) x subset(octlist) -> indices of nodes incident on octlist
      //    - map to read/set/add with subset and main list
      //  - abstraction for info to node traversal loop (given subset octlist and subset nodes)
      //  - DA create amat for subset octlist and subset nodes...


    }
#endif//BUILD_WITH_AMAT
  }

}

#endif//DENDRO_KT_DA_MATVEC_H
