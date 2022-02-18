
/**
 * @author Masado Ishii
 * @date 2022-02-17
 */

#ifndef DENDRO_KT_LERP_HPP
#define DENDRO_KT_LERP_HPP

#include "treeNode.h"
#include "distTree.h"
#include "oda.h"

namespace ot
{
  template <unsigned dim>
  void lerp(
      const DistTree<unsigned, dim> &from_dtree,
      const DA<dim> *from_da,
      const int ndofs,
      const std::vector<double> &from_vec,
      const DistTree<unsigned, dim> &to_dtree,
      const DA<dim> *to_da,
      std::vector<double> &to_vec);
}


// =======================================================================

namespace ot
{
  // lerp()
  template <unsigned dim>
  void lerp(
      const DistTree<unsigned, dim> &from_dtree,
      const DA<dim> *from_da,
      const int ndofs,
      const std::vector<double> &from_vec,
      const DistTree<unsigned, dim> &to_dtree,
      const DA<dim> *to_da,
      std::vector<double> &to_vec)
  {
    assert(from_vec.size() == ndofs * from_da->getLocalNodalSz());
    assert(to_vec.size() == ndofs * to_da->getLocalNodalSz());

  }

}//namespace ot

#endif//DENDRO_KT_LERP_HPP
