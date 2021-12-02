#ifndef DENDRO_LOOP_WRAPPER_H
#define DENDRO_LOOP_WRAPPER_H

#include "sfcTreeLoop_matvec_io.h"
#include "gridWrapper.h"
#include "vector.h"

namespace ot
{
  //
  // ElementLoopIn
  //
  template <unsigned int dim, typename ValT>
  class ElementLoopIn
  {
    private:
      ot::MatvecBaseIn<dim, ValT> m_loop;
    public:
      ElementLoopIn(const GridWrapper<dim> &mesh,
                    const Vector<ValT> &ghostedVec);
      ot::MatvecBaseIn<dim, ValT> & loop() { return m_loop; }
  };

  //
  // ElementLoopOut
  //
  template <unsigned int dim, typename ValT>
  class ElementLoopOut
  {
    private:
      ot::MatvecBaseOut<dim, ValT, true> m_loop;
    public:
      ElementLoopOut(const GridWrapper<dim> &mesh, unsigned int ndofs);
      ot::MatvecBaseOut<dim, ValT, true> & loop() { return m_loop; }
  };

  //
  // ElementLoopOutOverwrite
  //
  template <unsigned int dim, typename ValT>
  class ElementLoopOutOverwrite
  {
    private:
      ot::MatvecBaseOut<dim, ValT, false> m_loop;
    public:
      ElementLoopOutOverwrite(const GridWrapper<dim> &mesh, unsigned int ndofs);
      ot::MatvecBaseOut<dim, ValT, false> & loop() { return m_loop; }
  };
}


namespace ot
{
  // ElementLoopIn::ElementLoopIn()
  template <unsigned int dim, typename ValT>
  ElementLoopIn<dim, ValT>::ElementLoopIn(
      const GridWrapper<dim> &mesh,
      const Vector<ValT> &ghostedVec)
    :
      m_loop(mesh.da()->getTotalNodalSz(),
             ghostedVec.ndofs(),
             mesh.da()->getElementOrder(),
             false,
             0,
             mesh.da()->getTNCoords(),
             ghostedVec.ptr(),
             mesh.distTree()->getTreePartFiltered().data(),
             mesh.distTree()->getTreePartFiltered().size())
  { }

  // ElementLoopOut::ElementLoopOut()
  template <unsigned int dim, typename ValT>
  ElementLoopOut<dim, ValT>::ElementLoopOut(
      const GridWrapper<dim> &mesh, unsigned int ndofs)
    :
      m_loop(mesh.da()->getTotalNodalSz(),
             ndofs,
             mesh.da()->getElementOrder(),
             false,
             0,
             mesh.da()->getTNCoords(),
             mesh.distTree()->getTreePartFiltered().data(),
             mesh.distTree()->getTreePartFiltered().size())
  { }

  // ElementLoopOutOverwrite::ElementLoopOutOverwrite()
  template <unsigned int dim, typename ValT>
  ElementLoopOutOverwrite<dim, ValT>::ElementLoopOutOverwrite(
      const GridWrapper<dim> &mesh, unsigned int ndofs)
    :
      m_loop(mesh.da()->getTotalNodalSz(),
             ndofs,
             mesh.da()->getElementOrder(),
             false,
             0,
             mesh.da()->getTNCoords(),
             mesh.distTree()->getTreePartFiltered().data(),
             mesh.distTree()->getTreePartFiltered().size())
  { }
}

#endif//DENDRO_LOOP_WRAPPER_H
