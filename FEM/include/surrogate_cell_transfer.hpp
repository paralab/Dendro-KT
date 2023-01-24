

/**
 * @author Masado Ishii
 * @date 2022-03-03
 */

#ifndef DENDRO_KT_SURROGATE_CELL_TRANSFER_HPP
#define DENDRO_KT_SURROGATE_CELL_TRANSFER_HPP

#include "coarseToFine.hpp"
#include "parUtils.h"

namespace fem
{
  template <unsigned dim>
  void cell_transfer_refine(
      const ot::DA<dim> *coarse_da,
      const int cell_ndofs,
      const double *coarse_cell_dofs,
      const ot::DistTree<unsigned, dim> &surrogate_dtree,
      const ot::DA<dim> *surrogate_da,
      const ot::DistTree<unsigned, dim> &fine_dtree,
      const ot::DA<dim> *fine_da,
      double *fine_cell_dofs);

  // cell_transfer_coarsen()
  template <unsigned dim>
  void cell_transfer_coarsen(
      const ot::DistTree<unsigned, dim> &fine_dtree,
      const ot::DA<dim> *fine_da,
      const int cell_ndofs,
      const double *fine_cell_dofs,
      const ot::DistTree<unsigned, dim> &surrogate_dtree,
      const ot::DA<dim> *surrogate_da,
      const ot::DA<dim> *coarse_da,
      double *coarse_cell_dofs,
      CellCoarsen cell_coarsening);  // CellCoarsen::Copy or CellCoarsen::Sum
}


namespace fem
{
  // cell_transfer_refine()
  template <unsigned dim>
  void cell_transfer_refine(
      const ot::DA<dim> *coarse_da,
      const int cell_ndofs,
      const double *coarse_cell_dofs,
      const ot::DistTree<unsigned, dim> &surrogate_dtree,
      const ot::DA<dim> *surrogate_da,
      const ot::DistTree<unsigned, dim> &fine_dtree,
      const ot::DA<dim> *fine_da,
      double *fine_cell_dofs)
  {
    DOLLAR("cell_transfer_refine()");
    // Surrogate must be coarse partitioned by fine.
    assert(coarse_da->getGlobalElementSz() == surrogate_da->getGlobalElementSz());

    std::vector<double> surr_cell_dofs(surrogate_da->getLocalElementSz() * cell_ndofs);
    {DOLLAR("shift()");
    par::shift(
        coarse_da->getGlobalComm(),
        coarse_cell_dofs,
        coarse_da->getLocalElementSz(),
        coarse_da->getGlobalElementBegin(),
        surr_cell_dofs.data(),
        surrogate_da->getLocalElementSz(),
        surrogate_da->getGlobalElementBegin(),
        cell_ndofs);
    }

    local_cell_transfer(
        surrogate_dtree.getTreePartFiltered(),
        surr_cell_dofs.data(),
        cell_ndofs,
        fine_dtree.getTreePartFiltered(),
        fine_cell_dofs,
        CellCoarsen::Undefined);
  }


  // cell_transfer_coarsen()
  template <unsigned dim>
  void cell_transfer_coarsen(
      const ot::DistTree<unsigned, dim> &fine_dtree,
      const ot::DA<dim> *fine_da,
      const int cell_ndofs,
      const double *fine_cell_dofs,
      const ot::DistTree<unsigned, dim> &surrogate_dtree,
      const ot::DA<dim> *surrogate_da,
      const ot::DA<dim> *coarse_da,
      double *coarse_cell_dofs,
      CellCoarsen cell_coarsening)
  {
    // Surrogate must be coarse partitioned by fine.
    assert(coarse_da->getGlobalElementSz() == surrogate_da->getGlobalElementSz());

    std::vector<double> surr_cell_dofs(surrogate_da->getLocalElementSz() * cell_ndofs, 0);

    local_cell_transfer(
        fine_dtree.getTreePartFiltered(),
        fine_cell_dofs,
        cell_ndofs,
        surrogate_dtree.getTreePartFiltered(),
        surr_cell_dofs.data(),
        cell_coarsening);

    {DOLLAR("shift()");
    par::shift(
        coarse_da->getGlobalComm(),
        surr_cell_dofs.data(),
        surrogate_da->getLocalElementSz(),
        surrogate_da->getGlobalElementBegin(),
        coarse_cell_dofs,
        coarse_da->getLocalElementSz(),
        coarse_da->getGlobalElementBegin(),
        cell_ndofs);
    }
  }
}



#endif//DENDRO_KT_SURROGATE_CELL_TRANSFER_HPP
