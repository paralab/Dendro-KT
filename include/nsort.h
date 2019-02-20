/**
 * @file:nsort.h
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-02-20
 * @brief: Variations of the TreeSort algorithm (tsort.h) for fem-nodes in 4D+.
 */

#ifndef DENDRO_KT_NSORT_H
#define DENDRO_KT_NSORT_H

#include "tsort.h"
#include "treeNode.h"
#include "hcurvedata.h"
#include "parUtils.h"

#include <mpi.h>
#include <vector>
#include <stdio.h>

namespace ot {

  /**@brief Identification of face-interior/edge-interior etc., for nodes.
   * @description There are OuterDim+1 cell dimensions and pow(2,OuterDim) total cell orientations.
   *     The less-significant bits are used to store orientation, while the
   *     more-significant bits are used as redundant quick-access flags for the dimension.
   *     Orientation is represented as a bitvector, with a bit being set if that axis is part of the cell volume.
   *     A 0-cell (point) has all bits set to 0, while a OuterDim-cell (whole volume) has all bits set to 1.
   *     The cell dimension is how many bits are set to 1.
   * @note If the embedding dimension is OuterDim<=5, then unsigned char suffices.
   * @note The template information is not used right now since OuterDim is assumed to be 4 or less.
   *       However, making this a template means that CellType for higher OuterDims than 5 can be made
   *       as template instantiations later.
   */
  template <unsigned int OuterDim>
  struct CellType
  {
    using FlagType = unsigned char;

    /**@brief Constructs a 0-cell (point) inside the hypercube of dimension OuterDim. */
    CellType() { m_flag = 0; }

    // No checks are performed to make sure that c_orient is consistent with c_dim.
    operator FlagType&() { return m_flag; }
    operator FlagType() { return m_flag; }

    FlagType get_dim_flag() { return m_flag >> m_shift; }
    FlagType get_orient_flag() { return m_flag & m_mask; }

    void set_dimFlag(FlagType c_dim) { m_flag = (m_flag & m_mask) | (c_dim << m_shift); }
    void set_orientFlag(FlagType c_orient) { m_flag = (m_flag & (~m_mask)) | (c_orient & m_mask); }

    // TODO void set_flags(FlagType c_orient); // Counts 1s in c_orient and uses count for c_dim.

    FlagType m_flag;

    private:
      static const FlagType m_shift = 4u;
      static const FlagType m_mask = (1u << m_shift) - 1;
  };

  /**@brief TreeNode + extra attributes to keep track of node uniqueness. */
  template <typename T, unsigned int dim>
  class TNPoint : public TreeNode<T,dim>
  {
    public:
      /**
       * @brief Constructs a node at the extreme "lower-left" corner of the domain.
       */
      TNPoint();

      /**
        @brief Constructs a point.
        @param coords The coordinates of the point.
        @param level The level of the point (i.e. level of the element that spawned it).
        @note Uses the "dummy" overload of TreeNode() so that the coordinates are copied as-is.
        */
      TNPoint (const std::array<T,dim> coords, unsigned int level);

      /**@brief Copy constructor */
      TNPoint (const TNPoint & other);

      /**
        @brief Constructs an octant (without checks).
        @param dummy : not used yet.
        @param coords The coordinates of the point.
        @param level The level of the point (i.e. level of the element that spawned it).
      */
      TNPoint (const int dummy, const std::array<T,dim> coords, unsigned int level);

      /** @brief Assignment operator. No checks for dim or maxD are performed. It's ok to change dim and maxD of the object using the assignment operator.*/
      TNPoint & operator = (TNPoint const  & other);

      unsigned char get_isSelected() { return m_isSelected; }
      void set_isSelected(unsigned char isSelected) { m_isSelected = isSelected; }

      CellType<dim> get_cellType();

    protected:
      // Data members.
      /// CellType<dim> m_cellType;   // Can infer cell type from coordinates and level.
      unsigned char m_isSelected;
  };


  namespace util {
    //TODO define separate functions for counting, depending on whether the order is higher than 1 or not.
  }//namespace ot::util

  template <typename T, unsigned int dim>
  struct SFC_NodeSort
  {
    RankI countCGNodes(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order);

  }; // struct SFC_NodeSort



}//namespace ot

#include "nsort.tcc"


#endif//DENDRO_KT_NSORT_H
