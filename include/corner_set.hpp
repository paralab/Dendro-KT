
/**
 * @author Masado Ishii
 * @date   2023-03-29
 * @brief  A corner set is a 2x2x... grid, maybe children or vertices of a cell.
 */

#ifndef DENDRO_KT_CORNER_SET_HPP
#define DENDRO_KT_CORNER_SET_HPP

// Function declaration for linkage purposes.
inline void link_corner_set_tests() {};

#include "include/bitset_wrapper.hpp"

#include "doctest/doctest.h"
#include "include/mathUtils.h"

#include <iostream>
#include <sstream>

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  template <int dim>
  struct CornerSet : public BitsetWrapper<intPow(2, dim), CornerSet<dim>>
  {
    // BitsetWrapper::block is 2x2, 2x2x2, 2x2x2x2 local grids.

    using Base = BitsetWrapper<intPow(2, dim), CornerSet<dim>>;
    using Base::Base;  // inherit constructors

    // Class properties.
    static constexpr size_t n_corners();

    // Clear upper or lower hyperplane.
    inline CornerSet cleared_up(int axis) const;
    inline CornerSet cleared_down(int axis) const;

    // Shift along an axis, like Matlab circshift(), but overflow is truncated.
    inline CornerSet shifted_up(int axis) const;
    inline CornerSet shifted_down(int axis) const;
  };


  DOCTEST_TEST_SUITE("CornerSet")
  {
    DOCTEST_TEST_CASE("Shifted empty is empty")
    {
      for (int axis = 0; axis < 3; ++axis)
      {
        CHECK( CornerSet<3>::empty().shifted_up(axis).none() );
      }

      for (int axis = 0; axis < 3; ++axis)
      {
        CHECK( CornerSet<3>::empty().shifted_down(axis).none() );
      }
    }

    DOCTEST_TEST_CASE("Shifted full is neither full nor empty")
    {
      for (int axis = 0; axis < 3; ++axis)
      {
        auto shifted = CornerSet<3>::full().shifted_up(axis);
        CHECK_FALSE( shifted.all() );
        CHECK_FALSE( shifted.none() );
      }
    }

  }//DOCTEST_TEST_SUITE("CornerSet")




}


// =============================================================================
// Implementation
// =============================================================================

namespace ot
{

  // ---------------------------------------------------------------------------
  // Class properties.
  // ---------------------------------------------------------------------------
  template <int dim>
  constexpr size_t CornerSet<dim>::n_corners()
  {
    return intPow(2, dim);
  }


  // ---------------------------------------------------------------------------
  // Clear upper or lower hyperplane.
  // ---------------------------------------------------------------------------

  template <int dim>
  inline CornerSet<dim> CornerSet<dim>::cleared_up(int axis) const
  {
    return { this->block & ~detail::hyperplane<dim, 2>(axis, 1) };
  }

  template <int dim>
  inline CornerSet<dim> CornerSet<dim>::cleared_down(int axis) const
  {
    return { this->block & ~detail::hyperplane<dim, 2>(axis, 0) };
  }


  // ---------------------------------------------------------------------------
  // Shift along an axis.
  // ---------------------------------------------------------------------------

  template <int dim>
  inline CornerSet<dim> CornerSet<dim>::shifted_up(int axis) const
  {
    // Bits in the highest plane normal to axis are truncated.
    CornerSet<dim> result = this->cleared_up(axis);

    // Shift up.
    const int stride = intPow(2, axis);
    result.block <<= stride;

    return result;
  }

  template <int dim>
  inline CornerSet<dim> CornerSet<dim>::shifted_down(int axis) const
  {
    // Bits in the lowest plane normal to axis are truncated.
    CornerSet<dim> result = this->cleared_down(axis);

    // Shift down.
    const int stride = intPow(2, axis);
    result.block >>= stride;

    return result;
  }
}


#endif//DENDRO_KT_CORNER_SET_HPP
