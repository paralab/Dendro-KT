
/**
 * @author Masado Ishii
 * @date   2023-03-27
 * @brief  A neighborhood is a 3x3x... surrounding grid of same-level cells.
 */

//future: Rename to neighbor_set.hpp, NeighborSet

#ifndef DENDRO_KT_NEIGHBORHOOD_HPP
#define DENDRO_KT_NEIGHBORHOOD_HPP

// Function declaration for linkage purposes.
inline void link_neighborhood_tests() {};

#include "include/bitset_wrapper.hpp"

#include "include/mathUtils.h"

#include <bitset>
#include <iostream>
#include <sstream>


// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  /** @brief Multidimensional neighborhood, including by edges and corners. */
  template <int dim>
  struct Neighborhood : public BitsetWrapper<intPow(3, dim), Neighborhood<dim>>
  {
    // BitsetWrapper::block is 3x3, 3x3x3, 3x3x3x3 local grids.
      // Center is center bit, not 0th.

    using Base = BitsetWrapper<intPow(3, dim), Neighborhood<dim>>;
    using Base::Base;  // inherit constructors

    static inline Neighborhood solitary();
    static inline Neighborhood not_down(int axis);
    static inline Neighborhood not_up(int axis);
    static inline Neighborhood center_slab_mask(int axis);

    // Class properties.
    static constexpr size_t n_neighbors();

    inline bool center_occupied() const;

    // Clear upper or lower hyperplane.
    inline Neighborhood cleared_up(int axis) const;
    inline Neighborhood cleared_down(int axis) const;

    // Clear upper and lower hyperplane.
    inline Neighborhood center_slab(int axis) const;

    // Shift along an axis, like Matlab circshift(), but overflow is truncated.
    inline Neighborhood shifted_up(int axis) const;
    inline Neighborhood shifted_down(int axis) const;

    // Union with center_slab shifted up and down on each axis.
    inline Neighborhood spread_out() const;
  };


#ifdef DOCTEST_LIBRARY_INCLUDED
  DOCTEST_TEST_SUITE("Neighborhood")
  {
    DOCTEST_TEST_CASE("Full is full")
    {
      CHECK( Neighborhood<3>::full().all() );
    }

    DOCTEST_TEST_CASE("Empty is empty")
    {
      CHECK( Neighborhood<3>::empty().none() );
    }

    DOCTEST_TEST_CASE("Shifted empty is empty")
    {
      for (int axis = 0; axis < 3; ++axis)
      {
        CHECK( Neighborhood<3>::empty().shifted_up(axis).none() );
      }

      for (int axis = 0; axis < 3; ++axis)
      {
        CHECK( Neighborhood<3>::empty().shifted_down(axis).none() );
      }
    }

    DOCTEST_TEST_CASE("Shifted full is neither full nor empty")
    {
      for (int axis = 0; axis < 3; ++axis)
      {
        auto shifted = Neighborhood<3>::full().shifted_up(axis);
        CHECK_FALSE( shifted.all() );
        CHECK_FALSE( shifted.none() );
      }
    }

    DOCTEST_TEST_CASE("Full can be trimmed until solitary")
    {
      auto neighborhood = Neighborhood<3>::full();
      for (int axis = 0; axis < 3; ++axis)
      {
        neighborhood = neighborhood.shifted_up(axis).shifted_down(axis);
        neighborhood = neighborhood.shifted_down(axis).shifted_up(axis);
      }

      CHECK( neighborhood == Neighborhood<3>::solitary() );
    }

    DOCTEST_TEST_CASE("Solitary can be spread until full.")
    {
      auto neighborhood = Neighborhood<3>::solitary();
      for (int axis = 0; axis < 3; ++axis)
      {
        neighborhood |= neighborhood.shifted_up(axis);
        neighborhood |= neighborhood.shifted_down(axis);
      }

      CHECK( neighborhood == Neighborhood<3>::full() );
    }

  }//DOCTEST_TEST_SUITE("Neighborhood")
#endif//DOCTEST_LIBRARY_INCLUDED

}

namespace std
{
  template <int dim>
  inline std::ostream & operator<<(std::ostream &out, const ot::Neighborhood<dim> &neighborhood);

  template <int dim>
  inline std::string to_string(const ot::Neighborhood<dim> &neighborhood);
}




// =============================================================================
// Implementation
// =============================================================================

namespace ot
{

#ifdef DOCTEST_LIBRARY_INCLUDED
  DOCTEST_TEST_CASE("Can shift std::bitset")
  {
    DOCTEST_CHECK( (std::bitset<9>("000"
                                   "010"
                                   "000") << 3)
        .to_string() ==
                                   "010"
                                   "000"
                                   "000" );
  }
#endif//DOCTEST_LIBRARY_INCLUDED

  namespace detail
  {
    template <int dim>
    static inline size_t center_index() { return intPow(3, dim) / 2; }
  }

  // ---------------------------------------------------------------------------
  // Factory methods.
  // ---------------------------------------------------------------------------

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::solitary()
  {
    return {std::bitset<intPow(3, dim)>().set(detail::center_index<dim>())};
  }

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::not_down(int axis)
  {
    return {~detail::hyperplane<dim, 3>(axis, 0)};
  }

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::not_up(int axis)
  {
    return {~detail::hyperplane<dim, 3>(axis, 2)};
  }

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::center_slab_mask(int axis)
  {
    return {detail::hyperplane<dim, 3>(axis, 1)};
  }


  // ---------------------------------------------------------------------------
  // Class properties.
  // ---------------------------------------------------------------------------
  template <int dim>
  constexpr size_t Neighborhood<dim>::n_neighbors()
  {
    return intPow(3, dim);
  }


  // ---------------------------------------------------------------------------
  // Reductions.
  // ---------------------------------------------------------------------------

  template <int dim>
  inline bool Neighborhood<dim>::center_occupied() const
  {
    return this->block.test(detail::center_index<dim>());
  }



  // ---------------------------------------------------------------------------
  // Clear upper or lower hyperplane or both.
  // ---------------------------------------------------------------------------

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::cleared_up(int axis) const
  {
    return { this->block & ~detail::hyperplane<dim, 3>(axis, 2) };
  }

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::cleared_down(int axis) const
  {
    return { this->block & ~detail::hyperplane<dim, 3>(axis, 0) };
  }

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::center_slab(int axis) const
  {
    return { this->block & detail::hyperplane<dim, 3>(axis, 1) };
  }


  // ---------------------------------------------------------------------------
  // Shift along an axis.
  // ---------------------------------------------------------------------------

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::shifted_up(int axis) const
  {
    // Bits in the highest plane normal to axis are truncated.
    Neighborhood<dim> result = this->cleared_up(axis);

    // Shift up.
    const int stride = intPow(3, axis);
    result.block <<= stride;

    return result;
  }

  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::shifted_down(int axis) const
  {
    // Bits in the lowest plane normal to axis are truncated.
    Neighborhood<dim> result = this->cleared_down(axis);

    // Shift down.
    const int stride = intPow(3, axis);
    result.block >>= stride;

    return result;
  }


  // spread_out()
  template <int dim>
  inline Neighborhood<dim> Neighborhood<dim>::spread_out() const
  {
    Neighborhood<dim> result = *this;
    for (int axis = 0; axis < dim; ++axis)
    {
      const auto slab = result.center_slab(axis);
      const int stride = intPow(3, axis);
      result.block |= slab.block >> stride; // Spread down.
      result.block |= slab.block << stride; // Spread up.
    }
    return result;
  }

}


// Streamification
namespace std
{
  template <int dim>
  inline std::ostream & operator<<(std::ostream &out, const ot::Neighborhood<dim> &neighborhood)
  {
    return ot::detail::print_grid<dim, 3>(out, neighborhood);
  }


  template <int dim>
  inline std::string to_string(const ot::Neighborhood<dim> &neighborhood)
  {
    std::stringstream ss;
    ss << neighborhood;
    return ss.str();
  }
}



#endif//DENDRO_KT_NEIGHBORHOOD_HPP
