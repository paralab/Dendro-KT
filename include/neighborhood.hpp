
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

#include "doctest/doctest.h"
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

    // Class properties.
    static constexpr size_t n_neighbors();

    inline bool center_occupied() const;

    // Clear upper or lower hyperplane.
    inline Neighborhood cleared_up(int axis) const;
    inline Neighborhood cleared_down(int axis) const;

    // Shift along an axis, like Matlab circshift(), but overflow is truncated.
    inline Neighborhood shifted_up(int axis) const;
    inline Neighborhood shifted_down(int axis) const;
  };


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
  // Clear upper or lower hyperplane.
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
}


// Streamification
namespace std
{
  template <int dim>
  inline std::ostream & operator<<(std::ostream &out, const ot::Neighborhood<dim> &neighborhood)
  {
    // Up to 4D.
    // In horizontal axes, 0=left, 1=middle, 2=right.
    // In vertical axes, 0=bottom, 1=middle, 2=top.
    constexpr int N = intPow(3, dim);
    static_assert(dim <= 4, "Only supported up to dimension 4.");
    int i[4] = {};
    int index[4] = {};
    const int stride[4] = {1, 3, 9, 27};
    for (i[3] = 0; i[3] < 3 and i[3] * stride[3] < N; ++i[3])
    {
      if (i[3] > 0)  // new row block
        out << "\n\n";
      if (stride[3] < N)
        index[3] = (2 - i[3]) % 3;  // vertical: reverse and shift.

      for (i[1] = 0; i[1] < 3 and i[1] * stride[1] < N; ++i[1])
      {
        if (i[1] > 0)  // new row
          out << "\n";
        if (stride[1] < N)
          index[1] = (2 - i[1]) % 3;   // vertical: reverse and shift.

        for (i[2] = 0; i[2] < 3 and i[2] * stride[2] < N; ++i[2])
        {
          if (i[2] > 0)  // new column block
            out << "  ";
          if (stride[2] < N)
            index[2] = (i[2]) % 3;  // horizontal: shift.

          for (i[0] = 0; i[0] < 3 and i[0] * stride[0] < N; ++i[0])
          {
            if (i[0] > 0)  // new column
              out << " ";
            if (stride[0] < N)
              index[0] = (i[0]) % 3;  // horizontal: shift.

            const int flat = stride[3] * index[3] + 
                             stride[2] * index[2] +
                             stride[1] * index[1] +
                             stride[0] * index[0];

            out << neighborhood.block[flat];
          }
        }
      }
    }
    return out;
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
