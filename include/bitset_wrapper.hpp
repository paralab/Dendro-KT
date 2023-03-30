
/**
 * @author Masado Ishii
 * @date   2023-03-29
 * @brief  Wrap std::bitset bitwise ops in a base class of multidimensional sets.
 */

#ifndef DENDRO_KT_BITSET_WRAPPER_HPP
#define DENDRO_KT_BITSET_WRAPPER_HPP

// Function declaration for linkage purposes.
inline void link_bitset_wrapper_tests() {};

#include "doctest/doctest.h"
#include "include/mathUtils.h"
#include <bitset>

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  // This could be constexpr, if only std::bitset was constexpr pre c++23.

  template <size_t N, class Derived>
  struct BitsetWrapper
  {
    std::bitset<N> block;

    BitsetWrapper() = default;
    BitsetWrapper(std::bitset<N> block) : block(block) {}
    // Note: Derived should inherit these constructors with `using` declaration.

    // Factory methods.
    static inline Derived full();
    static inline Derived empty();
    template <typename IdxToBool>
      static inline Derived where(IdxToBool idx_to_bool);

    // Reductions.
    inline bool all() const;
    inline bool any() const;
    inline bool none() const;
    inline size_t count() const;

    // Scalar bit manipulation.
    inline void set_flat(size_t idx);
    inline void clear_flat(size_t idx);
    inline void flip_flat(size_t idx);
    inline bool test_flat(size_t idx) const;

    // Bitwise logic operators.
    inline Derived operator~() const;
    inline Derived operator&(const BitsetWrapper &other) const;
    inline Derived operator|(const BitsetWrapper &other) const;
    inline Derived operator^(const BitsetWrapper &other) const;

    inline Derived & operator&=(const BitsetWrapper &other);
    inline Derived & operator|=(const BitsetWrapper &other);
    inline Derived & operator^=(const BitsetWrapper &other);

    // Comparison operators.
    inline bool operator==(const BitsetWrapper &other) const;
    inline bool operator!=(const BitsetWrapper &other) const;
  };

  // For derived classes implementing multidimensional sets.
  namespace detail
  {
    template <int dim, int side>
    inline std::bitset<intPow(side, dim)> hyperplane(int axis, int pos = 0)
    {
      std::bitset<intPow(side, dim)> bitset;
      bitset[0] = true;
      for (int a = 0; a < dim; ++a)
      {
        const int stride = intPow(side, a);
        if (a != axis)
          bitset |= (bitset << stride) | (bitset << 2 * stride);
        else
          bitset <<= pos * stride;
      }
      return bitset;
    }
  }


}

// =============================================================================
// Implementation
// =============================================================================
namespace ot
{
  // ---------------------------------------------------------------------------
  // Factory methods.
  // ---------------------------------------------------------------------------

  template <size_t N, class Derived>
  inline Derived BitsetWrapper<N, Derived>::full()
  {
    return {std::bitset<N>().set()};
  }

  template <size_t N, class Derived>
  inline Derived BitsetWrapper<N, Derived>::empty()
  {
    return {std::bitset<N>().reset()};
  }

  template <size_t N, class Derived>
  template <typename IdxToBool>
  inline Derived BitsetWrapper<N, Derived>::where(IdxToBool idx_to_bool)
  {
    std::bitset<N> included;
    for (int i = 0; i < N; ++i)
      if (idx_to_bool(i))
        included.set(i);
    return {included};
  }

  // ---------------------------------------------------------------------------
  // Reductions.
  // ---------------------------------------------------------------------------

  template <size_t N, class Derived>
  inline bool BitsetWrapper<N, Derived>::all() const
  {
    return this->block.all();
  }

  template <size_t N, class Derived>
  inline bool BitsetWrapper<N, Derived>::any() const
  {
    return this->block.any();
  }

  template <size_t N, class Derived>
  inline bool BitsetWrapper<N, Derived>::none() const
  {
    return this->block.none();
  }

  template <size_t N, class Derived>
  inline size_t BitsetWrapper<N, Derived>::count() const
  {
    return this->block.count();
  }


  // ---------------------------------------------------------------------------
  // Scalar bit manipulation.
  // ---------------------------------------------------------------------------

  template <size_t N, class Derived>
  inline void BitsetWrapper<N, Derived>::set_flat(size_t idx)
  {
    this->block.set(idx);
  }

  template <size_t N, class Derived>
  inline void BitsetWrapper<N, Derived>::clear_flat(size_t idx)
  {
    this->block.clear(idx);
  }

  template <size_t N, class Derived>
  inline void BitsetWrapper<N, Derived>::flip_flat(size_t idx)
  {
    this->block.flip(idx);
  }

  template <size_t N, class Derived>
  inline bool BitsetWrapper<N, Derived>::test_flat(size_t idx) const
  {
    return this->block.test(idx);
  }

  // ---------------------------------------------------------------------------
  // Bitwise logic operators.
  // ---------------------------------------------------------------------------

  // Wrap std::bitset operators.

  template <size_t N, class Derived>
  inline Derived BitsetWrapper<N, Derived>::operator~() const
  {
    return {~this->block};
  }

  template <size_t N, class Derived>
  inline Derived BitsetWrapper<N, Derived>::operator&(const BitsetWrapper &other) const
  {
    return {this->block & other.block};
  }

  template <size_t N, class Derived>
  inline Derived BitsetWrapper<N, Derived>::operator|(const BitsetWrapper &other) const
  {
    return {this->block | other.block};
  }

  template <size_t N, class Derived>
  inline Derived BitsetWrapper<N, Derived>::operator^(const BitsetWrapper &other) const
  {
    return {this->block ^ other.block};
  }


  template <size_t N, class Derived>
  inline Derived & BitsetWrapper<N, Derived>::operator&=(const BitsetWrapper &other)
  {
    (*this) = (*this) & other;
    return static_cast<Derived &>(*this);
  }

  template <size_t N, class Derived>
  inline Derived & BitsetWrapper<N, Derived>::operator|=(const BitsetWrapper &other)
  {
    (*this) = (*this) | other;
    return static_cast<Derived &>(*this);
  }

  template <size_t N, class Derived>
  inline Derived & BitsetWrapper<N, Derived>::operator^=(const BitsetWrapper &other)
  {
    (*this) = (*this) ^ other;
    return static_cast<Derived &>(*this);
  }

  template <size_t N, class Derived>
  inline bool BitsetWrapper<N, Derived>::operator==(const BitsetWrapper &other) const
  {
    return this->block == other.block;
  }

  template <size_t N, class Derived>
  inline bool BitsetWrapper<N, Derived>::operator!=(const BitsetWrapper &other) const
  {
    return this->block != other.block;
  }


  // For derived classes implementing multidimensional sets.
  namespace detail
  {
    template <int dim, int side, class Derived>
    inline std::ostream & print_grid(std::ostream &out,
        const ot::BitsetWrapper<intPow(side, dim), Derived> &bitset)
    {
      // Up to 4D.
      // In horizontal axes, 0=left, 1=middle, 2=right.
      // In vertical axes, 0=bottom, 1=middle, 2=top.
      constexpr int N = intPow(side, dim);
      constexpr int max = side - 1;
      static_assert(dim <= 4, "Only supported up to dimension 4.");
      int i[4] = {};
      int index[4] = {};
      const int stride[4] = {1, side, side*side, side*side*side};
      for (i[3] = 0; i[3] < side and i[3] * stride[3] < N; ++i[3])
      {
        if (i[3] > 0)  // new row block
          out << "\n\n";
        if (stride[3] < N)
          index[3] = (max - i[3]) % side;  // vertical: reverse and shift.

        for (i[1] = 0; i[1] < side and i[1] * stride[1] < N; ++i[1])
        {
          if (i[1] > 0)  // new row
            out << "\n";
          if (stride[1] < N)
            index[1] = (max - i[1]) % side;   // vertical: reverse and shift.

          for (i[2] = 0; i[2] < side and i[2] * stride[2] < N; ++i[2])
          {
            if (i[2] > 0)  // new column block
              out << "  ";
            if (stride[2] < N)
              index[2] = (i[2]) % side;  // horizontal: shift.

            for (i[0] = 0; i[0] < side and i[0] * stride[0] < N; ++i[0])
            {
              if (i[0] > 0)  // new column
                out << " ";
              if (stride[0] < N)
                index[0] = (i[0]) % side;  // horizontal: shift.

              const int flat = stride[3] * index[3] + 
                               stride[2] * index[2] +
                               stride[1] * index[1] +
                               stride[0] * index[0];

              out << bitset.block[flat];
            }
          }
        }
      }
      return out;
    }
  }
}

#endif//DENDRO_KT_BITSET_WRAPPER_HPP
