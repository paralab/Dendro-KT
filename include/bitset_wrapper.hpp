
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

}

#endif//DENDRO_KT_BITSET_WRAPPER_HPP
