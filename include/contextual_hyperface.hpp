/**
 * @author Masado Ishii
 * @date   2023-04-12
 * @brief  Represent a (split?) hyperface of an octant with 2 bits per axis.
 *
 * @detail In context of an octant's position, coordinates are {0,1,2}^dim.
 *         For sibling octants, space is flipped, so that '0' always refers
 *         to the side shared with the parent and '2' is the side shared with
 *         siblings. This way, hyperface coordinates stay in the range of
 *         {0,1,2}^dim. The true meaning depends on the "siblilng" context.
 *         Although it is possible to avoid mirroring siblings
 *         by using full range of {0,1,2,3,0}^dim, the mirrored-sibling
 *         encoding opens up an extra "hidden bit." This "hidden bit" is
 *         encoded as an out-of-range coordinate, where one entry takes the
 *         value of '3', signifying that the hyperface is split. From the
 *         position of the value '3', acting as another channel, the lost
 *         entry can be recovered:
 *         - To decode, shift-rotate the entries until '3', is the least
 *           significant entry, then replace '3' with the number of entries
 *           shifted by.
 *         - To encode, save the value from the least significant entry,
 *           replacing it with '3', and then shift-rotate by the value saved.
 *
 * @note   Clearly, for this encoding to work, the coordinate must have at
 *         least 3 entries, that is, be 3D or 4D. In practice, we embed both
 *         2D and 3D inside a 4D coordinate that anyway occupies a single byte.
 */

#ifndef DENDRO_KT_CONTEXTUAL_HYPERFACE_HPP
#define DENDRO_KT_CONTEXTUAL_HYPERFACE_HPP

// Function declaration for linkage purposes.
inline void link_contextual_hyperface_tests() {};

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  class Hyperface4D;
  class SplitHyperface4D;

  class SplitHyperface4D
  {
    public:
      constexpr SplitHyperface4D(uint8_t encoding);
      constexpr uint8_t encoding() const;
      constexpr bool is_split() const;
      constexpr Hyperface4D decode() const;

    private:
      constexpr static uint8_t position_of_3_plus_one(uint8_t coordinate); // 0: none.

    private:
      uint8_t m_encoding;
  };

  class Hyperface4D
  {
    public:
      constexpr Hyperface4D(uint8_t coordinate);
      constexpr uint8_t coordinate() const;
      constexpr int coordinate(int axis) const;
      constexpr int flat() const;
      constexpr int dimension() const;
      constexpr Hyperface4D mirrored(int child_num) const;
      constexpr SplitHyperface4D encode_split(bool is_split) const;

    private:
      uint8_t m_coordinate;
  };

  template <int dim>
  class Hyperface;

  template <>
  class Hyperface<2> : public Hyperface4D { using Hyperface4D::Hyperface4D; };
  template <>
  class Hyperface<3> : public Hyperface4D { using Hyperface4D::Hyperface4D; };
  template <>
  class Hyperface<4> : public Hyperface4D { using Hyperface4D::Hyperface4D; };
}


// =============================================================================
// Tests
// =============================================================================
#ifdef DOCTEST_LIBRARY_INCLUDED
namespace ot
{
  DOCTEST_TEST_SUITE("Contextual Hyperface")
  {
    DOCTEST_TEST_CASE("Encode and decode")
    {
      for (bool split: {false, true})
        for (uint8_t t: {0u, 1u, 2u})
          for (uint8_t z: {0u, 1u, 2u})
            for (uint8_t y: {0u, 1u, 2u})
              for (uint8_t x: {0u, 1u, 2u})
              {
                const uint8_t coordinate = (t<<6)|(z<<4)|(y<<2)|(x<<0);
                const auto encoded = Hyperface<4>(coordinate).encode_split(split);
                const uint8_t decoded = encoded.decode().coordinate();
                const uint8_t ex = (encoded.encoding()>>0) & 3u;
                const uint8_t ey = (encoded.encoding()>>2) & 3u;
                const uint8_t ez = (encoded.encoding()>>4) & 3u;
                const uint8_t et = (encoded.encoding()>>6) & 3u;
                const uint8_t dx = (decoded>>0) & 3u;
                const uint8_t dy = (decoded>>2) & 3u;
                const uint8_t dz = (decoded>>4) & 3u;
                const uint8_t dt = (decoded>>6) & 3u;

                INFO("in:  x=", int(x),  " y=", int(y),  " z=", int(z),  " t=", int(t));
                INFO("enc: x=", int(ex), " y=", int(ey), " z=", int(ez), " t=", int(et));
                INFO("out: x=", int(dx), " y=", int(dy), " z=", int(dz), " t=", int(dt));
                CHECK( int(decoded) == int(coordinate) );
                CHECK( encoded.is_split() == split );
              }
    }

    DOCTEST_TEST_CASE("Mirror bijection")
    {
      constexpr int dim = 4;
      for (bool split: {false, true})
        for (int child = 0; child < nchild(dim); ++child)
          for (uint8_t t: {0u, 1u, 2u})
            for (uint8_t z: {0u, 1u, 2u})
              for (uint8_t y: {0u, 1u, 2u})
                for (uint8_t x: {0u, 1u, 2u})
                {
                  const uint8_t coordinate = (t<<6)|(z<<4)|(y<<2)|(x<<0);
                  const uint8_t mirrored =
                      Hyperface<dim>(coordinate).mirrored(child).coordinate();
                  const uint8_t double_mirrored =
                      Hyperface<dim>(mirrored).mirrored(child).coordinate();

                  if (not split)
                    CHECK( int(double_mirrored) == int(coordinate) );

                  // Semi-redundant with encoding checks, but confirms range.
                  const auto encoded = Hyperface<dim>(mirrored).encode_split(split);
                  const uint8_t decoded_unmirrored =
                      encoded.decode().mirrored(child).coordinate();
                  CHECK( int(decoded_unmirrored) == int(coordinate) );
                }
    }
  }
}
#endif//DOCTEST_LIBRARY_INCLUDED




// =============================================================================
// Implementation
// =============================================================================
namespace ot
{
  // future:
  // Almost all of the bit manipulation done to encode and decode
  // could be done on multiple bytes in parallel. The only non-SIMD
  // part is the amount of rotation, which will vary between bytes.

  // SplitHyperface4D::SplitHyperface4D()
  constexpr SplitHyperface4D::SplitHyperface4D(uint8_t encoding)
    :
      m_encoding(encoding)
  { }

  // SplitHyperface4D::encoding()
  constexpr uint8_t SplitHyperface4D::encoding() const
  {
    return m_encoding;
  }

  namespace detail
  {
    // rotl()
    // https://en.cppreference.com/w/cpp/numeric/rotl
    constexpr uint8_t rotl(uint8_t x, int s)
    {
      if (s < 0)
        throw std::range_error("shift amount cannot be negative, use rotr.");
      if (s >= 8)
        throw std::range_error("shift amount larger than 7 for uint8_t.");
      return (x << s) | (x >> (8 - s));
    }

    // rotr()
    // https://en.cppreference.com/w/cpp/numeric/rotr
    constexpr uint8_t rotr(uint8_t x, int s)
    {
      if (s < 0)
        throw std::range_error("shift amount cannot be negative, use rotl.");
      if (s >= 8)
        throw std::range_error("shift amount larger than 7 for uint8_t.");
      return (x >> s) | (x << (8 - s));
    }
  }

  // SplitHyperface4D::position_of_3_plus_one()
  constexpr uint8_t SplitHyperface4D::position_of_3_plus_one(uint8_t coordinate)
  {
    // Assume there occurs at most one pair of bits with value '3',
    // and if there is a '3', it occurs in the lowest 3 pairs of bits.

    // Select low and high bits from each pair of bits. Avoids interference.
    uint8_t lo = coordinate & 0b01010101u;
    uint8_t hi = coordinate & 0b10101010u;

    // Within each pair of bits, compute logical 'and'; spread to both bits.
    // The result is '3' -> '3' and {'0', '1', '2'} -> '0' for all pairs.
    lo &= hi >> 1;
    hi &= lo << 1;

    // Map each pair to its position plus 1.
    uint8_t pos = (lo | hi) & 0b00111001u;

    // Move all pairs (at most one should be nonzero) to the lowest pair; eval.
    pos = ((pos >> 4) | (pos >> 2) | pos) & 3u;
    return pos;
  }

  // SplitHyperface4D::is_split()
  constexpr bool SplitHyperface4D::is_split() const
  {
    return position_of_3_plus_one(m_encoding) > 0;
  }

  // SplitHyperface4D::decode()
  constexpr Hyperface4D SplitHyperface4D::decode() const
  {
    uint8_t coordinate = m_encoding;
    uint8_t pos = position_of_3_plus_one(m_encoding);
    if (pos > 0u)
    {
      // Shift left to encode, shift right to decode (arbitrary choice).
      pos -= 1u;
      coordinate = detail::rotr(coordinate, 2 * pos);
      coordinate &= 0b11111100u | pos;
    }
    return Hyperface4D(coordinate);
  }

  // Hyperface4D::Hyperface4D()
  constexpr Hyperface4D::Hyperface4D(uint8_t coordinate)
    :
      m_coordinate(coordinate)
  { }

  // Hyperface4D::coordinate()
  constexpr uint8_t Hyperface4D::coordinate() const
  {
    return m_coordinate;
  }

  // Hyperface4D::coordinate()
  constexpr int Hyperface4D::coordinate(int axis) const
  {
    return (m_coordinate >> (2 * axis)) & 3;
  }

  // Hyperface4D::flat()
  constexpr int Hyperface4D::flat() const
  {
    uint8_t index = 0;
    uint8_t stride = 1;
    uint8_t coord = m_coordinate;
    for (int d = 0; d < 4; ++d)
    {
      index += (coord & 3u) * stride;
      coord >>= 2;
      stride *= 3;
    }
    return index;
  }

  // Hyperface4D::dimension()
  constexpr int Hyperface4D::dimension() const
  {
    // The dimension is the number of odd coordinates.
    // This is true regardless of child number.
    uint8_t count = m_coordinate & 0b01010101u;
    count += count >> 4;
    count = ((count & 0b1100u) >> 2) + (count & 0b0011u);
    return count;
  }


  // Hyperface4D::mirrored()
  constexpr Hyperface4D Hyperface4D::mirrored(int child_num) const
  {
    // For each bit set in child_num, map bit-pair in coordinate 0 <--> 2.
    uint8_t mask = 0u;
    if ((child_num >> 0) & 1)
      mask |= 0b00000011u;
    if ((child_num >> 1) & 1)
      mask |= 0b00001100u;
    if ((child_num >> 2) & 1)
      mask |= 0b00110000u;
    if ((child_num >> 3) & 1)
      mask |= 0b11000000u;

    const uint8_t original = m_coordinate;
    uint8_t flipped = (0b10101010u - original) & mask;
    flipped |= original & (~mask);
    return Hyperface4D(flipped);
  }

  // Hyperface4D::encode_split()
  constexpr SplitHyperface4D Hyperface4D::encode_split(bool is_split) const
  {
    uint8_t encoding = m_coordinate;
    const uint8_t pos = (encoding & 3u);
    if (is_split)
    {
      // Shift left to encode, shift right to decode (arbitrary choice).
      encoding = detail::rotl(encoding | 3u, 2 * pos);
    }
    return SplitHyperface4D(encoding);
  }

}

#endif//DENDRO_KT_CONTEXTUAL_HYPERFACE_HPP
