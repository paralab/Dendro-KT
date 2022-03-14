/**
 * @author Masado Ishii
 * @date July 2, 2021
 * @brief Periodic coordinates
 */

#ifndef DENDRO_KT_PCOORD_H
#define DENDRO_KT_PCOORD_H

#include <array>
#include <assert.h>

namespace periodic
{
  enum { NO_PERIOD = 0 };

  /**
   * @brief Periodic integer coordinates glued by modulus operator.
   * @description Specifies the period of the lattice in each axis.
   * By default the period is the number of representable values,
   * i.e. effectively non-periodic.
   * If a finite period is specified, then each point is mapped
   * to it's canonical representative upon construction.
   */
  template <typename T, int dim>
  class PCoord
  {
    private:
      std::array<T, dim> m_coords;
      static std::array<T, dim> m_masks;

    public:
      // Set periods for each axis.
      // Default period for all axes is NO_PERIOD.
      inline static T period(int axis);
      inline static void period(int axis, T period);
      inline static std::array<T, dim> periods();
      inline static void periods(const std::array<T, dim> &periods);
      //
      inline static T map(int axis, T coord) { return m_masks[axis] & coord; }

      inline PCoord();
      inline PCoord(const std::array<T, dim> coords);

      inline PCoord(const PCoord &) = default;
      inline PCoord(PCoord &&) = default;
      inline PCoord & operator=(const PCoord &) = default;
      inline PCoord & operator=(PCoord &&) = default;

      class CoordTransfer
      {
        private:
          friend PCoord;
          T m_t;
          CoordTransfer() = default;
        public:
          CoordTransfer(const CoordTransfer &) = default;
          CoordTransfer(CoordTransfer &&) = default;
          CoordTransfer & operator=(const CoordTransfer &) = default;
          CoordTransfer & operator=(CoordTransfer &&) = default;
          operator T() const { return m_t; }
      };

      inline CoordTransfer coord(int axis) const;
      inline void coord(int axis, const CoordTransfer &coord);
      inline void coord(int axis, T coord);
      inline operator std::array<T, dim>() const { return m_coords; }
      inline const std::array<T, dim> & coords() const { return m_coords; }

      inline PCoord operator+(const PCoord &b) const;
      inline PCoord operator-(const PCoord &b) const;
      inline PCoord & operator+=(const PCoord &b);
      inline PCoord & operator-=(const PCoord &b);

      inline bool operator==(const PCoord &b) const;
      inline bool operator!=(const PCoord &b) const;

      inline void truncate(int bottomLevels);
      inline PCoord truncated(int bottomLevels) const;
  };


  // PRange
  template <typename T, int dim>
  class PRange
  {
    private:
      PCoord<T, dim> m_min;
      PCoord<T, dim> m_max;
      T m_side;

    public:
      inline PRange();
      inline PRange(const PCoord<T, dim> &min, T side);

      PRange(const PRange &) = default;
      PRange(PRange &&) = default;
      PRange & operator=(const PRange &) = default;
      PRange & operator=(PRange &&) = default;

      using CoordTransfer = typename PCoord<T, dim>::CoordTransfer;

      inline const PCoord<T, dim> & min() const;
      inline const PCoord<T, dim> & max() const;
      inline const CoordTransfer min(int d) const;
      inline const CoordTransfer max(int d) const;
      inline T side() const;

      inline bool openContains(int axis, const CoordTransfer &coord) const;
      inline bool halfClosedContains(int axis, const CoordTransfer &coord) const;
      inline bool upperEquals(int axis, const CoordTransfer &coord) const;
      inline bool closedContains(int axis, const CoordTransfer &coord) const;

      inline bool openContains(int axis, T coord) const;
      inline bool halfClosedContains(int axis, T coord) const;
      inline bool upperEquals(int axis, T coord) const;
      inline bool closedContains(int axis, T coord) const;

      inline bool closedContains(const PCoord<T, dim> &pcoord) const;
  };

}


namespace periodic
{
  // Representation:
  //   The period is stored as a static member of the class.
  //   A period of '0' encodes the NO_PERIOD condition. 

  // static accessors

  // period()
  template <typename T, int dim>
  inline T PCoord<T, dim>::period(int axis)
  {
    return m_masks[axis] + 1;
  }

  // period()
  template <typename T, int dim>
  inline void PCoord<T, dim>::period(int axis, T period)
  {
    // Also handles the case period = 0.
    const bool isPow2 = !bool((period - 1) & period);
    assert(isPow2);
    m_masks[axis] = period - 1;
  }

  // periods()
  template <typename T, int dim>
  inline std::array<T, dim> PCoord<T, dim>::periods()
  {
    std::array<T, dim> periods;
    for (int d = 0; d < dim; ++d)
      periods[d] = PCoord<T, dim>::period(d);
    return periods;
  }

  // periods()
  template <typename T, int dim>
  inline void PCoord<T, dim>::periods(const std::array<T, dim> &periods)
  {
    for (int d = 0; d < dim; ++d)
      PCoord<T, dim>::period(d, periods[d]);
  }

  // PCoord()
  template <typename T, int dim>
  inline PCoord<T, dim>::PCoord()
  {
    m_coords.fill(0);
  }

  // PCoord()
  template <typename T, int dim>
  inline PCoord<T, dim>::PCoord(const std::array<T, dim> coords)
  {
    for (int d = 0; d < dim; ++d)
      this->coord(d, coords[d]);
  }

  // coord()
  template <typename T, int dim>
  inline typename PCoord<T, dim>::CoordTransfer PCoord<T, dim>::coord(int axis) const
  {
    CoordTransfer result;
    result.m_t = m_coords[axis];
    return result;
  }

  // coord()
  template <typename T, int dim>
  inline void PCoord<T, dim>::coord(int axis, const PCoord<T, dim>::CoordTransfer &coord)
  {
    m_coords[axis] = coord;
  }

  // coord()
  template <typename T, int dim>
  inline void PCoord<T, dim>::coord(int axis, T coord)
  {
    m_coords[axis] = PCoord<T, dim>::map(axis, coord);
  }

  // operator+()
  template <typename T, int dim>
  inline PCoord<T, dim> PCoord<T, dim>::operator+(const PCoord &b) const
  {
    PCoord result;
    for (int d = 0; d < dim; ++d)
      result.coord(d, this->coord(d) + b.coord(d));
    return result;
  }

  // operator-()
  template <typename T, int dim>
  inline PCoord<T, dim> PCoord<T, dim>::operator-(const PCoord &b) const
  {
    PCoord result;
    for (int d = 0; d < dim; ++d)
    {
      const T ad = this->coord(d),  bd = b.coord(d);
      const bool negative = bd > ad;
      const T diff = (negative ? bd - ad : ad - bd);
      result.m_coords[d] = (negative ? period(d) - diff : diff);
    }
    return result;
  }

  // operator+=()
  template <typename T, int dim>
  inline PCoord<T, dim> & PCoord<T, dim>::operator+=(const PCoord &b)
  {
    return operator=(operator+(b));
  }

  // operator-=()
  template <typename T, int dim>
  inline PCoord<T, dim> & PCoord<T, dim>::operator-=(const PCoord &b)
  {
    return operator=(operator-(b));
  }

  // operator==()
  template <typename T, int dim>
  inline bool PCoord<T, dim>::operator==(const PCoord &b) const
  {
    return !operator!=(b);
  }

  // operator!=()
  template <typename T, int dim>
  inline bool PCoord<T, dim>::operator!=(const PCoord &b) const
  {
    for (int d = 0; d < dim; ++d)
      if (this->coord(d) != b.coord(d))
        return true;
    return false;
  }

  template <typename T, int dim>
  inline void PCoord<T, dim>::truncate(int bottomLevels)
  {
    const T levelMask = ~((1u << bottomLevels) - 1);
    for (int d = 0; d < dim; ++d)
      m_coords[d] &= levelMask;
  }

  template <typename T, int dim>
  inline PCoord<T, dim> PCoord<T, dim>::truncated(int bottomLevels) const
  {
    PCoord result = *this;
    result.truncate(bottomLevels);
    return result;
  }


  // PRange()
  template <typename T, int dim>
  PRange<T, dim>::PRange()
  : m_min(), m_max(), m_side(0)
  { }

  // PRange()
  template <typename T, int dim>
  PRange<T, dim>::PRange(const PCoord<T, dim> &min, T side)
  : m_min(min), m_side(side)
  {
    for (int d = 0; d < dim; ++d)
      m_max.coord(d, min.coord(d) + side);
  }

  // PRange::min()
  template <typename T, int dim>
  const PCoord<T, dim> & PRange<T, dim>::min() const
  {
    return m_min;
  }

  // PRange::max()
  template <typename T, int dim>
  const PCoord<T, dim> & PRange<T, dim>::max() const
  {
    return m_max;
  }

  template <typename T, int dim>
  const typename PCoord<T, dim>::CoordTransfer PRange<T, dim>::min(int d) const
  {
    return min().coord(d);
  }

  template <typename T, int dim>
  const typename PCoord<T, dim>::CoordTransfer PRange<T, dim>::max(int d) const
  {
    return max().coord(d);
  }

  // PRange::side()
  template <typename T, int dim>
  T PRange<T, dim>::side() const
  {
    return m_side;
  }

  // PRange::openContains()
  template <typename T, int dim>
  bool PRange<T, dim>::openContains(int axis, const CoordTransfer &coord) const
  {
    const T min_coord = m_min.coord(axis);
    return min_coord < coord && coord < min_coord + m_side;
  }

  // PRange::halfClosedContains()
  template <typename T, int dim>
  bool PRange<T, dim>::halfClosedContains(int axis, const CoordTransfer &coord) const
  {
    const T min_coord = m_min.coord(axis);
    return min_coord <= coord && coord < min_coord + m_side;
  }

  // PRange::upperEquals()
  template <typename T, int dim>
  bool PRange<T, dim>::upperEquals(int axis, const CoordTransfer &coord) const
  {
    const T max_coord = m_max.coord(axis);
    return coord == max_coord;
  }

  // PRange::closedContains()
  template <typename T, int dim>
  bool PRange<T, dim>::closedContains(int axis, const CoordTransfer &coord) const
  {
    return this->halfClosedContains(axis, coord) ||
        this->upperEquals(axis, coord);
  }

  // PRange::openContains()
  template <typename T, int dim>
  bool PRange<T, dim>::openContains(int axis, T coord) const
  {
    coord = PCoord<T, dim>::map(axis, coord);
    const T min_coord = m_min.coord(axis);
    return min_coord < coord && coord < min_coord + m_side;
  }

  // PRange::halfClosedContains()
  template <typename T, int dim>
  bool PRange<T, dim>::halfClosedContains(int axis, T coord) const
  {
    coord = PCoord<T, dim>::map(axis, coord);
    const T min_coord = m_min.coord(axis);
    return min_coord <= coord && coord < min_coord + m_side;
  }

  // PRange::upperEquals()
  template <typename T, int dim>
  bool PRange<T, dim>::upperEquals(int axis, T coord) const
  {
    coord = PCoord<T, dim>::map(axis, coord);
    const T max_coord = m_max.coord(axis);
    return coord == max_coord;
  }

  // PRange::closedContains()
  template <typename T, int dim>
  bool PRange<T, dim>::closedContains(int axis, T coord) const
  {
    return this->halfClosedContains(axis, coord) ||
        this->upperEquals(axis, coord);
  }

  // PRange::closedContains()
  template <typename T, int dim>
  bool PRange<T, dim>::closedContains(const PCoord<T, dim> &pcoord) const
  {
    bool allContains = true;
    for (int d = 0; d < dim; ++d)
      allContains &= this->closedContains(d, pcoord.coord(d));
    return allContains;
  }


}

#endif//DENDRO_KT_PCOORD_H
