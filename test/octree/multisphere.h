#ifndef DENDRO_KT_TEST_MULTISPHERE_H
#define DENDRO_KT_TEST_MULTISPHERE_H

#include <vector>
#include <array>
#include <algorithm>
#include <math.h>

#include "filterFunction.h"

namespace test
{
  template <int dim>
  struct SphereSet
  {
    inline ibm::Partition operator()(const double *elemPhysCoords, double elemPhysSize) const;
    // ------------------------------------------------------------
    inline void carveSphere(double radius, const std::array<double, dim> &center);
    inline void carveSphere(double radius, std::initializer_list<double> center);
    inline const std::array<double, dim> &center(int i) const;
    inline double radius(int i) const;
    inline int numSpheres() const;

    private:
      std::vector<double> m_radii;
      std::vector<std::array<double, dim>> m_centers;
  };
}

namespace test
{
  template <int dim>
  inline void SphereSet<dim>::carveSphere(double radius, const std::array<double, dim> &center)
  {
    m_radii.push_back(radius);
    m_centers.push_back(center);
  }

  template <int dim>
  inline void SphereSet<dim>::carveSphere(double radius, std::initializer_list<double> center)
  {
    std::array<double, dim> c;
    std::copy_n(center.begin(), dim, c.begin());
    this->carveSphere(radius, c);
  }

  // center()
  template <int dim>
  inline const std::array<double, dim> & SphereSet<dim>::center(int i) const
  {
    return m_centers[i];
  }

  // radius()
  template <int dim>
  inline double SphereSet<dim>::radius(int i) const
  {
    return m_radii[i];
  }

  // numSpheres()
  template <int dim>
  inline int SphereSet<dim>::numSpheres() const
  {
    return m_radii.size();
  }

  // SphereSet::operator()
  template <int dim>
  inline ibm::Partition SphereSet<dim>::operator()(
      const double *elemPhysCoords, double elemPhysSize) const
  {
    // For each sphere,
    //   find nearest point on box and test distance from center.
    //   find furtherst point on box and test distance from center.

    const int numSpheres = this->numSpheres();
    bool isIn = false, isOut = true;
    for (int i = 0; i < numSpheres; ++i)
    {
      double originToCenter[dim];
      for (int d = 0; d < dim; ++d)
        originToCenter[d] = this->center(i)[d] - elemPhysCoords[d];

      double nearest[dim];
      for (int d = 0; d < dim; ++d)
      {
        double clamped = originToCenter[d];
        if (clamped < 0)
          clamped = 0;
        else if (clamped > elemPhysSize)
          clamped = elemPhysSize;
        nearest[d] = clamped;
      }
      double nearestDist2 = 0;
      for (int d = 0; d < dim; ++d)
      {
        const double dist = nearest[d] - originToCenter[d];
        nearestDist2 += dist * dist;
      }

      double furthest[dim];
      for (int d = 0; d < dim; ++d)
      {
        double a = fabs(originToCenter[d] - 0);
        double b = fabs(originToCenter[d] - elemPhysSize);
        furthest[d] = (a >= b ? 0 : elemPhysSize);
      }
      double furthestDist2 = 0;
      for (int d = 0; d < dim; ++d)
      {
        const double dist = furthest[d] - originToCenter[d];
        furthestDist2 += dist * dist;
      }

      const double r2 = this->radius(i) * this->radius(i);
      isIn |= furthestDist2 <= r2;
      isOut &= nearestDist2 > r2;
    }

    ibm::Partition result;
    if (isIn && !isOut)
      result = ibm::IN;
    else if (isOut && !isIn)
      result = ibm::OUT;
    else
      result = ibm::INTERCEPTED;
    return result;
  }
}







/// constexpr int numSpheres = 2;
/// /// constexpr int numSpheres = 1;
/// 
/// // spheres()
/// const std::array<double, DIM> spheres(int i)
/// {
///   const std::array<double, DIM> spheres[numSpheres] = {
///     {0.0, 0.5}, 
///     {0.5, 0.5}
///   };
///   return spheres[i];
/// }
/// 
/// // radii()
/// const double radii(int i)
/// {
///   const double radius = 0.125;
///   return radius;
/// }


#endif//DENDRO_KT_TEST_MULTISPHERE_H
