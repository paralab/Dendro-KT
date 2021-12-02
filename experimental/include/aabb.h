#ifndef DENDRO_KT_AABB_H
#define DENDRO_KT_AABB_H

#include "point.h"

template <int dim>
class AABB
{
  protected:
    Point<dim> m_min;
    Point<dim> m_max;

  public:
    AABB(const Point<dim> &min, const Point<dim> &max) : m_min(min), m_max(max) { }
    const Point<dim> & min() const { return m_min; }
    const Point<dim> & max() const { return m_max; }
};

#endif//DENDRO_KT_AABB_H
