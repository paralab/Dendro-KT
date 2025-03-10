/***************************************************************************
 *   Copyright (C) 2005 by Hari sundar   *
 *   hsundar@seas.upenn.edu   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include "point.h"

#include <algorithm>


template <int dim>
Point<dim>::Point(double scale)
{
  std::fill_n(&_p[0], dim, scale);
}

template <int dim>
Point<dim>::Point(const std::array<double, dim> &newCoords)
{
  std::copy_n(&newCoords[0], dim, &_p[0]);
}

template <int dim>
Point<dim>::Point(const double * newCoords)
{
  std::copy_n(&newCoords[0], dim, &_p[0]);
}

template <int dim>
Point<dim>::Point(double newx, double newy, double newz)
{
  initialize3(newx, newy, newz);
}

template <int dim>
Point<dim>::Point(int newx, int newy, int newz)
{ 
  initialize3(static_cast<double>(newx),
      static_cast<double>(newy),
      static_cast<double>(newz));
}

template <int dim>
Point<dim>::Point(unsigned int newx, unsigned int newy, unsigned int newz)
{ 
  initialize3(static_cast<double>(newx),
      static_cast<double>(newy),
      static_cast<double>(newz));
}

/*
template <int dim>
Point<dim>::~Point()
{

}
*/

template <int dim>
inline void Point<dim>::initialize3(double newx, double newy, double newz)
{
  _p[0] = newx;  _p[1] = newy;  _p[2] = newz;
}

template <int dim>
Point<dim> Point<dim>::operator - () const {
  Point ret(*this);
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    ret._p[d] = -ret._p[d];
  return ret;
}

template <int dim>
void Point<dim>::operator *= (const int factor){
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    _p[d] *= factor;
}

template <int dim>
void Point<dim>::operator *= (const double factor){
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    _p[d] *= factor;
}

template <int dim>
void Point<dim>::operator /= (const int divisor){
  if (divisor == 0) return;
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    _p[d] /= static_cast<double>(divisor);
}

template <int dim>
void Point<dim>::operator /= (const double divisor){
  if (divisor == 0) return;
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    _p[d] /= divisor;
}

template <int dim>
void Point<dim>::operator += (const Point& other){
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    _p[d] += other._p[d];
}

template <int dim>
void Point<dim>::operator -= (const Point& other){
  #pragma unroll(dim)
  for (int d = 0; d < dim; d++)
    _p[d] -= other._p[d];
}

template <int dim>
Point<dim> Point<dim>::operator - (const Point &other) const{
  Point ret(*this);
  ret -= other;
  return ret;
}

template <int dim>
Point<dim> Point<dim>::operator + (const Point &other) const{
  Point ret(*this);
  ret += other;
  return ret;
}


template <int dim>
Point<dim> Point<dim>::operator /(const double divisor) const
{
  Point ret(*this);
  ret /= divisor;
  return ret;
}

template <int dim>
Point<dim> Point<dim>::operator *(const double factor) const
{
  Point ret(*this);
  ret *= factor;
  return ret;
}

template <int dim>
Point<dim> Point<dim>::TransMatMultiply3(double *transMat, Point inPoint)
{
  Point outPoint;

  outPoint._p[0] = transMat[ 0]*inPoint._p[0] +transMat[ 4]*inPoint._p[1] +transMat[8]
    *inPoint._p[2] +transMat[12];
  outPoint._p[1] = transMat[ 1]*inPoint._p[0] +transMat[ 5]*inPoint._p[1] +transMat[9]
    *inPoint._p[2] +transMat[13];
  outPoint._p[2] = transMat[ 2]*inPoint._p[0] +transMat[ 6]*inPoint._p[1]
    +transMat[10]*inPoint._p[2] +transMat[14];

  return outPoint;
}

template <int dim>
Point<dim> Point<dim>::TransMatMultiply(double *transMat, Point inPoint)
{
  if (dim == 3)
    return TransMatMultiply3(transMat, inPoint);

  Point outPoint;

  for (int i = 0; i < dim; i++)
  {
    outPoint._p[i] = transMat[dim*(dim+1) + i];
    for (int j = 0; j < dim; j++)
      outPoint._p[i] += transMat[j*(dim+1) + i] * inPoint._p[j];
  }

  return outPoint;
}



template <int dim>
void Point<dim>::normalize() {
  operator/=(abs());
}

template <int dim>
double Point<dim>::magnitude()
{
  return abs();
}

// Template instantiations.
template class Point<2u>;
template class Point<3u>;
template class Point<4u>;



