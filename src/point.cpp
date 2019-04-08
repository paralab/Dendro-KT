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

template <unsigned int dim>
Point<dim>::Point()
{
  _coords.fill(0.0);
}

template <unsigned int dim>
Point<dim>::Point(const std::array<double, dim> &newCoords)
{
  std::copy(&newCoords[0], &newCoords[dim], &_coords[0]);
  std::fill(&_coords[dim], &_coords[m_uiDim], 0.0);
}

template <unsigned int dim>
Point<dim>::Point(double newx, double newy, double newz)
{
  initialize3(newx, newy, newz);
}

template <unsigned int dim>
Point<dim>::Point(int newx, int newy, int newz)
{ 
  initialize3(static_cast<double>(newx),
      static_cast<double>(newy),
      static_cast<double>(newz));
}

template <unsigned int dim>
Point<dim>::Point(unsigned int newx, unsigned int newy, unsigned int newz)
{ 
  initialize3(static_cast<double>(newx),
      static_cast<double>(newy),
      static_cast<double>(newz));
}

template <unsigned int dim>
Point<dim>::Point(const Point &newposition)
{
  _coords = newposition._coords;
}
/*
template <unsigned int dim>
Point<dim>::~Point()
{

}
*/

template <unsigned int dim>
inline void Point<dim>::initialize3(double newx, double newy, double newz)
{
  _x = newx;  _y = newy;  _z = newz;
}

template <unsigned int dim>
Point<dim> Point<dim>::operator - () const {
	return Point(-_x, -_y, -_z);
}

template <unsigned int dim>
void Point<dim>::operator *= (const int factor){
	_x*=factor;
	_y*=factor;
	_z*=factor;
}

template <unsigned int dim>
void Point<dim>::operator *= (const double factor){
	_x*=factor;
	_y*=factor;
	_z*=factor;
}

template <unsigned int dim>
void Point<dim>::operator /= (const int divisor){
	if (divisor == 0) return;
	_x /= static_cast<double>(divisor);
	_y /= static_cast<double>(divisor);
	_z /= static_cast<double>(divisor);
}

template <unsigned int dim>
void Point<dim>::operator /= (const double divisor){
	if (divisor == 0) return;
	_x /= divisor;
	_y /= divisor;
	_z /= divisor;
}

template <unsigned int dim>
void Point<dim>::operator += (const Point& other){
	_x += other._x;
	_y += other._y;
	_z += other._z;
}

template <unsigned int dim>
Point<dim> Point<dim>::operator - (const Point &other){
	return Point(_x-other._x,_y-other._y, _z-other._z);
}

template <unsigned int dim>
Point<dim> Point<dim>::operator - (const Point &other) const {
	return Point(_x-other._x,_y-other._y, _z-other._z);
}

template <unsigned int dim>
Point<dim> Point<dim>::operator + (const Point &other){
	return Point(_x+other._x,_y+other._y, _z+other._z);
}

template <unsigned int dim>
Point<dim>& Point<dim>::operator=(const Point &other){
	_x = other._x;
	_y = other._y;
	_z = other._z;
	return *this;
}

template <unsigned int dim>
Point<dim> Point<dim>::operator /(const double divisor)
{
	return Point(_x/divisor,_y/divisor, _z/divisor);
}

template <unsigned int dim>
Point<dim> Point<dim>::operator *(const double factor)
{
	return Point(_x*factor,_y*factor, _z*factor);
}

template <unsigned int dim>
Point<dim> Point<dim>::TransMatMultiply(double *transMat, Point inPoint)
{
	Point outPoint;

	outPoint._x = transMat[ 0]*inPoint._x +transMat[ 4]*inPoint._y +transMat[8]
		*inPoint._z +transMat[12];
	outPoint._y = transMat[ 1]*inPoint._x +transMat[ 5]*inPoint._y +transMat[9]
		*inPoint._z +transMat[13];
	outPoint._z = transMat[ 2]*inPoint._x +transMat[ 6]*inPoint._y
		+transMat[10]*inPoint._z +transMat[14];

	return outPoint;
}

template <unsigned int dim>
void Point<dim>::normalize() {
	double abs = sqrt(_x*_x + _y*_y + _z*_z);
	_x /= abs; _y /= abs; _z /= abs;
}

template <unsigned int dim>
double Point<dim>::magnitude()
{
  double abs = sqrt(_x*_x + _y*_y + _z*_z);
  return abs;
}

