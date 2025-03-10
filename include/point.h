/**
  @file Point.h
  @brief A point class
  @author Hari Sundar
  */

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
#ifndef __POINT_H
#define __POINT_H

#include  <cmath>
#include  <array>

/**
  @brief A point class
  @author Hari Sundar
  @author Masado Ishii
  */
template <int dim>
class Point{
  public:
    static constexpr int m_uiDim = (dim > 3 ? dim : 3);

    /** @name Constructors and Destructor */
    //@{
    Point() = default;
    // virtual ~Point();

    Point(double scale);
    Point(const std::array<double, dim> &newCoords);
    Point(const double * newCoords);
    Point(double newx, double newy, double newz);
    Point(int newx, int newy, int newz);
    Point(unsigned int newx, unsigned int newy, unsigned int newz);
    Point(const Point &newpoint) = default;
    Point(Point &&newpoint) = default;
    //@}

    Point& operator=(const Point &other) = default;
    Point& operator=(Point &&other) = default;

    /** @name Getters */
    //@{
    const double& x() const {return _p[0]; };
    const double& y() const {return _p[1]; };
    const double& z() const {return _p[2]; };
    const double& x(int d) const { return _p[d]; }

    int xint() const {return static_cast<int>(_p[0]); };
    int yint() const {return static_cast<int>(_p[1]); };
    int zint() const {return static_cast<int>(_p[2]); };
    int xint(int d) const { return static_cast<int>(_p[d]); }
    //@}

    /** @name Overloaded Operators */
    //@{
    inline Point operator-() const;

    inline void operator += (const Point &other);
    inline void operator -= (const Point &other);
    inline void operator /= (const int divisor);
    inline void operator /= (const double divisor);
    inline void operator *= (const int factor);
    inline void operator *= (const double factor);

    inline Point  operator+(const Point &other) const;
    inline Point  operator-(const Point &other) const;

    inline Point  operator/(const double divisor) const;
    inline Point  operator*(const double factor) const;
    
    double magnitude();

    inline bool operator != (const Point &other) const
    {
      return this->_p != other._p;
    }

    inline bool operator == (const Point &other) const
    {
      return this->_p == other._p;
    }
    //@}

    inline double dot(Point other) const { 
      double sum = 0.0;
      #pragma unroll(dim)
      for (int d = 0; d < dim; d++)
        sum += _p[d] * other._p[d];
      return sum;
    }

    inline double dot3(Point Other) const {
      return  (_p[0]*Other._p[0]+_p[1]*Other._p[1]+_p[2]*Other._p[2]);
    };

    inline Point cross(Point  Other) const{
      return  Point(_p[1]*Other._p[2]-Other._p[1]*_p[2], _p[2]*Other._p[0]-_p[0]*Other._p[2], 
          _p[0]*Other._p[1]-_p[1]*Other._p[0]); 
    };

    inline double abs() const{
      return sqrt(dot(*this));
    };

    void normalize();

    static Point TransMatMultiply( double* transMat, Point inPoint);
    static Point TransMatMultiply3( double* transMat, Point inPoint);
  protected:
    inline void initialize3(double newx, double newy, double newz);

    std::array<double,m_uiDim> _p = {};
};

#endif // POINT_H
