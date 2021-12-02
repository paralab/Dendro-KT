#ifndef DENDRO_KT_VECTOR_H
#define DENDRO_KT_VECTOR_H

#include <vector>
#include <algorithm>

#include "gridWrapper.h"
#include "idx.h"
#include "oda.h"

namespace ot
{
  template <typename ValT>  class Vector;
  template <typename ValT>  class LocalVector;
  template <typename ValT>  class GhostedVector;

  //
  // Vector  (wrapper)
  //
  template <typename ValT>
  class Vector
  {
    private:
      std::vector<ValT> m_data;
      bool m_isGhosted = false;
      size_t m_ndofs = 1;
      unsigned long long m_globalNodeRankBegin = 0;

    protected:
      Vector(size_t size, bool isGhosted, size_t ndofs, unsigned long long globalRankBegin)
        : m_data(size), m_isGhosted(isGhosted), m_ndofs(ndofs), m_globalNodeRankBegin(globalRankBegin)
      { }

    public:
      // Vector constructor
      template <unsigned int dim>
      Vector(const GridWrapper<dim> &mesh, bool isGhosted, size_t ndofs, const ValT *input = nullptr)
      : m_isGhosted(isGhosted), m_ndofs(ndofs)
      {
        mesh.da()->createVector(m_data, false, isGhosted, ndofs);
        if (input != nullptr)
          std::copy_n(input, m_data.size(), m_data.begin());
        m_globalNodeRankBegin = mesh.da()->getGlobalRankBegin();
      }

      // data()
      std::vector<ValT> & data() { return m_data; }
      const std::vector<ValT> & data() const { return m_data; }

      // isGhosted()
      bool isGhosted() const { return m_isGhosted; }

      // ndofs()
      size_t ndofs() const { return m_ndofs; }

      // ptr()
      ValT * ptr() { return m_data.data(); }
      const ValT * ptr() const { return m_data.data(); }

      // size()
      size_t size() const { return m_data.size(); }

      // globalRankBegin();
      unsigned long long globalRankBegin() const { return m_globalNodeRankBegin; }
  };


  //
  // LocalVector
  //
  template <typename ValT>
  class LocalVector : public Vector<ValT>
  {
    protected:
      LocalVector(size_t size, size_t ndofs, unsigned long long globalRankBegin)
        : Vector<ValT>::Vector(size, false, ndofs, globalRankBegin)
      { }

    public:
      template <unsigned int dim>
      LocalVector(const GridWrapper<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
      : Vector<ValT>(mesh, false, ndofs, input)
      {}

      ValT & operator()(const idx::LocalIdx &local, size_t dof = 0) {
        return this->ptr()[local * this->ndofs() + dof];
      }
      const ValT & operator()(const idx::LocalIdx &local, size_t dof = 0) const {
        return this->ptr()[local * this->ndofs() + dof];
      }

      ValT & operator[](const idx::LocalIdx &local) { return (*this)(local, 0); }
      const ValT & operator[](const idx::LocalIdx &local) const { return (*this)(local, 0); }

      LocalVector operator+(const LocalVector &b) const {
        using idx::LocalIdx;
        const LocalVector &a = *this;
        LocalVector c(this->size(), this->ndofs(), this->globalRankBegin());
        assert(sizes_match(a, b));
        for (size_t i = 0; i < this->size(); ++i)
          c[LocalIdx(i)] = a[LocalIdx(i)] + b[LocalIdx(i)];
        return c;
      }

      LocalVector operator-(const LocalVector &b) const {
        using idx::LocalIdx;
        const LocalVector &a = *this;
        LocalVector c(this->size(), this->ndofs(), this->globalRankBegin());
        assert(sizes_match(a, b));
        for (size_t i = 0; i < this->size(); ++i)
          c[LocalIdx(i)] = a[LocalIdx(i)] - b[LocalIdx(i)];
        return c;
      }

      static bool sizes_match(const LocalVector &a, const LocalVector &b)
      {
        return a.size() == b.size();
      }
  };


  //
  // GhostedVector
  //
  template <typename ValT>
  class GhostedVector : public Vector<ValT>
  {
    public:
      template <unsigned int dim>
      GhostedVector(const GridWrapper<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
      : Vector<ValT>(mesh, true, ndofs, input)
      {}

      ValT & operator[](const idx::GhostedIdx &ghosted) { return this->ptr()[ghosted]; }
      const ValT & operator[](const idx::GhostedIdx &ghosted) const { return this->ptr()[ghosted]; }
  };
}

#endif//DENDRO_KT_VECTOR_H
