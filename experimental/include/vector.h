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
        else
          std::fill(m_data.begin(), m_data.end(), 0);
        m_globalNodeRankBegin = mesh.da()->getGlobalRankBegin();
      }

      // Vector constructor from temporary std::vector
      template <unsigned int dim>
      Vector(const GridWrapper<dim> &mesh, bool isGhosted, size_t ndofs, std::vector<ValT> &&input)
      : m_isGhosted(isGhosted), m_ndofs(ndofs)
      {
        std::swap(m_data, input);
        mesh.da()->createVector(m_data, false, isGhosted, ndofs); //ensure size
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

      template <unsigned int dim>
      LocalVector(const GridWrapper<dim> &mesh, size_t ndofs, std::vector<ValT> &&input)
      : Vector<ValT>(mesh, false, ndofs, std::move(input))
      {}

      ValT & operator()(const idx::LocalIdx &local, size_t dof = 0) {
        return this->ptr()[local * this->ndofs() + dof];
      }
      const ValT & operator()(const idx::LocalIdx &local, size_t dof = 0) const {
        return this->ptr()[local * this->ndofs() + dof];
      }

      ValT & operator[](const idx::LocalIdx &local) { return (*this)(local, 0); }
      const ValT & operator[](const idx::LocalIdx &local) const { return (*this)(local, 0); }

      // future: Use Eigen for value vectors and views.

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

      template <unsigned int dim>
      GhostedVector(const GridWrapper<dim> &mesh, size_t ndofs, std::vector<ValT> &&input)
      : Vector<ValT>(mesh, true, ndofs, std::move(input))
      {}

      ValT & operator[](const idx::GhostedIdx &ghosted) { return this->ptr()[ghosted]; }
      const ValT & operator[](const idx::GhostedIdx &ghosted) const { return this->ptr()[ghosted]; }
  };


  template <unsigned int dim, typename ValT>
  GhostedVector<ValT> makeGhosted(const GridWrapper<dim> &grid, LocalVector<ValT> &&local)
  {
    const DA<dim> *da = grid.da();
    std::vector<ValT> &vec = local.data();
    vec.reserve(da->getTotalNodalSz() * local.ndofs());
    vec.insert(vec.begin(), da->getPreNodalSz(), 0);
    vec.insert(vec.end(), da->getPostNodalSz(), 0);
    return GhostedVector<ValT>(grid, local.ndofs(), std::move(vec));
  }

  template <unsigned int dim, typename ValT>
  LocalVector<ValT> makeLocal(const GridWrapper<dim> &grid, GhostedVector<ValT> &&ghosted)
  {
    const DA<dim> *da = grid.da();
    const size_t localBegin = da->getLocalNodeBegin();
    const size_t localEnd = da->getLocalNodeEnd();
    const size_t ndofs = ghosted.ndofs();
    std::vector<ValT> &vec = ghosted.data();
    vec.erase(vec.begin() + localEnd * ndofs, vec.end());
    vec.erase(vec.begin(), vec.begin() + localBegin * ndofs);
    return LocalVector<ValT>(grid, ndofs, std::move(vec));
  }
}

#endif//DENDRO_KT_VECTOR_H
