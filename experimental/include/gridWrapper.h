#ifndef DENDRO_KT_GRID_WRAPPER_H
#define DENDRO_KT_GRID_WRAPPER_H

#include <array>

#include "distTree.h"
#include "oda.h"
#include "idx.h"
#include "aabb.h"

namespace ot
{
  //
  // GridWrapper
  //
  template <unsigned int dim>
  class GridWrapper
  {
    private:
      const ot::DistTree<unsigned int, dim> *m_distTree;
      const ot::MultiDA<dim> *m_multiDA;
      const unsigned m_stratum;

    public:
      // GridWrapper constructor
      GridWrapper(const ot::DistTree<unsigned int, dim> *distTree,
                        const ot::MultiDA<dim> *multiDA,
                        unsigned stratum = 0)
        : m_distTree(distTree), m_multiDA(multiDA), m_stratum(stratum)
      {}

      // Copy constructor and copy assignment.
      GridWrapper(const GridWrapper &other) = default;
      GridWrapper & operator=(const GridWrapper &other) = default;

      // distTree()
      const ot::DistTree<unsigned int, dim> * distTree() const { return m_distTree; }

      // octList()
      const std::vector<ot::TreeNode<unsigned int, dim>> & octList() const
      {
        return m_distTree->getTreePartFiltered(m_stratum);
      }

      // numElements()
      size_t numElements() const { return octList().size(); }

      // da()
      const ot::DA<dim> * da() const { return &((*m_multiDA)[m_stratum]); }

      // multiDA()
      const ot::MultiDA<dim> * multiDA() const { return m_multiDA; }

      unsigned stratum() const { return m_stratum; }

      // future: printSummary() -- see testRestriction.cpp

      // --------------------------------------------------------------------

      // local2ghosted
      idx::GhostedIdx local2ghosted(const idx::LocalIdx &local) const
      {
        return idx::GhostedIdx(da()->getLocalNodeBegin() + local);
      }

      std::array<double, dim> nodeCoord(const idx::LocalIdx &local, const AABB<dim> &aabb) const
      {
        // Floating point coordinates in the unit cube.
        std::array<double, dim> coord;
        ot::treeNode2Physical( da()->getTNCoords()[local2ghosted(local)],
                               da()->getElementOrder(),
                               coord.data() );

        // Coordinates in the box represented by aabb.
        for (int d = 0; d < dim; ++d)
          coord[d] = coord[d] * (aabb.max().x(d) - aabb.min().x(d)) + aabb.min().x(d);

        return coord;
      }

      std::array<double, dim> nodeCoord(const idx::GhostedIdx &ghosted, const AABB<dim> &aabb) const
      {
        // Floating point coordinates in the unit cube.
        std::array<double, dim> coord;
        ot::treeNode2Physical( da()->getTNCoords()[ghosted],
                               da()->getElementOrder(),
                               coord.data() );

        // Coordinates in the box represented by aabb.
        for (int d = 0; d < dim; ++d)
          coord[d] = coord[d] * (aabb.max().x(d) - aabb.min().x(d)) + aabb.min().x(d);

        return coord;
      }
  };
}

#endif//DENDRO_KT_GRID_WRAPPER_H
