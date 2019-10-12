/**
 * @file distTree.h
 * @author Masado Ishii, UofU SoC
 * @date 2019-10-04
 * @brief Struct to hold part of a distributed tree.
 */

#ifndef DENDRO_KT_DIST_TREE_H
#define DENDRO_KT_DIST_TREE_H


#include "treeNode.h"
#include "octUtils.h"


namespace ot
{
  //TODO the DA class and the eleTreeIterator class should now take
  //references to DistTree instead of references to std::vector<TreeNode<T,dim>>

  /**
   * @brief Intermediate container for filtering trees before creating the DA.
   *
   * @note  DistTree takes ownership of the provided TreeNodes and empties the
   *        provided std::vector.
   *
   * @note  It is intended that, during construction of the DA, the tree
   *        vector held by DistTree will be destroyed.
   *
   * @note  Create a DistTree from a partitioned complete tree, i.e. taking
   *        union of the TreeNodes across all processors should be the entire
   *        unit hypercube. If you want to filter the domain to a subset of the
   *        unit hypercube, use DistTree to accomplish that.
   *
   * @note  DistTree remembers the front and back TreeNode from the original partition.
   * @note  The partition cannot be changed without creating a new DistTree.
   */
  template <typename T, unsigned int dim>
  class DistTree
  {
    public:
      // Member functions.
      //
      DistTree();
      DistTree(std::vector<TreeNode<T, dim>> &treePart);
      // Using default copy constructor and assignment operator.

      // filterTree() has 2 overloads, depending on the type of your decider.
      void filterTree(
          const std::function<bool(const TreeNode<T, dim> &treeNodeElem)>
            &domainDecider);
      void filterTree(
          const std::function<bool(const double *elemPhysCoords, double elemPhysSize)>
            &domainDecider);

      void destroyTree();

      const std::function<bool(const TreeNode<T, dim> &treeNodeElem)>
        & getDomainDeciderTN() const;

      const std::function<bool(const double *elemPhysCoords, double elemPhysSize)>
        & getDomainDeciderPh() const;

      const std::vector<TreeNode<T, dim>> & getTreePartFiltered() const;
      size_t getOriginalTreePartSz() const;
      size_t getFilteredTreePartSz() const;
      TreeNode<T, dim> getTreePartFront() const;
      TreeNode<T, dim> getTreePartBack() const;


      // These deciders can be called directly.

      // Default domainDecider (treeNode)
      static bool defaultDomainDeciderTN(const TreeNode<T, dim> &tn)
      {
        bool isInside = true;

        const T domSz = 1u << m_uiMaxDepth;
        const T elemSz = 1u << (m_uiMaxDepth - tn.getLevel());
        #pragma unroll(dim)
        for (int d = 0; d < dim; d++)
          if (/*tn.getX(d) < 0 || */ tn.getX(d) + elemSz > domSz)
            isInside = false;

        return isInside;
      }

      // Default domainDecider (physical)
      static bool defaultDomainDeciderPh(const double * physCoords, double physSize)
      {
        bool isInside = true;

        #pragma unroll(dim)
        for (int d = 0; d < dim; d++)
          if (physCoords[d] < 0.0 || physCoords[d] + physSize > 1.0)
            isInside = false;

        return isInside;
      }


    protected:
      // Member variables.
      //
      std::function<bool(const TreeNode<T, dim> &treeNodeElem)> m_domainDeciderTN;
      std::function<bool(const double *elemPhysCoords, double elemPhysSize)> m_domainDeciderPh;

      bool m_usePhysCoordsDecider;

      size_t m_originalTreePartSz;
      size_t m_filteredTreePartSz;
      std::vector<TreeNode<T, dim>> m_treePartFiltered;
      TreeNode<T, dim> m_treePartFront;
      TreeNode<T, dim> m_treePartBack;


      //
      // Intrinsic Deciders (not callable directly).
      //

      // If given a decider on phys coords, can still test treeNodes.
      bool conversionDomainDeciderTN(const TreeNode<T, dim> &tn)
      {
        double physCoords[dim];
        double physSize;
        treeNode2Physical(tn, physCoords, physSize);

        return m_domainDeciderPh(physCoords, physSize);
      }

      // If given a decider on treeNodes, can still test physCoords.
      bool conversionDomainDeciderPh(const double * physCoords, double physSize)
      {
        return m_domainDeciderTN(physical2TreeNode<T,dim>(physCoords, physSize));
      }
  };


  //
  // DistTree() - default constructor
  //
  template <typename T, unsigned int dim>
  DistTree<T, dim>::DistTree()
  {
    m_usePhysCoordsDecider = false;

    m_domainDeciderTN = DistTree::defaultDomainDeciderTN;
    m_domainDeciderPh = DistTree::defaultDomainDeciderPh;
    m_originalTreePartSz = 0;
    m_filteredTreePartSz = 0;
  }


  //
  // DistTree() - constructor
  //
  template <typename T, unsigned int dim>
  DistTree<T, dim>::DistTree(std::vector<TreeNode<T, dim>> &treePart)
  {
    m_usePhysCoordsDecider = false;

    m_domainDeciderTN = DistTree::defaultDomainDeciderTN;
    m_domainDeciderPh = DistTree::defaultDomainDeciderPh;
    m_originalTreePartSz = treePart.size();
    m_filteredTreePartSz = treePart.size();

    m_treePartFiltered.clear();
    std::swap(m_treePartFiltered, treePart);  // Steal the tree vector.

    if (treePart.size())
    {
      m_treePartFront = treePart.front();
      m_treePartBack = treePart.back();
    }
  }


  //
  // destroyTree()
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::destroyTree()
  {
    m_treePartFiltered.clear();
    m_treePartFiltered.shrink_to_fit();
  }


  //
  // filterTree() (treeNode)
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::filterTree( const std::function<bool(const TreeNode<T, dim> &treeNodeElem)>
                                       &domainDecider)
  {
    m_usePhysCoordsDecider = false;
    m_domainDeciderTN = domainDecider;
    m_domainDeciderPh = this->conversionDomainDeciderPh;

    const size_t oldSz = m_treePartFiltered.size();
    size_t ii = 0;

    // Find first element to delete.
    while (ii < oldSz && domainDecider(m_treePartFiltered[ii]))
      ii++;

    m_filteredTreePartSz = ii;

    // Keep finding and deleting elements.
    for ( ; ii < oldSz ; ii++)
      if (!domainDecider(m_treePartFiltered[ii]))
        m_treePartFiltered[m_filteredTreePartSz++] = std::move(m_treePartFiltered[ii]);

    m_treePartFiltered.resize(m_filteredTreePartSz);
  }


  //
  // filterTree() (physical)
  //
  template <typename T, unsigned int dim>
  void DistTree<T, dim>::filterTree( const std::function<bool(const double *elemPhysCoords,
                                                              double elemPhysSize)>   &domainDecider)
  {
    m_usePhysCoordsDecider = true;
    m_domainDeciderPh = domainDecider;
    m_domainDeciderTN = this->conversionDomainDeciderTN;

    // Intermediate variables to pass treeNode2Physical()-->domainDecider().
    double physCoords[dim];
    double physSize;

    const size_t oldSz = m_treePartFiltered.size();
    size_t ii = 0;

    // Find first element to delete.
    while (ii < oldSz
        && (treeNode2Physical(m_treePartFiltered[ii], physCoords, physSize)
            , domainDecider(physCoords, physSize)))
      ii++;

    m_filteredTreePartSz = ii;

    // Keep finding and deleting elements.
    for ( ; ii < oldSz ; ii++)
      if ( !(treeNode2Physical(m_treePartFiltered[ii], physCoords, physSize)
            , domainDecider(physCoords, physSize)) )
        m_treePartFiltered[m_filteredTreePartSz++] = std::move(m_treePartFiltered[ii]);

    m_treePartFiltered.resize(m_filteredTreePartSz);
  }



  //
  // getDomainDeciderTN()
  //
  template <typename T, unsigned int dim>
  const std::function<bool(const TreeNode<T, dim> &treeNodeElem)> &
      DistTree<T, dim>::getDomainDeciderTN() const
  {
    return m_domainDeciderTN;
  }


  //
  // getDomainDeciderPh()
  //
  template <typename T, unsigned int dim>
  const std::function<bool(const double *elemPhysCoords, double elemPhysSize)> &
      DistTree<T, dim>::getDomainDeciderPh() const
  {
    return m_domainDeciderPh;
  }


  //
  // getTreePartFiltered()
  //
  template <typename T, unsigned int dim>
  const std::vector<TreeNode<T, dim>> &
      DistTree<T, dim>::getTreePartFiltered() const
  {
    return m_treePartFiltered;
  }


  //
  // getOriginalTreePartSz()
  //
  template <typename T, unsigned int dim>
  size_t DistTree<T, dim>::getOriginalTreePartSz() const
  {
    return m_originalTreePartSz;
  }


  //
  // getFilteredTreePartSz()
  //
  template <typename T, unsigned int dim>
  size_t DistTree<T, dim>::getFilteredTreePartSz() const
  {
    return m_filteredTreePartSz;
  }


  //
  // getTreePartFront()
  //
  template <typename T, unsigned int dim>
  TreeNode<T, dim> DistTree<T, dim>::getTreePartFront() const
  {
    return m_treePartFront;
  }


  //
  // getTreePartBack()
  //
  template <typename T, unsigned int dim>
  TreeNode<T, dim> DistTree<T, dim>::getTreePartBack() const
  {
    return m_treePartBack;
  }



}//namespace ot



#endif//DENDRO_KT_DIST_TREE_H
