#ifndef DENDRO_KT_SUBSET_H
#define DENDRO_KT_SUBSET_H

#include "oda.h"
#include "sfcTreeLoop_matvec_io.h"

namespace ot
{
  template <typename IteratorT>
  using IterRange = std::pair<IteratorT, IteratorT>;

  template <typename IteratorT>
  IterRange<IteratorT> iter_range(const IteratorT begin, const IteratorT end)
  {
    return IterRange<IteratorT>{begin, end};
  }

  // === filter_where() ===

  template <typename T, typename K>
  std::vector<T> filter_where(
      const std::vector<T> &source,
      const std::vector<K> &input,
      const K key);


  // === index_where() ===

  template <typename T>
  std::vector<size_t> index_where(const std::vector<T> &input, const T key);

  template <typename T>
  std::vector<size_t> index_where(const T *input, const size_t size, const T key);

  template <typename IteratorT>
  std::vector<size_t> index_where(const IterRange<IteratorT> &range, const decltype(*IteratorT()) key);


  // === index_nodes_where_element() ===

  template <unsigned int dim, typename T>
  std::vector<size_t> index_nodes_where_element(
      const DA<dim> &da,
      const std::vector<TreeNode<unsigned, dim>> &octList,
      const std::vector<T> &elem_input,
      const T elem_key);

  template <unsigned int dim, typename T>
  std::vector<size_t> index_nodes_where_element(
      const DA<dim> &da,
      const TreeNode<unsigned, dim> *octList,
      const T *elem_input,
      const T elem_key);

  template <unsigned int dim, typename OctIteratorT, typename IteratorT>
  std::vector<size_t> index_nodes_where_element(
      const DA<dim> &da,
      const OctIteratorT &octList_begin,
      const IteratorT &elem_input_begin,
      const decltype(*IteratorT()) elem_key);


  // === gather() ===

  // gather(): return input[idxs[i]]
  template <typename T>
  std::vector<T> gather(const T *input, const std::vector<size_t> &idxs);

  // gather(): return input[idxs[i]]
  template <typename T>
  std::vector<T> gather(const std::vector<T> &input, const std::vector<size_t> &idxs);


  // === gather_ndofs() ===

  // gather_ndofs(): return input[idxs[i]*ndofs + dof]
  template <typename T>
  std::vector<T> gather_ndofs(const T *input, const std::vector<size_t> &idxs, const size_t ndofs);


  // === scatter_ndofs() ===

  // scatter_ndofs(): output[idxs[i]*ndofs + dof] = input[i*ndofs + dof]
  template <typename T, typename MergeNewOld>
  void scatter_ndofs(const T *input, MergeNewOld merge, T *output, const std::vector<size_t> &idxs, const size_t ndofs);


  //
  // GlobalSubset
  //
  template <unsigned int dim>
  class GlobalSubset
  {
    public:
      using C = unsigned int;

    protected:
      const DA<dim> *m_da;
      std::vector<TreeNode<C, dim>> m_relevantOctList;
      std::vector<TreeNode<C, dim>> m_relevantNodes;
      std::vector<size_t> m_originalIndices;

      // within global subset, not referring to original array.
      size_t m_localNodalSz;
      size_t m_localNodeBegin;

      void readyMpiGlobalSizes() const;
      mutable bool m_global_sizes_ready = false;
      mutable DendroIntL m_globalNodeBegin = -1;
      mutable DendroIntL m_globalElementBegin = -1;
      mutable DendroIntL m_globalNodeSz = -1;
      mutable DendroIntL m_globalElementSz = -1;

    public:
      template <typename Label>
      GlobalSubset( const DA<dim> *da,
                    const std::vector<TreeNode<C, dim>> *octList,
                    const std::vector<Label> & labels,
                    const Label selectedLabel );

      GlobalSubset() = default;
      GlobalSubset(const GlobalSubset &) = default;
      GlobalSubset(GlobalSubset &&) = default;
      GlobalSubset & operator=(const GlobalSubset &) = default;
      GlobalSubset & operator=(GlobalSubset &&) = default;

      const DA<dim> * da() const;
      const std::vector<TreeNode<unsigned, dim>> & relevantOctList() const;
      const std::vector<TreeNode<unsigned, dim>> & relevantNodes() const;
      const std::vector<size_t> & originalIndices() const;


      // ----------------------------------------------------
      // DA proxy methods as if the subset were whole vector.
      // ----------------------------------------------------

      size_t getLocalElementSz() const;
      size_t getLocalNodalSz() const;
      size_t getLocalNodeBegin() const;
      size_t getLocalNodeEnd() const;
      size_t getPreNodalSz() const;
      size_t getPostNodalSz() const;
      size_t getTotalNodalSz() const;
      /// bool isActive() const;   // commented out for lack of active comm
      RankI getGlobalNodeSz() const;
      RankI getGlobalNodeBegin() const;
      DendroIntL getGlobalElementSz() const;
      DendroIntL getGlobalElementBegin() const;
      //TODO
      /// const std::vector<RankI> & getNodeLocalToGlobalMap();

  };


}





// implementations



namespace ot
{

  // === filter_where() ===

  template <typename T, typename K>
  std::vector<T> filter_where(
      const std::vector<T> &source,
      const std::vector<K> &input,
      const K key)
  {
    std::vector<T> matches;

    for (size_t ii = 0; ii < input.size(); ++ii)
      if (input[ii] == key)
        matches.push_back(source[ii]);

    return matches;
  }


  // === index_where() ===

  template <typename T>
  std::vector<size_t> index_where(const std::vector<T> &input, const T key)
  {
    return index_where(iter_range(input.begin(), input.end()), key);
  }

  template <typename T>
  std::vector<size_t> index_where(const T *input, const size_t size, const T key)
  {
    return index_where(iter_range(input, input + size), key);
  }

  template <typename IteratorT>
  std::vector<size_t> index_where(const IterRange<IteratorT> &range, const decltype(*IteratorT()) key)
  {
    const IteratorT begin = range.first,  end = range.second;

    std::vector<size_t> indices;
    indices.reserve(std::distance(begin, end));

    IteratorT it;
    size_t index;
    for (it = begin, index = 0; it != end; ++it, ++index)
      if (*it == key)
        indices.push_back(index);

    return indices;
  }



  // === index_nodes_where_element() ===

  template <unsigned int dim, typename T>
  std::vector<size_t> index_nodes_where_element(
      const DA<dim> &da,
      const std::vector<TreeNode<unsigned, dim>> &octList,
      const std::vector<T> &elem_input,
      const T elem_key)
  {
    return index_nodes_where_element(
        da, octList.begin(), elem_input.begin(), elem_key);
  }

  template <unsigned int dim, typename T>
  std::vector<size_t> index_nodes_where_element(
      const DA<dim> &da,
      const TreeNode<unsigned, dim> *octList,
      const T *elem_input,
      const T elem_key)
  {
    return index_nodes_where_element<dim, TreeNode<unsigned, dim>*, T*>(
        da, octList, elem_input, elem_key);
  }

  template <unsigned int dim, typename OctIteratorT, typename IteratorT>
  std::vector<size_t> index_nodes_where_element(
      const DA<dim> &da,
      const OctIteratorT &octList_begin,
      const IteratorT &elem_input_begin,
      const decltype(*IteratorT()) elem_key)
  {
    const size_t numElems = da.getLocalElementSz();
    const size_t totalNodes = da.getTotalNodalSz();

    IteratorT it = elem_input_begin;

    std::vector<char> selected(totalNodes, false);

    MatvecBaseOut<dim, char, true> loop(
        totalNodes,
        1,
        da.getElementOrder(),
        false,
        0,
        da.getTNCoords(),
        &(*octList_begin),
        numElems);

    const std::vector<char> nodesTrue(da.getNumNodesPerElement(), true);
    const std::vector<char> nodesFalse(da.getNumNodesPerElement(), false);

    while (!loop.isFinished())
    {
      if (loop.isPre() && loop.subtreeInfo().isLeaf())
      {
        if (*it == elem_key)
          loop.subtreeInfo().overwriteNodeValsOut(&(*nodesTrue.begin()));
        else
          loop.subtreeInfo().overwriteNodeValsOut(&(*nodesFalse.begin()));

        ++it;
        loop.next();
      }
      else
        loop.step();
    }

    /*const size_t writtenSz =*/
    loop.finalize(&(*selected.begin()));

    for (char & flag : selected)
      flag = bool(flag);

    return index_where(selected, char(true));
  }


  // === gather() ===

  template <typename T>
  std::vector<T> gather(const T *input, const std::vector<size_t> &idxs)
  {
    std::vector<T> gathered;
    for (size_t ii : idxs)
      gathered.emplace_back(input[ii]);
    return gathered;
  }

  template <typename T>
  std::vector<T> gather(const std::vector<T> &input, const std::vector<size_t> &idxs)
  {
    std::vector<T> gathered;
    for (size_t ii : idxs)
      gathered.emplace_back(input[ii]);
    return gathered;
  }


  // === gather_ndofs() ===

  template <typename T>
  std::vector<T> gather_ndofs(const T *input, const std::vector<size_t> &idxs, const size_t ndofs)
  {
    std::vector<T> gathered;
    for (size_t ii : idxs)
      for (size_t dof = 0; dof < ndofs; ++dof)
        gathered.emplace_back(input[ii * ndofs + dof]);
    return gathered;
  }


  // === scatter_ndofs() ===

  // scatter_ndofs(): output[idxs[i]*ndofs + dof] = input[i*ndofs + dof]
  template <typename T, typename MergeNewOld>
  void scatter_ndofs(const T *input, MergeNewOld merge, T *output, const std::vector<size_t> &idxs, const size_t ndofs)
  {
    const T * input_it = input;
    for (size_t ii : idxs)
      for (size_t dof = 0; dof < ndofs; ++dof)
      {
        T & out_v = output[ii*ndofs + dof];
        out_v = merge(*input_it, out_v);
        ++input_it;
      }
  }


  // GlobalSubset::GlobalSubset()
  template <unsigned int dim>
  template <typename Label>
  GlobalSubset<dim>::GlobalSubset(
      const DA<dim> *da,
      const std::vector<TreeNode<C, dim>> *octList,
      const std::vector<Label> & labels,
      const Label selectedLabel )
    :
      m_da(da),
      m_relevantOctList(
          filter_where(*octList, labels, selectedLabel)),
      m_originalIndices(
          index_nodes_where_element(*da, *octList, labels, selectedLabel))
  {
    m_relevantNodes = gather(da->getTNCoords(), m_originalIndices);

    const size_t localBegin =
        std::lower_bound(m_originalIndices.begin(),
                         m_originalIndices.end(),
                         da->getLocalNodeBegin()) - m_originalIndices.begin();

    const size_t localEnd =
        std::lower_bound(m_originalIndices.begin(),
                         m_originalIndices.end(),
                         da->getLocalNodeEnd()) - m_originalIndices.begin();

    m_localNodalSz = localEnd - localBegin;
    m_localNodeBegin = localBegin;

    //TODO boundary nodes search
  }

  // GlobalSubset::da()
  template <unsigned int dim>
  const DA<dim> * GlobalSubset<dim>::da() const
  {
    return m_da;
  }

  // GlobalSubset::relevantOctList()
  template <unsigned int dim>
  const std::vector<TreeNode<unsigned, dim>> & GlobalSubset<dim>::relevantOctList() const
  {
    return m_relevantOctList;
  }

  // GlobalSubset::relevantNodes()
  template <unsigned int dim>
  const std::vector<TreeNode<unsigned, dim>> & GlobalSubset<dim>::relevantNodes() const
  {
    return m_relevantNodes;
  }

  // GlobalSubset::originalIndices()
  template <unsigned int dim>
  const std::vector<size_t> & GlobalSubset<dim>::originalIndices() const
  {
    return m_originalIndices;
  }

  // --------------

  // GlobalSubset::getLocalElementSz()
  template <unsigned int dim>
  size_t GlobalSubset<dim>::getLocalElementSz() const
  {
    return m_relevantOctList.size();
  }

  // GlobalSubset::getTotalNodalSz()
  template <unsigned int dim>
  size_t GlobalSubset<dim>::getTotalNodalSz() const
  {
    return m_relevantNodes.size();
  }

  // GlobalSubset::getLocalNodalSz()
  template <unsigned int dim>
  size_t GlobalSubset<dim>::getLocalNodalSz() const
  {
    return m_localNodalSz;
  }

  // GlobalSubset::getLocalNodeBegin()
  template <unsigned int dim>
  size_t GlobalSubset<dim>::getLocalNodeBegin() const
  {
    return m_localNodeBegin;
  }

  // GlobalSubset::getLocalNodeEnd()
  template <unsigned int dim>
  size_t GlobalSubset<dim>::getLocalNodeEnd() const
  {
    return m_localNodeBegin + m_localNodalSz;
  }

  // GlobalSubset::getPreNodalSz()
  template <unsigned int dim>
  size_t GlobalSubset<dim>::getPreNodalSz() const
  {
    return m_localNodeBegin;
  }

  // GlobalSubset::getPostNodalSz()
  template <unsigned int dim>
  size_t GlobalSubset<dim>::getPostNodalSz() const
  {
    return getTotalNodalSz() - getLocalNodeEnd();
  }


  // commented out until Subset gets its own active comm (might be costly)
  /// // GlobalSubset::isActive()
  /// template <unsigned int dim>
  /// bool GlobalSubset<dim>::isActive() const
  /// {
  ///   return getLocalElementSz() > 0;
  /// }

  // GlobalSubset::getGlobalNodeSz()
  template <unsigned int dim>
  RankI GlobalSubset<dim>::getGlobalNodeSz() const
  {
    readyMpiGlobalSizes();
    return m_globalNodeSz;
  }

  // GlobalSubset::getGlobalNodeBegin()
  template <unsigned int dim>
  RankI GlobalSubset<dim>::getGlobalNodeBegin() const
  {
    readyMpiGlobalSizes();
    return m_globalNodeBegin;
  }

  // GlobalSubset::getGlobalElementSz()
  template <unsigned int dim>
  DendroIntL GlobalSubset<dim>::getGlobalElementSz() const
  {
    readyMpiGlobalSizes();
    return m_globalElementSz;
  }

  // GlobalSubset::getGlobalElementBegin()
  template <unsigned int dim>
  DendroIntL GlobalSubset<dim>::getGlobalElementBegin() const
  {
    readyMpiGlobalSizes();
    return m_globalElementBegin;
  }

  // GlobalSubset::readyMpiGlobalSizes()   (internal)
  template <unsigned int dim>
  void GlobalSubset<dim>::readyMpiGlobalSizes() const
  {
    if (m_global_sizes_ready)
      return;

    // Future: Use da->getCommActive() instead of global comm, IF
    //   every node is owned by a rank that owns the owning element AND
    //   every rank only has ghost nodes for owned elements.
    MPI_Comm comm = m_da->getGlobalComm();

    DendroIntL locNodes = getLocalNodalSz();
    par::Mpi_Allreduce(&locNodes, &m_globalNodeSz, 1, MPI_SUM, comm);
    par::Mpi_Scan(&locNodes, &m_globalNodeBegin, 1, MPI_SUM, comm);
    m_globalNodeBegin -= locNodes;

    DendroIntL locElements = getLocalElementSz();
    par::Mpi_Allreduce(&locElements, &m_globalElementSz, 1, MPI_SUM, comm);
    par::Mpi_Scan(&locElements, &m_globalElementBegin, 1, MPI_SUM, comm);
    m_globalElementBegin -= locElements;

    m_global_sizes_ready = true;
  }





}


#endif//DENDRO_KT_SUBSET_H
