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


}


#endif//DENDRO_KT_SUBSET_H
