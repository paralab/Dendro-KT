
#ifndef DENDRO_KT_OCTREE_TO_GRAPH_HPP
#define DENDRO_KT_OCTREE_TO_GRAPH_HPP

#include <set>
#include <ostream>
#include <mpi.h>
#include "include/oda.h"
#include "include/sfcTreeLoop_matvec_io.h"

// Quick and dirty (serial):
//  1. Construct DA 
//  2. MatVec loop -> e2n
//  3. e2n -> n2e
//  4. e2n + n2e -> e2e

namespace graph
{
  // Note that from and to can be two different sets.
  // I.e., it is assumed to be bipartite with zero-based indices for both sides.

  using Edge = std::pair<size_t, size_t>;

  struct ElementGraph
  {
    using Iterator = std::set<Edge>::const_iterator;

    std::set<Edge> m_forward;
    std::set<Edge> m_backward;

    size_t m_max_from = 0u;
    size_t m_max_to = 0u;

    // ---------------

    void insert_edge(size_t from, size_t to)
    {
      m_forward.insert(Edge{from, to});
      m_backward.insert(Edge{to, from});
      m_max_from = std::max(m_max_from, from);
      m_max_to = std::max(m_max_to, to);
    }

    size_t max_from() const { return m_max_from; }
    size_t max_to() const { return m_max_to; }
    size_t n_edges() const { return m_forward.size(); }

    std::pair<Iterator, Iterator> range_from(size_t from)
    {
      return {m_forward.lower_bound(Edge{from, 0}),
              m_forward.lower_bound(Edge{from + 1, 0})};
    }

    std::pair<Iterator, Iterator> range_to(size_t to)
    {
      return {m_backward.lower_bound(Edge{to, 0}),
              m_backward.lower_bound(Edge{to + 1, 0})};
    }

    friend std::ostream & operator<<(std::ostream &out, const ElementGraph &graph)
    {
      const size_t n_source = graph.max_from() + 1;
      out << n_source << "\n";
      out << graph.n_edges() << "\n";
      for (Edge e : graph.m_forward)
        out << e.first << "\t" << e.second << "\n";
      return out;
    }
  };

  enum SelfLoop { Keep, Remove };
}

namespace ot
{

  template <int dim>
  graph::ElementGraph octree_to_graph(
      const DistTree<unsigned, dim> &dtree,
      MPI_Comm comm,
      const graph::SelfLoop self_loop_policy)
  {
    const auto & octlist = dtree.getTreePartFiltered();
    const DA<dim> da(dtree, comm, 1);

    using LLU = long long unsigned;
    const LLU n_elements = da.getGlobalElementSz();
    const LLU n_nodes = da.getGlobalNodeSz();

    const unsigned int eleOrder = da.getElementOrder();
    const unsigned int nPe = da.getNumNodesPerElement();

    // Loop over all elements, adding row chunks from elemental matrices.
    // Get the node indices on an element using MatvecBaseIn<dim, unsigned int, false>.

    graph::ElementGraph e2n_graph;

    if (da.isActive())
    {
      using CoordT = typename ot::DA<dim>::C;
      using ScalarT = DendroScalar;
      using IndexT = long long unsigned;

      const size_t ghostedNodalSz = da.getTotalNodalSz();
      const ot::TreeNode<CoordT, dim> *odaCoords = da.getTNCoords();
      const std::vector<RankI> &ghostedGlobalNodeId = da.getNodeLocalToGlobalMap();

      const bool visitEmpty = false;
      const unsigned int padLevel = 0;
      ot::MatvecBaseIn<dim, RankI, false> treeLoopIn(ghostedNodalSz,
                                                     1,                // node id is scalar
                                                     eleOrder,
                                                     visitEmpty,
                                                     padLevel,
                                                     odaCoords,
                                                     &(*ghostedGlobalNodeId.cbegin()),
                                                     &(*octlist.cbegin()),
                                                     octlist.size(),
                                                     *da.getTreePartFront(),
                                                     *da.getTreePartBack());

      LLU element_id = da.getGlobalElementBegin();

      // Iterate over all leafs of the local part of the tree.
      while (!treeLoopIn.isFinished())
      {
        const ot::TreeNode<CoordT, dim> subtree = treeLoopIn.getCurrentSubtree();
        const auto subtreeInfo = treeLoopIn.subtreeInfo();

        if (treeLoopIn.isPre() && subtreeInfo.isLeaf())
        {
          const RankI * nodeIdsFlat = subtreeInfo.readNodeValsIn();
          const auto & nonhanging = subtreeInfo.readNodeNonhangingIn();

          for (size_t i = 0; i < nPe; ++i)
            if (nonhanging[i])
              e2n_graph.insert_edge(element_id, nodeIdsFlat[i]);

          // Note that even hanging faces contain a nonhanging node
          // from which we can infer element adjacency.
          // This simplification means the graph will not capture hanging
          // nodal dependencies, only strict spatial adjacencies.

          ++element_id;
        }
        treeLoopIn.step();
      }
    }


    // For now, assume sequential, to make it easier to invert e2n.
    assert(par::mpi_comm_size(comm) == 1);

    graph::ElementGraph e2e_graph;

    const LLU element_begin = da.getGlobalElementBegin();
    const LLU element_end = element_begin + da.getLocalElementSz();

    for (LLU element = element_begin; element < element_end; ++element)
    {
      const auto e2n_range = e2n_graph.range_from(element);
      for (auto e2n_edge = e2n_range.first; e2n_edge != e2n_range.second; ++e2n_edge)
      {
        const size_t node = e2n_edge->second;
        const auto n2e_range = e2n_graph.range_to(node);
        for (auto n2e_edge = n2e_range.first; n2e_edge != n2e_range.second; ++n2e_edge)
        {
          const size_t adjacent_element = n2e_edge->second;
          if (element != adjacent_element or
              self_loop_policy == graph::SelfLoop::Keep)
            e2e_graph.insert_edge(element, adjacent_element);
        }
      }
    }

    return e2e_graph;
  }

}


#endif//DENDRO_KT_OCTREE_TO_GRAPH_HPP
