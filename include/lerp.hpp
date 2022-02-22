
/**
 * @author Masado Ishii
 * @date 2022-02-17
 */

#ifndef DENDRO_KT_LERP_HPP
#define DENDRO_KT_LERP_HPP

#include "treeNode.h"
#include "distTree.h"
#include "oda.h"
#include "p2p.h"
#include "sfcTreeLoop_matvec_io.h"

namespace ot
{
  //future: rename because it's not just lerp if have cell data

  template <unsigned dim>
  void lerp(
      const DistTree<unsigned, dim> &from_dtree,
      const DA<dim> *from_da,
      const int ndofs,
      const std::vector<double> &from_local,
      const DistTree<unsigned, dim> &to_dtree,
      const DA<dim> *to_da,
      std::vector<double> &to_local);

  template <unsigned dim>
  void locLerp(
      const std::vector<TreeNode<unsigned, dim>> &from_octlist,
      const TreeNode<unsigned, dim> *from_nodes,
      const size_t from_nodes_sz,
      const double * from_dofs,
      const int ndofs,
      const std::vector<TreeNode<unsigned, dim>> &to_octlist,
      const TreeNode<unsigned, dim> *to_nodes,
      const size_t to_nodes_sz,
      double * to_dofs);
}


// =======================================================================

namespace ot
{
  // lerp()
  template <unsigned dim>
  void lerp(
      const DistTree<unsigned, dim> &from_dtree,
      const DA<dim> *from_da,
      const int ndofs,
      const std::vector<double> &from_local,
      const DistTree<unsigned, dim> &to_dtree,
      const DA<dim> *to_da,
      std::vector<double> &to_local)
  {
    using Oct = TreeNode<unsigned, dim>;
    using OctList = std::vector<Oct>;

    assert(from_local.size() == ndofs * from_da->getLocalNodalSz());
    assert(to_local.size() == ndofs * to_da->getLocalNodalSz());
    assert(from_da->getGlobalComm() == to_da->getGlobalComm());

    int comm_size, comm_rank;
    MPI_Comm comm = from_da->getGlobalComm();
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    const auto *from_loc_cells = &(*from_dtree.getTreePartFiltered().begin());
    const auto *from_loc_nodes = from_da->getTNCoords() + from_da->getLocalNodeBegin();
    const auto *to_cells = &(*to_dtree.getTreePartFiltered().begin());
    const auto *to_nodes = to_da->getTNCoords() + to_da->getLocalNodeBegin();
    const size_t from_loc_cell_sz = from_da->getLocalElementSz();
    const size_t from_loc_node_sz = from_da->getLocalNodalSz();
    const size_t to_cell_sz = to_da->getLocalElementSz();
    const size_t to_node_sz = to_da->getLocalNodalSz();

    // Find elements in coarse grid that overlap with fine partitions.

    /// // Splitters of active ranks.
    /// std::vector<int> active;
    /// const PartitionFrontBack<T, dim> partition =
    ///     allgatherSplitters(tree.size() > 0, tree.front(), tree.back(), comm, &active);

    /// std::map<int, int> invActive;
    /// for (int ar = 0; ar < active.size(); ++ar)
    ///   invActive[active[ar]] = ar;

    /// const int activeRank = (isActive ? invActive[commRank] : -1);

    /// std::vector<IntRange> insulationProcRanges =
    ///     treeNode2PartitionRanks(insulationOfOwned, partition, &active);

    // Ghost read coarse and send coarse element nodes to overlaps in fine grid.
    const size_t to_local_nodes = to_da->getLocalNodalSz();
    //TODO
      // stub: send all nodes to all ranks
    std::vector<int> all_but_self;
    for (int rank = 0; rank < comm_size; ++rank)
      if (rank != comm_rank)
        all_but_self.push_back(rank);
    par::P2PPartners p2p_partners(all_but_self, all_but_self, comm);
    par::P2PScalar<int, 2> p2p_scalar(&p2p_partners);
    par::P2PMeta p2p_cells(&p2p_partners);
    par::P2PMeta p2p_nodes(&p2p_partners);
    p2p_cells.reserve(comm_size - 1, comm_size - 1, 2);  // cells, cell dofs
    p2p_nodes.reserve(comm_size - 1, comm_size - 1, 2);  // nodes, node dofs
    // Communicate meta data.
    for (int rank = 0, s = 0; rank < comm_size; ++rank)
      if (rank != comm_rank)
      {
        p2p_scalar.send(s, int(from_loc_cell_sz), int(from_loc_node_sz));
        p2p_cells.schedule_send(s, int(from_loc_cell_sz), 0);
        p2p_nodes.schedule_send(s, int(from_loc_node_sz), 0);
        ++s;
      }
    for (int rank = 0, r = 0; rank < comm_size; ++rank)
    {
      if (rank != comm_rank)
      {
        p2p_cells.recv_size(r, p2p_scalar.recv<0>(r));
        p2p_nodes.recv_size(r, p2p_scalar.recv<1>(r));
        ++r;
      }
      else
      {
        p2p_cells.self_size(r, int(from_loc_cell_sz));
        p2p_nodes.self_size(r, int(from_loc_node_sz));
      }
    }
    p2p_cells.tally_recvs();
    p2p_nodes.tally_recvs();

    // Send
    p2p_cells.send(from_loc_cells);
    p2p_nodes.send(from_loc_nodes);
    //future: send_dofs(cell data)
    p2p_nodes.send_dofs(from_local.data(), ndofs);

    // Allocate
    OctList from_total_cells(p2p_cells.recv_total() + from_loc_cell_sz);
    OctList from_total_nodes(p2p_nodes.recv_total() + from_loc_node_sz);
    //future: allocate to receive cell data
    std::vector<double> from_total_dofs((p2p_nodes.recv_total() + from_loc_node_sz) * ndofs);

    // Copy self overlap
    const auto copy_self = [](const auto &from, auto &to, const par::P2PMeta &meta, int nd = 1) {
      std::copy_n(from, meta.self_size() * nd, &to[meta.self_offset() * nd]);
    };
    copy_self(from_loc_cells, from_total_cells, p2p_cells);
    copy_self(from_loc_nodes, from_total_nodes, p2p_nodes);
    //future: copy cell data
    copy_self(from_local.data(), from_total_dofs, p2p_nodes, ndofs);

    // Receive
    p2p_cells.recv(from_total_cells.data());
    p2p_nodes.recv(from_total_nodes.data());
    //future: recv cell data
    p2p_nodes.recv_dofs(from_total_dofs.data(), ndofs);

    // Wait on sends
    p2p_cells.wait_all();
    p2p_nodes.wait_all();

    // Fine interpolate from combination of local and remote nodes.
    assert(from_da->getElementOrder() == 1);
    locLerp(
        from_total_cells,
        from_total_nodes.data(), from_total_nodes.size(),
        from_total_dofs.data(),
        ndofs,
        to_dtree.getTreePartFiltered(),
        to_nodes, to_node_sz,
        to_local.data());

    //future: iterate _cell_ values to fine grid
  }


  //
  // locLerp()
  //
  template <unsigned dim>
  void locLerp(
      const std::vector<TreeNode<unsigned, dim>> &from_octlist,
      const TreeNode<unsigned, dim> *from_nodes,
      const size_t from_nodes_sz,
      const double * from_dofs,
      const int ndofs,
      const std::vector<TreeNode<unsigned, dim>> &to_octlist,
      const TreeNode<unsigned, dim> *to_nodes,
      const size_t to_nodes_sz,
      double * to_dofs)
  {
    using Oct = TreeNode<unsigned, dim>;
    using OctList = std::vector<Oct>;

    assert(from_octlist.size() > 0 or to_octlist.size() == 0);
    assert(to_octlist.size() > 0 or to_nodes_sz == 0);
    if (to_octlist.size() == 0)
      return;

    const int degree = 1;
    MatvecBaseIn<dim, double> from_loop(
        from_nodes_sz, ndofs, degree,
        false, 0,
        from_nodes, from_dofs,
        from_octlist.data(), from_octlist.size(),
        dummyOctant<dim>(),
        dummyOctant<dim>());

    MatvecBaseOut<dim, double, false> to_loop(
        to_nodes_sz, ndofs, degree,
        false, 0,
        to_nodes,
        to_octlist.data(), to_octlist.size(),
        dummyOctant<dim>(),
        dummyOctant<dim>());

    assert(degree == 1);
    const int npe = 1u << dim;
    std::vector<double> leaf_nodes(npe * ndofs, 0.0f);

    const auto subtree =[](const auto &loop) { return loop.getCurrentSubtree(); };
    const auto in = [](const Oct &a, const Oct &b) { return b.isAncestorInclusive(a); };

    size_t to_counter = 0;
    while (not from_loop.isFinished() and not to_loop.isFinished())
    {
      if (not in(subtree(to_loop), subtree(from_loop)))
      {
        from_loop.next();
      }
      else if (from_loop.isPre() and from_loop.isLeaf())  // Coarse leaf
      {
        const double * coarse_nodes = from_loop.subtreeInfo().readNodeValsIn();
        const std::array<unsigned, dim> coarse_origin = subtree(from_loop).getX();

        while (not to_loop.isFinished() and in(subtree(to_loop), subtree(from_loop)))
        {
          if (to_loop.isPre() and to_loop.isLeaf())  // Fine leaf
          {
            assert(degree == 1);  // Octant-based formula only valid for linear.

            const TreeNode<unsigned int, dim> * fine_nodes = to_loop.subtreeInfo().readNodeCoordsIn();
            const double ratio = 1.0 / (1u << (subtree(to_loop).getLevel() - subtree(from_loop).getLevel()));
            const int fine_height = m_uiMaxDepth - subtree(to_loop).getLevel();

            // Interpolate each fine node.
            for (unsigned to_vertex = 0; to_vertex < npe; ++to_vertex)
            {
              std::array<double, dim> t[2];  // t[0] = 1 - t[1]
              for (int d = 0; d < dim; ++d)
              {
                t[1][d] = ((fine_nodes[to_vertex].getX(d) - coarse_origin[d]) >> fine_height) * ratio;
                t[0][d] = 1.0 - t[1][d];
              }

              for (int dof = 0; dof < ndofs; ++dof)
                leaf_nodes[to_vertex * ndofs + dof] = 0;
              for (unsigned from_vertex = 0; from_vertex < npe; ++from_vertex)
              {
                double shape = 1.0;
                for (int d = 0; d < dim; ++d)
                  shape *= t[(from_vertex >> d) & 1u][d];
                for (int dof = 0; dof < ndofs; ++dof)
                {
                  leaf_nodes[to_vertex * ndofs + dof] +=
                      shape * coarse_nodes[from_vertex * ndofs + dof];
                }
              }
            }
            to_loop.subtreeInfo().overwriteNodeValsOut(leaf_nodes.data());

            //future: for cell data, consider overwriteNodeValsOutScalar() [if sfcTreeLoop]

            ++to_counter;
            to_loop.next();
          }
          else
          {
            to_loop.step();
          }
        }
        from_loop.next();
      }
      else
      {
        to_loop.step();
        from_loop.step();
      }
    }
    assert(to_loop.isFinished() and to_counter == to_octlist.size());
    // All fine cells must be accounted for.
    // Note that there may be unused coarse cells that get skipped.

    to_loop.finalize(to_dofs);
  }


}//namespace ot

#endif//DENDRO_KT_LERP_HPP
