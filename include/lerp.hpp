
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
      const DA<dim> *to_da,  //quick fix
      const TreeNode<unsigned, dim> *to_nodes,
      const size_t to_nodes_sz,
      double * to_dofs);

  template <unsigned dim>
  void flag_essential_nodes(
      const TreeNode<unsigned, dim> *octants,
      const size_t oct_sz,
      const DA<dim> *da,
      std::vector<char> &node_essential);

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

    const OctList & from_octlist = from_dtree.getTreePartFiltered();
    const OctList & to_octlist = to_dtree.getTreePartFiltered();
    const auto *to_nodes = to_da->getTNCoords() + to_da->getLocalNodeBegin();
    const size_t to_node_sz = to_da->getLocalNodalSz();

    // Ghost read begin.
    std::vector<double> from_nodes_ghosted;
    from_da->nodalVecToGhostedNodal(from_local, from_nodes_ghosted, false, ndofs);
    from_da->readFromGhostBegin(from_nodes_ghosted.data(), ndofs);

    // Find elements in coarse grid that overlap with fine partitions.

    using SizeRange = IntRange<size_t>;
    using IntRange = IntRange<>;

    // Map coarse cells to fine partitions.
    const bool is_active = (to_octlist.size() > 0);
    std::vector<int> active;
    const PartitionFrontBack<unsigned, dim> partition =
        SFC_Tree<unsigned, dim>::allgatherSplitters(
            is_active, to_octlist.front(), to_octlist.back(), comm, &active);
    std::map<int, int> inv_active;
    for (int ar = 0; ar < active.size(); ++ar)
      inv_active[active[ar]] = ar;
    const int active_rank = (is_active ? inv_active[comm_rank] : -1);
    std::vector<IntRange> to_proc_ranges =
        SFC_Tree<unsigned, dim>::treeNode2PartitionRanks(
            from_octlist, partition, &active);
    // to_proc_ranges may include self

    par::P2PPartners partners;
    std::vector<int> active_to_dest_idx(active.size(), -1);
    SizeRange self_range;
    {
      // Which active ranks are destinations, i.e., have our coarse cells mapped.
      std::vector<int> send_to_active;
      std::vector<char> is_dest(active.size(), false);
      for (size_t i = 0; i < from_octlist.size(); ++i)
      {
        IntRange to_range = to_proc_ranges[i];

        for (int a = to_range.min; a <= to_range.max; ++a)
          is_dest[a] = true;
        if (to_range.min <= active_rank and active_rank <= to_range.max)
          self_range.include(i);
      }
      if (is_active)
        is_dest[active_rank] = false;   // exclude self from destinations

      // Sets of destination and source ranks.
      send_to_active.reserve(std::count(is_dest.begin(), is_dest.end(), true));
      for (size_t a = 0; a < is_dest.size(); ++a)
        if (is_dest[a])
        {
          active_to_dest_idx[a] = send_to_active.size();
          send_to_active.push_back(a);
        }
      std::vector<int> recv_from_active = recvFromActive(active, send_to_active, comm);

      // Active index to raw rank.
      partners.reset(send_to_active.size(), recv_from_active.size(), comm);
      for (size_t d = 0; d < send_to_active.size(); ++d)
        partners.dest(d, active[send_to_active[d]]);
      for (size_t s = 0; s < recv_from_active.size(); ++s)
        partners.src(s, active[recv_from_active[s]]);
    }
    const size_t self_cells = self_range.length();

    par::P2PScalar<int, 2> p2p_scalar(&partners);
    par::P2PMeta p2p_cells(&partners, 2);
    par::P2PMeta p2p_nodes(&partners, 2);
    // layers=2 for keys + dofs, prevents copying MPI_Request

    // Count cells per destination.
    size_t cell_send_total = 0;
    {
      std::vector<size_t> send_counts(partners.nDest(), 0);
      for (IntRange to_range : to_proc_ranges)
        for (int a = to_range.min; a <= to_range.max; ++a)
          if (a != active_rank)
          {
            assert(active_to_dest_idx[a] >= 0);
            send_counts[active_to_dest_idx[a]]++;
          }
      for (int dst = 0; dst < partners.nDest(); ++dst)
      {
        p2p_cells.schedule_send(dst, send_counts[dst], cell_send_total);
        cell_send_total += send_counts[dst];
      }
    }

    // Stage cells for each destination.
    OctList send_cells(cell_send_total);
    //future: cell dofs
    {
      std::vector<size_t> send_offsets(
          p2p_cells.send_offsets(),
          p2p_cells.send_offsets() + partners.nDest());
      for (size_t i = 0; i < from_octlist.size(); ++i)
      {
        IntRange to_range = to_proc_ranges[i];
        for (int a = to_range.min; a <= to_range.max; ++a)
          if (a != active_rank)
            send_cells[send_offsets[active_to_dest_idx[a]]++] = from_octlist[i];
        //future: cell dofs
      }
    }

    // Ghost read end.
    from_da->readFromGhostEnd(from_nodes_ghosted.data(), ndofs);

    // Cell sets to node sets
    OctList send_nodes;
    std::vector<double> send_node_dofs;
    std::vector<char> node_essential;
    for (int dst = 0; dst < partners.nDest(); ++dst)
    {
      node_essential.clear();
      flag_essential_nodes(
          &send_cells[p2p_cells.send_offsets()[dst]],
          p2p_cells.send_sizes()[dst],
          from_da,
          node_essential);

      const size_t n_essential =
          std::count(node_essential.begin(), node_essential.end(), true);

      p2p_nodes.schedule_send(dst, n_essential, send_nodes.size());

      send_nodes.reserve(send_nodes.size() + n_essential);
      send_node_dofs.reserve(send_node_dofs.size() + n_essential * ndofs);

      for (size_t i = 0; i < from_da->getTotalNodalSz(); ++i)
        if (node_essential[i])
        {
          send_nodes.push_back(from_da->getTNCoords()[i]);
          for (int dof = 0; dof < ndofs; ++dof)
            send_node_dofs.push_back(from_nodes_ghosted[i * ndofs + dof]);
        }
    }

    // Send the sizes and payloads.
    for (int dst = 0; dst < partners.nDest(); ++dst)
      p2p_scalar.send(dst,
          p2p_cells.send_sizes()[dst],
          p2p_nodes.send_sizes()[dst]);
    p2p_cells.send(send_cells.data());
    p2p_nodes.send(send_nodes.data());
    //future: send_dofs(send_cell_dofs.data(), ndofs);
    p2p_nodes.send_dofs(send_node_dofs.data(), ndofs);

    // Self node size and selection.
    size_t self_nodes = 0;
    if (is_active and self_range.nonempty())
    {
      node_essential.clear();
      flag_essential_nodes(
          &from_octlist[self_range.min],
          self_cells,
          from_da,
          node_essential);
      // node_essential now contains flags for self nodes.

      self_nodes = std::count(node_essential.begin(), node_essential.end(), true);
    }

    // Receive sizes.
    {
      int src = 0;
      for (; src < partners.nSrc() and partners.src(src) < comm_rank; ++src)
      {
        p2p_cells.recv_size(src, p2p_scalar.recv<0>(src));
        p2p_nodes.recv_size(src, p2p_scalar.recv<1>(src));
      }
      p2p_cells.self_size(src, self_cells);
      p2p_nodes.self_size(src, self_nodes);
      for (; src < partners.nSrc(); ++src)
      {
        p2p_cells.recv_size(src, p2p_scalar.recv<0>(src));
        p2p_nodes.recv_size(src, p2p_scalar.recv<1>(src));
      }
      p2p_cells.tally_recvs();
      p2p_nodes.tally_recvs();
    }

    // Allocate.
    OctList from_total_cells(p2p_cells.recv_total() + self_cells);
    OctList from_total_nodes(p2p_nodes.recv_total() + self_nodes);
    //future: cells
    std::vector<double> from_total_node_dofs((p2p_nodes.recv_total() + self_nodes) * ndofs);

    // Copy self overlap.
    if (is_active and self_range.nonempty())
    {
      // Cells
      std::copy_n(
          &from_octlist[self_range.min],
          self_cells,
          &from_total_cells[p2p_cells.self_offset()]);
      //future: cell dofs

      // Nodes
      const size_t node_offset = p2p_nodes.self_offset();
      for (size_t i = 0, j = 0; i < from_da->getTotalNodalSz(); ++i)
        if (node_essential[i])
        {
          from_total_nodes[node_offset + j] = from_da->getTNCoords()[i];
          for (int dof = 0; dof < ndofs; ++dof)
            from_total_node_dofs[(node_offset + j) * ndofs + dof] =
                from_nodes_ghosted[i * ndofs + dof];
          ++j;
        }
    }

    // Receive payloads.
    p2p_cells.recv(from_total_cells.data());
    p2p_nodes.recv(from_total_nodes.data());
    //future: recv_dofs(from_total_cell_dofs.data(), ndofs);
    p2p_nodes.recv_dofs(from_total_node_dofs.data(), ndofs);

    // Wait on sends.
    p2p_cells.wait_all();
    p2p_nodes.wait_all();

    printf("[%d] cells: .in=%lu  .self_offset=%d  .self_size=%d=%lu\n",
        comm_rank, from_octlist.size(), p2p_cells.self_offset(), p2p_cells.self_size(), self_cells);

    quadTreeToGnuplot(from_total_cells, 8, "total.cells", comm);

    // Fine interpolate from combination of local and remote nodes.
    assert(from_da->getElementOrder() == 1);
    locLerp(
        from_total_cells,
        from_total_nodes.data(), from_total_nodes.size(),
        from_total_node_dofs.data(),
        ndofs,
        to_octlist,
        to_da,
        to_nodes, to_node_sz,
        to_local.data());

    //future: iterate cell values to fine grid
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
      const DA<dim> *to_da,  //quick fix
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

    // future: Fix sfcTreeLoop_* to define leafs based on octlist, not nodes.
    // For now, leafs are only identified correctly if pass DA _total_ nodes.
    // To write to local nodes, have to create ghosted vector then extract.
    std::vector<double> ghosted_to_dofs(to_da->getTotalNodalSz() * ndofs, 0);

    const int degree = 1;
    MatvecBaseIn<dim, double> from_loop(
        from_nodes_sz, ndofs, degree,
        false, 0,
        from_nodes, from_dofs,
        from_octlist.data(), from_octlist.size(),
        dummyOctant<dim>(),
        dummyOctant<dim>());

    MatvecBaseOut<dim, double, false> to_loop(
        to_da->getTotalNodalSz(), ndofs, degree,
        false, 0,
        to_da->getTNCoords(),
        to_octlist.data(), to_octlist.size(),
        dummyOctant<dim>(),
        dummyOctant<dim>());

    // local only, not ghosted (won't work until sfcTreeLoop_* fixed)
    /// MatvecBaseOut<dim, double, false> to_loop(
    ///     to_nodes_sz, ndofs, degree,
    ///     false, 0,
    ///     to_nodes,
    ///     to_octlist.data(), to_octlist.size(),
    ///     dummyOctant<dim>(),
    ///     dummyOctant<dim>());

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

    /// to_loop.finalize(to_dofs);
    to_loop.finalize(ghosted_to_dofs.data());
    std::copy_n(&ghosted_to_dofs[to_da->getLocalNodeBegin() * ndofs], to_nodes_sz * ndofs, to_dofs);
  }

  // ------------------

  // flag_essential_nodes()
  template <unsigned dim>
  void flag_essential_nodes(
      const TreeNode<unsigned, dim> *octants,
      const size_t oct_sz,
      const DA<dim> *da,
      std::vector<char> &node_essential)
  {
    MatvecBaseOut<dim, char, true> loop(
        da->getTotalNodalSz(),
        1,
        da->getElementOrder(),
        false, 0,
        da->getTNCoords(),
        octants,
        oct_sz,
        dummyOctant<dim>(),
        dummyOctant<dim>());

    const int npe = da->getNumNodesPerElement();
    const std::vector<char> leaf(npe, 1);

    while (not loop.isFinished())
    {
      if (loop.isPre() and loop.isLeaf())
      {
        loop.subtreeInfo().overwriteNodeValsOut(leaf.data());
        loop.next();
      }
      else
        loop.step();
    }

    node_essential.clear();
    node_essential.resize(da->getTotalNodalSz(), false);
    loop.finalize(node_essential.data());
    for (char &x : node_essential)
      x = bool(x);
  }



}//namespace ot

#endif//DENDRO_KT_LERP_HPP
