
/**
 * @author Masado Ishii
 * @date 2022-02-17
 */

#ifndef DENDRO_KT_COARSE_TO_FINE_HPP
#define DENDRO_KT_COARSE_TO_FINE_HPP

#include "treeNode.h"
#include "distTree.h"
#include "oda.h"
#include "p2p.h"
#include "sfcTreeLoop_matvec_io.h"

namespace fem
{
  template <unsigned dim>
  void coarse_to_fine(
      const ot::DistTree<unsigned, dim> &from_dtree,
      const ot::DA<dim> *from_da,
      const int node_dofs,  // can be positive or 0
      const int cell_dofs,  // can be positive or 0
      const std::vector<double> &from_node_dofs,
      const std::vector<double> &from_cell_dofs,
      const ot::DistTree<unsigned, dim> &to_dtree,
      const ot::DA<dim> *to_da,
      std::vector<double> &to_node_dofs,
      std::vector<double> &to_cell_dofs);

  template <unsigned dim>
  void local_lerp(
      const std::vector<ot::TreeNode<unsigned, dim>> &from_octlist,
      const ot::TreeNode<unsigned, dim> *from_nodes,
      const size_t from_nodes_sz,
      const double * from_dofs,
      const int ndofs,
      const std::vector<ot::TreeNode<unsigned, dim>> &to_octlist,
      const ot::DA<dim> *to_da,  //quick fix
      const ot::TreeNode<unsigned, dim> *to_nodes,
      const size_t to_nodes_sz,
      double * to_dofs);

  template <unsigned dim>
  void local_inherit(
          const std::vector<ot::TreeNode<unsigned, dim>> &from_octlist,
          const std::vector<double> &from_total_cell_dofs,
          const int ndofs,
          const std::vector<ot::TreeNode<unsigned, dim>> &to_octlist,
          std::vector<double> &to_cell_dofs);

  template <unsigned dim>
  void local_inherit(
          const std::vector<ot::TreeNode<unsigned, dim>> &from_octlist,
          const double *from_total_cell_dofs,
          const int ndofs,
          const std::vector<ot::TreeNode<unsigned, dim>> &to_octlist,
          double *to_cell_dofs);


  template <unsigned dim>
  void flag_essential_nodes(
      const ot::TreeNode<unsigned, dim> *octants,
      const size_t oct_sz,
      const ot::DA<dim> *da,
      std::vector<char> &node_essential);

}


// =======================================================================

namespace fem
{
  // coarse_to_fine()
  template <unsigned dim>
  void coarse_to_fine(
      const ot::DistTree<unsigned, dim> &from_dtree,
      const ot::DA<dim> *from_da,
      const int node_dofs,
      const int cell_dofs,
      const std::vector<double> &from_node_dofs,
      const std::vector<double> &from_cell_dofs,
      const ot::DistTree<unsigned, dim> &to_dtree,
      const ot::DA<dim> *to_da,
      std::vector<double> &to_node_dofs,
      std::vector<double> &to_cell_dofs)
  {
    using Oct = ot::TreeNode<unsigned, dim>;
    using OctList = std::vector<Oct>;

    int comm_size, comm_rank;
    MPI_Comm comm = from_da->getGlobalComm();
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    const OctList & from_octlist = from_dtree.getTreePartFiltered();
    const OctList & to_octlist = to_dtree.getTreePartFiltered();
    const Oct *to_nodes = to_da->getTNCoords() + to_da->getLocalNodeBegin();
    const size_t to_node_sz = to_da->getLocalNodalSz();

    assert(from_node_dofs.size() == node_dofs * from_da->getLocalNodalSz());
    assert(from_cell_dofs.size() == cell_dofs * from_octlist.size());
    assert(to_node_dofs.size() == node_dofs * to_da->getLocalNodalSz());
    assert(to_cell_dofs.size() == cell_dofs * to_octlist.size());
    assert(from_da->getGlobalComm() == to_da->getGlobalComm());

    // Ghost read begin.
    std::vector<double> from_nodes_ghosted;
    from_da->nodalVecToGhostedNodal(from_node_dofs, from_nodes_ghosted, false, node_dofs);
    from_da->readFromGhostBegin(from_nodes_ghosted.data(), node_dofs);

    // Find elements in coarse grid that overlap with fine partitions.

    using SizeRange = ot::IntRange<size_t>;
    using IntRange = ot::IntRange<>;

    // Map coarse cells to fine partitions.
    const bool is_active = (to_octlist.size() > 0);
    std::vector<int> active;
    const ot::PartitionFrontBack<unsigned, dim> partition =
        ot::SFC_Tree<unsigned, dim>::allgatherSplitters(
            is_active, to_octlist.front(), to_octlist.back(), comm, &active);
    std::map<int, int> inv_active;
    for (int ar = 0; ar < active.size(); ++ar)
      inv_active[active[ar]] = ar;
    const int active_rank = (is_active ? inv_active[comm_rank] : -1);
    std::vector<IntRange> to_proc_ranges =
        ot::SFC_Tree<unsigned, dim>::treeNode2PartitionRanks(
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
      std::vector<int> recv_from_active = ot::recvFromActive(active, send_to_active, comm);

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
    std::vector<SizeRange> send_ranges(partners.nDest());
    {
      for (size_t i = 0; i < from_octlist.size(); ++i)
      {
        IntRange to_range = to_proc_ranges[i];

        for (int a = to_range.min; a <= to_range.max; ++a)
          if (a != active_rank)
          {
            assert(active_to_dest_idx[a] >= 0);
            send_ranges[active_to_dest_idx[a]].include(i);
          }
      for (int dst = 0; dst < partners.nDest(); ++dst)
        p2p_cells.schedule_send(dst, send_ranges[dst].length(), send_ranges[dst].min);
      }
    }


    // Ghost read end.
    from_da->readFromGhostEnd(from_nodes_ghosted.data(), node_dofs);

    // Cell sets to node sets
    OctList send_nodes;
    std::vector<double> send_node_dofs;
    std::vector<char> node_essential;
    for (int dst = 0; dst < partners.nDest(); ++dst)
    {
      node_essential.clear();
      flag_essential_nodes(
          &from_octlist[p2p_cells.send_offsets()[dst]],
          p2p_cells.send_sizes()[dst],
          from_da,
          node_essential);

      const size_t n_essential =
          std::count(node_essential.begin(), node_essential.end(), true);

      p2p_nodes.schedule_send(dst, n_essential, send_nodes.size());

      send_nodes.reserve(send_nodes.size() + n_essential);
      send_node_dofs.reserve(send_node_dofs.size() + n_essential * node_dofs);

      for (size_t i = 0; i < from_da->getTotalNodalSz(); ++i)
        if (node_essential[i])
        {
          send_nodes.push_back(from_da->getTNCoords()[i]);
          for (int dof = 0; dof < node_dofs; ++dof)
            send_node_dofs.push_back(from_nodes_ghosted[i * node_dofs + dof]);
        }
    }

    // Send the sizes and payloads.
    for (int dst = 0; dst < partners.nDest(); ++dst)
      p2p_scalar.send(dst,
          p2p_cells.send_sizes()[dst],
          p2p_nodes.send_sizes()[dst]);
    p2p_cells.send(from_octlist.data());
    p2p_nodes.send(send_nodes.data());
    p2p_cells.send_dofs(from_cell_dofs.data(), cell_dofs);
    p2p_nodes.send_dofs(send_node_dofs.data(), node_dofs);

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
    std::vector<double> from_total_cell_dofs((p2p_cells.recv_total() + self_cells) * cell_dofs);
    std::vector<double> from_total_node_dofs((p2p_nodes.recv_total() + self_nodes) * node_dofs);

    // Copy self overlap.
    if (is_active and self_range.nonempty())
    {
      // Cells
      std::copy_n(
          &from_octlist[self_range.min],
          self_cells,
          &from_total_cells[p2p_cells.self_offset()]);
      std::copy_n(
          &from_cell_dofs[self_range.min * cell_dofs],
          self_cells * cell_dofs,
          &from_total_cell_dofs[p2p_cells.self_offset() * cell_dofs]);

      // Nodes
      const size_t node_offset = p2p_nodes.self_offset();
      for (size_t i = 0, j = 0; i < from_da->getTotalNodalSz(); ++i)
        if (node_essential[i])
        {
          from_total_nodes[node_offset + j] = from_da->getTNCoords()[i];
          for (int dof = 0; dof < node_dofs; ++dof)
            from_total_node_dofs[(node_offset + j) * node_dofs + dof] =
                from_nodes_ghosted[i * node_dofs + dof];
          ++j;
        }
    }

    // Receive payloads.
    p2p_cells.recv(from_total_cells.data());
    p2p_nodes.recv(from_total_nodes.data());
    p2p_cells.recv_dofs(from_total_cell_dofs.data(), cell_dofs);
    p2p_nodes.recv_dofs(from_total_node_dofs.data(), node_dofs);

    // Wait on sends.
    p2p_cells.wait_all();
    p2p_nodes.wait_all();

    /// quadTreeToGnuplot(from_total_cells, 8, "total.cells", comm);

    if (cell_dofs > 0)
      local_inherit(
          from_total_cells,
          from_total_cell_dofs,
          cell_dofs,
          to_octlist,
          to_cell_dofs);

    // Fine interpolate from combination of local and remote nodes.
    assert(from_da->getElementOrder() == 1);
    if (node_dofs > 0)
      local_lerp(
          from_total_cells,
          from_total_nodes.data(), from_total_nodes.size(),
          from_total_node_dofs.data(),
          node_dofs,
          to_octlist,
          to_da,
          to_nodes, to_node_sz,
          to_node_dofs.data());

    //future: iterate cell values to fine grid
  }


  //
  // local_lerp()
  //
  template <unsigned dim>
  void local_lerp(
      const std::vector<ot::TreeNode<unsigned, dim>> &from_octlist,
      const ot::TreeNode<unsigned, dim> *from_nodes,
      const size_t from_nodes_sz,
      const double * from_dofs,
      const int ndofs,
      const std::vector<ot::TreeNode<unsigned, dim>> &to_octlist,
      const ot::DA<dim> *to_da,  //quick fix
      const ot::TreeNode<unsigned, dim> *to_nodes,
      const size_t to_nodes_sz,
      double * to_dofs)
  {
    using Oct = ot::TreeNode<unsigned, dim>;
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
    ot::MatvecBaseIn<dim, double> from_loop(
        from_nodes_sz, ndofs, degree,
        false, 0,
        from_nodes, from_dofs,
        from_octlist.data(), from_octlist.size(),
        ot::dummyOctant<dim>(),
        ot::dummyOctant<dim>());

    ot::MatvecBaseOut<dim, double, false> to_loop(
        to_da->getTotalNodalSz(), ndofs, degree,
        false, 0,
        to_da->getTNCoords(),
        to_octlist.data(), to_octlist.size(),
        ot::dummyOctant<dim>(),
        ot::dummyOctant<dim>());

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

            const ot::TreeNode<unsigned int, dim> * fine_nodes = to_loop.subtreeInfo().readNodeCoordsIn();
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

  template <unsigned dim>
  void local_inherit_rec(
      ot::Segment<const ot::TreeNode<unsigned, dim>> &from_oct,
      ot::Segment<const ot::TreeNode<unsigned, dim>> &to_oct,
      const int ndofs,
      const double *from_dofs,
      double *to_dofs,
      ot::TreeNode<unsigned, dim> subtree = ot::TreeNode<unsigned, dim>(),
      ot::SFC_State<int(dim)> sfc = ot::SFC_State<int(dim)>::root())
  {
    using Oct = ot::TreeNode<unsigned, dim>;
    const auto overlaps = [](const Oct &a, const Oct &b) {
      return a.isAncestorInclusive(b) or b.isAncestorInclusive(a);
    };
    const auto transfer = [&]() {
      for (int dof = 0; dof < ndofs; ++dof)
        to_dofs[to_oct.begin * ndofs + dof] = from_dofs[from_oct.begin * ndofs + dof];
    };

    // Find intersection{to_oct, subtree} and transfer if possible.
    if (from_oct.nonempty() and overlaps(subtree, *from_oct))
    {
      while (to_oct.nonempty() and subtree == *to_oct)
      {
        transfer();
        ++to_oct;
      }
      if (to_oct.nonempty() and subtree.isAncestorInclusive(*to_oct))
        for (ot::sfc::SubIndex c(0); c < ot::nchild(dim); ++c)
          local_inherit_rec(
              from_oct, to_oct, ndofs, from_dofs, to_dofs,
              subtree.getChildMorton(sfc.child_num(c)),
              sfc.subcurve(c));
    }
    else
      while (to_oct.nonempty() and subtree.isAncestorInclusive(*to_oct))
        ++to_oct;

    // Move on
    while (from_oct.nonempty() and subtree.isAncestorInclusive(*from_oct))
      ++from_oct;
  }

  template <unsigned dim>
  void local_inherit(
          const std::vector<ot::TreeNode<unsigned, dim>> &from_octlist,
          const std::vector<double> &from_cell_dofs,
          const int ndofs,
          const std::vector<ot::TreeNode<unsigned, dim>> &to_octlist,
          std::vector<double> &to_cell_dofs)
  {
    assert(from_cell_dofs.size() == from_octlist.size() * ndofs);
    assert(to_cell_dofs.size() == to_octlist.size() * ndofs);

    local_inherit(
        from_octlist,
        from_cell_dofs.data(),
        ndofs,
        to_octlist,
        to_cell_dofs.data());
  }

  template <unsigned dim>
  void local_inherit(
          const std::vector<ot::TreeNode<unsigned, dim>> &from_octlist,
          const double *from_cell_dofs,
          const int ndofs,
          const std::vector<ot::TreeNode<unsigned, dim>> &to_octlist,
          double *to_cell_dofs)
  {
    ot::Segment<const ot::TreeNode<unsigned, dim>> from_seg = segment_all(from_octlist);
    ot::Segment<const ot::TreeNode<unsigned, dim>> to_seg = segment_all(to_octlist);

    local_inherit_rec(
        from_seg, to_seg, ndofs, from_cell_dofs, to_cell_dofs);

    assert(from_seg.empty());
    assert(to_seg.empty());
  }


  // ------------------

  // flag_essential_nodes()
  template <unsigned dim>
  void flag_essential_nodes(
      const ot::TreeNode<unsigned, dim> *octants,
      const size_t oct_sz,
      const ot::DA<dim> *da,
      std::vector<char> &node_essential)
  {
    ot::MatvecBaseOut<dim, char, true> loop(
        da->getTotalNodalSz(),
        1,
        da->getElementOrder(),
        false, 0,
        da->getTNCoords(),
        octants,
        oct_sz,
        ot::dummyOctant<dim>(),
        ot::dummyOctant<dim>());

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



}//namespace fem

#endif//DENDRO_KT_COARSE_TO_FINE_HPP
