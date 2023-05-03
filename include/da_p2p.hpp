/**
 * @author Masado Ishii
 * @date   2023-04-12
 * @brief  Create a DA using neighbor sets.
 */

#ifndef DENDRO_KT_DA_P2P_HPP
#define DENDRO_KT_DA_P2P_HPP

// Function declaration for linkage purposes.
inline void link_da_p2p_tests() {};

#include "include/treeNode.h"
#include "include/tsort.h"
#include "include/distTree.h"
#include "include/oda.h"
#include "include/parUtils.h"

#include "include/leaf_sets.hpp"
#include "include/neighbors_to_nodes.hpp"
#include "include/partition_border.hpp"
#include "include/contextual_hyperface.hpp"
/// #include "include/ghost_exchange.hpp"

/// #include "include/debug.hpp"

#include <vector>

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  namespace da_p2p
  {
    template <int dim>
    struct PointSet
    {
      int degree = 0;
      std::vector<TreeNode<uint32_t, dim>> points = {};
    };

    template <typename T, int dim>
    struct SplitDimCount
    {
      T *       unsplit()       { return &m_counts[0]; }
      const T * unsplit() const { return &m_counts[0]; }
      T *       split()         { return &m_counts[dim + 1]; }
      const T * split() const   { return &m_counts[dim + 1]; }

      std::array<T, (dim + 1) * 2> m_counts;
    };

    //future: rename as Topology or something, since it's aware of the octree.
    template <int dim>
    class DA
    {
      public:
        DA() = default;
        DA(const DistTree<uint32_t, dim> &dist_tree);
        ~DA();

        MPI_Comm global_comm() const;
        MPI_Comm active_comm() const;

        size_t n_local_nodes(int degree) const;
        DendroLLU n_global_nodes(int degree) const;
        DendroLLU global_node_offset(int degree) const;

        size_t n_total_nodes(int degree) const;
        size_t local_nodes_begin(int degree) const;
        size_t local_nodes_end(int degree) const;

        PointSet<dim> point_set(int degree) const;

        // The DA should not have changing state, except when the topology changes.
        // Asynchronous communication state and vector data should both be owned
        // by the caller, outside the DA.
        /// RemoteMap remote_map(int degree) const;

      private:
        size_t n_pre_ghost_nodes(int degree) const;
        size_t n_post_ghost_nodes(int degree) const;

        template <typename T>
        static T dimension_sum(
            const T *hyperfaces, int degree);

      private:
        MPI_Comm m_global_comm;
        MPI_Comm m_active_comm;
        bool m_active_comm_owned = false;

        // future: decompose this large struct into logical units

        // ---------------------------------------------------------------------
        // Details to produce and traverse point sets.
        // ---------------------------------------------------------------------
        size_t m_pre_ghost_octants = {};
        size_t m_post_ghost_octants = {};

        // Example
        //     ghosted octants     oct,       oct,    oct,           oct
        //     hyperface ranges    0,         2,      3,             6,      7
        //     hyperfaces          hf, hf,    hf,     hf, hf, hf,    hf
        std::vector<uint8_t> m_ghosted_hyperfaces;
        std::vector<size_t> m_ghosted_hyperface_ranges;

        std::vector<TreeNode<uint32_t, dim>> m_ghosted_octants;
        std::vector<TreeNode<uint32_t, dim>> m_scatter_octants;
        std::vector<size_t> m_scatter_octant_ids;//referring to local ids
        // ---------------------------------------------------------------------

        // ---------------------------------------------------------------------
        // Count the number of full unique hyperfaces of each dimensionality.
        // This makes it easy to return the ghost offsets, without a RemoteMap,
        // even when the vector degree is not known ahead of time.
        // ---------------------------------------------------------------------
        SplitDimCount<size_t, dim> m_local = {};
        SplitDimCount<size_t, dim> m_pre_ghost = {};
        SplitDimCount<size_t, dim> m_post_ghost = {};
        SplitDimCount<DendroLLU, dim> m_prefix_sum = {};
        SplitDimCount<DendroLLU, dim> m_reduction = {};
        // ---------------------------------------------------------------------
    };


    template <int dim>
    constexpr int compute_npe(int degree) { return intPow(dim, degree); }

    template <int dim>
    const TreeNode<uint32_t, dim> & dummy_octant()
    {
      static TreeNode<uint32_t, dim> dummy;
      return dummy;
    }

    template <int dim>
    struct WrapperData
    {
      PointSet<dim> m_points;
      ScatterMap m_sm;
      GatherMap m_gm;

        /// size_t n_local_elements();
        /// DendroLLU n_global_elements();
        /// DendroLLU global_element_offset();
    };

    template <int dim>
    class DA_Wrapper
    {
      public:
        //
        // Old interfaces
        //

        DA_Wrapper() = default;
        DA_Wrapper(DA_Wrapper &&moved_da) = default;
        DA_Wrapper & operator=(DA_Wrapper &&moved_da) = default;

        DA_Wrapper(
          const DistTree<uint32_t, dim> &inDistTree,
          int,  //ignored
          MPI_Comm comm,
          unsigned int order,
          size_t, //ignored
          double sfc_tol);

        DA_Wrapper(
          const DistTree<uint32_t, dim> &inDistTree,
          MPI_Comm comm,
          unsigned int order,
          size_t = {}, //ignored
          double sfc_tol = 0.3);

        inline size_t getLocalElementSz() const { return data()->n_local_elements(); }

        inline size_t getLocalNodalSz() const { return m_da.n_local_nodes(degree()); }

        inline size_t getLocalNodeBegin() const { return m_da.local_nodes_begin(degree()); }

        inline size_t getLocalNodeEnd() const { return m_da.local_nodes_end(degree()); }

        inline size_t getPreNodalSz() const { return getLocalNodeBegin(); }

        inline size_t getPostNodalSz() const { return getTotalNodalSz() - getLocalNodalSz() - getPreNodalSz(); }

        inline size_t getTotalNodalSz() const { return m_da.n_total_nodes(degree()); }

        inline RankI getGlobalNodeSz() const { return m_da.n_global_nodes(degree()); }

        inline RankI getGlobalRankBegin() const { return m_da.global_node_offset(degree()); }

        inline DendroIntL getGlobalElementSz() const { return data()->n_global_elements(); }

        inline DendroIntL getGlobalElementBegin() const { return m_da.global_element_offset(); }

        inline const std::vector<RankI> & getNodeLocalToGlobalMap();//ghosted //TODO

        inline bool isActive() const { return data()->n_local_elements() > 0; }

        size_t getTotalSendSz() const { return data()->m_sm.m_map.size(); }
        size_t getTotalRecvSz() const { return data()->m_gm.m_totalCount - data()->m_gm.m_locCount; }
        int getNumDestNeighbors() const { return data()->m_sm.m_sendProc.size(); }
        int getNumSrcNeighbors()  const { return data()->m_gm.m_recvProc.size(); }
        int getNumOutboundRanks() const { return data()->m_sm.m_sendProc.size(); }
        int getNumInboundRanks()  const { return data()->m_gm.m_recvProc.size(); }

        inline unsigned int getNumNodesPerElement() const { return compute_npe<dim>(degree()); }

        inline unsigned int getElementOrder() const { return degree(); }

        inline MPI_Comm getGlobalComm() const { return m_da.global_comm(); }

        inline MPI_Comm getCommActive() const { return m_da.active_comm(); }

        inline unsigned int getNpesAll() const {
          return par::mpi_comm_size(m_da.global_comm());
        };
        inline unsigned int getNpesActive() const {
          return isActive() ? par::mpi_comm_size(m_da.active_comm()) : 0;
        }

        inline unsigned int getRankAll() const {
          return par::mpi_comm_rank(m_da.global_comm());
        };

        inline unsigned int getRankActive() const {
          return isActive() ?
            par::mpi_comm_rank(m_da.active_comm()) : getRankAll();
        }

        inline const TreeNode<uint32_t, dim> * getTNCoords() const;//TODO

        inline const DendroIntL * getNodeOwnerElements() const;//TODO

        inline const TreeNode<uint32_t, dim> * getTreePartFront() const { return &dummy_octant<dim>(); }

        inline const TreeNode<uint32_t, dim> * getTreePartBack() const { return &dummy_octant<dim>(); }

        inline const RefElement * getReferenceElement() const;//TODO

        inline void getBoundaryNodeIndices(std::vector<size_t> &bdyIndex) const;//TODO

        inline const std::vector<size_t> & getBoundaryNodeIndices() const;//TODO

        /// inline const std::vector<int> & elements_per_node() const;//ghosted

        template <typename T>
        int createVector(
            T *&local,
            bool isElemental = false,
            bool isGhosted = false,
            unsigned int dof = 1) const;

        template <typename T>
        int createVector(
            std::vector<T> &local,
            bool isElemental = false,
            bool isGhosted = false,
            unsigned int dof = 1) const;

        template <typename T>
        void destroyVector(T *&local) const;

        template <typename T>
        void destroyVector(std::vector<T> &local) const;

        template <typename T>
        void readFromGhostBegin(
            T *vec,
            unsigned int dof = 1) const;

        template <typename T>
        void readFromGhostEnd(
            T *vec,
            unsigned int dof = 1) const;

        template <typename T>
        void writeToGhostsBegin(
            T *vec,
            unsigned int dof = 1,
            const char * isDirtyOut = nullptr) const;

        template <typename T>
        void writeToGhostsEnd(
            T *vec,
            unsigned int dof = 1,
            bool useAccumulation = true,
            const char * isDirtyOut = nullptr) const;

        template <typename T>
        void nodalVecToGhostedNodal(
            const T *in,
            T *&out,
            bool isAllocated = false,
            unsigned int dof = 1) const;

        template <typename T>
        void ghostedNodalToNodalVec(
            const T *gVec,
            T *&local,
            bool isAllocated = false,
            unsigned int dof = 1) const;


        template<typename T>
        void nodalVecToGhostedNodal(
            const std::vector<T> &in,
            std::vector<T> &out,
            bool isAllocated = false,
            unsigned int dof = 1) const;

        template<typename T>
        void ghostedNodalToNodalVec(
            const std::vector<T> gVec,
            std::vector<T> &local,
            bool isAllocated = false,
            unsigned int dof = 1) const;

        template <typename T>
        void setVectorByFunction(
            T *local,
            std::function<void( const T *, T *)> func,
            bool isElemental = false,
            bool isGhosted = false,
            unsigned int dof = 1) const;

        template <typename T>
        void setVectorByScalar(
            T *local,
            const T *value,
            bool isElemental = false,
            bool isGhosted = false,
            unsigned int dof = 1,
            unsigned int initDof = 1) const;

        template <typename T>
        void vecTopvtu(
            T *local,
            const char *fPrefix,
            char **nodalVarNames = NULL,
            bool isElemental = false,
            bool isGhosted = false,
            unsigned int dof = 1);


        void computeTreeNodeOwnerProc(
            const TreeNode<uint32_t, dim> * pNodes,
            unsigned int n,
            int* ownerRanks) const;


#ifdef BUILD_WITH_PETSC
        /// PetscErrorCode petscCreateVector(Vec &local, bool isElemental, bool isGhosted, unsigned int dof) const;
        /// PetscErrorCode createMatrix(Mat &M, MatType mtype, unsigned int dof = 1) const;

        /// template <typename T>
        /// void petscSetVectorByFunction(Vec &local, std::function<void(const T *, T *)> func, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /// void petscVecTopvtu(const Vec &local, const char *fPrefix, char **nodalVarNames = NULL, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1);

        /// PetscErrorCode petscDestroyVec(Vec & vec) const;
#endif


        // -----------------------------------------------------------


      private:
        WrapperData<dim> *data() { assert(m_data != nullptr); return &(*m_data); }
        const WrapperData<dim> *data() const { assert(m_data != nullptr); return &(*m_data); }

        int degree() const { return data()->m_points.degree; }

      private:
        DA<dim> m_da;
        std::unique_ptr<WrapperData<dim>> m_data = nullptr;
    };



    // future: fast edit for minor refinement or repartitioning.

    // future: separate loops for dependent, independent, and boundary elements.
  }

  template <int dim>
  using DA_P2P = da_p2p::DA_Wrapper<dim>;
}


// =============================================================================
// Tests
// =============================================================================
#ifdef DOCTEST_LIBRARY_INCLUDED
namespace ot
{
  DOCTEST_TEST_SUITE("Compare old and new DA")
  {

    // Helper routines
    template <int dim>
    std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_central(int max_depth);
    template <int dim>
    std::vector<ot::TreeNode<uint32_t, dim>>  grid_pattern_edges(int max_depth);

    auto make_next_comm_size(int max_comm_size) {
      auto &cout = std::cout;
      return [max_comm_size, &cout](int np) {
        if (np < max_comm_size)
        {
          if (((np-1) & np) == 0)                   // np is a power of 2
            np += 1;                                // -> power of 2 plus 1
          else if ((((np + 1)-1) & (np + 1)) == 0)  // np+1 is power of 2
            np += 1;                                // -> power of 2
          else                                      // np-1 is power of 2
            np = 2*(np - 1) - 1;                    // -> next power of 2 minus 1
          if (np > max_comm_size)                   // clamp
            np = max_comm_size;
          return np;
        }
        else
          return np + 1;
      };
    }

    DOCTEST_MPI_TEST_CASE("Number of nodes", 3)
    {
      /// dbg::wait_for_debugger(test_comm);
      constexpr int dim = 2;
      _InitializeHcurve(dim);
      const double sfc_tol = 0.3;

      std::stringstream ss;

      const auto next_comm_size = make_next_comm_size(test_nb_procs);
      for (int np = 1; np <= test_nb_procs; np = next_comm_size(np))
      /// for (int np = 1; np <= 2; np = next_comm_size(np))
      {
        MPI_Comm comm;
        MPI_Comm_split(test_comm, test_rank < np, 0, &comm);
        if (test_rank >= np)
        {
          MPI_Comm_free(&comm);
          continue;
        }

        INFO("mpi_size=", np, "  mpi_rank=", par::mpi_comm_rank(comm));
        const bool is_root = par::mpi_comm_rank(comm) == 0;

        enum Pattern { central, edges };
        for (Pattern grid_pattern : {central, edges})
        {
          INFO("grid_pattern=", std::string(grid_pattern == central? "central" : "edges"));
          for (int max_depth = 2; max_depth <= 5; ++max_depth)
          /// for (int max_depth = 2; max_depth <= 2; ++max_depth)
          {
            // Grid.
            std::vector<TreeNode<uint32_t, dim>> grid;
            if (is_root and grid_pattern == central)
              grid = grid_pattern_central<dim>(max_depth);
            else if (is_root and grid_pattern == edges)
              grid = grid_pattern_edges<dim>(max_depth);
            SFC_Tree<uint32_t, dim>::distTreeSort(grid, sfc_tol, comm);
            DistTree<uint32_t, dim> dtree(grid, comm);

            /// quadTreeToGnuplot(dtree.getTreePartFiltered(), max_depth, "new_da_tree", comm);

            for (int degree: {1, 2, 3})
            /// for (int degree: {1})
            {
              DA<dim> old_da(dtree, comm, degree);
              DA_P2P<dim> new_da(dtree, comm, degree);

              INFO("max_depth=", max_depth, "  degree=", degree);

              CHECK( new_da.getLocalNodalSz() == old_da.getLocalNodalSz() );
              CHECK( (new_da.getTotalNodalSz() - new_da.getLocalNodalSz())
                  == (old_da.getTotalNodalSz() - old_da.getLocalNodalSz()) );
                // ^ May fail for hilbert curve ordering; then, make other test.

              CHECK( new_da.getGlobalNodeSz() == old_da.getGlobalNodeSz() );
            }
          }
        }
        MPI_Comm_free(&comm);
      }

      _DestroyHcurve();
    }


    // =======================================
    //  Case 1 _ _ _ _      Case 2  _ _ _ _
    //        |_|_|_|_|            |+|+|+|+|
    //        |_|+|+|_|            |+|_|_|+|
    //        |_|+|+|_|            |+|_|_|+|
    //        |_|_|_|_|            |+|+|+|+|
    //   "Central"              "Edges"
    //   linear in max_depth    exponential in max_depth
    // =======================================

    //
    // grid_pattern_central()
    //
    template <int dim>
    std::vector<ot::TreeNode<uint32_t, dim>>
      grid_pattern_central(int max_depth)
    {
      using namespace ot;
      std::vector<TreeNode<uint32_t, dim>> grid = { TreeNode<uint32_t, dim>() };
      std::vector<TreeNode<uint32_t, dim>> queue;
      for (int level = 1; level <= max_depth; ++level)
      {
        queue.clear();
        const auto middle = TreeNode<uint32_t, dim>().getChildMorton(0).range().max();
        for (auto oct: grid)
        {
          // Case 1: Refine the center.
          if (oct.range().closedContains(middle))
            for (int child = 0; child < nchild(dim); ++child)
              queue.push_back(oct.getChildMorton(child));
          else
            queue.push_back(oct);
        }
        std::swap(grid, queue);
      }
      return grid;
    }

    //
    // grid_pattern_edges()
    //
    template <int dim>
    std::vector<ot::TreeNode<uint32_t, dim>>
      grid_pattern_edges(int max_depth)
    {
      using namespace ot;
      std::vector<TreeNode<uint32_t, dim>> grid = { TreeNode<uint32_t, dim>() };
      std::vector<TreeNode<uint32_t, dim>> queue;
      for (int level = 1; level <= max_depth; ++level)
      {
        queue.clear();
        const uint32_t maximum = TreeNode<uint32_t, dim>().range().side();
        for (auto oct: grid)
        {
          // Case 2: Refine the cube surface.
          const std::array<uint32_t, dim> min = oct.range().min();
          const std::array<uint32_t, dim> max = oct.range().max();
          if (*(std::min_element(min.begin(), min.end())) == 0 or
              *(std::max_element(max.begin(), max.end())) == maximum)
            for (int child = 0; child < nchild(dim); ++child)
              queue.push_back(oct.getChildMorton(child));
          else
            queue.push_back(oct);
        }
        std::swap(grid, queue);
      }
      return grid;
    }
  }
}
#endif//DOCTEST_LIBRARY_INCLUDED


// =============================================================================
// Implementation
// =============================================================================
namespace ot
{
  namespace da_p2p
  {
    // DA::DA()
    template <int dim>
    DA<dim>::DA(const DistTree<uint32_t, dim> &dist_tree)
    :
      m_global_comm(dist_tree.getComm()),
      m_active_comm(dist_tree.getComm())
    {
      using Octant = TreeNode<uint32_t, dim>;

      const std::vector<Octant> &local_octants =
          dist_tree.getTreePartFiltered();

      bool active = local_octants.size() > 0;

      MPI_Comm active_comm = dist_tree.getComm();
      if (not par::mpi_and(active, m_global_comm))
      {
        MPI_Comm_split(active_comm, active, 0, &active_comm);
        m_active_comm_owned = true;
        m_active_comm = active_comm;
      }
      assert(active_comm == m_active_comm);

      // TODO check whether all defaults are good from here.
      if (not active)
        return;

      // Rest of algorithm discovers and assigns ownership of octant hyperfaces.

      // Find all candidate neighbor partitions based on SFC ranges.  (any-any)
      LocalAdjacencyList adjacency_list;
      std::vector<LeafRange<dim>> adjacent_ranges;
      {
        // future: Have pre-gathered the partition splitters
        typename SFC_Tree<uint32_t, dim>::PartitionFrontBackRequest
            partition_request(
                active, local_octants.front(), local_octants.back(), active_comm);

        PartitionFrontBack<uint32_t, dim> partition;
        std::vector<int> active_list;  // not used outside this block scope.
        partition = std::move(partition_request).complete(&active_list);

        // Consider shared corner.
        //   owner (partition A) -> |___!___
        //                             _¡   |
        // dependent (partition C) -> |_|___| <- neighbor (partition B)
        //
        // It would not be correct to simply coarsen the partition ranges.
        // That would violate the leaf set assumption, and besides,
        // too many partitions can be overlapped (unscalable).
        //
        // Check daily note from 2023-04-24 for the solution.

        adjacency_list = sfc_partition<dim>(
            par::mpi_comm_rank(active_comm),
            active,
            active_list,
            partition);

        // Extract true octant ranges of partitions deemed as neighbors.
        for (int rank: adjacency_list.neighbor_ranks)
          adjacent_ranges.push_back( LeafRange<dim>::make(
              partition.m_fronts[rank], partition.m_backs[rank]) );
      }

      const int n_remote_parts = adjacency_list.neighbor_ranks.size();

      using VecReq = std::vector<MPI_Request>;
      const auto make_vec_req = [](size_t size) {
        return std::make_unique<VecReq>(size, MPI_REQUEST_NULL);
      };
      std::unique_ptr<VecReq> send_size_requests = make_vec_req(n_remote_parts);
      std::unique_ptr<VecReq> send_requests      = make_vec_req(n_remote_parts);
      std::unique_ptr<VecReq> recv_requests      = make_vec_req(2 * n_remote_parts);

      // Send
      //      local octants adjacent to remote partition ranges.   (all-any)
      // (Overlap communication with local computation).
      std::vector<int> send_sizes(n_remote_parts, -1);
      std::vector<int> recv_sizes(n_remote_parts, -1);
      std::vector<std::vector<size_t>> send_octant_ids(n_remote_parts);
      std::vector<std::vector<Octant>> send_octants(n_remote_parts);
      std::vector<std::vector<Octant>> recv_octants(n_remote_parts);
      for (size_t i = 0; i < n_remote_parts; ++i)
      {
        const int remote_rank = adjacency_list.neighbor_ranks[i];
        const LeafRange<dim> remote_range = adjacent_ranges[i];
        send_octants[i].reserve(100);

        // future: Search in pre-selected border octants, instead of whole list.
        std::vector<size_t> &send_octant_ids_i = send_octant_ids[i];
        std::vector<Octant> &send_octants_i = send_octants[i];
        const Octant *octants_ptr = local_octants.data();
        where_border(vec_leaf_list_view<dim>(local_octants), remote_range,
            [&](const Octant *owned_oct) {
                send_octants_i.push_back(*owned_oct);
                send_octant_ids_i.push_back(owned_oct - octants_ptr);
            });

        send_sizes[i] = send_octants[i].size();  // pointer stability

        par::Mpi_Isend<int>(&send_sizes[i], 1,
            remote_rank, {}, active_comm, &((*send_size_requests)[i]));

        par::Mpi_Isend(send_octants[i].data(), send_octants[i].size(),
            remote_rank, {}, active_comm, &((*send_requests)[i]));
      }

      //
      // Local neighbor sets
      //
      NeighborSetDict<dim> total_dict = neighbor_sets(local_octants);
      // future: Emit hyperfaces of independent elements before MPI_Wait

      NeighborSetDict<dim> border_dict;
      std::vector<NeighborSetDict<dim>> send_neighbors(n_remote_parts);
      std::vector<NeighborSetDict<dim>> recv_neighbors(n_remote_parts);

      // send_neighbors
      for (int i = 0; i < n_remote_parts; ++i)
      {
        NeighborSetDict<dim> neighbors = neighbor_sets(send_octants[i]);
        border_dict.concat(neighbors);
        send_neighbors[i] = std::move(neighbors);
      }

      // Receive
      //         partition-boundary octants.
      {
        MPI_Request *all_recv_requests = (*recv_requests).data();
        MPI_Request *size_requests = &((*recv_requests)[0]);
        MPI_Request *payload_requests = &((*recv_requests)[n_remote_parts]);

        // Initiate receiving of any payload size.
        for (int i = 0; i < n_remote_parts; ++i)
        {
          const int remote_rank = adjacency_list.neighbor_ranks[i];
          par::Mpi_Irecv<int>(&recv_sizes[i], 1,
              remote_rank, {}, active_comm, &size_requests[i]);
        }

        // As soon as a size is received, initiate receiving of payload itself.
        // As soon as a payload is received, process it (neighbor sets).

        for (int filled = 0; filled < 2 * n_remote_parts; ++filled)
        {
          const int index = par::Mpi_Waitany(2 * n_remote_parts, all_recv_requests);
          if (index < n_remote_parts)  // size
          {
            const int i = index;
            const int remote_rank = adjacency_list.neighbor_ranks[i];
            const size_t size = recv_sizes[i];
            recv_octants[i].resize(size);
            par::Mpi_Irecv(recv_octants[i].data(), size,
                remote_rank, {}, active_comm, &payload_requests[i]);
          }
          else // payload
          {
            const int i = index - n_remote_parts;
            // future: filter recv_octants for those adjacent to any send_octants[i];
            NeighborSetDict<dim> neighbors = neighbor_sets(recv_octants[i]);
            border_dict.concat(neighbors);
            recv_neighbors[i] = std::move(neighbors);
          }
        }
      }

      // Updated neighborhoods of all border octants, to emit scatter/gather.
      border_dict.reduce();

      //
      // let
      //   t: octant;
      //   unowned[t], split[t], shared[t]: set<hyperface>
      //   owned[t] = ~unowned[t]
      //
      // let (x, y, z) be neighbors of (t, the parent of t, a child of t):
      //   unowned[t] ∪= (x ∩ t) if x is prioritized over t.
      //   unowned[t] ∪= (y ∩ t)
      //
      //   shared[t] ∪= (x ∩ t)
      //   shared[t] ∪= (y ∩ t)
      //   shared[t] ∪= (parent(z) ∩ t)
      //
      //   split[t] ∪= (parent(z) ∩ t)

      // let
      //   r = this partition (mpi rank).
      //   P = partitions adjacent to r.
      //
      // r.local_octants: list<octant>
      // r.send_octants: list<octant>[|P|]
      // r.recv_octants: list<octant>[|P|]
      // r.send_octants[p] = {oct if adjacent(oct, p) for oct in local_octants}
      // r.recv_octants[p] = p.send_octants[r]
      // r.border_octants = ∪.p (send_octants[p] ∪ recv_octants[p])

      // Scattermap[p]: Emit faces "owned" and "shared" (with "split" status):
      //   neighbor_sets(border_octants)@send_octants[p]   -> "owned", "split"
      //   neighbor_sets(recv_octants[p])@send_octants[p]  -> "shared" to p
      for (int i = 0; i < n_remote_parts; ++i)
      {

        // Need a way to iterate through
        // - octants in send_octants[i]
        // - with greedy and split neighborhoods from border_dict
        // - with observer neighborhoods from remote_border_dicts[i]
      }

      std::vector<uint8_t> pre_ghost_hyperfaces;
      std::vector<uint8_t> post_ghost_hyperfaces;
      std::vector<size_t> pre_ghost_hyperface_offsets;
      std::vector<size_t> post_ghost_hyperface_offsets;
      std::vector<Octant> ghosted_octants;

      /// par::RemoteMapBuilder map_builder(adjacency_list.local_rank);

      // Gathermap[p]: Emit faces "owned" and "shared" (with "split" status):
      //   neighbor_sets(border_octants)@recv_octants[p]   -> "owned", "split"
      //   neighbor_sets(send_octants[p])@recv_octants[p]  -> "shared" to r
      for (int i = 0; i < n_remote_parts; ++i)
      {
        const int remote_rank = adjacency_list.neighbor_ranks[i];
        const bool pre_ghost = remote_rank < adjacency_list.local_rank;
        std::vector<uint8_t> &hyperfaces = (pre_ghost?
            pre_ghost_hyperfaces
            : post_ghost_hyperfaces);
        std::vector<size_t> &hyperface_offsets = (pre_ghost?
            pre_ghost_hyperface_offsets
            : post_ghost_hyperface_offsets);

        size_t count_hyperfaces = 0;

        ForLeafNeighborhoods<dim> border(border_dict, recv_octants[i]);
        ForLeafNeighborhoods<dim> send(send_neighbors[i], recv_octants[i]);
        for (auto border_it =  border.begin(), send_it  = send.begin(),
                  border_end = border.end(),   send_end = send.end();
                  border_it != border_end and  send_it != send_end;
                  ++border_it, ++send_it)
        {
          // Expect to loop over every leaf in the query set (recv_octants).
          assert(border_it.query_idx() == send_it.query_idx());

          const auto border_scope = *border_it;
          const auto send_scope = *send_it;
          const Octant key = border_scope.query_key;
          assert(key == send_scope.query_key);

          assert(border_scope.self_neighborhood.center_occupied());
          const auto priority = priority_neighbors<dim>();
          const auto owned = ~((border_scope.self_neighborhood & priority)
                               | border_scope.parent_neighborhood).spread_out();
          const auto split = border_scope.children_neighborhood.spread_out();
          const auto shared = (send_scope.self_neighborhood
                               | send_scope.children_neighborhood).spread_out();
          const auto emit = owned & shared;

          // future: Only list ghosted octants where there are ghosted faces.
          hyperface_offsets.push_back(hyperfaces.size());
          ghosted_octants.push_back(key);

          if (emit.none())
            continue;

          // Select hyperfaces to emit.
          tmp::nested_for<dim>(0, 3, [&](auto...idx_pack)
          {
            std::array<int, dim> idxs = {idx_pack...};  // 0..2 per axis.

            // Compute index to lookup bits.
            int hyperface_idx = 0;
            for (int d = 0, stride = 1; d < dim; ++d, stride *= 3)
              hyperface_idx += idxs[d] * stride;
            //future: ^ Fold expression (c++17)

            // Test.
            if(emit.test_flat(hyperface_idx))
            {
              int hyperface_coord = 0;
              for (int d = 0, stride = 1; d < dim; ++d, stride *= 4)  // 2b/face
                hyperface_coord += idxs[d] * stride;
              SplitHyperface4D hyperface = Hyperface4D(hyperface_coord)
                  .mirrored(key.getMortonIndex())
                  .encode_split(split.test_flat(hyperface_idx));
              hyperfaces.push_back(hyperface.encoding());
              ++count_hyperfaces;
            }
          });

          /// map_builder.increase_ghost_count(remote_rank, count_ghost);
        }
      }

      const size_t pre_ghost_octants = pre_ghost_hyperface_offsets.size();
      const size_t post_ghost_octants = post_ghost_hyperface_offsets.size();
      m_pre_ghost_octants = pre_ghost_octants;
      m_post_ghost_octants = post_ghost_octants;

      ghosted_octants.insert(ghosted_octants.begin() + pre_ghost_octants,
          local_octants.cbegin(), local_octants.cend());
      m_ghosted_octants = std::move(ghosted_octants);

      // Updated neighborhoods of all local octants, to emit local hyperfaces.
      //   future: Only update from recv_neighbors
      total_dict.concat(border_dict);
      total_dict.reduce();

      std::vector<uint8_t> local_hyperfaces;
      std::vector<size_t> local_hyperface_offsets;

      // Local hyperfaces: Emit faces "owned" (with "split" status):
      //   neighbor_sets(local_octants ∪ (∪.p recv_octants[p]))@local_octants
      size_t dbg_count = 0;
      size_t count_hyperfaces = pre_ghost_hyperfaces.size();

      // Another way to write the following, if it's a single loop.
      //   for (const auto &scope: ForLeafNeighborhoods<dim>(total_dict, local_octants))

      ForLeafNeighborhoods<dim> loop(total_dict, local_octants);
      for (auto it = loop.begin(), end = loop.end(); it != end; ++it)
      {
        const auto &scope = *it;
        const size_t leaf_index = scope.query_idx;
        const Octant key = scope.query_key;
        Neighborhood<dim> self_nbh = scope.self_neighborhood;
        Neighborhood<dim> parent_nbh = scope.parent_neighborhood;
        Neighborhood<dim> children_nbh = scope.children_neighborhood;

          // local_hyperface_offsets is indexed adjacent to local_octants
        assert(leaf_index == dbg_count);
        ++dbg_count;

        /// self_nbh |= Neighborhood<dim>::solitary();  // it is from local_octants
        assert(self_nbh.center_occupied());

        const auto priority = priority_neighbors<dim>();
        const auto owned = ~((self_nbh & priority) | parent_nbh).spread_out();
        const auto split = children_nbh.spread_out();

        local_hyperface_offsets.push_back(count_hyperfaces);

        // Select hyperfaces to emit.
        tmp::nested_for<dim>(0, 3, [&](auto...idx_pack)
        {
          std::array<int, dim> idxs = {idx_pack...};  // 0..2 per axis.

          // Compute index to lookup bits.
          int hyperface_idx = 0;
          for (int d = 0, stride = 1; d < dim; ++d, stride *= 3)
            hyperface_idx += idxs[d] * stride;
          //future: ^ Fold expression (c++17)

          // Test.
          if(owned.test_flat(hyperface_idx))
          {
            int hyperface_coord = 0;
            for (int d = 0, stride = 1; d < dim; ++d, stride *= 4)
              hyperface_coord += idxs[d] * stride;
            SplitHyperface4D hyperface = Hyperface4D(hyperface_coord)
                .mirrored(key.getMortonIndex())
                .encode_split(split.test_flat(hyperface_idx));
            local_hyperfaces.push_back(hyperface.encoding());
            ++count_hyperfaces;
          }
        });
      }

      for (size_t &i: post_ghost_hyperface_offsets)
        i += count_hyperfaces;

      // ghosted_hyperfaces
      std::vector<uint8_t> ghosted_hyperfaces;
      ghosted_hyperfaces.insert(ghosted_hyperfaces.end(),
          pre_ghost_hyperfaces.cbegin(), pre_ghost_hyperfaces.cend());
      ghosted_hyperfaces.insert(ghosted_hyperfaces.end(),
          local_hyperfaces.cbegin(), local_hyperfaces.cend());
      ghosted_hyperfaces.insert(ghosted_hyperfaces.end(),
          post_ghost_hyperfaces.cbegin(), post_ghost_hyperfaces.cend());

      // ghosted_hyperface_ranges
      std::vector<size_t> ghosted_hyperface_ranges;
      ghosted_hyperface_ranges.insert(ghosted_hyperface_ranges.end(),
          pre_ghost_hyperface_offsets.cbegin(), pre_ghost_hyperface_offsets.cend());
      ghosted_hyperface_ranges.insert(ghosted_hyperface_ranges.end(),
          local_hyperface_offsets.cbegin(), local_hyperface_offsets.cend());
      ghosted_hyperface_ranges.insert(ghosted_hyperface_ranges.end(),
          post_ghost_hyperface_offsets.cbegin(), post_ghost_hyperface_offsets.cend());
      ghosted_hyperface_ranges.push_back(count_hyperfaces);


      SplitDimCount<size_t, dim> hf_count_local = {};
      SplitDimCount<size_t, dim> hf_count_pre_ghost = {};
      SplitDimCount<size_t, dim> hf_count_post_ghost = {};

      const size_t ghost_begin = 0;
      const size_t local_begin = ghost_begin + pre_ghost_octants;
      const size_t local_end = local_begin + local_octants.size();
      const size_t ghost_end = local_end + post_ghost_octants;

      // pre ghost
      for (size_t ghost_idx = ghost_begin; ghost_idx < local_begin; ++ghost_idx)
      {
        for (size_t i = ghosted_hyperface_ranges[ghost_idx],
                  end = ghosted_hyperface_ranges[ghost_idx + 1]; i < end; ++i)
        {
          const SplitHyperface4D hf = {ghosted_hyperfaces[i]};
          size_t *count = (hf.is_split()?
              hf_count_pre_ghost.split() : hf_count_pre_ghost.unsplit());
          ++count[hf.decode().dimension()];
        }
      }

      // local
      for (size_t ghost_idx = local_begin; ghost_idx < local_end; ++ghost_idx)
      {
        for (size_t i = ghosted_hyperface_ranges[ghost_idx],
                  end = ghosted_hyperface_ranges[ghost_idx + 1]; i < end; ++i)
        {
          const SplitHyperface4D hf = {ghosted_hyperfaces[i]};
          size_t *count = (hf.is_split()?
              hf_count_local.split() : hf_count_local.unsplit());
          ++count[hf.decode().dimension()];
        }
      }

      // post ghost
      for (size_t ghost_idx = local_end; ghost_idx < ghost_end; ++ghost_idx)
      {
        for (size_t i = ghosted_hyperface_ranges[ghost_idx],
                  end = ghosted_hyperface_ranges[ghost_idx + 1]; i < end; ++i)
        {
          const SplitHyperface4D hf = {ghosted_hyperfaces[i]};
          size_t *count = (hf.is_split()?
              hf_count_post_ghost.split() : hf_count_post_ghost.unsplit());
          ++count[hf.decode().dimension()];
        }
      }

      m_ghosted_hyperfaces = std::move(ghosted_hyperfaces);
      m_ghosted_hyperface_ranges = std::move(ghosted_hyperface_ranges);

      m_local = hf_count_local;
      m_pre_ghost = hf_count_pre_ghost;
      m_post_ghost = hf_count_post_ghost;

      // Mpi scan and reduce local hyperface counts.
      SplitDimCount<DendroLLU, dim> combined = {};
      std::copy(
          hf_count_local.m_counts.cbegin(),
          hf_count_local.m_counts.cend(),
          combined.m_counts.begin());
      par::Mpi_Scan(
          combined.m_counts.data(), m_prefix_sum.m_counts.data(),
          combined.m_counts.size(), MPI_SUM, active_comm);
      par::Mpi_Allreduce(
          combined.m_counts.data(), m_reduction.m_counts.data(),
          combined.m_counts.size(), MPI_SUM, active_comm);

      // Wait for all outstanding requests.
      par::Mpi_Waitall(n_remote_parts, (*send_size_requests).data());
      par::Mpi_Waitall(n_remote_parts, (*send_requests).data());
    }

    // DA::~DA()
    template <int dim>
    DA<dim>::~DA()
    {
      if (m_active_comm_owned)
        MPI_Comm_free(&m_active_comm);
      //TODO also free on copy/move assignment. Best to wrap MPI_Comm.
      // future: should not split communicator on every DA.
    }

    // DA::global_comm()
    template <int dim>
    MPI_Comm DA<dim>::global_comm() const
    {
      return m_global_comm;
    }

    // DA::active_comm()
    template <int dim>
    MPI_Comm DA<dim>::active_comm() const
    {
      return m_active_comm;
    }

    // DA::n_local_nodes()
    template <int dim>
    size_t DA<dim>::n_local_nodes(int degree) const
    {
      // future: option for extra nodes on split faces.
      return dimension_sum(m_local.unsplit(), degree)
            + dimension_sum(m_local.split(), degree);
    }

    // DA::n_global_nodes()
    template <int dim>
    DendroLLU DA<dim>::n_global_nodes(int degree) const
    {
      // future: option for extra nodes on split faces.
      return dimension_sum(m_reduction.unsplit(), degree)
            + dimension_sum(m_reduction.split(), degree);
    }

    // DA::global_node_offset()
    template <int dim>
    DendroLLU DA<dim>::global_node_offset(int degree) const
    {
      // future: option for extra nodes on split faces.
      return dimension_sum(m_prefix_sum.unsplit(), degree)
            + dimension_sum(m_prefix_sum.split(), degree);
    }

    // DA::n_total_nodes()
    template <int dim>
    size_t DA<dim>::n_total_nodes(int degree) const
    {
      return n_pre_ghost_nodes(degree) +
             n_local_nodes(degree) +
             n_post_ghost_nodes(degree);
    }

    // DA::local_nodes_begin();
    template <int dim>
    size_t DA<dim>::local_nodes_begin(int degree) const
    {
      return n_pre_ghost_nodes(degree);
    }

    // DA::local_nodes_end()
    template <int dim>
    size_t DA<dim>::local_nodes_end(int degree) const
    {
      return n_pre_ghost_nodes(degree) + n_local_nodes(degree);
    }

    // DA::point_set()
    template <int dim>
    PointSet<dim> DA<dim>::point_set(int degree) const
    {
      std::vector<TreeNode<uint32_t, dim>> points;
#warning "Not implemented: DA::point_set()"
      //TODO

      return { degree, std::move(points) };
    }


    // Private

    // DA::n_pre_ghost_nodes()
    template <int dim>
    size_t DA<dim>::n_pre_ghost_nodes(int degree) const
    {
      // future: option for extra nodes on split faces.
      return dimension_sum(m_pre_ghost.unsplit(), degree)
            + dimension_sum(m_pre_ghost.split(), degree);
    }

    // DA::n_post_ghost_nodes()
    template <int dim>
    size_t DA<dim>::n_post_ghost_nodes(int degree) const
    {
      // future: option for extra nodes on split faces.
      return dimension_sum(m_post_ghost.unsplit(), degree)
            + dimension_sum(m_post_ghost.split(), degree);
    }

    // DA::dimension_sum()
    template <int dim>
    template <typename T>
    T DA<dim>::dimension_sum(const T *hyperfaces, int degree)
    {
      T count = 0;
      for (int d = 0; d <= dim; ++d)
        count += hyperfaces[d] * intPow(degree - 1, d);
      return count;
    }



    template <int dim>
    DA_Wrapper<dim>::DA_Wrapper(
        const DistTree<uint32_t, dim> &inDistTree,
        int,  //ignored
        MPI_Comm comm,
        unsigned int order,
        size_t, //ignored
        double sfc_tol)
    :
      m_da(inDistTree)
    {
      const int degree = order;

      //TODO
      ScatterMap scatter_map;
      GatherMap gather_map;

      m_data = std::unique_ptr<WrapperData<dim>>(new WrapperData<dim>({
        m_da.point_set(degree),
        scatter_map,
        gather_map
      }));
    }

    template <int dim>
    DA_Wrapper<dim>::DA_Wrapper(
      const DistTree<uint32_t, dim> &inDistTree,
      MPI_Comm comm,
      unsigned int order,
      size_t, //ignored
      double sfc_tol)
    :
      DA_Wrapper<dim>(inDistTree, 0, comm, order, {}, sfc_tol)  // delegate
    {}

  }//namespace da_p2p

}//namespace ot


#endif//DENDRO_KT_DA_P2P_HPP
