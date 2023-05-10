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
#include "include/tnUtils.h"

#include "include/leaf_sets.hpp"
#include "include/neighbors_to_nodes.hpp"
#include "include/partition_border.hpp"
#include "include/contextual_hyperface.hpp"
#include "include/ghost_exchange.hpp"

/// #include "include/debug.hpp"

#include <vector>
#include <unordered_map>  // for WrapperData

// =============================================================================
// Interfaces
// =============================================================================
namespace ot
{
  template <typename T>
  struct ConstRange
  {
    const T *begin() const { return m_begin; }
    const T *end()   const { return m_end; }
    size_t   size()  const { return m_end - m_begin; }

    const T *m_begin;
    const T *m_end;
  };

  template <typename T>
  struct Range
  {
    T *begin() const { return m_begin; }
    T *end()   const { return m_end; }
    size_t   size()  const { return m_end - m_begin; }
    operator ConstRange<T>() const { return { m_begin, m_end }; }

    T *m_begin;
    T *m_end;
  };


  namespace da_p2p
  {
    template <int dim>
    struct PointSet
    {
      int degree = 0;
      std::vector<TreeNode<uint32_t, dim>> ghosted_points = {};
      std::vector<size_t> local_boundary_indices = {};
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

        size_t n_local_cells() const;
        DendroLLU n_global_cells() const;
        DendroLLU global_cell_offset() const;
        ConstRange<TreeNode<uint32_t, dim>> local_cell_list() const;
        ConstRange<TreeNode<uint32_t, dim>> ghosted_cell_list() const;

        size_t n_local_nodes(int degree) const;
        DendroLLU n_global_nodes(int degree) const;
        DendroLLU global_node_offset(int degree) const;

        size_t n_total_nodes(int degree) const;
        size_t local_nodes_begin(int degree) const;
        size_t local_nodes_end(int degree) const;

        size_t n_local_cell_owned_nodes(int degree, size_t local_cell_id) const;

        PointSet<dim> point_set(int degree) const;

        const par::RemoteMap & remote_cell_map() const;

        // The DA should not have changing state, except when the topology changes.
        // Asynchronous communication state and vector data should both be owned
        // by the caller, outside the DA.
        par::RemoteMap remote_node_map(int degree) const;

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

          //future: consult m_cell_map for counts of pre and post ghost octants.

        // Example
        //     ghosted octants     oct,       oct,    oct,           oct
        //     hyperface ranges    0,         2,      3,             6,      7
        //     hyperfaces          hf, hf,    hf,     hf, hf, hf,    hf
        std::vector<uint8_t> m_ghosted_hyperfaces;
        std::vector<size_t> m_ghosted_hyperface_ranges;

        std::set<size_t> m_ghosted_hyperface_bdry_ids;

        std::vector<TreeNode<uint32_t, dim>> m_ghosted_octants;
        /// std::vector<TreeNode<uint32_t, dim>> m_scatter_octants;
        /// std::vector<size_t> m_scatter_octant_ids;//referring to local ids
        //future: consult m_cell_map
        // ---------------------------------------------------------------------

        // ---------------------------------------------------------------------
        // Maps needed for ghost exchanges.
        // ---------------------------------------------------------------------
        par::RemoteMap m_face_map = par::RemoteMapBuilder(0).finish();
        par::RemoteMap m_cell_map = par::RemoteMapBuilder(0).finish();
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

        // ---------------------------------------------------------------------
        // Reductions on the number of cells.
        // ---------------------------------------------------------------------
        DendroLLU m_cell_prefix_sum = {};
        DendroLLU m_cell_reduction = {};
        // ---------------------------------------------------------------------
    };


    template <int dim>
    constexpr int compute_npe(int degree) { return intPow(degree + 1, dim); }

    template <int dim>
    const TreeNode<uint32_t, dim> & dummy_octant()
    {
      static TreeNode<uint32_t, dim> dummy;
      return dummy;
    }

    template <int dim>
    struct WrapperData
    {
      PointSet<dim> points;

      std::vector<DendroIntL> ghosted_global_ids;
      std::vector<DendroIntL> ghosted_global_owning_elements;

      par::RemoteMap remote_map;

      // State
      std::unordered_map<const void *, par::GhostPullRequest>
        pull_requests;
      std::unordered_map<const void *, std::unique_ptr<par::GhostPushRequest>>
        push_requests;
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
          double);  // ignored

        DA_Wrapper(
          const DistTree<uint32_t, dim> &inDistTree,
          MPI_Comm comm,
          unsigned int order,
          size_t = {}, //ignored
          double = {});  // ignored

        inline size_t getLocalElementSz() const { return m_da.n_local_cells(); }

        inline size_t getLocalNodalSz() const { return m_da.n_local_nodes(degree()); }

        inline size_t getLocalNodeBegin() const { return m_da.local_nodes_begin(degree()); }

        inline size_t getLocalNodeEnd() const { return m_da.local_nodes_end(degree()); }

        inline size_t getPreNodalSz() const { return getLocalNodeBegin(); }

        inline size_t getPostNodalSz() const { return getTotalNodalSz() - getLocalNodalSz() - getPreNodalSz(); }

        inline size_t getTotalNodalSz() const { return m_da.n_total_nodes(degree()); }

        inline RankI getGlobalNodeSz() const { return m_da.n_global_nodes(degree()); }

        inline RankI getGlobalRankBegin() const { return m_da.global_node_offset(degree()); }

        inline DendroIntL getGlobalElementSz() const { return m_da.n_global_cells(); }

        inline DendroIntL getGlobalElementBegin() const { return m_da.global_cell_offset(); }

        inline const std::vector<RankI> & getNodeLocalToGlobalMap() const;//ghosted

        inline bool isActive() const { return m_da.n_local_cells() > 0; }

        int getNumDestNeighbors() const { return data()->remote_map.n_active_bound_links(); }
        int getNumSrcNeighbors()  const { return data()->remote_map.n_active_ghost_links(); }
        int getNumOutboundRanks() const { return data()->remote_map.n_active_bound_links(); }
        int getNumInboundRanks()  const { return data()->remote_map.n_active_ghost_links(); }
        size_t getTotalSendSz() const { return data()->remote_map.local_binding_total(); }
        size_t getTotalRecvSz() const { return data()->remote_map.total_count()
                                             - data()->remote_map.local_count(); }

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

        inline const TreeNode<uint32_t, dim> * getTNCoords() const;

        inline const DendroIntL * getNodeOwnerElements() const;

        inline const TreeNode<uint32_t, dim> * getTreePartFront() const { return &dummy_octant<dim>(); }

        inline const TreeNode<uint32_t, dim> * getTreePartBack() const { return &dummy_octant<dim>(); }

        inline const RefElement * getReferenceElement() const;

        /// inline void getBoundaryNodeIndices(std::vector<size_t> &bdyIndex) const;

        inline const std::vector<size_t> & getBoundaryNodeIndices() const;

        /// inline const std::vector<int> & elements_per_node() const;//ghosted

        template <typename T>
        void createVector(
             T *&local,
             bool isElemental = false,
             bool isGhosted = false,
             unsigned int dof = 1) const;

        template <typename T>
        void createVector(
             std::vector<T> &local,
             bool isElemental = false,
             bool isGhosted = false,
             unsigned int dof = 1) const;

        template <typename T>
        void destroyVector(T *&local) const;

        template <typename T>
        void destroyVector(std::vector<T> &local) const;

        // -----------------------------------------

        template <typename T>
        void readFromGhostBegin(
            T *vec,
            unsigned int dof = 1) const;

        template <typename T>
        void readFromGhostEnd(
            T *vec,
            unsigned int dof = 1) const;

        // -----------------------------------------

        template <typename T>
        void writeToGhostsBegin(    // useAccumulation=true.
            T *vec,
            unsigned int dof = 1,
            const char * isDirtyOut = nullptr) const;

        template <typename T>
        void writeToGhostsEnd(
            T *vec,
            unsigned int dof,
            bool useAccumulation,   // useAccumulation=false here is deprecated.
            const char * isDirtyOut) const;

        template <typename T>
        void writeToGhostsEnd(      // useAccumulation=true.
            T *vec,
            unsigned int dof = 1,
            const char * isDirtyOut = nullptr) const;

        // -----------------------------------------

        template <typename T>
        void overwriteToGhostsBegin(       // useAccumulation=false.
            T *vec,
            unsigned int dof = 1,
            const char * isDirtyOut = nullptr) const;

        template <typename T>
        void overwriteToGhostsEnd(         // useAccumulation=false.
            T *vec,
            unsigned int dof = 1,
            const char * isDirtyOut = nullptr) const;

        // -----------------------------------------

        template <typename T>
        void nodalVecToGhostedNodal(
            const T *in,
            T *&out,
            bool isAllocated = false,
            unsigned int dof = 1) const;

        template <typename T>
        void nodalVecToGhostedNodal(
            const T *in,
            T * &&out,     //rvalue ref
            bool isAllocated,
            unsigned int dof = 1) const;

        template <typename T>
        void ghostedNodalToNodalVec(
            const T *gVec,
            T *&local,
            bool isAllocated = false,
            unsigned int dof = 1) const;

        template <typename T>
        void ghostedNodalToNodalVec(
                const T *gVec,
                T *&&local,           //rvalue ref
                bool isAllocated,
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


        /// void computeTreeNodeOwnerProc(
        ///     const TreeNode<uint32_t, dim> * pNodes,
        ///     unsigned int n,
        ///     int* ownerRanks) const;

        // -----------------------------------------------------------

      public:
        struct ConstNodeRange;
        ConstNodeRange ghosted_nodes() const;
        ConstNodeRange local_nodes() const;

        ConstRange<TreeNode<uint32_t, dim>> local_cell_list() const;
        ConstRange<TreeNode<uint32_t, dim>> ghosted_cell_list() const;

      private:
        template <typename T, class Operation>
        void writeToGhostsBegin_impl(
            T *vec, unsigned int dof, const char * isDirtyOut, Operation op) const;

        template <typename T>
        void writeToGhostsEnd_impl(
            T *vec, unsigned int dof, const char * isDirtyOut) const;

      private:
        WrapperData<dim> *mutate() const { assert(m_data != nullptr); return &(*m_data); }
        WrapperData<dim> *data() { assert(m_data != nullptr); return &(*m_data); }
        const WrapperData<dim> *data() const { assert(m_data != nullptr); return &(*m_data); }

        int degree() const { return data()->points.degree; }

      private:
        DA<dim> m_da;
        std::unique_ptr<WrapperData<dim>> m_data = nullptr;
    };



    // future: fast edit for minor refinement or repartitioning.

    // future: separate loops for dependent, independent, and boundary elements.
  }


  template <typename DA_Type, typename T>
  inline ConstRange<T> ghost_range(const DA_Type &da, int ndofs, const T *a);

  template <typename DA_Type, typename T>
  inline ConstRange<T> local_range(const DA_Type &da, int ndofs, const T *a);

  template <typename DA_Type, typename T>
  inline Range<T> ghost_range(const DA_Type &da, int ndofs, T *a);

  template <typename DA_Type, typename T>
  inline Range<T> local_range(const DA_Type &da, int ndofs, T *a);


  template <typename DA_Type, unsigned dim>
  inline std::vector<DA_Type> multiLevelDA(
      const DistTree<uint32_t, dim> &dtree,
      MPI_Comm comm,
      unsigned order,
      size_t grain = 100,
      double sfc_tol = 0.3);


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
      constexpr int dim = 2;
      _InitializeHcurve(dim);
      const double sfc_tol = 0.3;

      const auto next_comm_size = make_next_comm_size(test_nb_procs);
      for (int np = 1; np <= test_nb_procs; np = next_comm_size(np))
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
          {
            // Grid.
            std::vector<TreeNode<uint32_t, dim>> grid;
            if (is_root and grid_pattern == central)
              grid = grid_pattern_central<dim>(max_depth);
            else if (is_root and grid_pattern == edges)
              grid = grid_pattern_edges<dim>(max_depth);
            SFC_Tree<uint32_t, dim>::distTreeSort(grid, sfc_tol, comm);
            DistTree<uint32_t, dim> dtree(grid, comm);

            for (int degree: {1, 2, 3})
            {
              DA<dim> old_da(dtree, comm, degree);
              DA_P2P<dim> new_da(dtree, comm, degree);

              INFO("max_depth=", max_depth, "  degree=", degree);

              CHECK( new_da.getReferenceElement()->getOrder()
                  == old_da.getReferenceElement()->getOrder() );
              CHECK( new_da.getReferenceElement()->getDim()
                  == old_da.getReferenceElement()->getDim() );
              CHECK( new_da.getReferenceElement()->get1DNumInterpolationPoints()
                  == old_da.getReferenceElement()->get1DNumInterpolationPoints() );

              CHECK( new_da.getNumNodesPerElement() == old_da.getNumNodesPerElement() );
              CHECK( new_da.getElementOrder() == old_da.getElementOrder() );

              CHECK( new_da.getLocalElementSz() == old_da.getLocalElementSz() );
              CHECK( new_da.getGlobalElementSz() == old_da.getGlobalElementSz() );
              CHECK( new_da.getGlobalElementBegin() == old_da.getGlobalElementBegin() );

              CHECK( new_da.getLocalNodalSz() == old_da.getLocalNodalSz() );
              CHECK( new_da.getTotalNodalSz() == old_da.getTotalNodalSz() );

              CHECK( (new_da.getTotalNodalSz() - new_da.getLocalNodalSz())
                  == (old_da.getTotalNodalSz() - old_da.getLocalNodalSz()) );
                // ^ May fail for hilbert curve ordering; then, make other test.

              CHECK( new_da.getGlobalNodeSz() == old_da.getGlobalNodeSz() );
              CHECK( new_da.getGlobalRankBegin() == old_da.getGlobalRankBegin() );

              // future: compare some kind of matvec
            }
          }
        }
        MPI_Comm_free(&comm);
      }

      _DestroyHcurve();
    }

    DOCTEST_MPI_TEST_CASE("Self consistency on small adaptive grids", 3)
    {
      dbg::wait_for_debugger(test_comm);
      constexpr int dim = 2;
      _InitializeHcurve(dim);
      const double sfc_tol = 0.3;
      MPI_Comm comm = test_comm;
      const bool is_root = par::mpi_comm_rank(comm) == 0;

      enum Pattern { central, edges };
      for (Pattern grid_pattern : {central, edges})
      {
        INFO("grid_pattern=", std::string(grid_pattern == central? "central" : "edges"));
        std::vector<TreeNode<uint32_t, dim>> grid;
        const int max_depth = 3;
        if (is_root and grid_pattern == central)
          grid = grid_pattern_central<dim>(max_depth);
        else if (is_root and grid_pattern == edges)
          grid = grid_pattern_edges<dim>(max_depth);
        SFC_Tree<uint32_t, dim>::distTreeSort(grid, sfc_tol, comm);
        DistTree<uint32_t, dim> dtree(grid, comm);

        for (int degree: {1, 2, 3})
        {
          INFO("max_depth=", max_depth, "  degree=", degree);
          DA_P2P<dim> new_da(dtree, comm, degree);

          // getNodeLocalToGlobalMap()
          {
            const std::vector<DendroIntL> &ghosted_to_global =
                new_da.getNodeLocalToGlobalMap();
            CHECK( ghosted_to_global.size() == new_da.getTotalNodalSz() );
            CHECK( std::is_sorted(ghosted_to_global.begin(),
                                  ghosted_to_global.end()) );
          }

          // ghost exchange properties
          {
            const size_t total_send_sz = new_da.getTotalSendSz();
            const size_t total_recv_sz = new_da.getTotalRecvSz();
            const int n_dest_neighbors = new_da.getNumDestNeighbors();
            const int n_src_neighbors  = new_da.getNumSrcNeighbors();
            const int n_outbound_ranks = new_da.getNumOutboundRanks();
            const int n_inbound_ranks  = new_da.getNumInboundRanks();

            CHECK( n_dest_neighbors == n_outbound_ranks );
            CHECK( n_src_neighbors == n_inbound_ranks );
            CHECK( par::mpi_sum(n_outbound_ranks, comm)
                == par::mpi_sum(n_inbound_ranks, comm) );
            CHECK( par::mpi_sum(total_send_sz, comm)
                == par::mpi_sum(total_recv_sz, comm) );
          }

          // getNodeOwnerElements()
          {
            const DendroIntL *node_owners = new_da.getNodeOwnerElements();
            const size_t n_nodes = new_da.getTotalNodalSz();
            REQUIRE( node_owners != nullptr );
            CHECK( std::is_sorted(node_owners, node_owners + n_nodes) );
          }

          // getTNCoords()
          {
            const TreeNode<uint32_t, dim> *nodes = new_da.getTNCoords();
            const size_t n_nodes = new_da.getTotalNodalSz();
            const auto leaf_set = vec_leaf_list_view<dim>(dtree.getTreePartFiltered());
            REQUIRE( nodes != nullptr );
            for (size_t i = 0; i < n_nodes; ++i)
            {
              const TreeNode<uint32_t, dim> tiny_cell(nodes[i].coords(), m_uiMaxDepth);
              CHECK( border_or_overlap_any<dim>(tiny_cell, leaf_set) );
            }

            // Since the point set is predicted communication-free,
            // check that the prediction matches truth by ghost_pull(points).
            std::vector<TreeNode<uint32_t, dim>> actual_nodes(
                nodes, nodes + n_nodes);
            new_da.readFromGhostBegin(actual_nodes.data(), 1);
            new_da.readFromGhostEnd(actual_nodes.data(), 1);
            CHECK( std::equal(nodes, nodes + n_nodes, actual_nodes.cbegin()) );
          }

          // getBoundaryNodeIndices()
          {
            const TreeNode<uint32_t, dim> *nodes = new_da.getTNCoords();
            REQUIRE( nodes != nullptr );
            const TreeNode<uint32_t, dim> *local_nodes =
                nodes + new_da.getLocalNodeBegin();

            const std::vector<size_t> local_boundary_node_indices =
                new_da.getBoundaryNodeIndices();
            const size_t local_size = new_da.getLocalNodalSz();
            CHECK( local_boundary_node_indices.size() <= local_size );
            CHECK( par::mpi_sum(local_boundary_node_indices.size(), comm) > 0 );
            for (size_t x: local_boundary_node_indices)
            {
              CHECK( x < local_size );
              CHECK( local_nodes[x].getIsOnTreeBdry() );
            }
          }
        }
      }

      _DestroyHcurve();
    }



    DOCTEST_MPI_TEST_CASE("Consistent ghost exchange", 3)
    {
      dbg::wait_for_debugger(test_comm);
      constexpr int dim = 2;
      _InitializeHcurve(dim);
      const double sfc_tol = 0.3;

      const auto next_comm_size = make_next_comm_size(test_nb_procs);
      for (int np = 1; np <= test_nb_procs; np = next_comm_size(np))
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
          {
            // Grid.
            std::vector<TreeNode<uint32_t, dim>> grid;
            if (is_root and grid_pattern == central)
              grid = grid_pattern_central<dim>(max_depth);
            else if (is_root and grid_pattern == edges)
              grid = grid_pattern_edges<dim>(max_depth);
            SFC_Tree<uint32_t, dim>::distTreeSort(grid, sfc_tol, comm);
            DistTree<uint32_t, dim> dtree(grid, comm);

            /// for (int degree: {1, 2, 3})  // node ordering different for 2+.
            for (int degree: {1})
            {
              DA<dim> old_da(dtree, comm, degree);
              DA_P2P<dim> new_da(dtree, comm, degree);

              INFO("max_depth=", max_depth, "  degree=", degree);

              std::vector<int> old_ghost_read, old_ghost_write;
              std::vector<int> new_ghost_read, new_ghost_write;
              const int unit = intPow(100, par::mpi_comm_rank(comm));
              for (int ndofs: {1, 2})
              {
                old_da.createVector(old_ghost_read, false, true, ndofs);
                old_da.createVector(old_ghost_write, false, true, ndofs);
                new_da.createVector(new_ghost_read, false, true, ndofs);
                new_da.createVector(new_ghost_write, false, true, ndofs);
                for (auto *v: {&old_ghost_read, &old_ghost_write,
                               &new_ghost_read, &new_ghost_write})
                  std::generate(v->begin(), v->end(),
                      [=, i=0]() mutable { return unit * (i++); });

                old_da.readFromGhostBegin(old_ghost_read.data(), ndofs);
                old_da.readFromGhostEnd(old_ghost_read.data(), ndofs);
                old_da.writeToGhostsBegin(old_ghost_write.data(), ndofs);
                old_da.writeToGhostsEnd(old_ghost_write.data(), ndofs);
                new_da.readFromGhostBegin(new_ghost_read.data(), ndofs);
                new_da.readFromGhostEnd(new_ghost_read.data(), ndofs);
                new_da.writeToGhostsBegin(new_ghost_write.data(), ndofs);
                new_da.writeToGhostsEnd(new_ghost_write.data(), ndofs);

                CHECK( new_ghost_read == old_ghost_read );
                CHECK( new_ghost_write == old_ghost_write );
              }

              // isDirtyOut   //future: Deprecate this feature
              for (int ndofs: {1, 2})
              {
                old_da.createVector(old_ghost_write, false, true, ndofs);
                new_da.createVector(new_ghost_write, false, true, ndofs);
                for (auto *v: {&old_ghost_write, &new_ghost_write})
                  std::generate(v->begin(), v->end(),
                      [=, i=0]() mutable { return unit * (i++); });

                std::vector<char> write_odd(old_ghost_write.size(), false);
                for (size_t i = 1; i < write_odd.size(); i += 2)
                  write_odd[i] = true;

                old_da.writeToGhostsBegin(old_ghost_write.data(), ndofs, write_odd.data());
                old_da.writeToGhostsEnd(old_ghost_write.data(), ndofs, true, write_odd.data());
                new_da.writeToGhostsBegin(new_ghost_write.data(), ndofs, write_odd.data());
                new_da.writeToGhostsEnd(new_ghost_write.data(), ndofs, write_odd.data());
                CHECK( new_ghost_write == old_ghost_write );
              }
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
    namespace detail
    {
      // nodes_internal_to_face()
      constexpr int nodes_internal_to_face(int degree, int dimension)
      {
        return intPow(degree - 1, dimension);
      }
    }


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

      if (not active)
      {
        //future: make the ranges arrays start with {0} and append range_end.
        // Then the default initialization can apply to both active and inactive.
        m_ghosted_hyperface_ranges = { 0 };
        return;
      }

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

      // Updated neighborhoods of all local octants, to emit local hyperfaces.
      //   future: Only update from recv_neighbors
      total_dict.concat(border_dict);
      total_dict.reduce();

      std::vector<uint8_t> local_hyperfaces;
      std::vector<size_t> local_hyperface_ranges;
      std::vector<size_t> local_boundary_hyperfaces;

      // Local hyperfaces: Emit faces "owned" (with "split" status):
      //   neighbor_sets(local_octants ∪ (∪.p recv_octants[p]))@local_octants
      size_t dbg_count = 0;
      size_t count_local_hyperfaces = 0;

      // Another way to write the following, if it's a single loop.
      //   for (const auto &scope: ForLeafNeighborhoods<dim>(total_dict, local_octants))

      ForLeafNeighborhoods<dim> loop(total_dict, local_octants);
      for (auto it = loop.begin(), end = loop.end(); it != end; ++it)
      {
        const auto &scope = *it;
        const size_t leaf_index = scope.query_idx;
        const Octant key = scope.query_key;
        const int corner = key.getMortonIndex();
        Neighborhood<dim> self_nbh = scope.self_neighborhood;
        Neighborhood<dim> parent_nbh = scope.parent_neighborhood;
        Neighborhood<dim> children_nbh = scope.children_neighborhood;

          // local_hyperface_ranges is indexed adjacent to local_octants
        assert(leaf_index == dbg_count);
        ++dbg_count;

        /// self_nbh |= Neighborhood<dim>::solitary();  // it is from local_octants
        assert(self_nbh.center_occupied());

        const auto priority = priority_neighbors<dim>();
        const auto owned    = ~((self_nbh & priority) | parent_nbh).spread_out();
        const auto split    = children_nbh.spread_out();

        const Neighborhood<dim> covered =
            parent_nbh.spread_out_directed(~corner) | self_nbh | children_nbh;
        const Neighborhood<dim> boundary = (~covered).spread_out();
        assert(not (owned & boundary).any() or key.getIsOnTreeBdry());

        local_hyperface_ranges.push_back(count_local_hyperfaces);

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
                .mirrored(corner)
                .encode_split(split.test_flat(hyperface_idx));
            local_hyperfaces.push_back(hyperface.encoding());
            if (boundary.test_flat(hyperface_idx))
              local_boundary_hyperfaces.push_back(count_local_hyperfaces);
            ++count_local_hyperfaces;
          }
        });
      }
      // Push end of last range.
      local_hyperface_ranges.push_back(count_local_hyperfaces);


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

      par::RemoteMapBuilder cell_map(adjacency_list.local_rank);
      par::RemoteMapBuilder face_map(adjacency_list.local_rank);
      cell_map.set_local_count(local_octants.size());
      face_map.set_local_count(count_local_hyperfaces);

      // Scattermap[p]: Emit faces "owned" and "shared" (with "split" status):
      //   neighbor_sets(border_octants)@send_octants[p]   -> "owned", "split"
      //   neighbor_sets(recv_octants[p])@send_octants[p]  -> "shared" to p
      for (int i = 0; i < n_remote_parts; ++i)
      {
        const int remote_rank = adjacency_list.neighbor_ranks[i];
        const std::vector<size_t> &send_ids = send_octant_ids[i];

        ForLeafNeighborhoods<dim> border(border_dict, send_octants[i]);
        ForLeafNeighborhoods<dim> recv(recv_neighbors[i], send_octants[i]);
        for (auto border_it =  border.begin(), recv_it  = recv.begin(),
                  border_end = border.end(),   recv_end = recv.end();
                  border_it != border_end and  recv_it != recv_end;
                  ++border_it, ++recv_it)
        {
          // Expect to loop over every leaf in the query set (send_octants).
          assert(border_it.query_idx() == recv_it.query_idx());
          const size_t query_idx = border_it.query_idx();

          const auto border_scope = *border_it;
          const auto recv_scope = *recv_it;
          const Octant key = border_scope.query_key;
          assert(key == recv_scope.query_key);

          assert(border_scope.self_neighborhood.center_occupied());
          const auto priority = priority_neighbors<dim>();
          const auto owned = ~((border_scope.self_neighborhood & priority)
                               | border_scope.parent_neighborhood).spread_out();
          const auto split = border_scope.children_neighborhood.spread_out();
          const auto shared = (recv_scope.self_neighborhood
                               | recv_scope.children_neighborhood).spread_out();
          const auto emit = owned & shared;

          if (emit.none())
            continue;

          const size_t octant_id = send_ids[query_idx];
          cell_map.bind_local_id(remote_rank, octant_id);

          // Search local_hyperfaces for exact indices to bind to remote.
          size_t count_hyperfaces = 0;
          const size_t oct_faces_begin = local_hyperface_ranges[octant_id];
          const size_t oct_faces_end = local_hyperface_ranges[octant_id + 1];
          for (size_t id = oct_faces_begin; id < oct_faces_end; ++id)
          {
            const Hyperface4D face = SplitHyperface4D(local_hyperfaces[id])
                                     .decode()
                                     .mirrored(key.getMortonIndex());
            if (emit.test_flat(face.flat()))
            {
              face_map.bind_local_id(remote_rank, id);
              ++count_hyperfaces;
            }
          }
          assert(count_hyperfaces == emit.count());
        }
      }

      std::vector<uint8_t> pre_ghost_hyperfaces;
      std::vector<uint8_t> post_ghost_hyperfaces;
      std::vector<size_t> pre_ghost_hyperface_ranges;
      std::vector<size_t> post_ghost_hyperface_ranges;
      std::vector<size_t> ghosted_boundary_hyperfaces;
      size_t count_pre_ghost_boundary_hyperfaces = 0;
      std::vector<Octant> ghosted_octants;

      // Gathermap[p]: Emit faces "owned" and "shared" (with "split" status):
      //   neighbor_sets(border_octants)@recv_octants[p]   -> "owned", "split"
      //   neighbor_sets(send_octants[p])@recv_octants[p]  -> "shared" to r
      for (int i = 0; i < n_remote_parts; ++i)
      {
        const int remote_rank = adjacency_list.neighbor_ranks[i];
        const bool pre_ghost = remote_rank < adjacency_list.local_rank;
        const size_t initial_count = (pre_ghost?
            0 :  pre_ghost_hyperfaces.size() + local_hyperfaces.size());
        std::vector<uint8_t> &hyperfaces = (pre_ghost?
            pre_ghost_hyperfaces
            : post_ghost_hyperfaces);
        std::vector<size_t> &hyperface_ranges = (pre_ghost?
            pre_ghost_hyperface_ranges
            : post_ghost_hyperface_ranges);

        size_t count_cells = 0;
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
          const int corner = key.getMortonIndex();
          assert(key == send_scope.query_key);

          assert(border_scope.self_neighborhood.center_occupied());

          const Neighborhood<dim> self_nbh = border_scope.self_neighborhood;
          const Neighborhood<dim> parent_nbh = border_scope.parent_neighborhood;
          const Neighborhood<dim> children_nbh = border_scope.children_neighborhood;

          const auto priority = priority_neighbors<dim>();
          const auto owned    = ~((self_nbh & priority) | parent_nbh).spread_out();
          const auto split    = children_nbh.spread_out();

          const auto shared = (send_scope.self_neighborhood
                               | send_scope.children_neighborhood).spread_out();
          const auto emit = owned & shared;

          // Only list ghosted octants where there are ghosted faces.
          if (emit.none())
            continue;

          const Neighborhood<dim> covered =
              parent_nbh.spread_out_directed(~corner) | self_nbh | children_nbh;
          const Neighborhood<dim> boundary = (~covered).spread_out();
          assert(not (emit & boundary).any() or key.getIsOnTreeBdry());

          hyperface_ranges.push_back(initial_count + hyperfaces.size());
          ghosted_octants.push_back(key);
          ++count_cells;

#warning "Does not match old DA node ordering that involved SFC-sorting points."
          // The nodal ordering differs on ownership for incomplete trees,
          // as well as intra-element order for degree>=2 complete trees.

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
            if (emit.test_flat(hyperface_idx))
            {
              int hyperface_coord = 0;
              for (int d = 0, stride = 1; d < dim; ++d, stride *= 4)  // 2b/face
                hyperface_coord += idxs[d] * stride;
              SplitHyperface4D hyperface = Hyperface4D(hyperface_coord)
                  .mirrored(corner)
                  .encode_split(split.test_flat(hyperface_idx));
              hyperfaces.push_back(hyperface.encoding());
              if (boundary.test_flat(hyperface_idx))
                ghosted_boundary_hyperfaces.push_back(initial_count + hyperfaces.size());
              ++count_hyperfaces;
            }
          });
        }

        if (count_cells > 0)
        {
          cell_map.increase_ghost_count(remote_rank, count_cells);
          face_map.increase_ghost_count(remote_rank, count_hyperfaces);
        }
      }

      m_cell_map = cell_map.finish();
      m_face_map = face_map.finish();

      const size_t pre_ghost_octants = pre_ghost_hyperface_ranges.size();
      const size_t post_ghost_octants = post_ghost_hyperface_ranges.size();
      // Push end of last range.
      pre_ghost_hyperface_ranges.push_back(pre_ghost_hyperfaces.size());
      post_ghost_hyperface_ranges.push_back( pre_ghost_hyperfaces.size()
                                           + local_hyperfaces.size()
                                           + post_ghost_hyperfaces.size());

      assert(pre_ghost_octants == m_cell_map.local_begin());
      assert(local_octants.size() == m_cell_map.local_count());

      ghosted_octants.insert(ghosted_octants.begin() + pre_ghost_octants,
          local_octants.cbegin(), local_octants.cend());
      m_ghosted_octants = std::move(ghosted_octants);

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
      for (auto &r: local_hyperface_ranges)
        r += pre_ghost_hyperfaces.size();
      // Exclude repeated end-of-prev == begin-of-next
      ghosted_hyperface_ranges.insert(ghosted_hyperface_ranges.end(),
          pre_ghost_hyperface_ranges.cbegin(), pre_ghost_hyperface_ranges.cend() - 1);
      ghosted_hyperface_ranges.insert(ghosted_hyperface_ranges.end(),
          local_hyperface_ranges.cbegin(), local_hyperface_ranges.cend() - 1);
      ghosted_hyperface_ranges.insert(ghosted_hyperface_ranges.end(),
          post_ghost_hyperface_ranges.cbegin(), post_ghost_hyperface_ranges.cend());
      // Keep end of last range in post_ghost_hyperface ranges.

      // ghosted_hyperface_bdry_ids
      std::set<size_t> ghosted_hyperface_bdry_ids;
      for (auto &r: local_boundary_hyperfaces)
        r += pre_ghost_hyperfaces.size();
      ghosted_hyperface_bdry_ids.insert(ghosted_boundary_hyperfaces.cbegin(),
                                        ghosted_boundary_hyperfaces.cend());
      ghosted_hyperface_bdry_ids.insert(local_boundary_hyperfaces.cbegin(),
                                        local_boundary_hyperfaces.cend());
      m_ghosted_hyperface_bdry_ids = std::move(ghosted_hyperface_bdry_ids);


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
      par::Mpi_Exscan(
          combined.m_counts.data(), m_prefix_sum.m_counts.data(), //initially 0
          combined.m_counts.size(), MPI_SUM, active_comm);
      par::Mpi_Allreduce(
          combined.m_counts.data(), m_reduction.m_counts.data(),
          combined.m_counts.size(), MPI_SUM, active_comm);

      // Mpi scan and reduce cell count.
      DendroLLU cell_count = local_octants.size();
      par::Mpi_Exscan(&cell_count, &m_cell_prefix_sum, 1, MPI_SUM, active_comm);
      par::Mpi_Allreduce(&cell_count, &m_cell_reduction, 1, MPI_SUM, active_comm);

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


    // DA::n_local_cells()
    template <int dim>
    size_t DA<dim>::n_local_cells() const
    {
      return m_cell_map.local_count();
    }

    // DA::n_global_cells()
    template <int dim>
    DendroLLU DA<dim>::n_global_cells() const
    {
      return m_cell_reduction;
    }

    // DA::global_cell_offset()
    template <int dim>
    DendroLLU DA<dim>::global_cell_offset() const
    {
      return m_cell_prefix_sum;
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

    // DA::n_local_cell_owned_nodes()
    template <int dim>
    size_t DA<dim>::n_local_cell_owned_nodes(int degree, size_t local_cell_id) const
    {
      const size_t ghosted_cell_id = m_cell_map.local_begin() + local_cell_id;
      const size_t face_begin = m_ghosted_hyperface_ranges[ghosted_cell_id];
      const size_t face_end   = m_ghosted_hyperface_ranges[ghosted_cell_id + 1];
      SplitDimCount<size_t, dim> owned_faces = {};
      for (size_t i = face_begin; i < face_end; ++i)
      {
        const SplitHyperface4D hf = {m_ghosted_hyperfaces[i]};
        size_t *count = (hf.is_split()?
            owned_faces.split() : owned_faces.unsplit());
        ++count[hf.decode().dimension()];  // independent of child number.
      }

      // future: option for extra nodes on split faces.
      if (degree == 1)
        return owned_faces.unsplit()[0] + owned_faces.split()[0];
      else
        return dimension_sum(owned_faces.unsplit(), degree)
              + dimension_sum(owned_faces.split(), degree);
    }

    // DA::point_set()
    template <int dim>
    PointSet<dim> DA<dim>::point_set(int degree) const
    {
      const size_t total_nodes = this->n_total_nodes(degree);
      const size_t local_nodes = this->n_local_nodes(degree);

      std::vector<TreeNode<uint32_t, dim>> ghosted_points;
      std::vector<size_t> local_boundary_indices;
      ghosted_points.reserve(total_nodes);
      local_boundary_indices.reserve(
          2 * dim * std::pow(local_nodes, double(dim - 1)/dim));

      const size_t ghosted_octants = this->remote_cell_map().total_count();
      const size_t local_cell_begin = this->remote_cell_map().local_begin();
      const size_t local_cell_end = this->remote_cell_map().local_end();
      const size_t local_nodes_begin = this->local_nodes_begin(degree);
      for (size_t oct = 0; oct < ghosted_octants; ++oct)
      {
        const bool owned_octant =
            (local_cell_begin <= oct and oct < local_cell_end);

        // Octant and owned face range.
        const TreeNode<uint32_t, dim> octant = m_ghosted_octants[oct];
        const size_t face_begin = m_ghosted_hyperface_ranges[oct];
        const size_t face_end   = m_ghosted_hyperface_ranges[oct + 1];

        // Write out nodes owned by this octant.
        for (size_t i = face_begin; i < face_end; ++i)
        {
          const SplitHyperface4D encoded = {m_ghosted_hyperfaces[i]};
          const bool split = encoded.is_split();
          const Hyperface4D face =
              encoded.decode().mirrored(octant.getMortonIndex());
          const int dimension = face.dimension();
          const size_t internal_nodes = detail::nodes_internal_to_face(degree, dimension);

          const bool is_boundary_face =
              m_ghosted_hyperface_bdry_ids.find(i)
              != m_ghosted_hyperface_bdry_ids.end();
          const bool local_boundary_face = owned_octant and is_boundary_face;

          // Prepare ranges of nested for loop. Restricted axes on the face
          // are mapped to single iterations on outermost loop levels.
          int axis[dim];
          int begin[dim];
          int end[dim];
          int free = 0;
          int fixed = dim - 1;
          for (int d = 0; d < dim; ++d)
          {
            const int coordinate = face.coordinate(d);
            if (coordinate == 1)
            {
              axis[d] = free;
              begin[free] = 1;
              end[free] = degree;
              ++free;
            }
            else
            {
              axis[d] = fixed;
              const int fixed_value = (coordinate == 0 ? 0 : degree);
              begin[fixed] = fixed_value;
              end[fixed] = fixed_value + 1;
              --fixed;
            }
          }

          // Map numerators (node indices) and denominator (degree) to coordinate.
          const auto create_node = [is_boundary_face](
              TreeNode<uint32_t, dim> octant, std::array<int, dim> idxs, int degree)
            -> TreeNode<uint32_t, dim>
          {
            periodic::PCoord<uint32_t, dim> node_pt = {};
            const uint32_t side = octant.range().side();
            for (int d = 0; d < dim; ++d)
              node_pt.coord(d, side * idxs[d] / degree);
            node_pt += octant.range().min();
            TreeNode<uint32_t, dim> result(node_pt, octant.getLevel());
            result.setIsOnTreeBdry(is_boundary_face);
            return result;
          };

          // dim-nested for loop.
          tmp::nested_for_rect<dim>(begin, end, [&](auto...idx_pack)
          {
            if (local_boundary_face)
            {
              assert(ghosted_points.size() >= local_nodes_begin);
              local_boundary_indices.push_back(
                  ghosted_points.size() - local_nodes_begin);
            }

            std::array<int, dim> loop_idxs = {idx_pack...};  // 0..degree per axis.
            std::array<int, dim> idxs;
            for (int d = 0; d < dim; ++d)
              idxs[d] = loop_idxs[axis[d]];
            ghosted_points.push_back(create_node(octant, idxs, degree));
          });

          //future: option for extra nodes on split faces
        }
      }

      assert(ghosted_points.size() == total_nodes);

      return { degree, std::move(ghosted_points), std::move(local_boundary_indices) };
    }

    // DA::remote_cell_map()
    template <int dim>
    const par::RemoteMap & DA<dim>::remote_cell_map() const
    {
      return m_cell_map;
    }

    // DA::remote_node_map()
    template <int dim>
    par::RemoteMap DA<dim>::remote_node_map(int degree) const
    {
      // Derive a new remote map from the base map: lookup faces, apply degree.
      const par::RemoteMap &map = m_face_map;
      par::RemoteMapBuilder builder(map.this_mpi_rank());
      builder.set_local_count(this->n_local_nodes(degree));
      for (int link = 0, n = map.n_links(); link < n; ++link)
      {
        const int remote_rank = map.mpi_rank(link);

        size_t count_ghost = 0;
        for (size_t i = map.ghost_begin(link),
                    e = map.ghost_end(link); i < e; ++i)
        {
          const int dimension = SplitHyperface4D(m_ghosted_hyperfaces[i])
                                .decode().dimension();
          count_ghost += detail::nodes_internal_to_face(degree, dimension);
        }
        builder.increase_ghost_count(remote_rank, count_ghost);

        size_t local_face_id = 0;
        size_t local_node_id = 0;
        const uint8_t *local_hyperfaces = &m_ghosted_hyperfaces[map.local_begin()];
        map.for_bound_local_id(link, [&](size_t, size_t bound_local_id)
        {
          for (; local_face_id < bound_local_id; ++local_face_id)
          {
            const int dimension = SplitHyperface4D(local_hyperfaces[local_face_id])
                                  .decode().dimension();
            local_node_id += detail::nodes_internal_to_face(degree, dimension);
          }

          const int dimension = SplitHyperface4D(local_hyperfaces[bound_local_id])
                                .decode().dimension();
          const int n_nodes_on_face = detail::nodes_internal_to_face(degree, dimension);
          for (int node = 0; node < n_nodes_on_face; ++node)
            builder.bind_local_id(remote_rank, local_node_id + node);
        });
      }

      return builder.finish();
    }

    // DA<dim>::local_cell_list()
    template <int dim>
    ConstRange<TreeNode<uint32_t, dim>> DA<dim>::local_cell_list() const
    {
      const TreeNode<uint32_t, dim> *ptr = m_ghosted_octants.data();
      return { ptr + this->remote_cell_map().local_begin(),
               ptr + this->remote_cell_map().local_end() };
    }

    // DA<dim>::ghosted_cell_list()
    template <int dim>
    ConstRange<TreeNode<uint32_t, dim>> DA<dim>::ghosted_cell_list() const
    {
      const TreeNode<uint32_t, dim> *ptr = m_ghosted_octants.data();
      return { ptr, ptr + this->remote_cell_map().total_count() };
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
        count += hyperfaces[d] * detail::nodes_internal_to_face(degree, d);
      return count;
    }


    // DA_Wrapper::DA_Wrapper
    template <int dim>
    DA_Wrapper<dim>::DA_Wrapper(
        const DistTree<uint32_t, dim> &inDistTree,
        int,  //ignored
        MPI_Comm comm,
        unsigned int order,
        size_t, //ignored
        double) //ignored
    :
      m_da(inDistTree)
    {
      const int degree = order;

      /// // Cell map.
      /// par::RemoteMap cell_map = m_da.remote_cell_map();
      /// const size_t total_cells = cell_map.total_count();
      /// const size_t local_cells = cell_map.local_count();
      /// const size_t local_cell_begin = cell_map.local_begin();
      /// const size_t local_cell_end = cell_map.local_end();
      /// // Cell ids at cells.
      /// std::vector<DendroIntL> global_cell_ids(cell_map.total_count());
      /// std::iota(&global_cell_ids[local_cell_begin],
      ///           &global_cell_ids[local_cell_end],
      ///           m_da.global_cell_offset());
      /// par::GhostPullRequest pull_cell_ids =
      ///     ghost_pull(comm, global_cell_ids.data(), cell_map, 1);

      // Node map.
      par::RemoteMap node_map = m_da.remote_node_map(degree);
      const size_t total_nodes = m_da.n_total_nodes(degree);
      const size_t local_nodes = m_da.n_local_nodes(degree);
      const size_t local_node_begin = m_da.local_nodes_begin(degree);
      const size_t local_node_end = m_da.local_nodes_end(degree);

      // Node ids at nodes.
      std::vector<DendroIntL> global_ids(total_nodes);
      std::iota(&global_ids[local_node_begin],
                &global_ids[local_node_end],
                m_da.global_node_offset(degree));
      par::GhostPullRequest pull_ids =
          par::ghost_pull(comm, global_ids.data(), node_map, 1);

      // Cell ids at nodes.
      const DendroIntL global_cell_offset = m_da.global_cell_offset();
      const size_t local_cells = m_da.n_local_cells();
      std::vector<DendroIntL> global_owner_ids;
      global_owner_ids.reserve(total_nodes);
      global_owner_ids.resize(local_node_begin);
      for (size_t i = 0; i < local_cells; ++i)
      {
        const size_t cell_owned_nodes = m_da.n_local_cell_owned_nodes(degree, i);
        global_owner_ids.insert(
            global_owner_ids.end(), cell_owned_nodes, i + global_cell_offset);
      }
      global_owner_ids.resize(total_nodes);
      par::GhostPullRequest pull_owners =
          par::ghost_pull(comm, global_owner_ids.data(), node_map, 1);

      // Point set.
      PointSet<dim> points = m_da.point_set(degree);

      // Wait for ghost exchanges to complete.
      pull_ids.wait_all();
      pull_owners.wait_all();

      // Finish initializing data class members.
      m_data = std::unique_ptr<WrapperData<dim>>(new WrapperData<dim>({
        std::move(points),
        std::move(global_ids),
        std::move(global_owner_ids),
        std::move(node_map),
        {}, {}  // empty request containers
      }));
    }

    // DA_Wrapper::DA_Wrapper
    template <int dim>
    DA_Wrapper<dim>::DA_Wrapper(
      const DistTree<uint32_t, dim> &inDistTree,
      MPI_Comm comm,
      unsigned int order,
      size_t, //ignored
      double) //ignored
    :
      DA_Wrapper<dim>(inDistTree, 0, comm, order, {}, {})  // delegate
    {}


    // DA_Wrapper::createVector()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::createVector(
         std::vector<T> &local,
         bool isElemental,
         bool isGhosted,
         unsigned int dof) const
    {
      assert(not isElemental);  //future: support elemental vectors
      local.resize(dof * (isGhosted? this->getTotalNodalSz()
                                   : this->getLocalNodalSz()));
    }

    // DA_Wrapper::createVector()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::createVector(
         T* &local,
         bool isElemental,
         bool isGhosted,
         unsigned int dof) const
    {
      assert(not isElemental);  //future: support elemental vectors
      const size_t size =
          dof * (isGhosted? this->getTotalNodalSz()
                          : this->getLocalNodalSz());
      local = (size > 0? new T[size] : nullptr);
    }

    // DA_Wrapper::destroyVector()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::destroyVector(T *&local) const
    {
      delete local;
      local = nullptr;
    }

    // DA_Wrapper::destroyVector()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::destroyVector(std::vector<T> &local) const
    {
      local.clear();
    }

    // DA_Wrapper::readFromGhostBegin()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::readFromGhostBegin(
        T *vec,
        unsigned int dof) const
    {
      //debug
      const int n_send = data()->remote_map.n_active_bound_links();
      const int n_recv = data()->remote_map.n_active_ghost_links();
      const MPI_Comm comm = m_da.active_comm();
      assert( par::mpi_sum(n_send, comm) == par::mpi_sum(n_recv, comm) );

      auto it_inserted =
          mutate()->pull_requests.emplace(vec,
            par::ghost_pull(m_da.active_comm(), vec, data()->remote_map, dof));

      const bool inserted = it_inserted.second;
      assert(inserted);  // otherwise, a request on vec is in progress.
    }

    // DA_Wrapper::readFromGhostEnd()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::readFromGhostEnd(
        T *vec,
        unsigned int dof) const
    {
      auto it = mutate()->pull_requests.find(vec);
      assert(it != mutate()->pull_requests.end());
      par::GhostPullRequest &request = it->second;
      request.wait_all();
      mutate()->pull_requests.erase(it);
    }


    // DA_Wrapper::writeToGhostBegin()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::writeToGhostsBegin(
        T *vec,
        unsigned int dof,
        const char * isDirtyOut) const
    {
      const auto add = [](const auto &x, const auto &y){ return x + y; };
      this->writeToGhostsBegin_impl(vec, dof, isDirtyOut, add);
    }

    // DA_Wrapper::overwriteToGhostBegin()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::overwriteToGhostsBegin(
        T *vec,
        unsigned int dof,
        const char * isDirtyOut) const
    {
      const auto replace = [](const auto &x, const auto &y) { return y; };
      this->writeToGhostsBegin_impl(vec, dof, isDirtyOut, replace);
    }

    // DA_Wrapper::writeToGhostBegin_impl()
    template <int dim>
    template <typename T, class Operation>
    void DA_Wrapper<dim>::writeToGhostsBegin_impl(
        T *vec,
        unsigned int dof,
        const char * isDirtyOut,
        Operation op) const
    {
      //debug
      const int n_send = data()->remote_map.n_active_ghost_links();
      const int n_recv = data()->remote_map.n_active_bound_links();
      const MPI_Comm comm = m_da.active_comm();
      assert( par::mpi_sum(n_send, comm) == par::mpi_sum(n_recv, comm) );

      const auto nop = [](const auto &x, const auto &y) { return x; };

      if (isDirtyOut == nullptr)
      {
        auto *payload_req = new auto (par::ghost_push(               //rval ctor
            m_da.active_comm(), data()->remote_map, dof, vec, op));
        const bool inserted =
            mutate()->push_requests.emplace(vec, payload_req).second;

        assert(inserted);  // otherwise, a request on vec is in progress.
      }
      else
      {
        // Store pointers to where flags and payload will be received.
        par::RemoteStage<char> flags_buffer(data()->remote_map, 1);  // per node
        par::RemoteStage<T> payload_buffer(data()->remote_map, dof);
        const char *staged_flag = flags_buffer.staged_data();
        const T *staged_payload = payload_buffer.staged_data();
        const size_t stage_size = payload_buffer.size();

        // This lambda only combines (x, y) if the corresponding flag is True.
        // To calculate offset and lookup flag, capture staging pointers.
        const auto combine_if =
          [=, dof=dof, op=std::move(op)](const T &x, const T &y)
        {
          const size_t offset = &y - staged_payload;
          assert(offset < stage_size);  // otherwise: probably a copy, not ref.
          const size_t node_offset = offset / dof;
          return (bool(staged_flag[node_offset])? op(x, y) : x);
        };

        // Push flags.
        auto *flag_req = new auto (
            std::move(flags_buffer)
              .ghost_push(m_da.active_comm(), (char*){}, isDirtyOut, nop));
        mutate()->push_requests.emplace(isDirtyOut, flag_req);

        // Push payload.
        auto *payload_req = new auto (
            std::move(payload_buffer)
              .ghost_push(m_da.active_comm(), vec, combine_if));
        const bool inserted =
            mutate()->push_requests.emplace(vec, payload_req).second;

        assert(inserted);  // otherwise, a request on vec is in progress.
      }
    }

    // DA_Wrapper::writeToGhostEnd()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::writeToGhostsEnd(
        T *vec,
        unsigned int dof,
        bool useAccumulation,
        const char * isDirtyOut) const
    {
      if (useAccumulation == false)
        throw std::invalid_argument("useAccumulation=false is deprecated. "
            "Try overwriteToGhostsBegin() / overwriteToGhostsEnd().");

      this->writeToGhostsEnd(vec, dof, isDirtyOut);
    }

    // DA_Wrapper::writeToGhostEnd()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::writeToGhostsEnd(
        T *vec,
        unsigned int dof,
        const char * isDirtyOut) const
    {
      this->writeToGhostsEnd_impl(vec, dof, isDirtyOut);
    }

    // DA_Wrapper::overwriteToGhostEnd()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::overwriteToGhostsEnd(
        T *vec,
        unsigned int dof,
        const char * isDirtyOut) const
    {
      this->writeToGhostsEnd_impl(vec, dof, isDirtyOut);
    }

    // DA_Wrapper::writeToGhostEnd_impl()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::writeToGhostsEnd_impl(
        T *vec,
        unsigned int dof,
        const char * isDirtyOut) const
    {
      // Extract GhostPushRequest for payload.  //future: extract()
      auto payload_it = mutate()->push_requests.find(vec);
      assert(payload_it != mutate()->push_requests.end());
      std::unique_ptr<par::GhostPushRequest> payload_req = std::move(payload_it->second);
      mutate()->push_requests.erase(payload_it);

      if (isDirtyOut != nullptr)
      {
        // Extract GhostPushRequest for flags.  //future: extract()
        auto flag_it = mutate()->push_requests.find(isDirtyOut);
        assert(flag_it != mutate()->push_requests.end());
        std::unique_ptr<par::GhostPushRequest> flag_req = std::move(flag_it->second);
        mutate()->push_requests.erase(flag_it);

        flag_req->wait_on_recv();

        payload_req->wait_on_recv();
        payload_req->update_local();
        payload_req->wait_on_send();

        // Do not update local flags.
        flag_req->wait_on_send();
      }
      else
      {
        payload_req->wait_on_recv();
        payload_req->update_local();
        payload_req->wait_on_send();
      }
    }


    //future: Create nodes on the fly, instead of using pointer to array.

    // struct DA_Wrapper::ConstNodeRange
    template <int dim>
    struct DA_Wrapper<dim>::ConstNodeRange
    {
      const TreeNode<uint32_t, dim> *begin() const { return m_begin; }
      const TreeNode<uint32_t, dim> *end()   const { return m_end; }
      size_t size() const { return m_end - m_begin; }

      const TreeNode<uint32_t, dim> *m_begin;
      const TreeNode<uint32_t, dim> *m_end;
    };

    // DA_Wrapper<dim>::ghosted_nodes()
    template <int dim>
    typename DA_Wrapper<dim>::ConstNodeRange DA_Wrapper<dim>::ghosted_nodes() const
    {
      const TreeNode<uint32_t, dim> *nodes = this->getTNCoords();
      return { nodes, nodes + this->getTotalNodalSz() };
    }

    // DA_Wrapper<dim>::local_nodes()
    template <int dim>
    typename DA_Wrapper<dim>::ConstNodeRange DA_Wrapper<dim>::local_nodes() const
    {
      const TreeNode<uint32_t, dim> *nodes = this->getTNCoords();
      return { nodes + this->getLocalNodeBegin(),
               nodes + this->getLocalNodeEnd() };
    }

    // DA_Wrapper<dim>::local_cell_list()
    template <int dim>
    ConstRange<TreeNode<uint32_t, dim>> DA_Wrapper<dim>::local_cell_list() const
    {
      return m_da.local_cell_list();
    }

    // DA_Wrapper<dim>::ghosted_cell_list()
    template <int dim>
    ConstRange<TreeNode<uint32_t, dim>> DA_Wrapper<dim>::ghosted_cell_list() const
    {
      return m_da.ghosted_cell_list();
    }



    // DA_Wrapper<dim>::getNodeLocalToGlobalMap()
    template <int dim>
    const std::vector<RankI> & DA_Wrapper<dim>::getNodeLocalToGlobalMap() const
    {
      return data()->ghosted_global_ids;
    }

    // DA_Wrapper<dim>::getTNCoords()
    template <int dim>
    const TreeNode<uint32_t, dim> * DA_Wrapper<dim>::getTNCoords() const
    {
      return data()->points.ghosted_points.data();
    }

    // DA_Wrapper<dim>::getNodeOwnerElements()
    template <int dim>
    const DendroIntL * DA_Wrapper<dim>::getNodeOwnerElements() const
    {
      return data()->ghosted_global_owning_elements.data();
    }

    // DA_Wrapper<dim>::getReferenceElement()
    template <int dim>
    const RefElement * DA_Wrapper<dim>::getReferenceElement() const
    {
      // Memoization avoids re-constructing RefElement for every DA.
      return memo_ref_element(dim, degree());
    }

    /// // DA_Wrapper<dim>::getBoundaryNodeIndices()
    /// template <int dim>
    /// void DA_Wrapper<dim>::getBoundaryNodeIndices(std::vector<size_t> &bdyIndex) const
    /// {
    ///   bdyIndex = this->getBoundaryNodeIndices();  // copy
    /// }

    // DA_Wrapper<dim>::getBoundaryNodeIndices()
    template <int dim>
    const std::vector<size_t> & DA_Wrapper<dim>::getBoundaryNodeIndices() const
    {
      return data()->points.local_boundary_indices;
    }


    // DA_Wrapper<dim>::nodalVecToGhostedNodal()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::nodalVecToGhostedNodal(
        const T *in,
        T *&&out,   // rvalue ref
        bool isAllocated,
        unsigned int dof) const
    {
      assert(isAllocated);
      this->nodalVecToGhostedNodal(in, out, true, dof);
    }

    // DA_Wrapper<dim>::nodalVecToGhostedNodal()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::nodalVecToGhostedNodal(
        const T *in,
        T *&out,
        bool isAllocated,
        unsigned int dof) const
    {
      if (not this->isActive())
        return;

      if (not isAllocated)
        this->createVector<T>(out, false, true, dof);

      // Assumes layout [abc][abc][...], so just need single shift.
      const size_t local_size = this->getLocalNodalSz();
      const size_t local_begin = this->getLocalNodeBegin();
      std::copy_n(in, dof * local_size, out + dof * local_begin);
    }

    // DA_Wrapper<dim>::ghostedNodalToNodalVec()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::ghostedNodalToNodalVec(
        const T *gVec,
        T *&&local,           //rvalue ref
        bool isAllocated,
        unsigned int dof) const
    {
      assert(isAllocated);  // can't pass a new pointer back.
      this->ghostedNodalToNodalVec(gVec, local, true, dof);
    }


    // DA_Wrapper<dim>::ghostedNodalToNodalVec()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::ghostedNodalToNodalVec(
        const T *gVec,
        T *&local,
        bool isAllocated,
        unsigned int dof) const
    {
      if (not this->isActive())
        return;

      if (not isAllocated)
        this->createVector(local, false, false, dof);

      // Assumes layout [abc][abc][...], so just need single shift.
      const size_t local_size = this->getLocalNodalSz();
      const size_t local_begin = this->getLocalNodeBegin();
      std::copy_n(gVec + dof * local_begin, dof * local_size, local);
    }

    // DA_Wrapper<dim>::nodalVecToGhostedNodal()
    template <int dim>
    template<typename T>
    void DA_Wrapper<dim>::nodalVecToGhostedNodal(
        const std::vector<T> &in,
        std::vector<T> &out,
        bool isAllocated,
        unsigned int dof) const
    {
      if (not this->isActive())
        return;

      if (not isAllocated)
        this->createVector<T>(out, false, true, dof);

      this->nodalVecToGhostedNodal(
          in.data(), out.data(), true, dof);
    }

    // DA_Wrapper<dim>::ghostedNodalToNodalVec()
    template <int dim>
    template<typename T>
    void DA_Wrapper<dim>::ghostedNodalToNodalVec(
        const std::vector<T> gVec,
        std::vector<T> &local,
        bool isAllocated,
        unsigned int dof) const
    {
      if (not this->isActive())
        return;

      if (not isAllocated)
        this->createVector(local, false, false, dof);

      this->ghostedNodalToNodalVec(gVec.data(), local.data(), true, dof);
    }

    // DA_Wrapper<dim>::setVectorByFunction()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::setVectorByFunction(
        T *dest,
        std::function<void( const T *, T *)> func,
        bool isElemental,
        bool isGhosted,
        unsigned int dof) const
    {
      if (isElemental)
      {
        throw std::logic_error("Elemental version not implemented!");
      }

      constexpr int maximum_dimension = 4;
      std::array<T, maximum_dimension> point = {};
      const int degree = this->degree();

      auto nodes = (isGhosted? this->ghosted_nodes() : this->local_nodes());
      for (const TreeNode<uint32_t, dim> &node: nodes)
      {
        treeNode2Physical(node, degree, point.data());
        func(point.data(), dest);
        dest += dof;
      }
    }

    // DA_Wrapper<dim>::setVectorByScalar()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::setVectorByScalar(
        T *dest,
        const T *value,
        bool isElemental,
        bool isGhosted,
        unsigned int dof,
        unsigned int initDof) const
    {
      if (isElemental)
      {
        throw std::logic_error("Elemental version not implemented!");
      }

      const size_t n_nodes = (isGhosted? this->getTotalNodalSz()
                                       : this->getLocalNodalSz());
      if (initDof == 1)
      {
        const T scalar_value = *value;
        for (size_t i = 0; i < n_nodes; ++i)
        {
          dest[0] = scalar_value;
          dest += dof;
        }
      }
      else
      {
        for (size_t i = 0; i < n_nodes; ++i)
        {
          for (int var = 0; var < initDof; ++var)
            dest[var] = value[var];
          dest += dof;
        }
      }
    }

    // DA_Wrapper<dim>::vecTopvtu()
    template <int dim>
    template <typename T>
    void DA_Wrapper<dim>::vecTopvtu(
        T *local,
        const char *fPrefix,
        char **nodalVarNames,
        bool isElemental,
        bool isGhosted,
        unsigned int dof)
    {
      throw std::logic_error("Not implemented");
    }

    /// // DA_Wrapper<dim>::computeTreeNodeOwnerProc()
    /// template <int dim>
    /// void DA_Wrapper<dim>::computeTreeNodeOwnerProc(
    ///     const TreeNode<uint32_t, dim> * pNodes,
    ///     unsigned int n,
    ///     int* ownerRanks) const
    /// {
    ///   // This seems to have been requested, for IBM.
    ///   // Possibly assumptions about it are broken.
    ///   // Ideally remove this until wew know what it's for.
    ///   throw std::logic_error("Not implemented");
    /// }


  }//namespace da_p2p


  // ghost_range() (const)
  template <typename DA_Type, typename T>
  ConstRange<T> ghost_range(const DA_Type &da, int ndofs, const T *a)
  {
    return { a, a + da.getTotalNodalSz() * ndofs };
  }

  // local_range() (const)
  template <typename DA_Type, typename T>
  ConstRange<T> local_range(const DA_Type &da, int ndofs, const T *a)
  {
    return { a + da.getLocalNodeBegin() * ndofs,
             a + da.getLocalNodeEnd() * ndofs };
  }

  // ghost_range()
  template <typename DA_Type, typename T>
  Range<T> ghost_range(const DA_Type &da, int ndofs, T *a)
  {
    return { a, a + da.getTotalNodalSz() * ndofs };
  }

  // local_range()
  template <typename DA_Type, typename T>
  Range<T> local_range(const DA_Type &da, int ndofs, T *a)
  {
    return { a + da.getLocalNodeBegin() * ndofs,
             a + da.getLocalNodeEnd() * ndofs };
  }


  // multiLevelDA()
  template <typename DA_Type, unsigned dim>
  std::vector<DA_Type> multiLevelDA(
      const DistTree<uint32_t, dim> &dtree,
      MPI_Comm comm,
      unsigned order,
      size_t grain,
      double sfc_tol)
  {
    const int numStrata = dtree.getNumStrata();
    std::vector<DA_Type> outDAPerStratum;
    outDAPerStratum.reserve(numStrata);
    for (int l = 0; l < numStrata; ++l)
      outDAPerStratum.emplace_back(dtree, l, comm, order, grain, sfc_tol);
    return outDAPerStratum;
  }








}//namespace ot



#ifdef BUILD_WITH_PETSC
    //future: these should go in a module that depends on Petsc and this one.
template <typename DA_Type>
inline PetscErrorCode petscCreateVector(
    const DA_Type &da,
    Vec &local,
    bool isElemental,
    bool isGhosted,
    unsigned int dof)
{
  PetscErrorCode status = 0;

  if (not da.isActive())
  {
    local = NULL;
    return status;
  }

  MPI_Comm active_comm = da.getCommActive();
  const size_t size = 
      dof * (isGhosted? da.getTotalNodalSz()
                      : da.getLocalNodalSz());

  VecCreate(active_comm, &local);
  status = VecSetSizes(local, size, PETSC_DECIDE);

  if (da.getNpesAll() > 1)
    VecSetType(local, VECMPI);
  else
    VecSetType(local, VECSEQ);

  return status;
}

template <typename DA_Type>
inline PetscErrorCode createMatrix(
    const DA_Type &da,
    Mat &M,
    MatType mtype,
    unsigned int dof)
{
  throw std::logic_error("Not implemented");
}

template <typename T, typename DA_Type>
inline void petscSetVectorByFunction(
    const DA_Type &da,
    Vec &local,
    std::function<void(const T *, T *)> func,
    bool isElemental,
    bool isGhosted,
    unsigned int dof)
{
  PetscScalar * arry = nullptr;
  VecGetArray(local, &arry);
  da.setVectorByFunction(arry, func, isElemental, isGhosted, dof);
  VecRestoreArray(local, &arry);
}

template <typename DA_Type>
inline void petscVecTopvtu(
    const DA_Type &da,
    const Vec &local,
    const char *fPrefix,
    char **nodalVarNames,
    bool isElemental,
    bool isGhosted,
    unsigned int dof)
{
  const PetscScalar *arry = nullptr;
  VecGetArrayRead(local, &arry);
  da.vecTopvtu(arry, fPrefix, nodalVarNames, isElemental, isGhosted, dof);
  VecRestoreArrayRead(local, &arry);
}

template <typename DA_Type>
inline PetscErrorCode petscDestroyVec(const DA_Type &da, Vec & vec)
{
  VecDestroy(&vec);
  vec = NULL;
  return 0;
}
#endif




#endif//DENDRO_KT_DA_P2P_HPP
