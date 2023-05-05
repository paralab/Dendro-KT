/**
 * @author Masado Ishii
 * @date   2023-04-28
 * @brief  Map and methods for pre-computed sending and receiving.
 */

#ifndef DENDRO_KT_GHOST_EXCHANGE_HPP
#define DENDRO_KT_GHOST_EXCHANGE_HPP

// Function declaration for linkage purposes.
inline void link_ghost_exchange_tests() {};

// =============================================================================
// Interfaces
// =============================================================================
namespace par
{
  // MpiRequestVec
  class MpiRequestVec
  {
    public:
      inline MpiRequestVec(size_t n_requests);
      inline MpiRequestVec(MpiRequestVec &&other);
      inline MpiRequestVec & operator=(MpiRequestVec &&other);
      inline ~MpiRequestVec();
      inline size_t size() const;
      inline MPI_Request & operator[](size_t i);
      inline const MPI_Request & operator[](size_t i) const;
      inline MPI_Request * data();
      inline const MPI_Request * data() const;
      inline void wait_all();

    private:
      inline void wipe_out();

    private:
      std::unique_ptr<std::vector<MPI_Request>> m_vec = nullptr;
  };

  class RemoteMap;

  // GhostPullRequest
  class GhostPullRequest
  {
    public:
      template <typename T>
      inline GhostPullRequest(MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs);  // infer local
      template <typename T>
      inline GhostPullRequest(MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs, const T *local);
      inline void wait_on_recv();   // Blocks until ghost is ready.
      inline void wait_on_stage();  // Blocks until local can be overwritten.
      inline void wait_on_send();   // Blocks until request can be deallocated.
      inline void wait_all();

    private:
      MpiRequestVec m_send_reqs;
      MpiRequestVec m_recv_reqs;
      std::shared_ptr<void> m_buffer;  // remembers type; deletes in dtor.
  };

  // GhostPushRequest
  class GhostPushRequest
  {
    public:
      virtual void wait_on_recv() = 0;
      virtual void wait_on_send() = 0;
      virtual void wait_all() = 0;
  };

  // GhostPushRequestTyped
  template <typename T, class Operation>
  class GhostPushRequestTyped : public GhostPushRequest
  {
    public:
      inline GhostPushRequestTyped(
          MPI_Comm comm, T *local, const RemoteMap &map, int ndofs, const T *ghost, Operation op);
      inline GhostPushRequestTyped(
          MPI_Comm comm, const RemoteMap &map, int ndofs, T *ghost, Operation op);  // infer local
      inline void wait_on_recv();
      inline void wait_on_send();
      inline void wait_all();

    private:
      inline void destage();

    private:
      MpiRequestVec m_send_reqs;
      MpiRequestVec m_recv_reqs;
      const RemoteMap *m_map;
      int m_ndofs;
      Operation m_op;
      std::vector<T> m_buffer;
      T *m_dest;
  };


  // ghost_pull()
  template <typename T>
  inline GhostPullRequest ghost_pull(MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs, const T *local);

  // ghost_pull()
  template <typename T>
  inline GhostPullRequest ghost_pull(MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs);  // infer local.

  // ghost_push()
  template <typename T, class Operation>
  inline GhostPushRequestTyped<T, Operation> ghost_push(
      MPI_Comm comm, T *local, const RemoteMap &map, int ndofs, const T *ghost, Operation op);

  // ghost_push()
  template <typename T, class Operation>
  inline GhostPushRequestTyped<T, Operation> ghost_push(
      MPI_Comm comm, const RemoteMap &map, int ndofs, T *ghost, Operation op);  // infer local.


  // RemoteMapData
  struct RemoteMapData
  {
    public:
      // Indexing: Local is between pre-links and post-links.
      // [0 .. n_pre_links) {n_pre_links} n_pre_links+1+[0 .. n_post_links)
      int local_index()        const { return n_pre_links; }
      int link_index(int link) const { return (link >= n_pre_links) + link; }

    public:
      int this_mpi_rank;

      int n_pre_links;
      int n_post_links;

      std::vector<int> mpi_ranks;
      std::vector<size_t> ghost_ranges;
      std::vector<size_t> binding_ranges;

      // Each active bound link gets a segment of increasing local ids.
      std::vector<size_t> bindings;
  };

  // RemoteMap
  class RemoteMap
  {
    public:
      inline int n_links() const;               // Not including self-link.
      inline int mpi_rank(int link) const;
      inline int this_mpi_rank() const;
      inline size_t total_count() const;     // sum of local + ghost
      inline size_t local_begin() const;
      inline size_t local_end() const;
      inline size_t local_count() const;
      inline size_t ghost_begin(int link) const;
      inline size_t ghost_end(int link) const;
      inline size_t ghost_count(int link) const;
      inline size_t local_bindings(int link) const;
      /// inline size_t local_binding(int link, size_t binding_index) const;
      inline size_t local_binding_total() const;
      inline int n_active_ghost_links() const;
      inline int n_active_bound_links() const;

      template <typename Body>  // body(size_t bound_idx, size_t local_id)
      inline void for_bound_local_id(int link, Body &&body) const;

    private:
      friend class RemoteMapBuilder;
      RemoteMap(RemoteMapData internals);
      const RemoteMapData & data() const;
      RemoteMapData m_data;
  };

  /**
   * @title RemoteMapBuilder
   * @brief Normalizes arbitrarily-ordered proto-RemoteMap data.
   */
  class RemoteMapBuilder
  {
    public:
      inline RemoteMapBuilder(int mpi_rank);
      inline RemoteMap finish();

      // Supports mutator chaining.
      inline RemoteMapBuilder & set_local_count(size_t count);
      inline RemoteMapBuilder & bind_local_id(int mpi_rank, size_t local_id);
      inline RemoteMapBuilder & increase_ghost_count(int mpi_rank, size_t count);

    private:
      int m_mpi_rank = 0;
      size_t m_local_count = 0;
      std::set<std::pair<int, size_t>> m_bindings;
      std::map<int, size_t> m_ghosts;
      int m_prev_bound_mpi_rank = -1;
  };


}

// =============================================================================
// Tests
// =============================================================================
#ifdef DOCTEST_LIBRARY_INCLUDED
#include "include/debug.hpp"
namespace par
{
  DOCTEST_TEST_SUITE("MpiRequestVec")
  {
  }


  DOCTEST_TEST_SUITE("Ghost exchange")
  {
    DOCTEST_MPI_TEST_CASE("Sender and receiver agree", 3)
    {
      const int this_mpi_rank = test_rank;
      MPI_Comm comm = test_comm;
      //     .                -     0
      //      \  0            1 -   0 0
      //   1   \_____.        1 1 - 0 | |
      //       /                    |
      //      /   2                 \ 2
      //     .                      \ 2 2
      RemoteMapBuilder builder[3] = {{0}, {1}, {2}};
      builder[0].set_local_count(4u);
      builder[1].set_local_count(3u);
      builder[2].set_local_count(3u);
      builder[0].increase_ghost_count(2, 2u);
      builder[1].increase_ghost_count(0, 3u);
      builder[2].increase_ghost_count(0, 1u)
                .increase_ghost_count(1, 2u);
      builder[0].bind_local_id(1, 0u)
                .bind_local_id(1, 1u)
                .bind_local_id(1, 3u)
                .bind_local_id(2, 0u);
      builder[1].bind_local_id(2, 0u)
                .bind_local_id(2, 1u);
      builder[2].bind_local_id(0, 1u)
                .bind_local_id(0, 2u);

      const RemoteMap map = builder[this_mpi_rank].finish();
      std::vector<int> vec(map.total_count(), 0);
      for (size_t i = map.local_begin(), end = map.local_end(); i < end; ++i)
        vec.at(i) = this_mpi_rank;
      /// GhostPullRequest &&pull = ghost_pull(comm, vec.data(), map);
      /// pull.wait_all();
      ghost_pull(comm, vec.data(), map, 1).wait_all();

      for (size_t i = map.local_begin(), end = map.local_end(); i < end; ++i)
        CHECK( vec.at(i) == this_mpi_rank );
      for (int link = 0, n = map.n_links(); link < n; ++link)
        for (size_t i = map.ghost_begin(link), end = map.ghost_end(link); i < end; ++i)
          CHECK( vec.at(i) == map.mpi_rank(link) );

      /// const auto add = [](auto x, auto y){ return x + y; };
      /// GhostPushRequest &&push = ghost_push(comm, map, vec.data(), add);
    }
  }


  DOCTEST_TEST_SUITE("RemoteMap")
  {
    DOCTEST_TEST_CASE("Empty remote map")
    {
      const RemoteMap map = RemoteMapBuilder(0).finish();
      CHECK( map.n_links() == 0 );
      CHECK( map.local_begin() == 0 );
      CHECK( map.local_end() == 0 );
      CHECK( map.local_count() == 0 );
      CHECK( map.local_binding_total() == 0 );
      CHECK( map.n_active_ghost_links() == 0 );
      CHECK( map.n_active_bound_links() == 0 );
    }

    DOCTEST_TEST_CASE("Remote map with 1 pre-ghost and 1 post-ghost")
    {
      const int pretend_mpi_rank = 5;
      const int lower_mpi_rank = 2;
      const int higher_mpi_rank = 8;
      RemoteMapBuilder builder(pretend_mpi_rank);

      const auto build_map = [&](std::initializer_list<int> events) {
        for (int event: events) switch (event)
        {
          case 0: builder.set_local_count(10);                        break;
          case 1: builder.increase_ghost_count(lower_mpi_rank, 1);    break;
          case 2: builder.increase_ghost_count(higher_mpi_rank, 100); break;
          case 3: builder.bind_local_id(lower_mpi_rank, 0);           break;
          case 4: builder.bind_local_id(lower_mpi_rank, 2);           break;
          case 5: builder.bind_local_id(higher_mpi_rank, 7);          break;
          case 6: builder.bind_local_id(higher_mpi_rank, 9);          break;
        }
      };
      DOCTEST_SUBCASE("")
      { build_map({0, 1, 2, 3, 4, 5, 6}); }
      DOCTEST_SUBCASE("")
      { build_map({6, 5, 2, 1, 0, 4, 3}); }
      DOCTEST_SUBCASE("")
      { build_map({2, 4, 5, 6, 0, 3, 1}); }
      DOCTEST_SUBCASE("")
      { build_map({3, 2, 6, 4, 0, 1, 5}); }
      DOCTEST_SUBCASE("")
      { build_map({4, 5, 3, 0, 1, 6, 2}); }

      const RemoteMap map = builder.finish();

      CHECK( map.n_links() == 2 );
      CHECK( map.mpi_rank(0) == 2 );
      CHECK( map.mpi_rank(1) == 8 );
      CHECK( map.local_count() == 10 );
      CHECK( map.ghost_count(0) == 1 );
      CHECK( map.ghost_count(1) == 100 );

      CHECK( map.ghost_begin(0) == 0 );
      CHECK( map.ghost_end(0) == 1 );
      CHECK( map.local_begin() == 1 );
      CHECK( map.local_end() == 11 );
      CHECK( map.ghost_begin(1) == 11 );
      CHECK( map.ghost_end(1) == 111 );

      CHECK( map.local_binding_total() == 4 );
      CHECK( map.local_bindings(0) == 2 );
      CHECK( map.local_bindings(1) == 2 );

      CHECK( map.n_active_ghost_links() == 2 );
      CHECK( map.n_active_bound_links() == 2 );
    }


    DOCTEST_TEST_CASE("for_bound_local_id() identity")
    {
      //  6 |   7    | 8        Assume this is mpi rank 4,
      // ___|________|___       owning a 10x10 grid, exporting
      //    |        |          its sides and corners to 
      //  3 |   4    | 5        neighboring partitions.
      //    |        |
      // ___|________|___
      //  0 |   1    | 2
      //    |        |
      RemoteMapBuilder builder(4);
      for (size_t i = 0; i < 10; ++i)
      {
        for (size_t j = 0; j < 10; ++j)
        {
          const size_t local_id = 10*i + j;
          if (i == 0 and j == 0)  builder.bind_local_id(0, local_id);
          if (i == 0)             builder.bind_local_id(1, local_id);
          if (i == 0 and j == 9)  builder.bind_local_id(2, local_id);
          if (j == 0)             builder.bind_local_id(3, local_id);
          if (j == 9)             builder.bind_local_id(5, local_id);
          if (i == 9 and j == 0)  builder.bind_local_id(6, local_id);
          if (i == 9)             builder.bind_local_id(7, local_id);
          if (i == 9 and j == 9)  builder.bind_local_id(8, local_id);
        }
      }
      const RemoteMap map = builder.finish();
      CHECK( map.n_links() == 8 );
      CHECK( map.n_active_ghost_links() == 0 );
      CHECK( map.n_active_bound_links() == 8 );
      CHECK( map.local_binding_total() == (4 * 10 + 4 * 1) );
      std::vector<size_t> bindings[8];
      using Vec = std::vector<size_t>;
      for (int link = 0; link < map.n_links(); ++link)
      {
        map.for_bound_local_id(link, [&](size_t idx, size_t local_id) {
          CHECK( idx == bindings[link].size() );
          bindings[link].push_back(local_id);
        });
      }
      using Vec = std::vector<size_t>;
      CHECK( bindings[0] == Vec{0} );
      CHECK( bindings[2] == Vec{9} );
      CHECK( bindings[5] == Vec{90} );   // link index, not mpi rank
      CHECK( bindings[7] == Vec{99} );   //
      CHECK( bindings[1] == Vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9} );
      CHECK( bindings[3] == Vec{0, 10, 20, 30, 40, 50, 60, 70, 80, 90} );
      CHECK( bindings[4] == Vec{9, 19, 29, 39, 49, 59, 69, 79, 89, 99} );
      CHECK( bindings[6] == Vec{90, 91, 92, 93, 94, 95, 96, 97, 98, 99} );
    }
  }
}
#endif// DOCTEST_LIBRARY_INCLUDED


// =============================================================================
// Implementations
// =============================================================================
namespace par
{
  // MpiRequestVec::MpiRequestVec()
  MpiRequestVec::MpiRequestVec(size_t n_requests)
    : m_vec(std::make_unique<std::vector<MPI_Request>>(n_requests, MPI_REQUEST_NULL))
  { }

  // MpiRequestVec::MpiRequestVec()
  MpiRequestVec::MpiRequestVec(MpiRequestVec &&other)
    : m_vec(std::move(other.m_vec))
  { }

  // MpiRequestVec::operator=
  MpiRequestVec & MpiRequestVec::operator=(MpiRequestVec &&other)
  {
    this->wipe_out();
    m_vec = std::move(other.m_vec);
    return *this;
  }

  // MpiRequestVec::~MpiRequestVec()
  MpiRequestVec::~MpiRequestVec()
  {
    this->wipe_out();
  }

  // MpiRequestVec::wipe_out()
  void MpiRequestVec::wipe_out()
  {
    if (m_vec != nullptr)
    {
      for (size_t i = 0, e = this->size(); i < e; ++i)
      {
        assert((*this)[i] == MPI_REQUEST_NULL);
      }
      m_vec->clear();
    }
  }

  // MpiRequestVec::size()
  size_t MpiRequestVec::size() const
  {
    assert(m_vec != nullptr);
    return m_vec->size();
  }

  // MpiRequestVec::operator[]
  MPI_Request & MpiRequestVec::operator[](size_t i)
  {
    assert(m_vec != nullptr);
    return (*m_vec)[i];
  }

  // MpiRequestVec::operator[]
  const MPI_Request & MpiRequestVec::operator[](size_t i) const
  {
    assert(m_vec != nullptr);
    return (*m_vec)[i];
  }

  // MpiRequestVec::data()
  MPI_Request * MpiRequestVec::data()
  {
    assert(m_vec != nullptr);
    return m_vec->data();
  }

  // MpiRequestVec::data()
  const MPI_Request * MpiRequestVec::data() const
  {
    assert(m_vec != nullptr);
    return m_vec->data();
  }

  // MpiRequestVec::wait_all()
  void MpiRequestVec::wait_all()
  {
    assert(m_vec != nullptr);
    Mpi_Waitall(this->size(), this->data());
  }

  // -------------------------------------------------------------------

  // ghost_pull()
  template <typename T>
  GhostPullRequest ghost_pull(MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs, const T *local)
  {
    return { comm, ghost, map, ndofs, local };
  }

  // ghost_pull()
  template <typename T>
  GhostPullRequest ghost_pull(MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs)
  {
    return { comm, ghost, map, ndofs };
  }

  // ghost_push()
  template <typename T, class Operation>
  GhostPushRequestTyped<T, Operation> ghost_push(
      MPI_Comm comm, T *local, const RemoteMap &map, int ndofs, const T *ghost, Operation op)
  {
    return { comm, local, map, ndofs, ghost, std::move(op) };
  }

  // ghost_push()
  template <typename T, class Operation>
  GhostPushRequestTyped<T, Operation> ghost_push(
      MPI_Comm comm, const RemoteMap &map, int ndofs, T *ghost, Operation op)
  {
    return { comm, map, ndofs, ghost, std::move(op) };
  }

  // -------------------------------------------------------------------

  // GhostPullRequest::GhostPullRequest()
  template <typename T>
  GhostPullRequest::GhostPullRequest(
      MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs)
  : GhostPullRequest(comm, ghost, map, ndofs, ghost + ndofs * map.local_begin())
  { }

  // GhostPullRequest::GhostPullRequest()
  template <typename T>
  GhostPullRequest::GhostPullRequest(
      MPI_Comm comm, T *ghost, const RemoteMap &map, int ndofs, const T *local)
    : m_send_reqs(map.n_links()), m_recv_reqs(map.n_links())
  {
    //future: Message pools. Reuse previously allocated pool if done sending.

    if (map.n_active_bound_links() > 0) { assert(local != nullptr); }
    if (map.n_active_ghost_links() > 0)     { assert(ghost != nullptr); }

    const int n_links = map.n_links();

    // Receive ghost.
    for (int i = 0; i < n_links; ++i)
    {
      const size_t begin = ndofs * map.ghost_begin(i);
      const size_t end = ndofs * map.ghost_end(i);
      const size_t count = end - begin;
      if (count > 0)
        Mpi_Irecv( ghost + begin, count,
            map.mpi_rank(i), {}, comm, &m_recv_reqs[i]);
    }

    // Allocate buffer to stage local data.
    const size_t stage_total = ndofs * map.local_binding_total();
    T *local_stage = new T[stage_total];
    m_buffer = std::shared_ptr<void>(local_stage);  // store for deletion.

    // Stage and send local data.
    size_t stage_offset = 0;
    for (int i = 0; i < n_links; ++i)
    {
      const size_t count = ndofs * map.local_bindings(i);
      map.for_bound_local_id(i, [=](size_t bound_idx, size_t local_id)
      {
          bound_idx *= ndofs;
          local_id *= ndofs;
          for (int dof = 0; dof < ndofs; ++dof)
            local_stage[stage_offset + bound_idx + dof] = local[local_id + dof];
      });

      if (count > 0)
        Mpi_Isend( local_stage + stage_offset, count,
            map.mpi_rank(i), {}, comm, &m_send_reqs[i]);
      stage_offset += count;
    }
  }

  // GhostPushRequestTyped::GhostPushRequestTyped()
  template <typename T, class Operation>
  GhostPushRequestTyped<T, Operation>::GhostPushRequestTyped(
      MPI_Comm comm, const RemoteMap &map, int ndofs, T *ghost, Operation op)
    : GhostPushRequestTyped(
        comm, ghost + ndofs * map.local_begin(), map, ndofs, ghost, std::move(op))
  { }

  // GhostPushRequestTyped::GhostPushRequestTyped()
  template <typename T, class Operation>
  GhostPushRequestTyped<T, Operation>::GhostPushRequestTyped(
      MPI_Comm comm, T *local, const RemoteMap &map, int ndofs, const T *ghost, Operation op)
    : m_send_reqs(map.n_links()), m_recv_reqs(map.n_links()),
      m_map(&map),
      m_ndofs(ndofs),
      m_op(std::move(op)),
      m_dest(local)
  {
    if (map.n_active_bound_links() > 0) { assert(local != nullptr); }
    if (map.n_active_ghost_links() > 0)     { assert(ghost != nullptr); }

    const int n_links = map.n_links();

    // Allocate buffer to receive staged local data.
    const size_t stage_total = ndofs * map.local_binding_total();
    m_buffer.resize(stage_total);
    T *local_stage = m_buffer.data();

    // Receive into staged local data.
    size_t stage_offset = 0;
    for (int i = 0; i < n_links; ++i)
    {
      const size_t count = ndofs * map.local_bindings(i);
      if (count > 0)
        Mpi_Irecv( local_stage + stage_offset, count,
            map.mpi_rank(i), {}, comm, &m_recv_reqs[i]);
      stage_offset += count;
    }

    // Send ghost.
    for (int i = 0; i < n_links; ++i)
    {
      const size_t begin = ndofs * map.ghost_begin(i);
      const size_t end = ndofs * map.ghost_end(i);
      const size_t count = end - begin;
      if (count > 0)
        Mpi_Isend( ghost + begin, count,
            map.mpi_rank(i), {}, comm, &m_send_reqs[i]);
    }
  }

  // GhostPushRequestTyped::destage()
  template <typename T, class Operation>
  void GhostPushRequestTyped<T, Operation>::destage()
  {
    const RemoteMap &map = *m_map;
    const int ndofs = m_ndofs;
    const T *local_stage = m_buffer.data();
    T *local = m_dest;

    const int n_links = map.n_links();

    // De-stage to local.  Accumulate with operation op.
    size_t stage_offset = 0;
    for (int i = 0; i < n_links; ++i)
    {
      const size_t count = ndofs * map.local_bindings(i);
      map.for_bound_local_id(i, [=, op=std::move(m_op)](
            size_t bound_idx, size_t local_id)
      {
        bound_idx *= ndofs;
        local_id *= ndofs;
        for (int dof = 0; dof < ndofs; ++dof)
        {
          const T staged_value = local_stage[stage_offset + bound_idx + dof];
          local[local_id + dof] = op(local[local_id + dof], staged_value);
        }
      });
      stage_offset += count;
    }
  }

  // GhostPullRequest::wait_on_recv()
  void GhostPullRequest::wait_on_recv()
  {
    m_recv_reqs.wait_all();
  }

  // GhostPullRequest::wait_on_stage()
  void GhostPullRequest::wait_on_stage()
  {
    // Input vector was copied to an internal buffer, so return immediately.
  }

  // GhostPullRequest::wait_on_send()
  void GhostPullRequest::wait_on_send()
  {
    m_send_reqs.wait_all();
  }

  // GhostPullRequest::wait_all()
  void GhostPullRequest::wait_all()
  {
    this->wait_on_recv();
    this->wait_on_stage();
    this->wait_on_send();
  }

  // GhostPushRequestTyped<T, Operation>::wait_on_recv()
  template <typename T, class Operation>
  void GhostPushRequestTyped<T, Operation>::wait_on_recv()
  {
    m_recv_reqs.wait_all();
    this->destage();
  }

  // GhostPushRequestTyped<T, Operation>::wait_on_send()
  template <typename T, class Operation>
  void GhostPushRequestTyped<T, Operation>::wait_on_send()
  {
    m_send_reqs.wait_all();
  }

  // GhostPushRequestTyped<T, Operation>::wait_all()
  template <typename T, class Operation>
  void GhostPushRequestTyped<T, Operation>::wait_all()
  {
    this->wait_on_recv();
    this->wait_on_send();
  }

  // -------------------------------------------------------------------


  // RemoteMap::RemoteMap()
  RemoteMap::RemoteMap(RemoteMapData internals)
    : m_data(internals)
  { }

  // RemoteMap::RemoteMapData()
  const RemoteMapData & RemoteMap::data() const
  {
    return m_data;
  }

  // RemoteMap::n_links()
  int RemoteMap::n_links() const
  {
    return data().n_pre_links + data().n_post_links;
  }

  // RemoteMap::mpi_rank()
  int RemoteMap::mpi_rank(int link) const
  {
    return data().mpi_ranks[ data().link_index(link) ];
  }

  // RemoteMap::this_mpi_rank()
  int RemoteMap::this_mpi_rank() const
  {
    return data().this_mpi_rank;
  }

  // RemoteMap::ghost_begin()
  size_t RemoteMap::ghost_begin(int link) const
  {
    assert(data().ghost_ranges.size() >= 2);
    return data().ghost_ranges[ data().link_index(link) ];
  }

  // RemoteMap::ghost_end()
  size_t RemoteMap::ghost_end(int link) const
  {
    assert(data().ghost_ranges.size() >= 2);
    return data().ghost_ranges[ data().link_index(link) + 1 ];
  }

  // RemoteMap::ghost_count()
  size_t RemoteMap::ghost_count(int link) const
  {
    assert(data().ghost_ranges.size() >= 2);
    return this->ghost_end(link) - this->ghost_begin(link);
  }

  // RemoteMap::total_count() 
  size_t RemoteMap::total_count() const
  {
    assert(data().ghost_ranges.size() >= 2);
    return data().ghost_ranges.back();
  }

  // RemoteMap::local_begin()
  size_t RemoteMap::local_begin() const
  {
    assert(data().ghost_ranges.size() >= 2);
    return data().ghost_ranges[ data().local_index() ];
  }

  // RemoteMap::local_end()
  size_t RemoteMap::local_end() const
  {
    assert(data().ghost_ranges.size() >= 2);
    return data().ghost_ranges[ data().local_index() + 1 ];
  }

  // RemoteMap::local_count()
  size_t RemoteMap::local_count() const
  {
    assert(data().ghost_ranges.size() >= 2);
    return this->local_end() - this->local_begin();
  }

  // RemoteMap::local_bindings()
  size_t RemoteMap::local_bindings(int link) const
  {
    assert(data().binding_ranges.size() >= 2);
    const int index = data().link_index(link);
    return data().binding_ranges[index + 1] - data().binding_ranges[index];
  }

  // RemoteMap::local_binding_total()
  size_t RemoteMap::local_binding_total() const
  {
    return data().bindings.size();
  }

  // RemoteMap::n_active_ghost_links()
  int RemoteMap::n_active_ghost_links() const
  {
    int active = 0;
    for (int link = 0, n = this->n_links(); link < n; ++link)
      if (this->ghost_begin(link) < this->ghost_end(link))
        ++active;
    return active;
  }

  // RemoteMap::n_active_bound_links()
  int RemoteMap::n_active_bound_links() const
  {
    int active = 0;
    for (int link = 0, n = this->n_links(); link < n; ++link)
      if (this->local_bindings(link) > 0)
        ++active;
    return active;
  }

  // RemoteMap::for_bound_local_id()
  template <typename Body>  // body(size_t bound_idx, size_t local_id)
  void RemoteMap::for_bound_local_id(int link, Body &&body) const
  {
    const int link_index = data().link_index(link);
    const size_t begin = data().binding_ranges[link_index];
    const size_t end = data().binding_ranges[link_index + 1];
    const size_t count = end - begin;
    for (size_t idx = 0; idx < count; ++idx)
    {
      body(idx, data().bindings[begin + idx]);
    }
  }


  // -------------------------------------------------------------------

  // RemoteMapBuilder::RemoteMapBuilder()
  RemoteMapBuilder::RemoteMapBuilder(int mpi_rank)
    : m_mpi_rank(mpi_rank)
  { }

  // RemoteMapBuilder::set_local_count()
  RemoteMapBuilder & RemoteMapBuilder::set_local_count(size_t count)
  {
    m_local_count = count;
    return *this;
  }

  // RemoteMapBuilder::bind_local_id()
  RemoteMapBuilder & RemoteMapBuilder::bind_local_id(int mpi_rank, size_t local_id)
  {
    assert(mpi_rank >= 0);
    m_bindings.insert(m_bindings.end(), {mpi_rank, local_id});
    if (m_prev_bound_mpi_rank != mpi_rank)
    {
      m_ghosts.insert({mpi_rank, 0});  // nop if mpi_rank already inserted.
      m_prev_bound_mpi_rank = mpi_rank;
    }
    return *this;
  }

  // RemoteMapBuilder::increase_ghost_count()
  RemoteMapBuilder & RemoteMapBuilder::increase_ghost_count(int mpi_rank, size_t count)
  {
    m_ghosts[mpi_rank] += count;
    return *this;
  }

  // RemoteMapBuilder::finish()
  RemoteMap RemoteMapBuilder::finish()
  {
    const int n_links = m_ghosts.size();
    const auto ghost_pre_end = m_ghosts.lower_bound(m_mpi_rank);
    const auto ghost_post_begin = m_ghosts.upper_bound(m_mpi_rank);
    assert(ghost_pre_end == ghost_post_begin);  // else, self is ghost!

    // Insert mpi ranks and ghost segment ranges of links (+ local).
    std::vector<int> mpi_ranks;
    std::vector<size_t> ghost_ranges;
    size_t offset = 0;
    ghost_ranges.push_back(offset);

    // Pre-ghost.
    int n_pre_links = 0;
    for (auto it = m_ghosts.begin(); it != ghost_pre_end; ++it, ++n_pre_links)
    {
      const int mpi_rank = it->first;
      const size_t count = it->second;
      mpi_ranks.push_back(mpi_rank);
      offset += count;
      ghost_ranges.push_back(offset);
    }

    // Local.
    mpi_ranks.push_back(m_mpi_rank);
    offset += m_local_count;
    ghost_ranges.push_back(offset);

    // Post-ghost.
    int n_post_links = 0;
    for (auto it = ghost_post_begin; it != m_ghosts.end(); ++it, ++n_post_links)
    {
      const int mpi_rank = it->first;
      const size_t count = it->second;
      mpi_ranks.push_back(mpi_rank);
      offset += count;
      ghost_ranges.push_back(offset);
    }

    // Insert bindings.
    std::vector<size_t> binding_ranges;
    std::vector<size_t> bindings;
    binding_ranges.reserve(ghost_ranges.size());
    bindings.reserve(m_bindings.size());
    auto binding_it = m_bindings.begin();
    const auto binding_end = m_bindings.end();
    binding_ranges.push_back(bindings.size());
    for (size_t i = 0, e = mpi_ranks.size(); i < e; ++i)
    {
      const int mpi_rank = mpi_ranks[i];
      while (binding_it != binding_end and binding_it->first == mpi_rank)
      {
        const size_t local_id = binding_it->second;
        bindings.push_back(local_id);
        ++binding_it;
      }
      binding_ranges.push_back(bindings.size());
    }

    assert((n_pre_links + n_post_links == n_links));
    return RemoteMap( RemoteMapData {
        m_mpi_rank, n_pre_links, n_post_links,
        std::move(mpi_ranks),
        std::move(ghost_ranges),
        std::move(binding_ranges),
        std::move(bindings) } );
  }

  // -------------------------------------------------------------------

};

#endif//DENDRO_KT_GHOST_EXCHANGE_HPP
