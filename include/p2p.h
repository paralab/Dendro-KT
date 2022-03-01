/**
 * @author Masado Ishii
 * @date 2022-02-17
 */

#ifndef DENDRO_KT_P2P_H
#define DENDRO_KT_P2P_H

#include <vector>
#include <mpi.h>
#include <assert.h>

#include "parUtils.h"

namespace par
{

  struct P2PPartners
  {
    std::vector<int> m_dest;
    std::vector<int> m_src;
    MPI_Comm m_comm;
    int m_comm_size;
    int m_comm_rank;

    P2PPartners() = default;
    P2PPartners(const std::vector<int> &dest,
                const std::vector<int> &src,
                MPI_Comm comm);

    void reserve(size_t ndest, size_t nsrc);
    void reset(size_t ndest, size_t nsrc, MPI_Comm comm);

    inline MPI_Comm comm() const;
    inline int comm_size() const;
    inline int comm_rank() const;

    inline size_t nDest() const;
    inline size_t nSrc()  const;

    inline void nDest(size_t size);
    inline void nSrc(size_t size) ;

    inline int dest(size_t i) const;
    inline int src(size_t i)  const;

    inline void dest(size_t i, int d);
    inline void src(size_t i, int s);
  };

  template <typename Tag>
  struct P2PRequest  // Wrapper to detect MPI_Request copying. (Shouldn't be.)
  {
    MPI_Request m_request;
    MPI_Request & operator*() { return m_request; }
    P2PRequest() = default;
    P2PRequest(const P2PRequest&) { assert(!"Tried to copy MPI_Request. Try reserve()."); }
    P2PRequest(P2PRequest&&) { assert(!"Tried to move MPI_Request. Try reserve()."); }
    //future: Prevent active MPI_Request's from being copied. Smart pointers?
  };

  // future: refactor into p2p.tcc and p2p.cpp
  template <typename ScalarT = int, int LEN = 1>
  struct P2PScalar
  {
    using Request = P2PRequest<P2PScalar>;
    static_assert(LEN >= 0, "Number of scalars cannot be negative.");

    const P2PPartners *m_partners = nullptr;
    MPI_Comm comm() const    { assert(m_partners != nullptr);  return m_partners->comm(); }
    int dest(size_t i) const { assert(m_partners != nullptr);  return m_partners->dest(i); }
    int src(size_t i)  const { assert(m_partners != nullptr);  return m_partners->src(i); }

    std::vector<ScalarT> m_sendScalar;
    std::vector<ScalarT> m_recvScalar;
    std::vector<Request> m_requests;
    std::vector<char> m_sent;
    std::vector<char> m_rcvd;

    void reserve(int ndest, int nsrc)
    {
      assert(ndest >= 0);
      assert(nsrc >= 0);
      m_sendScalar.reserve(LEN * ndest);
      m_recvScalar.reserve(LEN * nsrc);
      m_requests.reserve(ndest);
      m_sent.reserve(ndest);
      m_rcvd.reserve(nsrc);
    }

    P2PScalar() = default;
    P2PScalar(const P2PPartners *partners) { reset(partners); }

    void reset(const P2PPartners *partners)
    {
      assert(partners != nullptr);
      m_partners = partners;
      m_sendScalar.resize(LEN * partners->nDest());
      m_recvScalar.resize(LEN * partners->nSrc());
      m_requests.resize(partners->nDest());
      assert(std::find(m_sent.begin(), m_sent.end(), false) == m_sent.end());
      assert(std::find(m_rcvd.begin(), m_rcvd.end(), false) == m_rcvd.end());
      m_sent.clear();  m_sent.resize(partners->nDest(), false);
      m_rcvd.clear();  m_rcvd.resize(partners->nSrc(), false);
    }

    template <typename...X>
    void send(int destIdx, const X&...scalars) {
      static_assert(sizeof...(X) == LEN, "Must send all LEN scalars in a single call to send().");
      assert(destIdx < m_partners->nDest());
      assert(not m_sent[destIdx]);

      ScalarT *stage = &m_sendScalar[destIdx * LEN];
      DENDRO_FOR_PACK( *(stage++) = scalars );
      par::Mpi_Isend(&(m_sendScalar[destIdx * LEN]), LEN, dest(destIdx), 0, comm(), &(*m_requests[destIdx]));
      m_sent[destIdx] = true;
    }

    template <typename...X>
    void recv_all(int srcIdx, X&...scalars) {
      assert(srcIdx < m_partners->nSrc());

      MPI_Status status;
      if (not m_rcvd[srcIdx])
        par::Mpi_Recv(&(m_recvScalar[srcIdx * LEN]), LEN, src(srcIdx), 0, comm(), &status);
      m_rcvd[srcIdx] = true;
      ScalarT *stage = &(m_recvScalar[srcIdx * LEN]);
      DENDRO_FOR_PACK( scalars = *(stage++) );
    }

    template <int IDX = 0>
    ScalarT recv(int srcIdx) {
      recv_all(srcIdx);
      return m_recvScalar[srcIdx * LEN + IDX];
    }

    void wait_all() {
      for (Request &request_wrapper : m_requests)
      {
        MPI_Request &request = *request_wrapper;
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
      m_requests.clear();
    }
  };


  // future: refactor into p2p.tcc and p2p.cpp
  template <typename X>
  struct P2PVector
  {
    const std::vector<int> &m_dest;
    const std::vector<int> &m_src;
    MPI_Comm m_comm;

    std::vector<MPI_Request> m_requests;

    P2PVector(const P2PPartners *partners)
      : m_dest(partners->m_dest), m_src(partners->m_src), m_comm(partners->m_comm),
        m_requests(partners->nDest())
    { }

    void send(int destIdx, const std::vector<X> &vector) {
      par::Mpi_Isend(&(*vector.cbegin()), vector.size(), m_dest[destIdx], 0, m_comm, &m_requests[destIdx]);
    }

    void recv(int srcIdx, std::vector<X> &vector) {
      MPI_Status status;
      par::Mpi_Recv(&(*vector.begin()), vector.size(), m_src[srcIdx], 0, m_comm, &status);
    }

    void wait_all() {
      for (MPI_Request &request : m_requests)
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
  };


  // P2PMeta
  struct P2PMeta
  {
    const P2PPartners *m_partners = nullptr;

    using Request = P2PRequest<P2PMeta>;

    std::vector<int> m_send_meta;
    std::vector<int> m_recv_meta;
    std::vector<Request> m_requests;  // size == # of sends since wait_all()
    int m_recv_total = 0;  // Does not include self segment count.

    int m_self_size = 0;
    int m_self_offset = 0;
    int m_self_pos = 0;  // Self is between (m_self_pos-1) and (m_self_pos).

    int m_send_tag = 0;
    int m_recv_tag = 0;

    long long unsigned m_bytes_sent = 0;
    long long unsigned m_bytes_rcvd = 0;

    inline MPI_Comm comm() const;
    inline int dest(size_t i) const;
    inline int src(size_t i)  const;

    inline int * send_sizes();
    inline int * send_offsets();
    inline int * recv_sizes();
    inline int * recv_offsets();

    inline int recv_total() const;
    inline int self_size() const;
    inline int self_offset() const;

    inline long long unsigned bytes_sent() const;
    inline long long unsigned bytes_rcvd() const;

    void reserve(int ndest, int nsrc, int layers = 1);

    P2PMeta() = default;
    P2PMeta(const P2PPartners *partners, int layers = 1) {
      reset(partners);
      reserve(partners->nDest(), partners->nSrc(), layers);
    }

    void reset(const P2PPartners *partners);

    // Usage: schedule_send(), recv_size() ... tally_recvs(), send(), recv()

    void schedule_send(int destIdx, int size, int offset) {
      send_sizes()[destIdx] = size;
      send_offsets()[destIdx] = offset;
    }

    void recv_size(int srcIdx, int size) {
      recv_sizes()[srcIdx] = size;
    }

    void self_size(int srcIdx, int size) {
      m_self_pos = srcIdx;
      m_self_size = size;
    }

    void tally_recvs();

    template <typename X>
    void send(const X *send_buffer);

    template <typename X>
    void send_dofs(const X *send_buffer, const int ndofs);

    template <typename X>
    void recv(X *recv_buffer);

    template <typename X>
    void recv_dofs(X *recv_buffer, const int ndofs);

    void wait_all();
  };

}//namespace par

#include "p2p.tcc"

#endif//DENDRO_KT_P2P_H
