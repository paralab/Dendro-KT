/**
 * @author Masado Ishii
 * @date 2022-02-17
 */

#include "p2p.h"

namespace par
{
  // P2PPartners::reserve()
  void P2PPartners::reserve(size_t ndest, size_t nsrc)
  {
    assert(ndest >= 0);
    assert(nsrc >= 0);
    m_dest.reserve(ndest);
    m_src.reserve(nsrc);
  }

  // P2PPartners::P2PPartners()
  P2PPartners::P2PPartners(
      const std::vector<int> &dest,
      const std::vector<int> &src,
      MPI_Comm comm)
    : m_dest(dest), m_src(src), m_comm(comm)
  {}

  // P2PPartners::reset()
  void P2PPartners::reset(size_t ndest, size_t nsrc, MPI_Comm comm)
  {
    nDest(ndest);
    nSrc(nsrc);

    m_comm = comm;
    MPI_Comm_size(comm, &m_comm_size);
    MPI_Comm_rank(comm, &m_comm_rank);
  }


  // P2PMeta::reserve()
  void P2PMeta::reserve(int ndest, int nsrc, int layers)
  {
    m_send_meta.reserve(2 * ndest);
    m_recv_meta.reserve(2 * nsrc);
    m_requests.reserve(ndest * layers);
  }

  // P2PMeta::reset()
  void P2PMeta::reset(const P2PPartners *partners)
  {
    m_send_meta.clear();
    m_recv_meta.clear();
    wait_all();

    m_partners = partners;
    m_send_meta.resize(2 * partners->nDest(), 0);
    m_recv_meta.resize(2 * partners->nSrc(), 0);
  }

  // P2PMeta::tally_recvs()
  void P2PMeta::tally_recvs()
  {
    int sum = 0;
    for (int i = 0; i < m_self_pos; ++i) {
      recv_offsets()[i] = sum;
      sum += recv_sizes()[i];
    }
    m_self_offset = sum;
    sum += m_self_size;
    for (int i = m_self_pos; i < m_partners->nSrc(); ++i) {
      recv_offsets()[i] = sum;
      sum += recv_sizes()[i];
    }
    m_recv_total = sum - m_self_size;
  }

  // P2PMeta::wait_all()
  void P2PMeta::wait_all()
  {
    for (Request &request_wrapper : m_requests)
    {
      MPI_Request &request = *request_wrapper;
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    m_requests.clear();
  }

}//namspace par
