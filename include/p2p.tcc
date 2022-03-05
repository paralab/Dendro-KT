/**
 * @author Masado Ishii
 * @date 2022-02-17
 */

namespace par
{

  // P2PPartners::comm()
  MPI_Comm P2PPartners::comm() const
  {
    return m_comm;
  }

  // P2PPartners::comm_size()
  int P2PPartners::comm_size() const
  {
    return m_comm_size;
  }

  // P2PPartners::comm_rank()
  int P2PPartners::comm_rank() const
  {
    return m_comm_rank;
  }

  // P2PPartners::nDest()
  size_t P2PPartners::nDest() const
  {
    return m_dest.size();
  }

  // P2PPartners::nSrc()
  size_t P2PPartners::nSrc()  const
  {
    return m_src.size();
  }

  // P2PPartners::nDest()
  void P2PPartners::nDest(size_t size)
  {
    m_dest.resize(size);
  }

  // P2PPartners::nSrc()
  void P2PPartners::nSrc(size_t size)
  {
    m_src.resize(size);
  }

  // P2PPartners::dest()
  int P2PPartners::dest(size_t i) const
  {
    assert(i < m_dest.size());
    return m_dest[i];
  }

  // P2PPartners::src()
  int P2PPartners::src(size_t i)  const
  {
    assert(i < m_src.size());
    return m_src[i];
  }

  // P2PPartners::dest()
  void P2PPartners::dest(size_t i, int d)
  {
    assert(i < m_dest.size());
    m_dest[i] = d;
  }

  // P2PPartners::src()
  void P2PPartners::src(size_t i, int s)
  {
    assert(i < m_src.size());
    m_src[i] = s;
  }


  // P2PMeta accessors
  MPI_Comm P2PMeta::comm() const    { assert(m_partners != nullptr);  return m_partners->comm(); }
  int P2PMeta::dest(size_t i) const { assert(m_partners != nullptr);  return m_partners->dest(i); }
  int P2PMeta::src(size_t i)  const { assert(m_partners != nullptr);  return m_partners->src(i); }
  int * P2PMeta::send_sizes()   { return &m_send_meta[0]; }
  int * P2PMeta::send_offsets() { return &m_send_meta[m_partners->nDest()]; }
  int * P2PMeta::recv_sizes()   { return &m_recv_meta[0]; }
  int * P2PMeta::recv_offsets() { return &m_recv_meta[m_partners->nSrc()]; }
  int P2PMeta::recv_total() const { return m_recv_total; }
  int P2PMeta::self_size() const { return m_self_size; }
  int P2PMeta::self_offset() const { return m_self_offset; }
  long long unsigned P2PMeta::bytes_sent() const { return m_bytes_sent; }
  long long unsigned P2PMeta::bytes_rcvd() const { return m_bytes_rcvd; }

  // P2PMeta::send()
  template <typename X>
  void P2PMeta::send(const X *send_buffer)
  {
    for (int i = 0; i < m_partners->nDest(); ++i)
    {
      m_requests.emplace_back();
      const auto code = par::Mpi_Isend(
          &send_buffer[send_offsets()[i]], send_sizes()[i],
          dest(i), m_send_tag, comm(), &(*m_requests.back()));
      m_bytes_sent += send_sizes()[i] * sizeof(X);
    }
    ++m_send_tag;
  }

  // P2PMeta::send_dofs()
  template <typename X>
  void P2PMeta::send_dofs(const X *send_buffer, const int ndofs)
  {
    for (int i = 0; i < m_partners->nDest(); ++i)
    {
      m_requests.emplace_back();
      const auto code = par::Mpi_Isend(
          &send_buffer[ndofs * send_offsets()[i]], ndofs * send_sizes()[i],
          dest(i), m_send_tag, comm(), &(*m_requests.back()));
      m_bytes_sent += ndofs * send_sizes()[i] * sizeof(X);
    }
    ++m_send_tag;
  }

  // P2PMeta::recv()
  template <typename X>
  void P2PMeta::recv(X *recv_buffer)
  {
    for (int i = 0; i < m_partners->nSrc(); ++i)
    {
      assert(recv_offsets()[i] < m_recv_total + m_self_size or recv_sizes()[i] == 0);

      const int code = par::Mpi_Recv(
          &recv_buffer[recv_offsets()[i]], recv_sizes()[i],
          src(i), m_recv_tag, comm(), MPI_STATUS_IGNORE);
      m_bytes_rcvd += recv_sizes()[i] * sizeof(X);
    }
    ++m_recv_tag;
  }

  // P2PMeta::recv_dofs()
  template <typename X>
  void P2PMeta::recv_dofs(X *recv_buffer, const int ndofs)
  {
    for (int i = 0; i < m_partners->nSrc(); ++i)
    {
      assert(recv_offsets()[i] < m_recv_total + m_self_size or recv_sizes()[i] == 0);

      const int code = par::Mpi_Recv(
          &recv_buffer[ndofs * recv_offsets()[i]], ndofs * recv_sizes()[i],
          src(i), m_recv_tag, comm(), MPI_STATUS_IGNORE);
      m_bytes_rcvd += ndofs * recv_sizes()[i] * sizeof(X);
    }
    ++m_recv_tag;
  }

}//namespace par
