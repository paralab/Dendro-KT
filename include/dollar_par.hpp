/**
 * @author Masado Ishii (University of Utah)
 * @date 2022-01-14
 */

#ifndef DENDRO_KT_DOLLAR_PAR_HPP
#define DENDRO_KT_DOLLAR_PAR_HPP

#include <mpi.h>
#include <dollar.hpp>

/**
 * @note On collectives, neither the total number of messages
 *       nor total data transfer are available. Therefore,
 *       a collective is counted as a single exchange,
 *       and the number of bytes is counted by the high-level
 *       arguments.
 */

namespace dollar
{
  namespace detail
  {
    static inline int rank(MPI_Comm comm) { int r; ::MPI_Comm_rank(comm, &r); return r; }
    static inline int size(MPI_Comm comm) { int s; ::MPI_Comm_size(comm, &s); return s; }
    static inline int tsize(MPI_Datatype t) { int s; ::MPI_Type_size(t, &s); return s; }
    static inline int sum(MPI_Comm comm, const int *counts) {
      int t = 0, n = size(comm); while (n-- > 0) { t += *(counts++); } return t;
    }
  }


  inline
  int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                    int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
  {
    int ret = ::MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf,
                    recvcount, recvtype, comm);
    namespace d = detail;
    msg_coll(sendcount                 * d::tsize(sendtype),
             recvcount * d::size(comm) * d::tsize(recvtype));
    return ret;
  }


  inline
  int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                     const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
  {
    int ret = ::MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                     recvcounts, displs, recvtype, comm);
    namespace d = detail;
    msg_coll(sendcount                * d::tsize(sendtype),
             d::sum(comm, recvcounts) * d::tsize(recvtype));
    return ret;
  }


  inline
  int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, MPI_Comm comm)
  {
    int ret = ::MPI_Allreduce(sendbuf, recvbuf, count, datatype,
                    op, comm);
    namespace d = detail;
    msg_coll(count * d::tsize(datatype),
             count * d::tsize(datatype));
    return ret;
  }


  inline
  int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
  {
    int ret = ::MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                   recvcount, recvtype, comm);
    namespace d = detail;
    msg_coll(sendcount * d::size(comm) * d::tsize(sendtype),
             recvcount * d::size(comm) * d::tsize(recvtype));
    return ret;
  }


  inline
  int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls,
                    MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
                    const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
  {
    int ret = ::MPI_Alltoallv(sendbuf, sendcounts, sdispls,
                    sendtype, recvbuf, recvcounts,
                    rdispls, recvtype, comm);
    namespace d = detail;
    msg_coll(d::sum(comm, sendcounts) * d::tsize(sendtype),
             d::sum(comm, recvcounts) * d::tsize(recvtype));
    return ret;
  }


  inline
  int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
  {
    int ret = ::MPI_Bcast(buffer, count, datatype, root, comm);
    namespace d = detail;
    msg_coll((d::rank(comm) == root) ? (count * d::tsize(datatype)) : 0,
                                        count * d::tsize(datatype));
    return ret;
  }


  inline
  int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
  {
    int ret = ::MPI_Gather(sendbuf, sendcount, sendtype, recvbuf,
                 recvcount, recvtype, root, comm);
    namespace d = detail;
    msg_coll(                           sendcount                 * d::tsize(sendtype),
             (d::rank(comm) == root) ? (recvcount * d::size(comm) * d::tsize(recvtype)) : 0);
    return ret;
  }


  inline
  int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root,
                  MPI_Comm comm)
  {
    int ret = ::MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf,
                  recvcounts, displs, recvtype, root, comm);
    namespace d = detail;
    msg_coll(sendcount * d::tsize(sendtype),
             (d::rank(comm) == root) ? (d::sum(comm, recvcounts) * d::tsize(recvtype)) : 0);
    return ret;
  }


  inline
  int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                MPI_Comm comm, MPI_Request *request)
  {
    int ret = ::MPI_Irecv(buf, count, datatype, source, tag, comm, request);
    msg_p2p(0, count * detail::tsize(datatype));
    return ret;
  }

  inline
  int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                MPI_Comm comm, MPI_Request *request)
  {
    int ret = ::MPI_Isend(buf, count, datatype, dest, tag, comm, request);
    msg_p2p(count * detail::tsize(datatype), 0);
    return ret;
  }

  inline
  int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                 MPI_Comm comm, MPI_Request *request)
  {
    int ret = ::MPI_Issend(buf, count, datatype, dest, tag, comm, request);
    msg_p2p(count * detail::tsize(datatype), 0);
    return ret;
  }

  inline
  int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
               MPI_Comm comm, MPI_Status *status)
  {
    int ret = ::MPI_Recv(buf, count, datatype, source, tag, comm, status);
    msg_p2p(0, count * detail::tsize(datatype));
    return ret;
  }

  inline
  int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                 MPI_Op op, int root, MPI_Comm comm)
  {
    int ret = ::MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    namespace d = detail;
    msg_coll(count * d::tsize(datatype), count * d::tsize(datatype));
    return ret;
  }


  inline
  int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
               MPI_Comm comm)
  {
    int ret = ::MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
    namespace d = detail;
    msg_coll(count * d::tsize(datatype), count * d::tsize(datatype));
    return ret;
  }

  inline
  int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm)
  {
    int ret = ::MPI_Send(buf, count, datatype, dest, tag, comm);
    msg_p2p(count * detail::tsize(datatype), 0);
    return ret;
  }

  inline
  int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                   int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int source, int recvtag, MPI_Comm comm, MPI_Status *status)
  {
    int ret = ::MPI_Sendrecv(sendbuf, sendcount, sendtype, dest,
                   sendtag, recvbuf, recvcount, recvtype,
                   source, recvtag, comm, status);
    msg_p2p(sendcount * detail::tsize(sendtype),
            recvcount * detail::tsize(recvtype));
    return ret;
  }
}

#endif//DENDRO_KT_DOLLAR_PAR_HPP
