/**
  @file parUtils.txx
  @brief Definitions of the templated functions in the par module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
 */

#include "binUtils.h"
#include "seqUtils.h"
#include "dtypes.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include "dendro.h"
#include "ompUtils.h"
#include <chrono>
#include <assert.h>
#include <memory>


#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

#define MILLISECOND_CONVERSION 1e3

namespace par {

  inline int mpi_comm_size(MPI_Comm comm)
  {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    return comm_size;
  }

  inline int mpi_comm_rank(MPI_Comm comm)
  {
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    return comm_rank;
  }

  template<typename T>
  inline int Mpi_Isend(const T *buf, int count, int dest, int tag,
                       MPI_Comm comm, MPI_Request *request) {

    return MPI_Isend(buf, count, par::Mpi_datatype<T>::value(),
              dest, tag, comm, request);

  }

  template<typename T>
  inline int Mpi_Issend(const T *buf, int count, int dest, int tag,
                        MPI_Comm comm, MPI_Request *request) {

    return MPI_Issend(buf, count, par::Mpi_datatype<T>::value(),
               dest, tag, comm, request);

  }

  template<typename T>
  inline int Mpi_Recv(T *buf, int count, int source, int tag,
                      MPI_Comm comm, MPI_Status *status) {

    return MPI_Recv(buf, count, par::Mpi_datatype<T>::value(),
             source, tag, comm, status);

  }

  template<typename T>
  inline int Mpi_Irecv(T *buf, int count, int source, int tag,
                       MPI_Comm comm, MPI_Request *request) {

    return MPI_Irecv(buf, count, par::Mpi_datatype<T>::value(),
              source, tag, comm, request);

  }

  template <typename T>
  inline int Mpi_Mrecv(T* buf, int count, MPI_Message *message, MPI_Status *status)
  {
    return MPI_Mrecv(buf, count, Mpi_datatype<T>::value(), message, status);
  }

  template <typename T>
  inline int Mpi_Imrecv(T* buf, int count, MPI_Message *message, MPI_Request* request)
  {
    return MPI_Imrecv(buf, count, Mpi_datatype<T>::value(), message, request);
  }


  template<typename T, typename S>
  inline int Mpi_Sendrecv(const T *sendBuf, int sendCount, int dest, int sendTag,
                          S *recvBuf, int recvCount, int source, int recvTag,
                          MPI_Comm comm, MPI_Status *status) {
    PROF_PAR_SENDRECV_BEGIN

    MPI_Sendrecv(sendBuf, sendCount, par::Mpi_datatype<T>::value(), dest, sendTag,
                 recvBuf, recvCount, par::Mpi_datatype<S>::value(), source, recvTag, comm, status);

    PROF_PAR_SENDRECV_END
  }

  template<typename T>
  inline int Mpi_Scan(const T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    // PROF_PAR_SCAN_BEGIN

    MPI_Scan(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);

    // PROF_PAR_SCAN_END
  }

  template<typename T>
  inline int Mpi_Exscan(const T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_SCAN_BEGIN

    MPI_Exscan(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);

    PROF_PAR_SCAN_END
  }


  template<typename T>
  inline int Mpi_Allreduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_ALLREDUCE_BEGIN

    MPI_Allreduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);

    PROF_PAR_ALLREDUCE_END
  }

  template<typename T>
  inline int Mpi_Alltoall(const T *sendbuf, T *recvbuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_ALL2ALL_BEGIN

    MPI_Alltoall(sendbuf, count, par::Mpi_datatype<T>::value(),
                 recvbuf, count, par::Mpi_datatype<T>::value(), comm);

    PROF_PAR_ALL2ALL_END
  }

  template<typename T>
  inline int Mpi_Alltoallv
      (const T *sendbuf, int *sendcnts, int *sdispls,
       T *recvbuf, int *recvcnts, int *rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif

    MPI_Alltoallv(
        sendbuf, sendcnts, sdispls, par::Mpi_datatype<T>::value(),
        recvbuf, recvcnts, rdispls, par::Mpi_datatype<T>::value(),
        comm);
    return 0;
  }

  template<typename T>
  inline int Mpi_Gather(const T *sendBuffer, T *recvBuffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_GATHER_BEGIN

    MPI_Gather(sendBuffer, count, par::Mpi_datatype<T>::value(),
               recvBuffer, count, par::Mpi_datatype<T>::value(), root, comm);

    PROF_PAR_GATHER_END
  }

  template<typename T>
  inline int Mpi_Bcast(T *buffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_BCAST_BEGIN

    MPI_Bcast(buffer, count, par::Mpi_datatype<T>::value(), root, comm);

    PROF_PAR_BCAST_END
  }

  template<typename T>
  inline int Mpi_Reduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_REDUCE_BEGIN

    MPI_Reduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, root, comm);

    PROF_PAR_REDUCE_END
  }

  template<typename T>
  int Mpi_Allgatherv(const T *sendBuf, int sendCount, T *recvBuf,
                     int *recvCounts, int *displs, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_ALLGATHERV_BEGIN

#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

    int maxSendCount;
  int npes, rank;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  par::Mpi_Allreduce<int>(&sendCount, &maxSendCount, 1, MPI_MAX, comm);

  T* dummySendBuf = new T[maxSendCount*npes];
  assert(dummySendBuf);

#pragma omp parallel for
  for(int i = 0; i < npes; i++) {
    for(int j = 0; j < sendCount; j++) {
      dummySendBuf[(i*maxSendCount) + j] = sendBuf[j];
    }
  }

  T* dummyRecvBuf = new T[maxSendCount*npes];
  assert(dummyRecvBuf);

  par::Mpi_Alltoall<T>(dummySendBuf, dummyRecvBuf, maxSendCount, comm);

#pragma omp parallel for
  for(int i = 0; i < npes; i++) {
    for(int j = 0; j < recvCounts[i]; j++) {
      recvBuf[displs[i] + j] = dummyRecvBuf[(i*maxSendCount) + j];
    }
  }

  delete [] dummySendBuf;
  delete [] dummyRecvBuf;

#else

    MPI_Allgatherv(sendBuf, sendCount, par::Mpi_datatype<T>::value(),
                   recvBuf, recvCounts, displs, par::Mpi_datatype<T>::value(), comm);

#endif

    PROF_PAR_ALLGATHERV_END
  }

  /** @author Masado Ishii */
  template <typename T>
  int Mpi_Iallgather(const T* sendbuf, T* recvbuf, int count,
        MPI_Comm comm, MPI_Request *request)
  {
    return MPI_Iallgather(sendbuf, count, Mpi_datatype<T>::value(),
                          recvbuf, count, Mpi_datatype<T>::value(),
                          comm, request);
  }





  template<typename T>
  int Mpi_Allgather(const T *sendBuf, T *recvBuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_ALLGATHER_BEGIN

#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

    int npes;
  MPI_Comm_size(comm, &npes);
  T* dummySendBuf = new T[count*npes];
  assert(dummySendBuf);
#pragma omp parallel for
  for(int i = 0; i < npes; i++) {
    for(int j = 0; j < count; j++) {
      dummySendBuf[(i*count) + j] = sendBuf[j];
    }
  }
  par::Mpi_Alltoall<T>(dummySendBuf, recvBuf, count, comm);
  delete [] dummySendBuf;

#else

    MPI_Allgather(sendBuf, count, par::Mpi_datatype<T>::value(),
                  recvBuf, count, par::Mpi_datatype<T>::value(), comm);

#endif

    PROF_PAR_ALLGATHER_END
  }

  template<typename T>
  int Mpi_Alltoallv_sparse(const T *sendbuf, int *sendcnts, int *sdispls,
                           T *recvbuf, int *recvcnts, int *rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_ALL2ALLV_SPARSE_BEGIN

/*#ifndef ALLTOALLV_FIX
    Mpi_Alltoallv
        (sendbuf, sendcnts, sdispls,
         recvbuf, recvcnts, rdispls, comm);
#else*/

    int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  int commCnt = 0;

#pragma omp parallel for reduction(+:commCnt)
  for(int i = 0; i < rank; i++) {
    if(sendcnts[i] > 0) {
      commCnt++;
    }
    if(recvcnts[i] > 0) {
      commCnt++;
    }
  }

#pragma omp parallel for reduction(+:commCnt)
  for(int i = (rank+1); i < npes; i++) {
    if(sendcnts[i] > 0) {
      commCnt++;
    }
    if(recvcnts[i] > 0) {
      commCnt++;
    }
  }

  MPI_Request* requests = new MPI_Request[commCnt];
  assert(requests);

  MPI_Status* statuses = new MPI_Status[commCnt];
  assert(statuses);

  commCnt = 0;

  //First place all recv requests. Do not recv from self.
  for(int i = 0; i < rank; i++) {
    if(recvcnts[i] > 0) {
      par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
          comm, &(requests[commCnt]) );
      commCnt++;
    }
  }

  for(int i = (rank + 1); i < npes; i++) {
    if(recvcnts[i] > 0) {
      par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
          comm, &(requests[commCnt]) );
      commCnt++;
    }
  }

  //Next send the messages. Do not send to self.
  for(int i = 0; i < rank; i++) {
    if(sendcnts[i] > 0) {
      par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], i, 1,
          comm, &(requests[commCnt]) );
      commCnt++;
    }
  }

  for(int i = (rank + 1); i < npes; i++) {
    if(sendcnts[i] > 0) {
      par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i],
          i, 1, comm, &(requests[commCnt]) );
      commCnt++;
    }
  }

  //Now copy local portion.
#ifdef __DEBUG_PAR__
  assert(sendcnts[rank] == recvcnts[rank]);
#endif

#pragma omp parallel for
  for(int i = 0; i < sendcnts[rank]; i++) {
    recvbuf[rdispls[rank] + i] = sendbuf[sdispls[rank] + i];
  }

  PROF_A2AV_WAIT_BEGIN

    MPI_Waitall(commCnt, requests, statuses);

  PROF_A2AV_WAIT_END

  delete [] requests;
  delete [] statuses;
//#endif

    PROF_PAR_ALL2ALLV_SPARSE_END
  }

  template<typename T>
  int Mpi_Alltoallv_dense(const T *sendbuf, int *sendcnts, int *sdispls,
                          T *recvbuf, int *recvcnts, int *rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_ALL2ALLV_DENSE_BEGIN

#ifndef ALLTOALLV_FIX
    Mpi_Alltoallv
        (sendbuf, sendcnts, sdispls,
         recvbuf, recvcnts, rdispls, comm);
#else
    int npes, rank;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    //Processors may send a lot of information to themselves and a lesser
    //amount to others. If so, we don't want to waste communication by
    //including the local copy size in the max message size.
    int maxNumElemSend = 0;
    for(int i = 0; i < rank; i++) {
      if(sendcnts[i] > maxNumElemSend) {
        maxNumElemSend = sendcnts[i];
      }
    }

    for(int i = (rank + 1); i < npes; i++) {
      if(sendcnts[i] > maxNumElemSend) {
        maxNumElemSend = sendcnts[i];
      }
    }

    int allToAllCount;
    par::Mpi_Allreduce<int>(&maxNumElemSend, &allToAllCount, 1, MPI_MAX, comm);

    T* tmpSendBuf = new T[allToAllCount*npes];
    assert(tmpSendBuf);

    T* tmpRecvBuf = new T[allToAllCount*npes];
    assert(tmpRecvBuf);

    for(int i = 0; i < rank; i++) {
      for(int j = 0; j < sendcnts[i]; j++) {
        tmpSendBuf[(allToAllCount*i) + j] = sendbuf[sdispls[i] + j];
      }
    }

    for(int i = (rank + 1); i < npes; i++) {
      for(int j = 0; j < sendcnts[i]; j++) {
        tmpSendBuf[(allToAllCount*i) + j] = sendbuf[sdispls[i] + j];
      }
    }

    par::Mpi_Alltoall<T>(tmpSendBuf, tmpRecvBuf, allToAllCount, comm);

    for(int i = 0; i < rank; i++) {
      for(int j = 0; j < recvcnts[i]; j++) {
        recvbuf[rdispls[i] + j] = tmpRecvBuf[(allToAllCount*i) + j];
      }
    }

    //Now copy local portion.
#ifdef __DEBUG_PAR__
    assert(sendcnts[rank] == recvcnts[rank]);
#endif

    for(int j = 0; j < recvcnts[rank]; j++) {
      recvbuf[rdispls[rank] + j] = sendbuf[sdispls[rank] + j];
    }

    for(int i = (rank + 1); i < npes; i++) {
      for(int j = 0; j < recvcnts[i]; j++) {
        recvbuf[rdispls[i] + j] = tmpRecvBuf[(allToAllCount*i) + j];
      }
    }

    delete [] tmpSendBuf;
    delete [] tmpRecvBuf;
#endif

    PROF_PAR_ALL2ALLV_DENSE_END
  }





    template <typename T>
    int Mpi_Alltoallv_Kway(const T* sbuff_, int* s_cnt_, int* sdisp_,
                            T* rbuff_, int* r_cnt_, int* rdisp_, MPI_Comm c){

        //std::vector<double> tt(4096*200,0);
        //std::vector<double> tt_wait(4096*200,0);

#ifdef __PROFILE_WITH_BARRIER__
        MPI_Barrier(comm);
#endif
        PROF_PAR_ALL2ALLV_DENSE_BEGIN

#ifndef ALLTOALLV_FIX
        Mpi_Alltoallv
                (sbuff_, s_cnt_, sdisp_,
                 rbuff_, r_cnt_, rdisp_, c);
#else

  int kway = KWAY;
  int np, pid;
  MPI_Comm_size(c, &np);
  MPI_Comm_rank(c, &pid);
  //std::cout<<" Kway: "<<kway<<std::endl;
  int range[2]={0,np};
  int split_id, partner;

  std::vector<int> s_cnt(np);
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    s_cnt[i]=s_cnt_[i]*sizeof(T)+2*sizeof(int);
  }
  std::vector<int> sdisp(np); sdisp[0]=0;
  omp_par::scan(&s_cnt[0],&sdisp[0],np);

  char* sbuff=new char[sdisp[np-1]+s_cnt[np-1]];
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    ((int*)&sbuff[sdisp[i]])[0]=s_cnt[i];
    ((int*)&sbuff[sdisp[i]])[1]=pid;
    memcpy(&sbuff[sdisp[i]]+2*sizeof(int),&sbuff_[sdisp_[i]],s_cnt[i]-2*sizeof(int));
  }

  //int t_indx=0;
  int iter_cnt=0;
  while(range[1]-range[0]>1){
    iter_cnt++;
    if(kway>range[1]-range[0])
      kway=range[1]-range[0];

    std::vector<int> new_range(kway+1);
    for(int i=0;i<=kway;i++)
      new_range[i]=(range[0]*(kway-i)+range[1]*i)/kway;
    int p_class=(std::upper_bound(&new_range[0],&new_range[kway],pid)-&new_range[0]-1);
    int new_np=new_range[p_class+1]-new_range[p_class];
    int new_pid=pid-new_range[p_class];

    //Communication.
    {
      std::vector<int> r_cnt    (new_np*kway, 0);
      std::vector<int> r_cnt_ext(new_np*kway, 0);
      //Exchange send sizes.
      for(int i=0;i<kway;i++){
        MPI_Status status;
        int cmp_np=new_range[i+1]-new_range[i];
        int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;
        assert(     (new_pid<cmp_np? true: new_range[i]+new_pid==new_range[i+1]  )); //Remove this.
        MPI_Sendrecv(&s_cnt[new_range[i]-new_range[0]], cmp_np, MPI_INT, partner, 0,
                     &r_cnt[new_np   *i ], new_np, MPI_INT, partner, 0, c, &status);

        //Handle extra communication.
        if(new_pid==new_np-1 && cmp_np>new_np){
          int partner=new_range[i+1]-1;
          std::vector<int> s_cnt_ext(cmp_np, 0);
          MPI_Sendrecv(&s_cnt_ext[       0], cmp_np, MPI_INT, partner, 0,
                       &r_cnt_ext[new_np*i], new_np, MPI_INT, partner, 0, c, &status);
        }
      }

      //Allocate receive buffer.
      std::vector<int> rdisp    (new_np*kway, 0);
      std::vector<int> rdisp_ext(new_np*kway, 0);
      int rbuff_size, rbuff_size_ext;
      char *rbuff, *rbuff_ext;
      {
        omp_par::scan(&r_cnt    [0], &rdisp    [0],new_np*kway);
        omp_par::scan(&r_cnt_ext[0], &rdisp_ext[0],new_np*kway);
        rbuff_size     = rdisp    [new_np*kway-1] + r_cnt    [new_np*kway-1];
        rbuff_size_ext = rdisp_ext[new_np*kway-1] + r_cnt_ext[new_np*kway-1];
        rbuff     = new char[rbuff_size    ];
        rbuff_ext = new char[rbuff_size_ext];
      }

      //Sendrecv data.
      //*
      int my_block=kway;
      while(pid<new_range[my_block]) my_block--;
//      MPI_Barrier(c);
      for(int i_=0;i_<=kway/2;i_++){
        int i1=(my_block+i_)%kway;
        int i2=(my_block+kway-i_)%kway;

        for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
          int i=(i_==0?i1:((j+my_block/i_)%2?i1:i2));
          MPI_Status status;
          int cmp_np=new_range[i+1]-new_range[i];
          int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;

          int send_dsp     =sdisp[new_range[i  ]-new_range[0]  ];
          int send_dsp_last=sdisp[new_range[i+1]-new_range[0]-1];
          int send_cnt     =s_cnt[new_range[i+1]-new_range[0]-1]+send_dsp_last-send_dsp;

//          double ttt=omp_get_wtime();
//          MPI_Sendrecv(&sbuff[send_dsp], send_cnt>0?1:0, MPI_BYTE, partner, 0,
//                       &rbuff[rdisp[new_np  * i ]], (r_cnt[new_np  *(i+1)-1]+rdisp[new_np  *(i+1)-1]-rdisp[new_np  * i ])>0?1:0, MPI_BYTE, partner, 0, c, &status);
//          tt_wait[200*pid+t_indx]=omp_get_wtime()-ttt;
//
//          ttt=omp_get_wtime();
          MPI_Sendrecv(&sbuff[send_dsp], send_cnt, MPI_BYTE, partner, 0,
                       &rbuff[rdisp[new_np  * i ]], r_cnt[new_np  *(i+1)-1]+rdisp[new_np  *(i+1)-1]-rdisp[new_np  * i ], MPI_BYTE, partner, 0, c, &status);
//          tt[200*pid+t_indx]=omp_get_wtime()-ttt;
//          t_indx++;

          //Handle extra communication.
          if(pid==new_np-1 && cmp_np>new_np){
            int partner=new_range[i+1]-1;
            std::vector<int> s_cnt_ext(cmp_np, 0);
            MPI_Sendrecv(                       NULL,                                                                       0, MPI_BYTE, partner, 0,
                         &rbuff[rdisp_ext[new_np*i]], r_cnt_ext[new_np*(i+1)-1]+rdisp_ext[new_np*(i+1)-1]-rdisp_ext[new_np*i], MPI_BYTE, partner, 0, c, &status);
          }
        }
      }
      /*/
      {
        MPI_Request* requests = new MPI_Request[4*kway];
        MPI_Status * statuses = new MPI_Status[4*kway];
        int commCnt=0;
        for(int i=0;i<kway;i++){
          MPI_Status status;
          int cmp_np=new_range[i+1]-new_range[i];
          int partner=              new_range[i]+new_pid;
          if(partner<new_range[i+1]){
            MPI_Irecv(&rbuff    [rdisp    [new_np*i]], r_cnt    [new_np*(i+1)-1]+rdisp    [new_np*(i+1)-1]-rdisp    [new_np*i ], MPI_BYTE, partner, 0, c, &requests[commCnt]); commCnt++;
          }

          //Handle extra recv.
          if(new_pid==new_np-1 && cmp_np>new_np){
            int partner=new_range[i+1]-1;
            MPI_Irecv(&rbuff_ext[rdisp_ext[new_np*i]], r_cnt_ext[new_np*(i+1)-1]+rdisp_ext[new_np*(i+1)-1]-rdisp_ext[new_np*i ], MPI_BYTE, partner, 0, c, &requests[commCnt]); commCnt++;
          }
        }
        for(int i=0;i<kway;i++){
          MPI_Status status;
          int cmp_np=new_range[i+1]-new_range[i];
          int partner=(new_pid<cmp_np?  new_range[i]+new_pid: new_range[i+1]-1);
          int send_dsp     =sdisp[new_range[i  ]-new_range[0]  ];
          int send_dsp_last=sdisp[new_range[i+1]-new_range[0]-1];
          int send_cnt     =s_cnt[new_range[i+1]-new_range[0]-1]+send_dsp_last-send_dsp;
          MPI_Issend (&sbuff[send_dsp], send_cnt, MPI_BYTE, partner, 0, c, &requests[commCnt]); commCnt++;
        }
        MPI_Waitall(commCnt, requests, statuses);
        delete[] requests;
        delete[] statuses;
      }// */

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      //Rearrange received data.
      {
        if(sbuff!=NULL) delete[] sbuff;
        sbuff=new char[rbuff_size+rbuff_size_ext];

        std::vector<int>  cnt_new(2*new_np*kway, 0);
        std::vector<int> disp_new(2*new_np*kway, 0);
        for(int i=0;i<new_np;i++)
        for(int j=0;j<kway;j++){
          cnt_new[(i*2  )*kway+j]=r_cnt    [j*new_np+i];
          cnt_new[(i*2+1)*kway+j]=r_cnt_ext[j*new_np+i];
        }
        omp_par::scan(&cnt_new[0], &disp_new[0],2*new_np*kway);

        #pragma omp parallel for
        for(int i=0;i<new_np;i++)
        for(int j=0;j<kway;j++){
          memcpy(&sbuff[disp_new[(i*2  )*kway+j]], &rbuff    [rdisp    [j*new_np+i]], r_cnt    [j*new_np+i]);
          memcpy(&sbuff[disp_new[(i*2+1)*kway+j]], &rbuff_ext[rdisp_ext[j*new_np+i]], r_cnt_ext[j*new_np+i]);
        }

        //Free memory.
        if(rbuff    !=NULL) delete[] rbuff    ;
        if(rbuff_ext!=NULL) delete[] rbuff_ext;

        s_cnt.clear();
        s_cnt.resize(new_np,0);
        sdisp.resize(new_np);
        for(int i=0;i<new_np;i++){
          for(int j=0;j<2*kway;j++)
            s_cnt[i]+=cnt_new[i*2*kway+j];
          sdisp[i]=disp_new[i*2*kway];
        }
      }

/*
      //Rearrange received data.
      {
        int * s_cnt_old=&s_cnt[new_range[0]-range[0]];
        int * sdisp_old=&sdisp[new_range[0]-range[0]];

        std::vector<int> s_cnt_new(&s_cnt_old[0],&s_cnt_old[new_np]);
        std::vector<int> sdisp_new(new_np       ,0                 );
        #pragma omp parallel for
        for(int i=0;i<new_np;i++){
          s_cnt_new[i]+=r_cnt[i];
        }
        omp_par::scan(&s_cnt_new[0],&sdisp_new[0],new_np);

        //Copy data to sbuff_new.
        char* sbuff_new=new char[sdisp_new[new_np-1]+s_cnt_new[new_np-1]];
        #pragma omp parallel for
        for(int i=0;i<new_np;i++){
          memcpy(&sbuff_new[sdisp_new[i]                      ],&sbuff   [sdisp_old[i]],s_cnt_old[i]);
          memcpy(&sbuff_new[sdisp_new[i]+s_cnt_old[i]         ],&rbuff   [rdisp    [i]],r_cnt    [i]);
        }

        //Free memory.
        if(sbuff   !=NULL) delete[] sbuff   ;
        if(rbuff   !=NULL) delete[] rbuff   ;

        //Substitute data for next iteration.
        s_cnt=s_cnt_new;
        sdisp=sdisp_new;
        sbuff=sbuff_new;
      }*/
    }

    range[0]=new_range[p_class  ];
    range[1]=new_range[p_class+1];
    //range[0]=new_range[0];
    //range[1]=new_range[1];
  }

  //Copy data to rbuff_.
  std::vector<char*> buff_ptr(np);
  char* tmp_ptr=sbuff;
  for(int i=0;i<np;i++){
    int& blk_size=((int*)tmp_ptr)[0];
    buff_ptr[i]=tmp_ptr;
    tmp_ptr+=blk_size;
  }
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    int& blk_size=((int*)buff_ptr[i])[0];
    int& src_pid=((int*)buff_ptr[i])[1];
    assert(blk_size-2*sizeof(int)<=r_cnt_[src_pid]*sizeof(T));
    memcpy(&rbuff_[rdisp_[src_pid]],buff_ptr[i]+2*sizeof(int),blk_size-2*sizeof(int));
  }
/*
  std::vector<double> tt_sum(4096*200,0);
  std::vector<double> tt_wait_sum(4096*200,0);
  MPI_Reduce(&tt[0], &tt_sum[0], 4096*200, MPI_DOUBLE, MPI_SUM, 0, c);
  MPI_Reduce(&tt_wait[0], &tt_wait_sum[0], 4096*200, MPI_DOUBLE, MPI_SUM, 0, c);

#define MAX_PROCS 4096

  if(np==MAX_PROCS){
    if(!pid) std::cout<<"Tw=[";
    size_t j=0;
    for(size_t i=0;i<200;i++){
      for(j=0;j<MAX_PROCS;j++)
        if(!pid) std::cout<<tt_wait_sum[j*200+i]<<' ';
      if(!pid) std::cout<<";\n";
      MPI_Barrier(c);
    }
    if(!pid) std::cout<<"];\n\n\n";
  }

  MPI_Barrier(c);

  if(np==MAX_PROCS){
    if(!pid) std::cout<<"Tc=[";
    size_t j=0;
    for(size_t i=0;i<200;i++){
      for(j=0;j<MAX_PROCS;j++)
        if(!pid) std::cout<<tt_sum[j*200+i]<<' ';
      if(!pid) std::cout<<";\n";
      MPI_Barrier(c);
    }
    if(!pid) std::cout<<"];\n\n\n";
  }
  // */
  //Free memory.
  if(sbuff   !=NULL) delete[] sbuff;
#endif

        PROF_PAR_ALL2ALLV_DENSE_END
    }






  /** @brief Nonblocking Consensus due to (Hoefler et al., 2010).
   * Allows alltoall sparse exchange without calling any alltoall.
   */
  inline int Mpi_NBX_neighbors(
      const int *destinations, const int ndest,
      std::vector<int> &sources,
      MPI_Comm comm,
      bool sort_sources)
  {
    std::vector<int> dummy_recvbuf;
    return Mpi_NBX<int>(
        destinations, ndest, nullptr, 0, sources, dummy_recvbuf, comm,
        sort_sources);
  }

  inline int Mpi_NBX_sizes(
      const int *destinations, const int *send_sizes, const int ndest,
      std::vector<int> &sources, std::vector<int> &recv_sizes,
      MPI_Comm comm,
      bool sort_sources)
  {
    const int count = 1;
    return Mpi_NBX<int>(
        destinations, ndest, send_sizes, count, sources, recv_sizes, comm,
        sort_sources);
  }

  inline int Mpi_NBX_neighbors(
      const int *destinations, const int ndest,
      std::vector<int> &sources,
      MPI_Comm comm,
      NbxSynch synch_p2p,     // array of length ndest
      NbxSynch synch_global,
      bool sort_sources)
  {
    std::vector<int> dummy_recvbuf;
    return Mpi_NBX<int>(
        destinations, ndest, nullptr, 0, sources, dummy_recvbuf, comm,
        synch_p2p, synch_global,
        sort_sources);
  }

  inline int Mpi_NBX_sizes(
      const int *destinations, const int *send_sizes, const int ndest,
      std::vector<int> &sources, std::vector<int> &recv_sizes,
      MPI_Comm comm,
      NbxSynch synch_p2p,     // array of length ndest
      NbxSynch synch_global,
      bool sort_sources)
  {
    const int count = 1;
    return Mpi_NBX<int>(
        destinations, ndest, send_sizes, count, sources, recv_sizes, comm,
        synch_p2p, synch_global,
        sort_sources);
  }

  template <typename T>
  inline int Mpi_NBX(
      const int *destinations, const int ndest,
      const T *sendbuf, const int count,
      std::vector<int> &sources, std::vector<T> &recvbuf,
      MPI_Comm comm,
      bool sort_sources)
  {
    MPI_Request final_barrier;
    int err = Mpi_NBX(destinations, ndest, sendbuf, count, sources, recvbuf, comm,
        NbxSynch::disable(), NbxSynch::enable(&final_barrier),
        sort_sources);
    MPI_Wait(&final_barrier, MPI_STATUS_IGNORE);
    return err;
  }

  template <typename T>
  inline int Mpi_NBX(
      const int *destinations, const int ndest,
      const T *sendbuf, const int count,
      std::vector<int> &sources, std::vector<T> &recvbuf,
      MPI_Comm comm,
      NbxSynch synch_p2p,     // array of length ndest
      NbxSynch synch_global,
      bool sort_sources)
  {
    /** (Hoefler et al., 2010)
     * https://doi-org.ezproxy.lib.utah.edu/10.1145/1693453.1693476
     */

    // Pseudocode (Algorithm 2) from the paper.
    // Input: List I of destinations and data
    // Output: List O of received data and sources
    //
    // done=false;
    // barr_act=false;
    // foreach i ∈ I do
    //   start nonblocking synchronous send to process dest(i);
    // while not done do
    //   msg = nonblocking probe for incoming message;
    //   if msg found then
    //     allocate buffer, receive message, add buffer to O;
    //   if barr_act then
    //     comp = test barrier for completion;
    //     if comp then done=true;
    //   else
    //     if all sends are finished then
    //       start nonblocking barrier;
    //       barr act=true;

    int err = MPI_SUCCESS;
#define ABORT_ERR(mpi_call) if ((err = (mpi_call)) != MPI_SUCCESS) { assert(false); return err; }

    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    const int nbx_tag = 1;
    const int synch_tag = 2;

    static std::vector<MPI_Request> send_requests(
        2 * ndest);
    send_requests.clear();
    send_requests.resize(ndest, MPI_REQUEST_NULL);
    for (int dest_idx = 0; dest_idx < ndest; ++dest_idx)
      ABORT_ERR( par::Mpi_Issend(
                     sendbuf + dest_idx * count, count, destinations[dest_idx],
                     nbx_tag, comm, &send_requests[dest_idx]) );

    sources.clear();
    recvbuf.clear();
    sources.reserve(ndest);
    recvbuf.reserve(ndest * count);

    int done = false,  barrier_active = false;
    MPI_Request barrier_request = MPI_REQUEST_NULL;
    while (not done)
    {
      // Nonblocking matching probe, with matched receive, if there is message.
      int there_is_message = false;
      MPI_Message msg;
      MPI_Status status;
      ABORT_ERR(MPI_Improbe(MPI_ANY_SOURCE, nbx_tag, comm, &there_is_message, &msg, &status));
      if (there_is_message)
      {
        sources.push_back(status.MPI_SOURCE);
        recvbuf.resize(sources.size() * count);
        ABORT_ERR(Mpi_Mrecv(&(*recvbuf.end()) - count, count, &msg, &status));
        // Chose blocking recv here, otherwise next recv could resize buffer.
      }

      // Exit if the barrier has completed, i.e. all dependents have received.
      if (barrier_active)
      {
        int barrier_complete = false;
        ABORT_ERR(MPI_Test(&barrier_request, &barrier_complete, MPI_STATUS_IGNORE));
        if (barrier_complete)
          done = true;
      }
      else
      {
        // Enter barrier only once Ssend's complete, i.e. own dependents received.
        int ssends_complete = false;
        ABORT_ERR(MPI_Testall(ndest, send_requests.data(), &ssends_complete, MPI_STATUSES_IGNORE));
        if (ssends_complete)
        {
          ABORT_ERR(MPI_Ibarrier(comm, &barrier_request));
          barrier_active = true;
        }
      }
    }

    const int local_nbr[2] = {ndest, static_cast<int>(sources.size())};
    const auto consistent_srcs = [comm](const int local[2]) -> bool {
      int global[2] = {-1, 1};
      Mpi_Allreduce(local, global, 2, MPI_SUM, comm);
      return global[0] == global[1];
    };
    assert(consistent_srcs(local_nbr));

    // Need extra synch to avoid race condition due to MPI_ANY_SOURCE:
    // Maybe the ibarrier first completes on process, A, which exits the loop
    // and sends some data, and then another process, B, receives that data
    // within the ibarrier loop.

    std::vector<MPI_Request> send_ack_req(sources.size(), MPI_REQUEST_NULL);
    if (synch_p2p.enabled)
    {
      // Send ACK to sources, so they know that we know we are done.
      for (int src_idx = 0, nsrc = sources.size(); src_idx < nsrc; ++src_idx)
        ABORT_ERR( MPI_Isend(nullptr, 0, MPI_INT, sources[src_idx],
              synch_tag, comm, &send_ack_req[src_idx]) );

      // The requests to receive ACK are returned for the caller to complete,
      // e.g., MPI_Waitall(ndest, synch_p2p.ptr, MPI_STATUSES_IGNORE)
      for (int dest_idx = 0; dest_idx < ndest; ++dest_idx)
        ABORT_ERR( MPI_Irecv(nullptr, 0, MPI_INT, destinations[dest_idx],
              synch_tag, comm, &synch_p2p.ptr[dest_idx]) );

      // Note that synch_tag != nbx_tag. Otherwise, original race comes here.
    }

    if (synch_global.enabled)
    {
      ABORT_ERR( MPI_Ibarrier(comm, synch_global.ptr) );
    }

    if (sort_sources)
    {
      // future: factor out to a non-template non-inline function

      // Sparse .. average num sources is small .. sort with std::sort().
      const int nsrc = sources.size();
      static std::vector<int> permutation(
          2 * nsrc);
      permutation.clear();
      permutation.resize(nsrc);
      std::iota(permutation.begin(), permutation.end(), 0);
      std::sort(permutation.begin(), permutation.end(),
          [&sources](int i, int j) { return sources[i] < sources[j]; });

      // Permute sources.
      union Union { int i; T t; };
      const size_t szU = sizeof(Union);
      static std::vector<Union> reorder;

      reorder.clear();
      reorder.resize((nsrc * sizeof(int) + szU - 1) / szU);
      int *reorder_src = reinterpret_cast<int *>(reorder.data());
      for (int i = 0; i < nsrc; ++i)
        reorder_src[i] = sources[permutation[i]];
      for (int i = 0; i < nsrc; ++i)
        sources[i] = reorder_src[i];

      // Permute recv data, maintaining order within each message.
      reorder.clear();
      reorder.resize((nsrc * sizeof(T) * count + szU - 1) / szU);
      T *reorder_buf = reinterpret_cast<T *>(reorder.data());
      for (int i = 0; i < nsrc; ++i)
        for (int j = 0; j < count; ++j)
          reorder_buf[i * count + j] = recvbuf[permutation[i] * count + j];
      for (int i = 0; i < nsrc * count; ++i)
        recvbuf[i] = reorder_buf[i];

      assert(std::is_sorted(sources.begin(), sources.end()));
    }

    ABORT_ERR(MPI_Waitall(sources.size(), send_ack_req.data(), MPI_STATUSES_IGNORE));

#undef ABORT_ERR

    return MPI_SUCCESS;
  }





    template <typename T>
    std::vector<T> sendAll(const std::vector<T> &sdata, const std::vector<int> &sdest, MPI_Comm comm)
    {
      if (sdata.size() != sdest.size())
        throw std::logic_error("sdata and sdest must have the same size.");

      // future: Instead of sendAll, index destinations within neighborhood.

      int comm_size, comm_rank;
      MPI_Comm_size(comm, &comm_size);
      MPI_Comm_rank(comm, &comm_rank);

      // Validate input
      assert(sdata.size() == sdest.size());
      for (size_t i = 0; i < sdest.size(); ++i)
      {
        assert(0 <= sdest[i]);
        assert(sdest[i] < comm_size);
      }

      const auto to_set = [](std::vector<int> v) { return std::set<int>{v.begin(), v.end()}; };
      const auto to_vector = [](std::set<int> s) { return std::vector<int>{s.begin(), s.end()}; };
      const std::vector<int> destinations = to_vector(to_set(sdest));
      const int ndest = destinations.size();

      std::map<int, int> dest2idx;
      for (int i = 0; i < ndest; ++i)
        dest2idx[destinations[i]] = i;

      std::vector<int> sdest_idx(sdest.size());
      for (int i = 0, e = sdest.size(); i < e; ++i)
        sdest_idx[i] = dest2idx[sdest[i]];

      std::vector<int> send_sizes(ndest, 0);
      for (int dest_idx : sdest_idx)
        ++send_sizes[dest_idx];

      std::vector<int> sources;
      std::vector<int> recv_sizes;
      std::vector<MPI_Request> nbx_synch_p2p(ndest, MPI_REQUEST_NULL);
      MPI_Request nbx_synch_global = MPI_REQUEST_NULL;
      Mpi_NBX_sizes(destinations.data(), send_sizes.data(), ndest,
          sources, recv_sizes, comm,
          NbxSynch::enable(nbx_synch_p2p.data()),
          NbxSynch::enable(&nbx_synch_global));
      const int nsrc = sources.size();

      const auto make_offsets = [](const std::vector<int> &sizes) {
          std::vector<int> offsets(sizes.size(), 0);
          int sum_sz = 0;
          for (int i = 0, e = sizes.size(); i < e; ++i)
          {
            offsets[i] = sum_sz;
            sum_sz += sizes[i];
          }
          return offsets;
      };

      const int send_total = sdata.size();
      const int recv_total = std::accumulate(recv_sizes.begin(), recv_sizes.end(), int{});
      const std::vector<int> send_offsets = make_offsets(send_sizes);
      const std::vector<int> recv_offsets = make_offsets(recv_sizes);
      std::vector<T> rdata(recv_total);

      std::vector<MPI_Request> send_req(ndest, MPI_REQUEST_NULL);
      std::vector<MPI_Request> recv_req(nsrc, MPI_REQUEST_NULL);

      // Permute data
      std::vector<T> ordered(send_total);
      {
        std::vector<int> pos = send_offsets;
        for (int i = 0; i < send_total; ++i)
          ordered[pos[sdest_idx[i]]++] = sdata[i];
      }

      int self_dest_idx = -1;
      int self_src_idx = -1;

      // Must wait for ACKs from destinations before send again.
      MPI_Waitall(ndest, nbx_synch_p2p.data(), MPI_STATUSES_IGNORE);

      // Post receives.
      for (int si = 0; si < nsrc; ++si)
        if (sources[si] != comm_rank)
          Mpi_Irecv( &rdata[recv_offsets[si]], recv_sizes[si], sources[si],
              int{}, comm, &recv_req[si]);
        else
          self_src_idx = si;

      // Initiate sends.
      for (int di = 0; di < ndest; ++di)
        if (destinations[di] != comm_rank)
          Mpi_Isend( &ordered[send_offsets[di]], send_sizes[di], destinations[di],
              int{}, comm, &send_req[di]);
        else
          self_dest_idx = di;

      // Copy self.
      assert((self_src_idx != -1) == (self_dest_idx != -1));
      if (self_dest_idx != -1 and self_src_idx != -1)
      {
        assert(send_sizes[self_dest_idx] == recv_sizes[self_src_idx]);
        std::copy_n( &ordered[send_offsets[self_dest_idx]],
            send_sizes[self_dest_idx],
            &rdata[recv_offsets[self_src_idx]] );
      }

      MPI_Waitall(ndest, send_req.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(nsrc, recv_req.data(), MPI_STATUSES_IGNORE);
      MPI_Wait(&nbx_synch_global, MPI_STATUS_IGNORE);

      return rdata;
    }





  template<typename T>
  unsigned int defaultWeight(const T *a) {
    return 1;
  }

  template<typename T>
  int scatterValues(const std::vector<T> &in, std::vector<T> &out,
                    DendroIntL outSz, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_SCATTER_BEGIN

    int rank, npes;

    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    MPI_Request request;
    MPI_Status status;

    DendroIntL inSz = in.size();
    out.resize(outSz);

    DendroIntL off1 = 0, off2 = 0;
    DendroIntL *scnIn = NULL;
    if (inSz) {
      scnIn = new DendroIntL[inSz];
      assert(scnIn);
    }

    // perform a local scan first ...
    DendroIntL zero = 0;
    if (inSz) {
      scnIn[0] = 1;
      for (DendroIntL i = 1; i < inSz; i++) {
        scnIn[i] = scnIn[i - 1] + 1;
      }//end for
      // now scan with the final members of
      par::Mpi_Scan<DendroIntL>(scnIn + inSz - 1, &off1, 1, MPI_SUM, comm);
    } else {
      par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm);
    }

    // communicate the offsets ...
    if (rank < (npes - 1)) {
      par::Mpi_Issend<DendroIntL>(&off1, 1, (rank + 1), 0, comm, &request);
    }
    if (rank) {
      par::Mpi_Recv<DendroIntL>(&off2, 1, (rank - 1), 0, comm, &status);
    } else {
      off2 = 0;
    }

    // add offset to local array
    for (DendroIntL i = 0; i < inSz; i++) {
      scnIn[i] = scnIn[i] + off2;  // This has the global scan results now ...
    }//end for

    //Gather Scan of outCnts
    DendroIntL *outCnts;
    outCnts = new DendroIntL[npes];
    assert(outCnts);

    if (rank < (npes - 1)) {
      MPI_Status statusWait;
      MPI_Wait(&request, &statusWait);
    }

    if (outSz) {
      par::Mpi_Scan<DendroIntL>(&outSz, &off1, 1, MPI_SUM, comm);
    } else {
      par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm);
    }

    par::Mpi_Allgather<DendroIntL>(&off1, outCnts, 1, comm);

    int *sendSz = new int[npes];
    assert(sendSz);

    int *recvSz = new int[npes];
    assert(recvSz);

    int *sendOff = new int[npes];
    assert(sendOff);

    int *recvOff = new int[npes];
    assert(recvOff);

    // compute the partition offsets and sizes so that All2Allv can be performed.
    // initialize ...
    for (int i = 0; i < npes; i++) {
      sendSz[i] = 0;
    }

    //The Heart of the algorithm....
    //scnIn and outCnts are both sorted
    DendroIntL inCnt = 0;
    int pCnt = 0;
    while ((inCnt < inSz) && (pCnt < npes)) {
      if (scnIn[inCnt] <= outCnts[pCnt]) {
        sendSz[pCnt]++;
        inCnt++;
      } else {
        pCnt++;
      }
    }

    // communicate with other procs how many you shall be sending and get how
    // many to recieve from whom.
    par::Mpi_Alltoall<int>(sendSz, recvSz, 1, comm);

    int nn = 0; // new value of nlSize, ie the local nodes.
    for (int i = 0; i < npes; i++) {
      nn += recvSz[i];
    }

    // compute offsets ...
    sendOff[0] = 0;
    recvOff[0] = 0;
    for (int i = 1; i < npes; i++) {
      sendOff[i] = sendOff[i - 1] + sendSz[i - 1];
      recvOff[i] = recvOff[i - 1] + recvSz[i - 1];
    }

    assert(static_cast<unsigned int>(nn) == outSz);
    // perform All2All  ...
    const T *inPtr = NULL;
    T *outPtr = NULL;
    if (!in.empty()) {
      inPtr = &(*(in.cbegin()));
    }
    if (!out.empty()) {
      outPtr = &(*(out.begin()));
    }
    par::Mpi_Alltoallv_sparse<T>(inPtr, sendSz, sendOff,
                                 outPtr, recvSz, recvOff, comm);

    // clean up...
    if (scnIn) {
      delete[] scnIn;
      scnIn = NULL;
    }

    delete[] outCnts;
    outCnts = NULL;

    delete[] sendSz;
    sendSz = NULL;

    delete[] sendOff;
    sendOff = NULL;

    delete[] recvSz;
    recvSz = NULL;

    delete[] recvOff;
    recvOff = NULL;

    PROF_PAR_SCATTER_END
  }


  template<typename T>
  int concatenate(std::vector<T> &listA, std::vector<T> &listB,
                  MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PAR_CONCAT_BEGIN

    int rank;
    int npes;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    assert(!(listA.empty()));

    //1. First perform Allreduce to get total listA size
    //and total listB size;

    DendroIntL locAsz_locBsz[2];
    DendroIntL globAsz_globBsz[2];

    locAsz_locBsz[0] = listA.size();
    locAsz_locBsz[1] = listB.size();
    globAsz_globBsz[0] = 0;
    globAsz_globBsz[1] = 0;

    par::Mpi_Allreduce<DendroIntL>(locAsz_locBsz, globAsz_globBsz, 2, MPI_SUM, comm);

    //2. Re-distribute A and B independently so that
    //B is distributed only on the high rank processors
    //and A is distribute only on the low rank processors.

    DendroIntL avgTotalSize = ((globAsz_globBsz[0] + globAsz_globBsz[1]) / npes);

    //since listA is not empty on any of the active procs,
    //globASz > npes so avgTotalSize >= 1

    DendroIntL remTotalSize = ((globAsz_globBsz[0] + globAsz_globBsz[1]) % npes);

    int numSmallProcs = (npes - remTotalSize);

    //In the final merged list, there will be exactly remTotalSize number
    //of processors each having (avgTotalSize + 1) elements and there will
    //be exactly numSmallProcs number of processors each having
    //avgTotalSize elements.
    //Also, len(A) + len(B) = (numSmallProcs*avg) + (remTotalSize*(avg+1))

    std::vector<T> tmpA;
    std::vector<T> tmpB;

    int numAhighProcs;
    int numAlowProcs;
    int numBothProcs;
    int numBhighProcs;
    int numBlowProcs;
    DendroIntL aSizeForBoth;
    DendroIntL bSizeForBoth;

    if (globAsz_globBsz[1] <= (numSmallProcs * avgTotalSize)) {
      numBhighProcs = 0;
      numBlowProcs = ((globAsz_globBsz[1]) / avgTotalSize);
      bSizeForBoth = ((globAsz_globBsz[1]) % avgTotalSize);

      assert(numBlowProcs <= numSmallProcs);

      //remBsize is < avgTotalSize. So it will fit on one proc.
      if (bSizeForBoth) {
        numBothProcs = 1;
        if (numBlowProcs < numSmallProcs) {
          //We don't know if remTotalSize is 0 or not.
          //So, let the common proc be a low proc.
          aSizeForBoth = (avgTotalSize - bSizeForBoth);
          numAhighProcs = remTotalSize;
          numAlowProcs = (numSmallProcs - (1 + numBlowProcs));
        } else {
          //No more room for small procs. The common has to be a high proc.
          aSizeForBoth = ((avgTotalSize + 1) - bSizeForBoth);
          numAhighProcs = (remTotalSize - 1);
          numAlowProcs = 0;
        }
      } else {
        numBothProcs = 0;
        aSizeForBoth = 0;
        numAhighProcs = remTotalSize;
        numAlowProcs = (numSmallProcs - numBlowProcs);
      }
    } else {
      //Some B procs will have (avgTotalSize+1) elements
      DendroIntL numBusingAvgPlus1 = ((globAsz_globBsz[1]) / (avgTotalSize + 1));
      DendroIntL remBusingAvgPlus1 = ((globAsz_globBsz[1]) % (avgTotalSize + 1));
      if (numBusingAvgPlus1 <= remTotalSize) {
        //Each block can use (avg+1) elements each, since there will be some
        //remaining for A
        numBhighProcs = numBusingAvgPlus1;
        numBlowProcs = 0;
        bSizeForBoth = remBusingAvgPlus1;
        if (bSizeForBoth) {
          numBothProcs = 1;
          if (numBhighProcs < remTotalSize) {
            //We don't know if numSmallProcs is 0 or not.
            //So, let the common proc be a high proc
            aSizeForBoth = ((avgTotalSize + 1) - bSizeForBoth);
            numAhighProcs = (remTotalSize - (numBhighProcs + 1));
            numAlowProcs = numSmallProcs;
          } else {
            //No more room for high procs. The common has to be a low proc.
            aSizeForBoth = (avgTotalSize - bSizeForBoth);
            numAhighProcs = 0;
            numAlowProcs = (numSmallProcs - 1);
          }
        } else {
          numBothProcs = 0;
          aSizeForBoth = 0;
          numAhighProcs = (remTotalSize - numBhighProcs);
          numAlowProcs = numSmallProcs;
        }
      } else {
        //Since numBusingAvgPlus1 > remTotalSize*(avg+1)
        //=> len(B) > remTotalSize*(avg+1)
        //=> len(A) < numSmallProcs*avg
        //This is identical to the first case (except for
        //the equality), with A and B swapped.

        assert(globAsz_globBsz[0] < (numSmallProcs * avgTotalSize));

        numAhighProcs = 0;
        numAlowProcs = ((globAsz_globBsz[0]) / avgTotalSize);
        aSizeForBoth = ((globAsz_globBsz[0]) % avgTotalSize);

        assert(numAlowProcs < numSmallProcs);

        //remAsize is < avgTotalSize. So it will fit on one proc.
        if (aSizeForBoth) {
          numBothProcs = 1;
          //We don't know if remTotalSize is 0 or not.
          //So, let the common proc be a low proc.
          bSizeForBoth = (avgTotalSize - aSizeForBoth);
          numBhighProcs = remTotalSize;
          numBlowProcs = (numSmallProcs - (1 + numAlowProcs));
        } else {
          numBothProcs = 0;
          bSizeForBoth = 0;
          numBhighProcs = remTotalSize;
          numBlowProcs = (numSmallProcs - numAlowProcs);
        }
      }
    }

    assert((numAhighProcs + numAlowProcs + numBothProcs
            + numBhighProcs + numBlowProcs) == npes);

    assert((aSizeForBoth + bSizeForBoth) <= (avgTotalSize + 1));

    if (numBothProcs) {
      assert((aSizeForBoth + bSizeForBoth) >= avgTotalSize);
    } else {
      assert(aSizeForBoth == 0);
      assert(bSizeForBoth == 0);
    }

    if ((aSizeForBoth + bSizeForBoth) == (avgTotalSize + 1)) {
      assert((numAhighProcs + numBothProcs + numBhighProcs) == remTotalSize);
      assert((numAlowProcs + numBlowProcs) == numSmallProcs);
    } else {
      assert((numAhighProcs + numBhighProcs) == remTotalSize);
      assert((numAlowProcs + numBothProcs + numBlowProcs) == numSmallProcs);
    }

    //The partition is as follow:
    //1. numAhighProcs with (avg+1) elements each exclusively from A,
    //2. numAlowProcs with avg elements each exclusively from A
    //3. numBothProcs with aSizeForBoth elements from A and
    // bSizeForBoth elements from B
    //4. numBhighProcs with (avg+1) elements each exclusively from B.
    //5. numBlowProcs with avg elements each exclusively from B.

    if (rank < numAhighProcs) {
      par::scatterValues<T>(listA, tmpA, (avgTotalSize + 1), comm);
      par::scatterValues<T>(listB, tmpB, 0, comm);
    } else if (rank < (numAhighProcs + numAlowProcs)) {
      par::scatterValues<T>(listA, tmpA, avgTotalSize, comm);
      par::scatterValues<T>(listB, tmpB, 0, comm);
    } else if (rank < (numAhighProcs + numAlowProcs + numBothProcs)) {
      par::scatterValues<T>(listA, tmpA, aSizeForBoth, comm);
      par::scatterValues<T>(listB, tmpB, bSizeForBoth, comm);
    } else if (rank <
               (numAhighProcs + numAlowProcs + numBothProcs + numBhighProcs)) {
      par::scatterValues<T>(listA, tmpA, 0, comm);
      par::scatterValues<T>(listB, tmpB, (avgTotalSize + 1), comm);
    } else {
      par::scatterValues<T>(listA, tmpA, 0, comm);
      par::scatterValues<T>(listB, tmpB, avgTotalSize, comm);
    }

    listA = tmpA;
    listB = tmpB;
    tmpA.clear();
    tmpB.clear();

    //3. Finally do a simple concatenation A = A + B. If the previous step
    //was performed correctly, there will be atmost 1 processor, which has both
    //non-empty A and non-empty B. On other processors one of the two lists
    //will be empty
    if (listA.empty()) {
      listA = listB;
    } else {
      if (!(listB.empty())) {
        listA.insert(listA.end(), listB.begin(), listB.end());
      }
    }

    listB.clear();

    PROF_PAR_CONCAT_END
  }

  template<typename T>
  int maxLowerBound(const std::vector<T> &keys, const std::vector<T> &searchList,
                    std::vector<T> &results, MPI_Comm comm) {
    PROF_SEARCH_BEGIN

    int rank, npes;

    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    // allocate memory for the mins array
    std::vector<T> mins(npes);
    assert(!searchList.empty());

    T *searchListPtr = NULL;
    T *minsPtr = NULL;
    if (!searchList.empty()) {
      searchListPtr = &(*(searchList.begin()));
    }
    if (!mins.empty()) {
      minsPtr = &(*(mins.begin()));
    }
    par::Mpi_Allgather<T>(searchListPtr, minsPtr, 1, comm);

    //For each key decide which processor to send to
    unsigned int *part = NULL;

    if (keys.size()) {
      part = new unsigned int[keys.size()];
      assert(part);
    }

    for (unsigned int i = 0; i < keys.size(); i++) {
      //maxLB returns the smallest index in a sorted array such
      //that a[ind] <= key and  a[index +1] > key
      bool found = par::maxLowerBound<T>(mins, keys[i], part + i, NULL, NULL);
      if (!found) {
        //This key is smaller than the mins from every processor.
        //No point in searching.
        part[i] = rank;
      }
    }

    mins.clear();

    int *numKeysSend = new int[npes];
    assert(numKeysSend);

    int *numKeysRecv = new int[npes];
    assert(numKeysRecv);

    for (int i = 0; i < npes; i++) {
      numKeysSend[i] = 0;
    }

    // calculate the number of keys to send ...
    for (unsigned int i = 0; i < keys.size(); i++) {
      numKeysSend[part[i]]++;
    }

    // Now do an All2All to get numKeysRecv
    par::Mpi_Alltoall<int>(numKeysSend, numKeysRecv, 1, comm);

    unsigned int totalKeys = 0;        // total number of local keys ...
    for (int i = 0; i < npes; i++) {
      totalKeys += numKeysRecv[i];
    }

    // create the send and recv buffers ...
    std::vector<T> sendK(keys.size());
    std::vector<T> recvK(totalKeys);

    // the mapping ..
    unsigned int *comm_map = NULL;

    if (keys.size()) {
      comm_map = new unsigned int[keys.size()];
      assert(comm_map);
    }

    // Now create sendK
    int *sendOffsets = new int[npes];
    assert(sendOffsets);
    sendOffsets[0] = 0;

    int *recvOffsets = new int[npes];
    assert(recvOffsets);
    recvOffsets[0] = 0;

    int *numKeysTmp = new int[npes];
    assert(numKeysTmp);
    numKeysTmp[0] = 0;

    // compute offsets ...
    for (int i = 1; i < npes; i++) {
      sendOffsets[i] = sendOffsets[i - 1] + numKeysSend[i - 1];
      recvOffsets[i] = recvOffsets[i - 1] + numKeysRecv[i - 1];
      numKeysTmp[i] = 0;
    }

    for (unsigned int i = 0; i < keys.size(); i++) {
      unsigned int ni = numKeysTmp[part[i]];
      numKeysTmp[part[i]]++;
      // set entry ...
      sendK[sendOffsets[part[i]] + ni] = keys[i];
      // save mapping .. will need it later ...
      comm_map[i] = sendOffsets[part[i]] + ni;
    }

    if (part) {
      delete[] part;
    }

    assert(numKeysTmp);
    delete[] numKeysTmp;
    numKeysTmp = NULL;

    T *sendKptr = NULL;
    T *recvKptr = NULL;
    if (!sendK.empty()) {
      sendKptr = &(*(sendK.begin()));
    }
    if (!recvK.empty()) {
      recvKptr = &(*(recvK.begin()));
    }

    par::Mpi_Alltoallv_sparse<T>(sendKptr, numKeysSend, sendOffsets,
                                 recvKptr, numKeysRecv, recvOffsets, comm);


    std::vector<T> resSend(totalKeys);
    std::vector<T> resRecv(keys.size());

    //Final local search.
    for (unsigned int i = 0; i < totalKeys; i++) {
      unsigned int idx;
      bool found = par::maxLowerBound<T>(searchList, recvK[i], &idx, NULL, NULL);
      if (found) {
        resSend[i] = searchList[idx];
      }
    }//end for i

    //Exchange Results
    //Return what you received in the earlier communication.
    T *resSendPtr = NULL;
    T *resRecvPtr = NULL;
    if (!resSend.empty()) {
      resSendPtr = &(*(resSend.begin()));
    }
    if (!resRecv.empty()) {
      resRecvPtr = &(*(resRecv.begin()));
    }
    par::Mpi_Alltoallv_sparse<T>(resSendPtr, numKeysRecv, recvOffsets,
                                 resRecvPtr, numKeysSend, sendOffsets, comm);

    assert(sendOffsets);
    delete[] sendOffsets;
    sendOffsets = NULL;

    assert(recvOffsets);
    delete[] recvOffsets;
    recvOffsets = NULL;

    assert(numKeysSend);
    delete[] numKeysSend;
    numKeysSend = NULL;

    assert(numKeysRecv);
    delete[] numKeysRecv;
    numKeysRecv = NULL;

    for (unsigned int i = 0; i < keys.size(); i++) {
      results[i] = resRecv[comm_map[i]];
    }//end for

    // Clean up ...
    if (comm_map) {
      delete[] comm_map;
    }

    PROF_SEARCH_END
  }

  template<typename T>
  int partitionW(std::vector<T> &nodeList, unsigned int (*getWeight)(const T *), MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif
    PROF_PARTW_BEGIN

    int npes;

    MPI_Comm_size(comm, &npes);

    if (npes == 1) {
      PROF_PARTW_END
    }

    if (getWeight == NULL) {
      getWeight = par::defaultWeight<T>;
    }

    int rank;

    MPI_Comm_rank(comm, &rank);

    MPI_Request request;
    MPI_Status status;
    const bool nEmpty = nodeList.empty();

    DendroIntL off1 = 0, off2 = 0, localWt = 0, totalWt = 0;

    DendroIntL *wts = NULL;
    DendroIntL *lscn = NULL;
    DendroIntL nlSize = nodeList.size();
    if (nlSize) {
      wts = new DendroIntL[nlSize];
      assert(wts);

      lscn = new DendroIntL[nlSize];
      assert(lscn);
    }

    // First construct arrays of id and wts.
#pragma omp parallel for reduction(+:localWt)
    for (DendroIntL i = 0; i < nlSize; i++) {
      wts[i] = (*getWeight)(&(nodeList[i]));
      localWt += wts[i];
    }

#ifdef __DEBUG_PAR__
    MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"Partition: Stage-1 passed."<<std::endl;
    }
    MPI_Barrier(comm);
#endif

    // compute the total weight of the problem ...
    par::Mpi_Allreduce<DendroIntL>(&localWt, &totalWt, 1, MPI_SUM, comm);

    // perform a local scan on the weights first ...
    DendroIntL zero = 0;
    if (!nEmpty) {
      lscn[0] = wts[0];
//        for (DendroIntL i = 1; i < nlSize; i++) {
//          lscn[i] = wts[i] + lscn[i-1];
//        }//end for
      omp_par::scan(&wts[1], lscn, nlSize);
      // now scan with the final members of
      par::Mpi_Scan<DendroIntL>(lscn + nlSize - 1, &off1, 1, MPI_SUM, comm);
    } else {
      par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm);
    }

    // communicate the offsets ...
    if (rank < (npes - 1)) {
      par::Mpi_Issend<DendroIntL>(&off1, 1, rank + 1, 0, comm, &request);
    }
    if (rank) {
      par::Mpi_Recv<DendroIntL>(&off2, 1, rank - 1, 0, comm, &status);
    }
    else {
      off2 = 0;
    }

    // add offset to local array
#pragma omp parallel for
    for (DendroIntL i = 0; i < nlSize; i++) {
      lscn[i] = lscn[i] + off2;       // This has the global scan results now ...
    }//end for

#ifdef __DEBUG_PAR__
    MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"Partition: Stage-2 passed."<<std::endl;
    }
    MPI_Barrier(comm);
#endif

    int *sendSz = new int[npes];
    assert(sendSz);

    int *recvSz = new int[npes];
    assert(recvSz);

    int *sendOff = new int[npes];
    assert(sendOff);
    sendOff[0] = 0;

    int *recvOff = new int[npes];
    assert(recvOff);
    recvOff[0] = 0;

    // compute the partition offsets and sizes so that All2Allv can be performed.
    // initialize ...

#pragma omp parallel for
    for (int i = 0; i < npes; i++) {
      sendSz[i] = 0;
    }

    // Now determine the average load ...
    DendroIntL npesLong = npes;
    DendroIntL avgLoad = (totalWt / npesLong);

    DendroIntL extra = (totalWt % npesLong);

    //The Heart of the algorithm....
    if (avgLoad > 0) {
      for (DendroIntL i = 0; i < nlSize; i++) {
        if (lscn[i] == 0) {
          sendSz[0]++;
        } else {
          int ind = 0;
          if (lscn[i] <= (extra * (avgLoad + 1))) {
            ind = ((lscn[i] - 1) / (avgLoad + 1));
          } else {
            ind = ((lscn[i] - (1 + extra)) / avgLoad);
          }
          assert(ind < npes);
          sendSz[ind]++;
        }//end if-else
      }//end for */


#ifdef __DEBUG_PAR__
      int tmp_sum=0;
      for(int i=0;i<npes;i++) tmp_sum+=sendSz[i];
      assert(tmp_sum==nlSize);
#endif

    } else {
      sendSz[0] += nlSize;
    }//end if-else

#ifdef __DEBUG_PAR__
    MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"Partition: Stage-3 passed."<<std::endl;
    }
    MPI_Barrier(comm);
#endif

    if (rank < (npes - 1)) {
      MPI_Status statusWait;
      MPI_Wait(&request, &statusWait);
    }

    // communicate with other procs how many you shall be sending and get how
    // many to recieve from whom.
    par::Mpi_Alltoall<int>(sendSz, recvSz, 1, comm);

#ifdef __DEBUG_PAR__
    DendroIntL totSendToOthers = 0;
    DendroIntL totRecvFromOthers = 0;
    for (int i = 0; i < npes; i++) {
      if(rank != i) {
        totSendToOthers += sendSz[i];
        totRecvFromOthers += recvSz[i];
      }
    }
#endif

    DendroIntL nn = 0; // new value of nlSize, ie the local nodes.
#pragma omp parallel for reduction(+:nn)
    for (int i = 0; i < npes; i++) {
      nn += recvSz[i];
    }

    // compute offsets ...
//      for (int i = 1; i < npes; i++) {
//        sendOff[i] = sendOff[i-1] + sendSz[i-1];
//        recvOff[i] = recvOff[i-1] + recvSz[i-1];
//      }
    omp_par::scan(sendSz, sendOff, npes);
    omp_par::scan(recvSz, recvOff, npes);

#ifdef __DEBUG_PAR__
    MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"Partition: Stage-4 passed."<<std::endl;
    }
    MPI_Barrier(comm);
    /*
       std::cout<<rank<<": newSize: "<<nn<<" oldSize: "<<(nodeList.size())
       <<" send: "<<totSendToOthers<<" recv: "<<totRecvFromOthers<<std::endl;
     */
    MPI_Barrier(comm);
#endif

    // allocate memory for the new arrays ...
    std::vector<T> newNodes(nn);

#ifdef __DEBUG_PAR__
    MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"Partition: Final alloc successful."<<std::endl;
    }
    MPI_Barrier(comm);
#endif

    // perform All2All  ...
    T *nodeListPtr = NULL;
    T *newNodesPtr = NULL;
    if (!nodeList.empty()) {
      nodeListPtr = &(*(nodeList.begin()));
    }
    if (!newNodes.empty()) {
      newNodesPtr = &(*(newNodes.begin()));
    }
    par::Mpi_Alltoallv_sparse<T>(nodeListPtr, sendSz, sendOff,
                                 newNodesPtr, recvSz, recvOff, comm);

#ifdef __DEBUG_PAR__
    MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"Partition: Stage-5 passed."<<std::endl;
    }
    MPI_Barrier(comm);
#endif

    // reset the pointer ...
    swap(nodeList, newNodes);
    newNodes.clear();

    // clean up...
    if (!nEmpty) {
      delete[] lscn;
      delete[] wts;
    }
    delete[] sendSz;
    sendSz = NULL;

    delete[] sendOff;
    sendOff = NULL;

    delete[] recvSz;
    recvSz = NULL;

    delete[] recvOff;
    recvOff = NULL;

#ifdef __DEBUG_PAR__
    MPI_Barrier(comm);
    if(!rank) {
      std::cout<<"Partition: Stage-6 passed."<<std::endl;
    }
    MPI_Barrier(comm);
#endif

    PROF_PARTW_END
  }//end function


  /********************************************************************/
  /*
   * which_keys is one of KEEP_HIGH or KEEP_LOW
   * partner    is the processor with which to Merge and Split.
   *
   */
  template<typename T>
  void MergeSplit(std::vector<T> &local_list, int which_keys, int partner, MPI_Comm comm) {

    MPI_Status status;
    int send_size = local_list.size();
    int recv_size = 0;

    // first communicate how many you will send and how many you will receive ...

    par::Mpi_Sendrecv<int, int>(&send_size, 1, partner, 0,
                                &recv_size, 1, partner, 0, comm, &status);

    std::vector<T> temp_list(recv_size);

    T *local_listPtr = NULL;
    T *temp_listPtr = NULL;
    if (!local_list.empty()) {
      local_listPtr = &(*(local_list.begin()));
    }
    if (!temp_list.empty()) {
      temp_listPtr = &(*(temp_list.begin()));
    }

    par::Mpi_Sendrecv<T, T>(local_listPtr, send_size, partner,
                            1, temp_listPtr, recv_size, partner, 1, comm, &status);


    MergeLists<T>(local_list, temp_list, which_keys);


    temp_list.clear();
  } // Merge_split

  template<typename T>
  void Par_bitonic_sort_incr(std::vector<T> &local_list, int proc_set_size, MPI_Comm comm) {
    int eor_bit;
    int proc_set_dim;
    int stage;
    int partner;
    int my_rank;

    MPI_Comm_rank(comm, &my_rank);

    proc_set_dim = 0;
    int x = proc_set_size;
    while (x > 1) {
      x = x >> 1;
      proc_set_dim++;
    }

    eor_bit = (1 << (proc_set_dim - 1));
    for (stage = 0; stage < proc_set_dim; stage++) {
      partner = (my_rank ^ eor_bit);

      if (my_rank < partner) {
        MergeSplit<T>(local_list, KEEP_LOW, partner, comm);
      } else {
        MergeSplit<T>(local_list, KEEP_HIGH, partner, comm);
      }

      eor_bit = (eor_bit >> 1);
    }
  }  // Par_bitonic_sort_incr


  template<typename T>
  void Par_bitonic_sort_decr(std::vector<T> &local_list, int proc_set_size, MPI_Comm comm) {
    int eor_bit;
    int proc_set_dim;
    int stage;
    int partner;
    int my_rank;

    MPI_Comm_rank(comm, &my_rank);

    proc_set_dim = 0;
    int x = proc_set_size;
    while (x > 1) {
      x = x >> 1;
      proc_set_dim++;
    }

    eor_bit = (1 << (proc_set_dim - 1));
    for (stage = 0; stage < proc_set_dim; stage++) {
      partner = my_rank ^ eor_bit;

      if (my_rank > partner) {
        MergeSplit<T>(local_list, KEEP_LOW, partner, comm);
      } else {
        MergeSplit<T>(local_list, KEEP_HIGH, partner, comm);
      }

      eor_bit = (eor_bit >> 1);
    }

  } // Par_bitonic_sort_decr

  template<typename T>
  void Par_bitonic_merge_incr(std::vector<T> &local_list, int proc_set_size, MPI_Comm comm) {
    int partner;
    int rank, npes;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    unsigned int num_left = binOp::getPrevHighestPowerOfTwo(npes);
    unsigned int num_right = npes - num_left;

    // 1, Do merge between the k right procs and the highest k left procs.
    if ((static_cast<unsigned int>(rank) < num_left) &&
        (static_cast<unsigned int>(rank) >= (num_left - num_right))) {
      partner = static_cast<unsigned int>(rank) + num_right;
      MergeSplit<T>(local_list, KEEP_LOW, partner, comm);
    } else if (static_cast<unsigned int>(rank) >= num_left) {
      partner = static_cast<unsigned int>(rank) - num_right;
      MergeSplit<T>(local_list, KEEP_HIGH, partner, comm);
    }
  }

  template<typename T>
  void bitonicSort_binary(std::vector<T> &in, MPI_Comm comm) {
    int proc_set_size;
    unsigned int and_bit;
    int rank;
    int npes;

    MPI_Comm_size(comm, &npes);

#ifdef __DEBUG_PAR__
    assert(npes > 1);
    assert(!(npes & (npes-1)));
    assert(!(in.empty()));
#endif

    MPI_Comm_rank(comm, &rank);

    for (proc_set_size = 2, and_bit = 2;
         proc_set_size <= npes;
         proc_set_size = proc_set_size * 2,
             and_bit = and_bit << 1) {

      if ((rank & and_bit) == 0) {
        Par_bitonic_sort_incr<T>(in, proc_set_size, comm);
      } else {
        Par_bitonic_sort_decr<T>(in, proc_set_size, comm);
      }
    }//end for
  }

  template<typename T>
  void bitonicSort(std::vector<T> &in, MPI_Comm comm) {
    int rank;
    int npes;

    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    assert(!(in.empty()));

    //Local Sort first
    //std::sort(in.begin(),in.end());
    omp_par::merge_sort(in.begin(), in.end());

    if (npes > 1) {

      // check if npes is a power of two ...
      bool isPower = (!(npes & (npes - 1)));

      if (isPower) {
        bitonicSort_binary<T>(in, comm);
      } else {
        MPI_Comm new_comm;

        // Since npes is not a power of two, we shall split the problem in two ...
        //
        // 1. Create 2 comm groups ... one for the 2^d portion and one for the
        // remainder.
        unsigned int splitter = splitCommBinary(comm, &new_comm);

        if (static_cast<unsigned int>(rank) < splitter) {
          bitonicSort_binary<T>(in, new_comm);
        } else {
          bitonicSort<T>(in, new_comm);
        }

        // 3. Do a special merge of the two segments. (original comm).
        Par_bitonic_merge_incr(in, binOp::getNextHighestPowerOfTwo(npes), comm);
        MPI_Comm_free(&new_comm);
        splitter = splitCommBinaryNoFlip(comm, &new_comm);

        // 4. Now a final sort on the segments.
        if (static_cast<unsigned int>(rank) < splitter) {
          bitonicSort_binary<T>(in, new_comm);
        } else {
          bitonicSort<T>(in, new_comm);
        }
        MPI_Comm_free(&new_comm);
      }//end if isPower of 2
    }//end if single processor
  }//end function

  template<typename T>
  void MergeLists(std::vector<T> &listA, std::vector<T> &listB,
                  int KEEP_WHAT) {

    T _low, _high;

    assert(!(listA.empty()));
    assert(!(listB.empty()));


    // max ( min(A,B) )
    _low = ((listA[0] > listB[0]) ? listA[0] : listB[0]);
    // min ( max(A,B) )
    _high = ((listA[listA.size() - 1] < listB[listB.size() - 1]) ?
             listA[listA.size() - 1] : listB[listB.size() - 1]);

    // We will do a full merge first ...
    size_t list_size = listA.size() + listB.size();

    std::vector<T> scratch_list(list_size);

    unsigned int index1 = 0;
    unsigned int index2 = 0;

    for (size_t i = 0; i < list_size; i++) {
      //The order of (A || B) is important here,
      //so that index2 remains within bounds
      if ((index1 < listA.size()) &&
          ((index2 >= listB.size()) ||
           (listA[index1] <= listB[index2]))) {
        scratch_list[i] = listA[index1];
        index1++;
      } else {
        scratch_list[i] = listB[index2];
        index2++;
      }
    }


    listA.clear();
    listB.clear();

    if (KEEP_WHAT == KEEP_LOW) {
      int ii = 0;

      while (((scratch_list[ii] < _low) ||
              (ii < (list_size / 2)))
             && (scratch_list[ii] <= _high)) {
        ii++;
      }
      if (ii) {
        listA.insert(listA.end(), scratch_list.begin(),
                     (scratch_list.begin() + ii));
      }
    } else {
      int ii = (list_size - 1);
      while (((ii >= (list_size / 2))
              && (scratch_list[ii] >= _low))
             || (scratch_list[ii] > _high)) {
        ii--;
      }
      if (ii < (list_size - 1)) {
        listA.insert(listA.begin(), (scratch_list.begin() + (ii + 1)),
                     (scratch_list.begin() + list_size));
      }
    }
    scratch_list.clear();
  }//end function



  namespace detail
  {
    template <typename T>
    class ShiftP2PContext;
    class ShiftP2PSchedule;

    //
    // class ShiftP2PSchedule
    //
    class ShiftP2PSchedule
    {
      public:
        inline ShiftP2PSchedule( MPI_Comm comm,
                          size_t srcSizeLocal, DendroIntL srcBeginGlobal,
                          size_t dstSizeLocal, DendroIntL dstBeginGlobal );

        template <typename T>
        ShiftP2PContext<T> context(const T *srcLocal, T *dstLocal, int ndofs) const;

      private:
        static constexpr bool printDebug = false;
        template <typename T>  friend class ShiftP2PContext;

        MPI_Comm comm = MPI_COMM_NULL;  // must be in ctor member init list
        int nProc = mpi_comm_size(comm);
        int rProc = mpi_comm_rank(comm);
        std::vector<int> dstTo;
        std::vector<int> srcFrom;
        std::vector<int> dstCount;
        std::vector<int> srcCount;
        std::vector<int> dstDspls;
        std::vector<int> srcDspls;
        std::string dstRangeStr;
        std::string srcRangeStr;
        int selfSrcI = -1;
        int selfDstI = -1;
    };

    //
    // class ShiftP2PContext
    //
    template <typename T>
    class ShiftP2PContext
    {
      public:
        // Borrows schedule; schedule must outlive activity that uses context.
        ShiftP2PContext(const ShiftP2PSchedule *schedule, const T *srcLocal, T *dstLocal, int ndofs)
          : s(schedule), srcLocal(srcLocal), dstLocal(dstLocal), ndofs(ndofs)
        {
          if (printDebug)
          { std::stringstream ss;
            ss << "[" << s->rProc << "] ";
            rankPrefix = ss.str();
          }
        }

        // Moveable, not copyable.
        ShiftP2PContext(const ShiftP2PContext &) = delete;
        ShiftP2PContext(ShiftP2PContext &&) = default;

        ShiftP2PContext & isendrecv();
        ShiftP2PContext & local();
        ShiftP2PContext & wait();

        ShiftP2PContext & operator=(const ShiftP2PContext &) = delete;
        ShiftP2PContext & operator=(ShiftP2PContext &&) = default;

        // Above operations must be done in-order on any ShiftP2PContext instance:
        // A constructed object must call isendrecv() then local() then wait().
        // An object can be optionally move-assigned, but only after calling wait().

      private:
        static constexpr bool printDebug = ShiftP2PSchedule::printDebug;

        const ShiftP2PSchedule *s = nullptr;
        const T * const srcLocal;  T * const dstLocal;  int ndofs;

        // Init'n is in order of declaration: Ctor member init list MUST init s.
        const std::vector<int> & dstTo    = s->dstTo;
        const std::vector<int> & srcFrom  = s->srcFrom;
        const std::vector<int> & dstCount = s->dstCount;
        const std::vector<int> & srcCount = s->srcCount;
        const std::vector<int> & dstDspls = s->dstDspls;
        const std::vector<int> & srcDspls = s->srcDspls;

        // make_unique() needs c++14
        using VecReq = std::vector<MPI_Request>;
        std::unique_ptr<VecReq> requests = std::make_unique<VecReq>(
            srcFrom.size() + dstTo.size(), MPI_REQUEST_NULL);

        int commCnt = 0;
        std::string rankPrefix;
    };

    template <typename T>
    ShiftP2PContext<T> ShiftP2PSchedule::context(const T *srcLocal, T *dstLocal, int ndofs) const
    {
      return ShiftP2PContext<T>(this, srcLocal, dstLocal, ndofs);
    }
  }


  template <typename T>
  void shift(
      MPI_Comm comm,
      const T *srcLocal, size_t srcSizeLocal, DendroIntL srcBeginGlobal,
            T *dstLocal, size_t dstSizeLocal, DendroIntL dstBeginGlobal,
      int ndofs)
  {
    using namespace detail;
    const ShiftP2PSchedule schedule(
        comm, srcSizeLocal, srcBeginGlobal, dstSizeLocal, dstBeginGlobal); 
    schedule.context(srcLocal, dstLocal, ndofs)
      .isendrecv()
      .local()
      .wait();
  }


  namespace detail
  {

    //
    // ShiftP2PSchedule::ShiftP2PSchedule()
    //
    ShiftP2PSchedule::ShiftP2PSchedule(
        MPI_Comm comm,
        size_t srcSizeLocal, DendroIntL srcBeginGlobal,
        size_t dstSizeLocal, DendroIntL dstBeginGlobal )
      :
        comm(comm)
    {
      // Upfront Allgather, locally make pairs, point-to-point.

      std::string rankPrefix;
      if (printDebug)
      { std::stringstream ss;
        ss << "[" << rProc << "] ";
        rankPrefix = ss.str();
      }

      const DendroIntL mySrcGlobBegin = srcBeginGlobal;
      const DendroIntL myDstGlobBegin = dstBeginGlobal;
      const DendroIntL mySrcGlobEnd = srcBeginGlobal + srcSizeLocal;
      const DendroIntL myDstGlobEnd = dstBeginGlobal + dstSizeLocal;

      // Includes boundaries, [0]=0 and [n]=(total global number nodes)
      std::vector<DendroIntL> allSrcSplit(nProc+1, 0);
      std::vector<DendroIntL> allDstSplit(nProc+1, 0);
      if(printDebug) std::cerr << rankPrefix << "Begin Allgather (#1)\n";
      par::Mpi_Allgather(&mySrcGlobEnd, &allSrcSplit[1], 1, comm);
      if(printDebug) std::cerr << rankPrefix << "Begin Allgather (#2)\n";
      par::Mpi_Allgather(&myDstGlobEnd, &allDstSplit[1], 1, comm);
      if(printDebug) std::cerr << rankPrefix << "Concluded Allgather\n";

      const bool isActiveSrc = srcSizeLocal > 0;
      const bool isActiveDst = dstSizeLocal > 0;

      if (printDebug && rProc == 0)
      {
        std::stringstream ss;
        ss << "allSrcSplit == ";
        for (DendroIntL split : allSrcSplit)
          ss << " " << split;
        ss << "\n";
        ss << "allDstSplit == ";
        for (DendroIntL split : allDstSplit)
          ss << " " << split;
        ss << "\n";
        fprintf(stderr, "\n%s\n", ss.str().c_str());
      }

      if (!isActiveSrc && !isActiveDst)
      {
        if(printDebug) std::cerr << rankPrefix << "Not active. Returning.\n";
        return;
      }


      // Figure out whom to send to / recv from.
      // - otherDstBegin[r0] <= mySrcBegin <= mySrcRank[x] < mySrcEnd <= otherDstEnd[r1];
      // - otherSrcBegin[r0] <= myDstBegin <= myDstRank[x] < myDstEnd <= otherSrcEnd[r1];

      int dst_r0, dst_r1;
      int src_r0, src_r1;

      if(printDebug) std::cerr << rankPrefix << "Begin binary search (#1)\n";

      // Binary search to determine  dst_r0, dst_r1,  src_r0, src_r1.
      constexpr int bSplit = 0;
      constexpr int eSplit = 1;
      // Search for last rank : dstBegin[rank] <= mySrcBegin
      int r_lb = 0, r_ub = nProc-1;  // Inclusive lower/upper bounds.
      { while (r_lb < r_ub)
        {
          const int test = (r_lb + r_ub + 1)/2;
          if (allDstSplit[test + bSplit] <= mySrcGlobBegin)
          {
            r_lb = test;
          }
          else
          {
            r_ub = test - 1;
          }
        }
      }
      dst_r0 = r_lb;

      if(printDebug) std::cerr << rankPrefix << "Begin binary search (#2)\n";
      // Search for first rank after dst_r0: mySrcEnd <= dstEnd[rank]
      r_ub = nProc - 1;
      { while (r_lb < r_ub)
        {
          const int test = (r_lb + r_ub)/2;
          if (mySrcGlobEnd <= allDstSplit[test + eSplit])
          {
            r_ub = test;
          }
          else
          {
            r_lb = test + 1;
          }
        }
      }
      dst_r1 = r_lb;

      if(printDebug) std::cerr << rankPrefix << "Begin binary search (#3)\n";
      // Search for last rank : srcBegin[rank] <= myDstBegin
      r_lb = 0;  r_ub = nProc-1;
      { while (r_lb < r_ub)
        {
          const int test = (r_lb + r_ub + 1)/2;
          if (allSrcSplit[test + bSplit] <= myDstGlobBegin)
          {
            r_lb = test;
          }
          else
          {
            r_ub = test - 1;
          }
        }
      }
      src_r0 = r_lb;

      if(printDebug) std::cerr << rankPrefix << "Begin binary search (#4)\n";
      // Search for first rank after src_r0: myDstEnd <= srcEnd[rank]
      r_ub = nProc - 1;
      { while (r_lb < r_ub)
        {
          const int test = (r_lb + r_ub)/2;
          if (myDstGlobEnd <= allSrcSplit[test + eSplit])
          {
            r_ub = test;
          }
          else
          {
            r_lb = test + 1;
          }
        }
      }
      src_r1 = r_lb;

      if(printDebug) std::cerr << rankPrefix << "Concluded binary search\n";

      // The binary search should automaticallly take care of the case
      // that some ranks are not active, but if it doesn't, give error.
      if (!(
            !(dst_r0+1 < nProc   && allDstSplit[dst_r0+1 + bSplit] <= mySrcGlobBegin) &&
            !(dst_r1-1 >= dst_r0 && allDstSplit[dst_r1-1 + eSplit] >= mySrcGlobEnd) &&
            !(src_r0+1 < nProc   && allSrcSplit[src_r0+1 + bSplit] <= myDstGlobBegin) &&
            !(src_r1-1 >= src_r0 && allSrcSplit[src_r1-1 + eSplit] >= myDstGlobEnd)
           ))
      {
        fprintf(stderr, "%s:\n"
                        "  mySrcGlobBegin--End == %llu--%llu\n"
                        "  myDstGlobBegin--End == %llu--%llu\n"
                        "  dst_r0+1 + bSplit == %u    -->   allDstSplit[] == %llu\n"
                        "  dst_r1-1 + eSplit == %u    -->   allDstSplit[] == %llu\n"
                        "  src_r0+1 + bSplit == %u    -->   allSrcSplit[] == %llu\n"
                        "  src_r1-1 + eSplit == %u    -->   allSrcSplit[] == %llu\n",
                        rankPrefix.c_str(),
                        mySrcGlobBegin, mySrcGlobEnd,
                        myDstGlobBegin, myDstGlobEnd,
                        dst_r0+1 + bSplit, allDstSplit[dst_r0+1 + bSplit],
                        dst_r1-1 + eSplit, allDstSplit[dst_r1-1 + eSplit],
                        src_r0+1 + bSplit, allSrcSplit[src_r0+1 + bSplit],
                        src_r1-1 + eSplit, allSrcSplit[src_r1-1 + eSplit]
            );

        assert(!(dst_r0+1 < nProc   && allDstSplit[dst_r0+1 + bSplit] <= mySrcGlobBegin));
        assert(!(dst_r1-1 >= dst_r0 && allDstSplit[dst_r1-1 + eSplit] >= mySrcGlobEnd));
        assert(!(src_r0+1 < nProc   && allSrcSplit[src_r0+1 + bSplit] <= myDstGlobBegin));
        assert(!(src_r1-1 >= src_r0 && allSrcSplit[src_r1-1 + eSplit] >= myDstGlobEnd));
      }

      { std::stringstream ss;
        ss << "dst[" << dst_r0 << "," << dst_r1 << "]  ";
        dstRangeStr = ss.str();
        ss.str("");
        ss << "src[" << src_r0 << "," << src_r1 << "]  ";
        srcRangeStr = ss.str();
      }

      if(printDebug) std::cerr << rankPrefix << dstRangeStr << srcRangeStr << "\n";

      // We know whom to exchange with. Next find distribution.
      const int src_range_sz = src_r1 - src_r0 + 1;
      const int dst_range_sz = dst_r1 - dst_r0 + 1;

      dstTo = std::vector<int> (dst_range_sz);
      srcFrom = std::vector<int> (src_range_sz);
      std::iota(dstTo.begin(), dstTo.end(), dst_r0);
      std::iota(srcFrom.begin(), srcFrom.end(), src_r0);
      dstCount = std::vector<int> (dst_range_sz, 0);
      srcCount = std::vector<int> (src_range_sz, 0);
      // otherDstBegin[r], mySrcBegin  <=  mySrcRank[x]  <  mySrcEnd, otherDstEnd[r];
      // otherSrcBegin[r], myDstBegin  <=  myDstRank[x]  <  myDstEnd, otherSrcEnd[r];
      for (int i = 0; i < dstTo.size(); ++i)
        dstCount[i] = (std::min(mySrcGlobEnd, allDstSplit[dstTo[i] + eSplit])
                               - std::max(mySrcGlobBegin, allDstSplit[dstTo[i] + bSplit]));
      for (int i = 0; i < srcFrom.size(); ++i)
        srcCount[i] = (std::min(myDstGlobEnd, allSrcSplit[srcFrom[i] + eSplit])
                               - std::max(myDstGlobBegin, allSrcSplit[srcFrom[i] + bSplit]));
      dstDspls = std::vector<int> (1, 0);
      srcDspls = std::vector<int> (1, 0);
      for (int c : dstCount)
        dstDspls.emplace_back(dstDspls.back() + c);
      for (int c : srcCount)
        srcDspls.emplace_back(srcDspls.back() + c);

      std::string dstCountStr, srcCountStr;
      { std::stringstream ss;
        ss << "[";
        for (int c : dstCount) { ss << c << ", "; }
        ss << "]";
        dstCountStr = ss.str();
        ss.str("");
        ss << "[";
        for (int c : srcCount) { ss << c << ", "; }
        ss << "]";
        srcCountStr = ss.str();
      }
      if(printDebug) std::cerr << rankPrefix << "dstCount==" << dstCountStr
                              << "    srcCount==" << srcCountStr << "\n";

      for (int i = 0; i < srcFrom.size(); ++i)
        if (srcFrom[i] == rProc)
          selfSrcI = i;

      for (int i = 0; i < dstTo.size(); ++i)
        if (dstTo[i] == rProc)
          selfDstI = i;

      // Empty self-sends/recvs don't count.
      if (selfSrcI != -1 && srcCount[selfSrcI] == 0)
        selfSrcI = -1;
      if (selfDstI != -1 && dstCount[selfDstI] == 0)
        selfDstI = -1;

      const bool selfConsistentSendRecv = (selfSrcI != -1) == (selfDstI != -1);
      if (!selfConsistentSendRecv)
      {
        std::cerr << rankPrefix << "**Violates self consistency. "
                  << dstRangeStr << srcRangeStr << "\n";
        assert(!"Violates self consistency");
      }
      if (selfSrcI != -1)
      {
        if (!(srcCount[selfSrcI] == dstCount[selfDstI]))
        {
          std::cerr << rankPrefix << "Sending different number to self than receiving from self.\n";
          assert(!"Sending different number to self than receiving from self.\n");
        }
      }
    }


    //
    // ShiftP2PContext::isendrecv()
    //
    template <typename T>
    ShiftP2PContext<T> & ShiftP2PContext<T>::isendrecv()
    {
      // ---------------------------------------------------------------
      // Point-to-point exchanges. Based on par::Mpi_Alltoallv_sparse().
      // ---------------------------------------------------------------

      if(printDebug) std::cerr << rankPrefix << "Begin point-to-point.\n";

      const int src_range_sz = srcFrom.size();
      const int dst_range_sz = dstTo.size();
      assert(requests->size() == src_range_sz + dst_range_sz);

      //First place all recv requests. Do not recv from self.
      int i = 0;
      for(i = 0; i < src_range_sz && srcFrom[i] < s->rProc; i++) {
        if(srcCount[i] > 0) {
          par::Mpi_Irecv( &(dstLocal[ndofs * srcDspls[i]]) , ndofs * srcCount[i], srcFrom[i], 1,
              s->comm, &((*requests)[commCnt++]) );
        }
      }
      if (i < src_range_sz && srcFrom[i] == s->rProc)
        i++;
      for(i; i < src_range_sz; i++) {
        if(srcCount[i] > 0) {
          par::Mpi_Irecv( &(dstLocal[ndofs * srcDspls[i]]) , ndofs * srcCount[i], srcFrom[i], 1,
              s->comm, &((*requests)[commCnt++]) );
        }
      }

      //Next send the messages. Do not send to self.
      for(i = 0; i < dst_range_sz && dstTo[i] < s->rProc; i++) {
        if(dstCount[i] > 0) {
          par::Mpi_Issend( &(srcLocal[ndofs * dstDspls[i]]), ndofs * dstCount[i], dstTo[i], 1,
              s->comm, &((*requests)[commCnt++]) );
        }
      }
      if (i < dst_range_sz && dstTo[i] == s->rProc)
        i++;
      for(; i < dst_range_sz; i++) {
        if(dstCount[i] > 0) {
          par::Mpi_Issend( &(srcLocal[ndofs * dstDspls[i]]), ndofs * dstCount[i], dstTo[i], 1,
              s->comm, &((*requests)[commCnt++]) );
        }
      }

      return *this;
    }


    //
    // ShiftP2PContext::local()
    //
    template <typename T>
    ShiftP2PContext<T> & ShiftP2PContext<T>::local()
    {
      // Now copy local portion.
      const int selfSrcI = s->selfSrcI;
      const int selfDstI = s->selfDstI;
      if (selfSrcI != -1 && srcCount[selfSrcI] > 0)
        std::copy_n( &srcLocal[ndofs * dstDspls[selfDstI]],
                     ndofs * dstCount[selfDstI],
                     &dstLocal[ndofs * srcDspls[selfSrcI]]);

      return *this;
    }


    //
    // ShiftP2PContext::wait()
    //
    template <typename T>
    ShiftP2PContext<T> & ShiftP2PContext<T>::wait()
    {
      if(printDebug) std::cerr << rankPrefix << "Waiting...\n";

      MPI_Waitall(commCnt, requests->data(), MPI_STATUSES_IGNORE);

      if(printDebug) std::cerr << rankPrefix << "  Done.\n";

      return *this;
    }

  }//namespace detail


  template <class TN>
  void dbg_printTNList(const TN *tnList, unsigned int listSz, unsigned int lev, MPI_Comm comm)
  {
    int rProc, nProc;
    MPI_Comm_rank(comm, &rProc);
    MPI_Comm_size(comm, &nProc);

    for (unsigned int r = 0; r < nProc; r++)
    {
      int sync;
      Mpi_Bcast(&sync, 1, r, comm);

      if (r != rProc)
        continue;

      fprintf(stderr, "Begin [%d].\n", rProc);
      for (unsigned int ii = 0; ii < listSz; ii++)
        fprintf(stderr, "[%d] treeNode %u: (%u)|%s\n", rProc, ii, tnList[ii].getLevel(), tnList[ii].getBase32Hex(lev).data());
    }

    int sync;
    Mpi_Bcast(&sync, 1, nProc-1, comm);
  }


  template <typename WeightT>
  double loadImbalance(WeightT localWeight, MPI_Comm comm)
  {
    int nProc;
    MPI_Comm_size(comm, &nProc);

    WeightT sumWeight = 0;
    par::Mpi_Allreduce(&localWeight, &sumWeight, 1, MPI_SUM, comm);

    double localImbalance = fabs((double) localWeight / (double) sumWeight - 1.0 / nProc);

    double maxImbalance;
    par::Mpi_Allreduce(&localImbalance, &maxImbalance, 1, MPI_MAX, comm);

    return maxImbalance;
  }


}//end namespace

