//
// Created by milinda on 11/19/18.
//

/**
 * @brief Simple class to manage async data transfer in the ODA class.
 * */


#ifndef DENDRO_KT_UPDATECTX_H
#define DENDRO_KT_UPDATECTX_H

#include "nsort.h"

#include "mpi.h"
#include <vector>
#include <typeinfo>

namespace ot {

    class AsyncExchangeContex {

        private :
            size_t m_bufferType;

            /** pointer to the variable which perform the ghost exchange */
            void* m_uiBuffer;

            /** pointer to the send buffer*/
            void* m_uiSendBuf;

            /** pointer to the send buffer*/
            void* m_uiRecvBuf;

            std::vector<MPI_Request> m_uiUpstRequests;
            std::vector<MPI_Request> m_uiDnstRequests;

            // Note: MPI_Request is not copy-assignable, but we assume ownership to avoid lots of new().
            // If you want to add these the below constructors, use something like std::shared_pointer
            // to manage m_uiUpstRequests and m_uiDnstRequests.
            AsyncExchangeContex(const AsyncExchangeContex &) = delete;
            void operator= (const AsyncExchangeContex &) = delete;

        public:
            /**@brief creates an async ghost exchange contex*/
            AsyncExchangeContex(const void* var, size_t bufferType, unsigned int nUpstProcs, unsigned int nDnstProcs)
            {
                m_bufferType = bufferType;
                m_uiBuffer=(void*)var;
                m_uiSendBuf=NULL;
                m_uiRecvBuf=NULL;
                m_uiUpstRequests.resize(nUpstProcs);
                m_uiDnstRequests.resize(nDnstProcs);
            }

            /**@brief allocates send buffer for ghost exchange*/
            inline void allocateSendBuffer(size_t bytes)
            {
                m_uiSendBuf=malloc(bytes);
            }

            /**@brief allocates recv buffer for ghost exchange*/
            inline void allocateRecvBuffer(size_t bytes)
            {
                m_uiRecvBuf=malloc(bytes);
            }

            /**@brief allocates send buffer for ghost exchange*/
            inline void deAllocateSendBuffer()
            {
                free(m_uiSendBuf);
                m_uiSendBuf=NULL;
            }

            /**@brief allocates recv buffer for ghost exchange*/
            inline void deAllocateRecvBuffer()
            {
                free(m_uiRecvBuf);
                m_uiRecvBuf=NULL;
            }

            inline size_t getBufferType() { return m_bufferType; }

            inline void* getSendBuffer() { return m_uiSendBuf;}
            inline void* getRecvBuffer() { return m_uiRecvBuf;}

            inline void* getBuffer() const {return m_uiBuffer;}

            /** @note Remember not to copy-assign MPI_Request. */
            inline MPI_Request * getUpstRequestList(){ return m_uiUpstRequests.data();}
            inline MPI_Request * getDnstRequestList(){ return m_uiDnstRequests.data();}

            bool operator== (const AsyncExchangeContex &other) const{
                return( m_uiBuffer == other.m_uiBuffer );
            }

            ~AsyncExchangeContex() {

               /* for(unsigned int i=0;i<m_uiRequests.size();i++)
                {
                    delete m_uiRequests[i];
                    m_uiRequests[i]=NULL;
                }

                m_uiRequests.clear();*/

            }

    };

} //end namespace

#endif //DENDRO_KT_UPDATECTX_H
