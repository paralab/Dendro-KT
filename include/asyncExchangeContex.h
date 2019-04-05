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

            std::vector<MPI_Request> m_uiUpstrRequests;
            std::vector<MPI_Request> m_uiDnstrRequests;

        public:
            /**@brief creates an async ghost exchange contex*/
            AsyncExchangeContex(const void* var, size_t bufferType, unsigned int nUpstrProcs, unsigned int nDnstrProcs)
            {
                m_bufferType = bufferType;
                m_uiBuffer=(void*)var;
                m_uiSendBuf=NULL;
                m_uiRecvBuf=NULL;
                m_uiUpstrRequests.resize(nUpstrProcs);
                m_uiDnstrRequests.resize(nDnstrProcs);
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

            inline const void* getBuffer() {return m_uiBuffer;}

            inline MPI_Request * getUpstrRequestList(){ return m_uiUpstrRequests.data();}
            inline MPI_Request * getDnstrRequestList(){ return m_uiDnstrRequests.data();}

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
