/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando. 
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class. 
 * @date 04/04/2019
 **/

namespace ot
{

    
    template <unsigned int dim>
    template <typename T>
    int DA<dim>::createVector(T*& local, bool isElemental, bool isGhosted, unsigned int dof) const
    {

        if(!(m_uiIsActive))
        {
            local=NULL;

        }else {

            if(isElemental)
            {
                if(isGhosted)
                    local=new T[dof*m_uiTotalElementSz];
                else
                    local=new T[dof*m_uiLocalElementSz];

            }else {

                if(isGhosted)
                    local=new T[dof*m_uiTotalNodalSz];
                else
                    local=new T[dof*m_uiLocalNodalSz];
            }
        }

        return 0;


    }

    template <unsigned int dim>
    template<typename T>
    int DA<dim>::createVector(std::vector<T>& local, bool isElemental, bool isGhosted, unsigned int dof) const
    {
        if(!(m_uiIsActive))
        {
            local.clear();

        }else {

            if(isElemental)
            {
                if(isGhosted)
                    local.resize(dof*m_uiTotalElementSz);
                else
                    local.resize(dof*m_uiLocalElementSz);

            }else {

                if(isGhosted)
                    local.resize(dof*m_uiTotalNodalSz);
                else
                    local.resize(dof*m_uiLocalNodalSz);
            }
        }

        return 0;
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::destroyVector(T*& local) const
    {
        delete [] local;
        local=NULL;
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::destroyVector(std::vector<T>& local) const
    {
        local.clear();
    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::nodalVecToGhostedNodal(const T* in, T*& out,bool isAllocated,unsigned int dof) const
    {

        if(!(m_uiIsActive))
            return;

        if(!isAllocated)
            createVector<T>(out,false,true,dof);

        for(unsigned int var=0;var<dof;var++)
        {
            std::memcpy((out+var*m_uiTotalNodalSz+m_uiLocalNodeBegin),(in+var*m_uiLocalNodalSz),sizeof(T)*(m_uiLocalNodalSz));
        }


    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::ghostedNodalToNodalVec(const T* gVec,T*& local,bool isAllocated,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return;

        if(!isAllocated)
            createVector(local,false,false,dof);

        for(unsigned int var=0;var<dof;var++)
            std::memcpy((local + var*m_uiLocalNodalSz ),(gVec+(var*m_uiTotalNodalSz)+m_uiLocalNodeBegin),sizeof(T)*(m_uiLocalNodalSz));

    }



    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostBegin(T* vec,unsigned int dof)
    {

        if(m_uiGlobalNpes==1)
            return;

        // send recv buffers.
        T* sendB = NULL;
        T* recvB = NULL;
        // todo: @masado could you write this part using your scatter map, (I left the commented code as a reference, once you done please remove it)
        /*if(m_uiIsActive)
        {
            const std::vector<unsigned int> nodeSendCount=m_uiMesh->getNodalSendCounts();
            const std::vector<unsigned int> nodeSendOffset=m_uiMesh->getNodalSendOffsets();

            const std::vector<unsigned int> nodeRecvCount=m_uiMesh->getNodalRecvCounts();
            const std::vector<unsigned int> nodeRecvOffset=m_uiMesh->getNodalRecvOffsets();

            const std::vector<unsigned int> sendProcList=m_uiMesh->getSendProcList();
            const std::vector<unsigned int> recvProcList=m_uiMesh->getRecvProcList();

            const std::vector<unsigned int> sendNodeSM=m_uiMesh->getSendNodeSM();
            const std::vector<unsigned int> recvNodeSM=m_uiMesh->getRecvNodeSM();


            const unsigned int activeNpes=m_uiMesh->getMPICommSize();

            const unsigned int sendBSz=nodeSendOffset[activeNpes-1] + nodeSendCount[activeNpes-1];
            const unsigned int recvBSz=nodeRecvOffset[activeNpes-1] + nodeRecvCount[activeNpes-1];
            unsigned int proc_id;

            AsyncExchangeContex ctx(vec);
            MPI_Comm commActive=m_uiMesh->getMPICommunicator();


            if(recvBSz)
            {
                ctx.allocateRecvBuffer((sizeof(T)*recvBSz*dof));
                recvB=(T*)ctx.getRecvBuffer();

                // active recv procs
                for(unsigned int recv_p=0;recv_p<recvProcList.size();recv_p++)
                {
                    proc_id=recvProcList[recv_p];
                    MPI_Request* req=new MPI_Request();
                    par::Mpi_Irecv((recvB+dof*nodeRecvOffset[proc_id]),dof*nodeRecvCount[proc_id],proc_id,m_uiCommTag,commActive,req);
                    ctx.getRequestList().push_back(req);

                }

            }

            if(sendBSz)
            {
                ctx.allocateSendBuffer(sizeof(T)*dof*sendBSz);
                sendB=(T*)ctx.getSendBuffer();

                for(unsigned int send_p=0;send_p<sendProcList.size();send_p++) {
                    proc_id=sendProcList[send_p];

                    for(unsigned int var=0;var<dof;var++)
                    {
                        for (unsigned int k = nodeSendOffset[proc_id]; k < (nodeSendOffset[proc_id] + nodeSendCount[proc_id]); k++)
                        {
                            sendB[dof*(nodeSendOffset[proc_id]) + (var*nodeSendCount[proc_id])+(k-nodeSendOffset[proc_id])] = (vec+var*m_uiTotalNodalSz)[sendNodeSM[k]];
                        }

                    }



                }

                // active send procs
                for(unsigned int send_p=0;send_p<sendProcList.size();send_p++)
                {
                    proc_id=sendProcList[send_p];
                    MPI_Request * req=new MPI_Request();
                    par::Mpi_Isend(sendB+dof*nodeSendOffset[proc_id],dof*nodeSendCount[proc_id],proc_id,m_uiCommTag,commActive,req);
                    ctx.getRequestList().push_back(req);

                }


            }

            m_uiCommTag++;
            m_uiMPIContexts.push_back(ctx);


        }*/

        return;

    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostEnd(T *vec,unsigned int dof)
    {
        if(m_uiGlobalNpes==1)
            return;

        // send recv buffers.
        /*T* sendB = NULL;
        T* recvB = NULL;

        if(m_uiIsActive)
        {
            const std::vector<unsigned int> nodeSendCount=m_uiMesh->getNodalSendCounts();
            const std::vector<unsigned int> nodeSendOffset=m_uiMesh->getNodalSendOffsets();

            const std::vector<unsigned int> nodeRecvCount=m_uiMesh->getNodalRecvCounts();
            const std::vector<unsigned int> nodeRecvOffset=m_uiMesh->getNodalRecvOffsets();

            const std::vector<unsigned int> sendProcList=m_uiMesh->getSendProcList();
            const std::vector<unsigned int> recvProcList=m_uiMesh->getRecvProcList();

            const std::vector<unsigned int> sendNodeSM=m_uiMesh->getSendNodeSM();
            const std::vector<unsigned int> recvNodeSM=m_uiMesh->getRecvNodeSM();


            const unsigned int activeNpes=m_uiMesh->getMPICommSize();

            const unsigned int sendBSz=nodeSendOffset[activeNpes-1] + nodeSendCount[activeNpes-1];
            const unsigned int recvBSz=nodeRecvOffset[activeNpes-1] + nodeRecvCount[activeNpes-1];
            unsigned int proc_id;

            unsigned int ctxIndex=0;
            for(unsigned int i=0;i<m_uiMPIContexts.size();i++)
            {
                if(m_uiMPIContexts[i].getBuffer()==vec)
                {
                    ctxIndex=i;
                    break;
                }

            }


            MPI_Status status;
            // need to wait for the commns to finish ...
            for (unsigned int i = 0; i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++) {
                MPI_Wait(m_uiMPIContexts[ctxIndex].getRequestList()[i], &status);
            }

            if(recvBSz)
            {
                // copy the recv data to the vec
                recvB=(T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();

                for(unsigned int recv_p=0;recv_p<recvProcList.size();recv_p++){
                    proc_id=recvProcList[recv_p];

                    for(unsigned int var=0;var<dof;var++)
                    {
                        for (unsigned int k = nodeRecvOffset[proc_id]; k < (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]); k++)
                        {
                            (vec+var*m_uiTotalNodalSz)[recvNodeSM[k]]=recvB[dof*(nodeRecvOffset[proc_id]) + (var*nodeRecvCount[proc_id])+(k-nodeRecvOffset[proc_id])];
                        }
                    }

                }

            }



            m_uiMPIContexts[ctxIndex].deAllocateSendBuffer();
            m_uiMPIContexts[ctxIndex].deAllocateRecvBuffer();

            for (unsigned int i = 0; i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++)
                delete m_uiMPIContexts[ctxIndex].getRequestList()[i];

            m_uiMPIContexts[ctxIndex].getRequestList().clear();

            // remove the context ...
            m_uiMPIContexts.erase(m_uiMPIContexts.begin() + ctxIndex);


        }

        return;*/


    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::writeToGhostsBegin(T *vec, unsigned int dof)
    {
        // todo @massado can you please write this part. 

    }

    
    template <unsigned int dim>
    template <typename T>
    void DA<dim>::writeToGhostsEnd(T *vec, unsigned int dof)
    {
        // todo @massado can you please write this part. 
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByFunction(T* local,std::function<void(T,T,T,T*)>func,bool isElemental, bool isGhosted, unsigned int dof) const
    {
        // todo @massado can you please write this part. 
        
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByScalar(T* local,const T* value,bool isElemental, bool isGhosted, unsigned int dof) const
    {

        unsigned int arrSz;

        if(!isElemental)
        {

            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }

            for(unsigned int var=0;var<dof;var++)
            {
                for(unsigned int node=0;node<arrSz;node++)
                    local[ (var*arrSz) + node]=value[var];
            }


        }else{

            if(isGhosted) {
                arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

            for(unsigned int var=0;var<dof;var++)
            {
                for(unsigned int ele=0;ele<arrSz;ele++)
                    local[ (var*arrSz) + ele]=value[var];
            }

        }


    }

    template <unsigned int dim>
    template<typename T>
    T* DA<dim>::getVecPointerToDof(T* in ,unsigned int dofInex, bool isElemental,bool isGhosted) const
    {

        if(!(m_uiIsActive))
            return NULL;

        unsigned int arrSz;

        if(!isElemental)
        {
            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }


        }else{

            if(isGhosted) {
                arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

        }

        return (T*)(&in[dofInex*arrSz]);

    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::vecTopvtu(T* local, const char * fPrefix,char** nodalVarNames,bool isElemental,bool isGhosted,unsigned int dof) 
    {

       
    }


    template <unsigned int dim>
    template<typename T>
    void DA<dim>::copyVector(T* dest,const T* source,bool isElemental,bool isGhosted) const
    {
        if(!(m_uiIsActive))
            return ;

        unsigned int arrSz;

        if(!isElemental)
        {
            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }


        }else{

            if(isGhosted) {
                arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

        }


        std::memcpy(dest,source,sizeof(T)*arrSz);

    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::copyVectors(T* dest,const T* source,bool isElemental,bool isGhosted,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return ;

        unsigned int arrSz;

        if(!isElemental)
        {
            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }


        }else{

            if(isGhosted) {
                arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

        }


        std::memcpy(dest,source,sizeof(T)*arrSz*dof);
    }


   


#ifdef BUILD_WITH_PETSC


    template <unsigned int dim>
    template<typename T>
    void DA<dim>::petscSetVectorByFunction(Vec& local,std::function<void(T,T,T,T*)>func,bool isElemental, bool isGhosted, unsigned int dof) const
    {

        PetscScalar * arry=NULL;
        VecGetArray(local,&arry);

        setVectorByFunction(arry,func,isElemental,isGhosted,dof);

        VecRestoreArray(local,&arry);


    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::petscSetVectorByScalar(Vec& local,const T* value,bool isElemental, bool isGhosted, unsigned int dof) const
    {

        PetscScalar * arry=NULL;
        VecGetArray(local,&arry);

        setVectorByScalar(arry,value,isElemental,isGhosted,dof);

        VecRestoreArray(local,&arry);

    }

#endif



}



