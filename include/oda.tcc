/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando.
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class.
 * @date 04/04/2019
 **/

namespace ot
{

    namespace util
    {
      template <typename C, unsigned int dim>
      void constructRegularGrid(MPI_Comm comm, unsigned int grainSz, double sfc_tol, std::vector<ot::TreeNode<C,dim>> &outTree)
      {
        int nProc, rProc;
        MPI_Comm_size(comm, &nProc);
        MPI_Comm_rank(comm, &rProc);

        if (grainSz == 0)
          grainSz = 1;

        // numElements == pow(2, dim*endL); --> endL = roundUp(log(numElements)/dim);
        const unsigned int endL = (binOp::binLength(nProc*grainSz - 1) + dim - 1) / dim;
        const unsigned int numElem1D = 1u << endL;
        const unsigned int globNumElem = 1u << (endL*dim);
        const unsigned int len = 1u << (m_uiMaxDepth - endL);

        // To make a distributed regular grid (regular implies balanced),
        // follow the lexicographic order and then use distTreeSort().
        const unsigned int locEleCount = globNumElem / nProc + (rProc < globNumElem % nProc ? 1 : 0);
        const unsigned int locEleRank = globNumElem / nProc * rProc + (rProc < globNumElem % nProc ? rProc : globNumElem % nProc);

        std::array<C,dim> eleMultiIdx;
        unsigned int r = locEleRank, q = 0;
        eleMultiIdx[0] = 1;
        for (int d = 1; d < dim; d++)                // Build up strides.
          eleMultiIdx[d] = eleMultiIdx[d-1] * numElem1D;

        for (int d = 0; d < dim; d++)                // Compute start coords.
        {
          q = r / eleMultiIdx[dim-1 - d];
          r = r % eleMultiIdx[dim-1 - d];
          eleMultiIdx[dim-1 - d] = q;
        }

        // Create part of the tree in lexicographic order.
        outTree.resize(locEleCount);
        for (unsigned int ii = 0; ii < locEleCount; ii++)
        {
          std::array<C,dim> eleCoords;
          for (int d = 0; d < dim; d++)
            eleCoords[d] = len * eleMultiIdx[d];
          outTree[ii] = ot::TreeNode<C,dim>(1, eleCoords, endL);

          incrementBaseB<C,dim>(eleMultiIdx, numElem1D);    // Lexicographic advancement.
        }

        SFC_Tree<C,dim>::distTreeSort(outTree, sfc_tol, comm);
      }

    }//namespace ot::util



    template <unsigned int dim>
    template <typename T>
    DA<dim>::DA(std::function<void(const T *, T *)> func, unsigned int dofSz, MPI_Comm comm, unsigned int order, double interp_tol, unsigned int grainSz, double sfc_tol)
        : m_refel{dim, order}
    {
      std::vector<unsigned int> varIndex(dofSz);
      for (unsigned int ii = 0; ii < dofSz; ii++)
        varIndex[ii] = ii;

      // Get a complete tree sufficiently granular to represent func with accuracy interp_tol.
      std::vector<ot::TreeNode<C,dim>> completeTree;
      function2Octree<C,dim>(func, dofSz, &(*varIndex.cbegin()), dofSz, completeTree, m_uiMaxDepth, interp_tol, sfc_tol, order, comm);

      // Make the tree balanced, using completeTree as a minimal set of TreeNodes.
      // Calling distTreeBalancing() on a complete tree with ptsPerElement==1
      // should do exactly what we want.
      std::vector<ot::TreeNode<C,dim>> balancedTree;
      ot::SFC_Tree<C,dim>::distTreeBalancing(completeTree, balancedTree, 1, sfc_tol, comm);

      // Create ODA based on balancedTree.
      construct(&(*balancedTree.cbegin()), (RankI) balancedTree.size(), comm, order, grainSz, sfc_tol);
    }

    template <unsigned int dim>
    DA<dim>::DA(MPI_Comm comm, unsigned int order, unsigned int grainSz, double sfc_tol)
        : m_refel{dim, order}
    {
        // Ignore interp_tol and just pick a uniform refinement level to satisfy grainSz.
        std::vector<ot::TreeNode<C,dim>> tree;
        util::constructRegularGrid<C,dim>(comm, grainSz, sfc_tol, tree);

        construct(&(*tree.cbegin()), (unsigned int) tree.size(), comm, order, grainSz, sfc_tol);
    }

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

        // Assumes layout [abc][abc][...], so just need single shift.
        std::copy(in, in + dof*m_uiLocalNodalSz, out + dof*m_uiLocalNodeBegin);
    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::ghostedNodalToNodalVec(const T* gVec,T*& local,bool isAllocated,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return;

        if(!isAllocated)
            createVector(local,false,false,dof);

        // Assumes layout [abc][abc][...], so just need single shift.
        const T* srcStart = gVec + dof*m_uiLocalNodeBegin;
        std::copy(srcStart, srcStart + dof*m_uiLocalNodalSz, local);
    }



    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostBegin(T* vec,unsigned int dof)
    {
        // Send to downstream, recv from upstream.

        if (m_uiGlobalNpes==1)
            return;

        if (m_uiIsActive)
        {
          // send recv buffers.
          T* dnstB = NULL;
          T* upstB = NULL;

          // 1. Prepare asynchronous exchange context.
          const unsigned int nUpstProcs = m_gm.m_recvProc.size();
          const unsigned int nDnstProcs = m_sm.m_sendProc.size();
          m_uiMPIContexts.emplace_back(vec, typeid(T).hash_code(), nUpstProcs, nDnstProcs);
          AsyncExchangeContex &ctx = m_uiMPIContexts.back();

          const unsigned int upstBSz = m_uiTotalNodalSz - m_uiLocalNodalSz;
          const unsigned int dnstBSz = m_sm.m_map.size();

          // 2. Initiate recvs. Since vec is collated [abc abc], can receive into vec.
          if (upstBSz)
          {
            /// ctx.allocateRecvBuffer(sizeof(T)*upstBSz*dof);
            /// upstB = (T*) ctx.getRecvBuffer();
            upstB = vec;
            MPI_Request *reql = ctx.getUpstRequestList();

            for (unsigned int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
            {
              T *upstProcStart = upstB + dof*m_gm.m_recvOffsets[upstIdx];
              unsigned int upstCount = dof*m_gm.m_recvCounts[upstIdx];
              unsigned int upstProc = m_gm.m_recvProc[upstIdx];
              par::Mpi_Irecv(upstProcStart, upstCount, upstProc, m_uiCommTag, m_uiActiveComm, &reql[upstIdx]);
            }
          }

          // 3. Send data.
          if (dnstBSz)
          {
            ctx.allocateSendBuffer(sizeof(T)*dnstBSz*dof);
            dnstB = (T*) ctx.getSendBuffer();
            MPI_Request *reql = ctx.getDnstRequestList();

            // 3a. Stage the send data.
            for (unsigned int k = 0; k < dnstBSz; k++)
            {
              const T *nodeSrc = vec + dof * m_sm.m_map[k];
              std::copy(nodeSrc, nodeSrc + dof, dnstB + dof * k);
            }

            // 3b. Fire the sends.
            for (unsigned int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
            {
              T *dnstProcStart = dnstB + dof * m_sm.m_sendOffsets[dnstIdx];
              unsigned int dnstCount = dof*m_sm.m_sendCounts[dnstIdx];
              unsigned int dnstProc = m_sm.m_sendProc[dnstIdx];
              par::Mpi_Isend(dnstProcStart, dnstCount, dnstProc, m_uiCommTag, m_uiActiveComm, &reql[dnstIdx]);
            }
          }
        }

        m_uiCommTag++;   // inactive procs also advance tag.
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostEnd(T *vec,unsigned int dof)
    {
        if (m_uiGlobalNpes==1)
            return;

        if (!m_uiIsActive)
            return;

        // 1. Find asynchronous exchange context.
        MPI_Request *reql;
        MPI_Status status;
        auto const ctxPtr = std::find_if(
              m_uiMPIContexts.begin(), m_uiMPIContexts.end(),
              [vec](const AsyncExchangeContex &c){ return ((T*) c.getBuffer()) == vec; });

        // Weak type safety check.
        assert(ctxPtr->getBufferType() == typeid(T).hash_code());

        const unsigned int nUpstProcs = m_gm.m_recvProc.size();
        const unsigned int nDnstProcs = m_sm.m_sendProc.size();

        // 2. Wait on recvs and sends.
        reql = ctxPtr->getUpstRequestList();
        for (int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
          MPI_Wait(&reql[upstIdx], &status);

        reql = ctxPtr->getDnstRequestList();
        for (int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
          MPI_Wait(&reql[dnstIdx], &status);

        // 3. Release the asynchronous exchange context.
        /// ctxPtr->deAllocateRecvBuffer();
        ctxPtr->deAllocateSendBuffer();
        m_uiMPIContexts.erase(ctxPtr);
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::writeToGhostsBegin(T *vec, unsigned int dof)
    {
        // The same as readFromGhosts, but roles reversed:
        // Send to upstream, recv from downstream.

        if (m_uiGlobalNpes==1)
            return;

        if (m_uiIsActive)
        {
          // send recv buffers.
          T* dnstB = NULL;
          T* upstB = NULL;

          // 1. Prepare asynchronous exchange context.
          const unsigned int nUpstProcs = m_gm.m_recvProc.size();
          const unsigned int nDnstProcs = m_sm.m_sendProc.size();
          m_uiMPIContexts.emplace_back(vec, typeid(T).hash_code(), nUpstProcs, nDnstProcs);
          AsyncExchangeContex &ctx = m_uiMPIContexts.back();

          const unsigned int upstBSz = m_uiTotalNodalSz - m_uiLocalNodalSz;
          const unsigned int dnstBSz = m_sm.m_map.size();

          // 2. Initiate receives. (De-staging done in writeToGhostEnd().)
          if (dnstBSz)
          {
            ctx.allocateSendBuffer(sizeof(T)*dnstBSz*dof);  // Re-use the send buffer for receiving.
            dnstB = (T*) ctx.getSendBuffer();
            MPI_Request *reql = ctx.getDnstRequestList();

            for (unsigned int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
            {
              T *dnstProcStart = dnstB + dof * m_sm.m_sendOffsets[dnstIdx];
              unsigned int dnstCount = dof*m_sm.m_sendCounts[dnstIdx];
              unsigned int dnstProc = m_sm.m_sendProc[dnstIdx];
              par::Mpi_Irecv(dnstProcStart, dnstCount, dnstProc, m_uiCommTag, m_uiActiveComm, &reql[dnstIdx]);
            }
          }

          // 3. Send data. Since vec is collated [abc abc], ghosts can be sent directly from vec.
          if (upstBSz)
          {
            /// ctx.allocateRecvBuffer(sizeof(T)*upstBSz*dof);
            /// upstB = (T*) ctx.getRecvBuffer();
            upstB = vec;
            MPI_Request *reql = ctx.getUpstRequestList();

            for (unsigned int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
            {
              T *upstProcStart = upstB + dof*m_gm.m_recvOffsets[upstIdx];
              unsigned int upstCount = dof*m_gm.m_recvCounts[upstIdx];
              unsigned int upstProc = m_gm.m_recvProc[upstIdx];
              par::Mpi_Isend(upstProcStart, upstCount, upstProc, m_uiCommTag, m_uiActiveComm, &reql[upstIdx]);
            }
          }
        }

        m_uiCommTag++;   // inactive procs also advance tag.
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::writeToGhostsEnd(T *vec, unsigned int dof)
    {
        // The same as readFromGhosts, but roles reversed:
        // Send to upstream, recv from downstream.

        if (m_uiGlobalNpes==1)
            return;

        if (!m_uiIsActive)
            return;

        T* dnstB = NULL;

        // 1. Find asynchronous exchange context.
        MPI_Request *reql;
        MPI_Status status;
        auto const ctxPtr = std::find_if(
              m_uiMPIContexts.begin(), m_uiMPIContexts.end(),
              [vec](const AsyncExchangeContex &c){ return ((T*) c.getBuffer()) == vec; });

        // Weak type safety check.
        assert(ctxPtr->getBufferType() == typeid(T).hash_code());

        dnstB = (T*) ctxPtr->getSendBuffer();

        const unsigned int nUpstProcs = m_gm.m_recvProc.size();
        const unsigned int nDnstProcs = m_sm.m_sendProc.size();
        const unsigned int dnstBSz = m_sm.m_map.size();

        // 2. Wait on recvs.
        reql = ctxPtr->getDnstRequestList();
        for (int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
          MPI_Wait(&reql[dnstIdx], &status);

        // 3. "De-stage" the received downstream data.
        for (unsigned int k = 0; k < dnstBSz; k++)
        {
          // Instead of simply copying from the downstream data, we need to accumulate it.
          const T *nodeSrc = dnstB + dof * k;
          /// std::copy(nodeSrc, nodeSrc + dof, vec + dof * m_sm.m_map[k]);
          for (unsigned int v = 0; v < dof; v++)
            vec[dof * m_sm.m_map[k] + v] += nodeSrc[v];
        }

        // 4. Wait on sends.
        reql = ctxPtr->getUpstRequestList();
        for (int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
          MPI_Wait(&reql[upstIdx], &status);

        // 5. Release the asynchronous exchange context.
        /// ctxPtr->deAllocateRecvBuffer();
        ctxPtr->deAllocateSendBuffer();
        m_uiMPIContexts.erase(ctxPtr);
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByFunction(T* local,std::function<void(const T *, T*)>func,bool isElemental, bool isGhosted, unsigned int dof) const
    {
        // todo @massado can you please write this part.

        // TODO allow multiple dimensionality of input function. Until then, workaround:
        constexpr unsigned int edim = (dim < 3 ? dim : 3);
        T fCoords[3] = {0, 0, 0};
        std::array<C,dim> tnCoords;

        const T scale = 1.0 / (1u << m_uiMaxDepth);

        //TODO initialize the ghost segments too? I will for now...

        if (!isElemental)
        {
            const unsigned int nodalSz = (isGhosted ? m_uiTotalNodalSz : m_uiLocalNodalSz);
            // Assumes interleaved variables, [abc][abc].
            for (unsigned int k = 0; k < nodalSz; k++)
            {
                m_tnCoords[k].getAnchor(tnCoords);
                #pragma unroll(edim)
                for (int d = 0; d < edim; d++)
                  fCoords[d] = scale * tnCoords[d];

                func(fCoords, &local[dof*k]);
            }
        }
        else
        {
          //TODO
        }
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByScalar(T* local,const T* value,bool isElemental, bool isGhosted, unsigned int dof, unsigned int initDof) const
    {

        unsigned int arrSz;

        if(!isElemental)
        {

            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }

            for(unsigned int node=0;node<arrSz;node++)
                for(unsigned int var = 0; var < initDof; var++)
                    local[dof*node + var] = value[var];

        }else{
            // TODO, haven't considered elemental case.

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
    T* DA<dim>::getVecPointerToDof(T* in ,unsigned int dofInex, bool isElemental,bool isGhosted, unsigned int dof) const
    {

        if(!(m_uiIsActive))
            return NULL;

        unsigned int arrSz;

        /// if(!isElemental)
        /// {
        ///     if(isGhosted) {
        ///         arrSz = m_uiTotalNodalSz;
        ///     }else{
        ///         arrSz=m_uiLocalNodalSz;
        ///     }


        /// }else{

        ///     if(isGhosted) {
        ///         arrSz = m_uiTotalElementSz;
        ///     }else{
        ///         arrSz=m_uiLocalElementSz;
        ///     }

        /// }

        return in + dofInex;
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::vecTopvtu(T* local, const char * fPrefix,char** nodalVarNames,bool isElemental,bool isGhosted,unsigned int dof)
    {
      //TODO

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
    void DA<dim>::petscSetVectorByFunction(Vec& local,std::function<void(const T *, T*)>func,bool isElemental, bool isGhosted, unsigned int dof) const
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



