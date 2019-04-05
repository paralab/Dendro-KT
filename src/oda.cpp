/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando.
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class.
 * @date 04/04/2019
 **/

#include "oda.h"

namespace ot
{
    template <unsigned int dim>
    DA<dim>::DA(){
        // Does nothing!
        m_uiTotalNodalSz = 0;
        m_uiLocalNodalSz = 0;
        m_uiLocalElementSz = 0;
        m_uiTotalElementSz = 0;
        m_uiPreNodeBegin = 0;
        m_uiPreNodeEnd = 0;
        m_uiLocalNodeBegin = 0;
        m_uiLocalNodeEnd = 0;
        m_uiPostNodeBegin = 0;
        m_uiPostNodeEnd = 0;
        m_uiCommTag = 0;
        m_uiGlobalNodeSz = 0;
        m_uiElementOrder = 0;
        m_uiNpE = 0;
        m_uiActiveNpes = 0;
        m_uiGlobalNpes = 0;
        m_uiRankActive = 0;
        m_uiRankGlobal = 0;
    }


    /**@brief: Constructor for the DA data structures
      * @param [in] inTree : input octree, need to be 2:1 balanced unique sorted octree.
      * @param [in] nEle : size of input octree.
      * @param [in] comm: MPI global communicator for mesh generation.
      * @param [in] order: order of the element.
     * */
    template <unsigned int dim>
    template <typename TN>
    DA<dim>::DA(const TN *inTree, unsigned int nEle, MPI_Comm comm, unsigned int order)
    {
        m_uiElementOrder = order;
        m_uiNpE = intPow(order + 1, dim);

        unsigned int intNodesPerEle = intPow(order - 1, dim);

        int nProc, rProc;
        MPI_Comm_size(comm, &nProc);
        MPI_Comm_rank(comm, &rProc);
        m_uiGlobalNpes = nProc;
        m_uiRankGlobal = rProc;
        m_uiGlobalComm = comm;

        // Splitters for distributed exchanges.
        const ot::TreeNode<C,dim> treeFront = inTree[0];
        const ot::TreeNode<C,dim> treeBack = inTree[nEle-1];

        // Generate nodes from the tree. First, element-exterior nodes.
        std::vector<ot::TNPoint<C,dim>> nodeList;
        for (const TN &tn : inTree)
            ot::Element<C,dim>(tn).appendExteriorNodes(order, nodeList);

        // Count unique element-exterior nodes.
        unsigned int glbExtNodes = ot::SFC_NodeSort<C,dim>::dist_countCGNodes(nodeList, order, &treeFront, &treeBack, comm);

        // Finish generating nodes from the tree - element-interior nodes.
        // TODO measure if keeping interior nodes at end of list good/bad for performance.
        for (const ot::TreeNode<C,dim> &tn : inTree)
            ot::Element<C,dim>(tn).appendInteriorNodes(order, nodeList);

        unsigned int locIntNodes = intNodesPerEle * nEle;
        unsigned int glbIntNodes = 0;
        par::Mpi_Allreduce(&locIntNodes, &glbIntNodes, 1, MPI_SUM, comm);

        m_uiLocalNodalSz = nodeList.size();
        m_uiGlobalNodeSz = glbExtNodes + glbIntNodes;

        //TODO I don't quite understand the AsyncExchangeContex class...
        m_uiMPIContexts.push_back({nullptr});  //TODO

        // Create scatter/gather maps.
        m_uiMPIContexts[0].getScatterMap() = ot::SFC_NodeSort<C,dim>::computeScattermap(nodeList, &treeFront, comm);
        m_uiMPIContexts[0].getGatherMap() = ot::SFC_NodeSort<C,dim>::scatter2gather(m_uiMPIContexts[0].getScatterMap(), m_uiLocalNodalSz, comm);

        // Import from gm: dividers between local and ghost segments.
        const ot::GatherMap &gm = m_uiMPIContexts[0].getGatherMap();
        m_uiTotalNodalSz   = gm.m_totalCount;
        m_uiPreNodeBegin   = 0;
        m_uiPreNodeEnd     = gm.m_locOffset;
        m_uiLocalNodeBegin = gm.m_locOffset;
        m_uiLocalNodeEnd   = gm.m_locOffset + gm.m_locCount;
        m_uiPostNodeBegin  = gm.m_locOffset + gm.m_locCount;;
        m_uiPostNodeEnd    = gm.m_totalCount;

        // Create vector of node coordinates, with ghost segments allocated.
        m_tnCoords.resize(m_uiTotalNodalSz);
        for (unsigned int ii = 0; ii < m_uiLocalNodalSz; ii++)
          m_tnCoords[m_uiLocalNodeBegin + ii] = nodeList[ii];
        nodeList.clear();

        // Fill ghost segments of node coordinates vector.
        const ot::ScatterMap &sm = m_uiMPIContexts[0].getScatterMap();
        std::vector<ot::TreeNode<C,dim>> tmpSendBuf(sm.m_map.size());
        ot::SFC_NodeSort<C,dim>::template ghostExchange<ot::TreeNode<C,dim>>(
            &(*m_tnCoords.begin()), &(*tmpSendBuf.begin()), sm, gm, comm);
        //TODO transfer ghostExchange into this class, then use new method.

        // ???  leftover uninitialized member variables.
        //
        /// std::vector<unsigned int> m_uiBdyNodeIds;
        /// bool m_uiIsActive;
        /// MPI_Comm m_uiActiveComm;
        /// unsigned int m_uiCommTag;
        /// unsigned int m_uiActiveNpes;
        /// unsigned int m_uiRankActive;
        /// unsigned int m_uiLocalElementSz;
        /// unsigned int m_uiTotalElementSz;
    }


    template <unsigned int dim>
    DA<dim>::~DA()
    {
    }


    // all the petsc functionalities goes below.
    #ifdef BUILD_WITH_PETSC

    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscCreateVector(Vec &local, bool isElemental, bool isGhosted, unsigned int dof) const
    {
        unsigned int sz=0;
        MPI_Comm globalComm=this->getGlobalComm();
        if(!m_uiIsActive)
        {
            local=NULL;

        }else {

            if(isElemental)
            {
                if(isGhosted)
                    sz=dof*m_uiTotalElementSz;
                else
                    sz=dof*m_uiLocalElementSz;

            }else {

                if(isGhosted)
                    sz=dof*m_uiTotalNodalSz;
                else
                    sz=dof*m_uiLocalNodalSz;
            }

        }

        VecCreate(globalComm,&local);
        PetscErrorCode status=VecSetSizes(local,sz,PETSC_DECIDE);

        if (this->getNpesAll() > 1) {
            VecSetType(local,VECMPI);
        } else {
            VecSetType(local,VECSEQ);
        }


        return status;


    }

    template <unsigned int dim>
    PetscErrorCode DA<dim>::createMatrix(Mat &M, MatType mtype, unsigned int dof) const
    {



        if(m_uiIsActive)
        {

            const unsigned int npesAll=m_uiGlobalNpes;
            const unsigned int eleOrder=m_uiElementOrder;
            // in linear cases, 53 can be generated with 27 + 27 -1(self) node.
            const unsigned int preAllocFactor=dof*(53*(eleOrder+1));

            // first determine the size ...
            unsigned int lSz = dof*(m_uiLocalNodalSz);
            MPI_Comm activeComm=m_uiActiveComm;

            PetscBool isAij, isAijSeq, isAijPrl, isSuperLU, isSuperLU_Dist;
            PetscStrcmp(mtype,MATAIJ,&isAij);
            PetscStrcmp(mtype,MATSEQAIJ,&isAijSeq);
            PetscStrcmp(mtype,MATMPIAIJ,&isAijPrl);
            isSuperLU = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU,&isSuperLU);
            isSuperLU_Dist = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU_DIST,&isSuperLU_Dist);

            MatCreate(activeComm, &M);
            MatSetSizes(M, lSz,lSz, PETSC_DETERMINE, PETSC_DETERMINE);
            MatSetType(M,mtype);


            if(isAij || isAijSeq || isAijPrl || isSuperLU || isSuperLU_Dist) {
                if(npesAll > 1) {
                    MatMPIAIJSetPreallocation(M, preAllocFactor , PETSC_NULL, preAllocFactor , PETSC_NULL);
                }else {
                    MatSeqAIJSetPreallocation(M, preAllocFactor, PETSC_NULL);
                }
            }

        }



        return 0;
    }



    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscNodalVecToGhostedNodal(const Vec& in,Vec& out,bool isAllocated,unsigned int dof) const
    {

        if(!(m_uiIsActive))
            return 0 ;

        unsigned int status=0;
        if(!isAllocated)
            status=petscCreateVector(out,false,true,dof);

        PetscScalar * inArry=NULL;
        PetscScalar * outArry=NULL;

        VecGetArray(in,&inArry);
        VecGetArray(out,&outArry);

        
        for(unsigned int var=0;var<dof;var++)
            std::memcpy((outArry+var*m_uiTotalNodalSz+m_uiLocalNodeBegin),(inArry+var*m_uiLocalNodalSz),sizeof(PetscScalar)*(m_uiLocalNodalSz));

        VecRestoreArray(in,&inArry);
        VecRestoreArray(out,&outArry);

        return status;

    }


    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscGhostedNodalToNodalVec(const Vec& gVec,Vec& local,bool isAllocated,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return 0;

        unsigned int status=0;
        if(!isAllocated)
            status=petscCreateVector(local,false,false,dof);

        PetscScalar * gVecArry=NULL;
        PetscScalar * localArry=NULL;

        VecGetArray(gVec,&gVecArry);
        VecGetArray(local,&localArry);

        for(unsigned int var=0;var<dof;var++)
            std::memcpy((localArry + var*m_uiLocalNodalSz ),(gVecArry+var*m_uiTotalNodalSz+m_uiLocalNodeBegin),sizeof(PetscScalar)*(m_uiLocalNodalSz));

        VecRestoreArray(gVec,&gVecArry);
        VecRestoreArray(local,&localArry);

        return status;

    }


    template <unsigned int dim>
    void DA<dim>::petscReadFromGhostBegin(PetscScalar* vecArry, unsigned int dof) 
    {
        if(!m_uiIsActive)
            return;

        readFromGhostBegin(vecArry,dof);

        return;

    }

    template <unsigned int dim>
    void DA<dim>::petscReadFromGhostEnd(PetscScalar* vecArry, unsigned int dof) 
    {
        if(!m_uiIsActive)
            return;

        readFromGhostEnd(vecArry,dof);

        return;

    }


    template <unsigned int dim>
    void DA<dim>::petscVecTopvtu(const Vec& local, const char * fPrefix,char** nodalVarNames,bool isElemental,bool isGhosted,unsigned int dof) 
    {

        PetscScalar *arry=NULL;
        VecGetArray(local,&arry);

        vecTopvtu(arry,fPrefix,nodalVarNames,isElemental,isGhosted,dof);

        VecRestoreArray(local,&arry);

    }


    
    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscChangeVecToMatBased(Vec& v1,bool isElemental,bool isGhosted, unsigned int dof) const
    {
        Vec tmp;
        petscCreateVector(tmp,isElemental,isGhosted,dof);
        unsigned int sz;
        if(isElemental)
        {
            if(isGhosted)
                sz=m_uiTotalElementSz;
            else
                sz=m_uiLocalElementSz;

        }else {

            if(isGhosted)
                sz=m_uiTotalNodalSz;
            else
                sz=m_uiLocalNodalSz;
        }
        
        PetscScalar * tmpArry=NULL;
        PetscScalar * v1Arry=NULL;

        VecGetArray(tmp,&tmpArry);
        VecGetArray(v1,&v1Arry);
        
        for(unsigned int node=0;node<sz;node++)
        {
            for(unsigned int var=0;var<dof;var++)
            {
                tmpArry[dof*node+var]=v1Arry[var*sz+node];
            }
        }
        
        VecRestoreArray(tmp,&tmpArry);
        VecRestoreArray(v1,&v1Arry);
        
       
        std::swap(tmp,v1);
        VecDestroy(&tmp);
        
        return 0;
    }

    
    
    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscChangeVecToMatFree(Vec& v1,bool isElemental,bool isGhosted,unsigned int dof) const
    {
        Vec tmp;
        petscCreateVector(tmp,isElemental,isGhosted,dof);
        unsigned int sz;
        if(isElemental)
        {
            if(isGhosted)
                sz=m_uiTotalElementSz;
            else
                sz=m_uiLocalElementSz;

        }else {

            if(isGhosted)
                sz=m_uiTotalNodalSz;
            else
                sz=m_uiLocalNodalSz;
        }
        
        PetscScalar * tmpArry=NULL;
        PetscScalar * v1Arry=NULL;

        VecGetArray(tmp,&tmpArry);
        VecGetArray(v1,&v1Arry);
        
        for(unsigned int node=0;node<sz;node++)
        {
            for(unsigned int var=0;var<dof;var++)
            {
                tmpArry[var*sz+node]=v1Arry[dof*node+var];
            }
        }
        
        VecRestoreArray(tmp,&tmpArry);
        VecRestoreArray(v1,&v1Arry);
        
       
        std::swap(tmp,v1);
        VecDestroy(&tmp);
        
        return 0;
    }


    template <unsigned int dim>
    PetscErrorCode DA<dim>::petscDestroyVec(Vec & vec)
    {
            VecDestroy(&vec);
            vec=NULL;
            return 0;
    }
    
    
    
#endif

}



