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

    /**@brief: Constructor for the DA data structures
      * @param [in] in : input octree, need to be 2:1 balanced unique sorted octree.
      * @param [in] comm: MPI global communicator for mesh generation.
      * @param [in] order: order of the element.
      * @param [in] grainSz: Number of suggested elements per processor,
      * @param [in] sfc_tol: SFC partitioning tolerance,
     * */
    DA::DA(){

        // todo: @masado, I think you should write the oda constructor. 
        // initialize all the vars related to cg nodes. 

    }

  
    DA::~DA()
    {
    }

    
    // all the petsc functionalities goes below.
    #ifdef BUILD_WITH_PETSC

    PetscErrorCode DA::petscCreateVector(Vec &local, bool isElemental, bool isGhosted, unsigned int dof) const
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

    PetscErrorCode DA::createMatrix(Mat &M, MatType mtype, unsigned int dof) const
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



    PetscErrorCode DA::petscNodalVecToGhostedNodal(const Vec& in,Vec& out,bool isAllocated,unsigned int dof) const
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


    PetscErrorCode DA::petscGhostedNodalToNodalVec(const Vec& gVec,Vec& local,bool isAllocated,unsigned int dof) const
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


    void DA::petscReadFromGhostBegin(PetscScalar* vecArry, unsigned int dof) 
    {
        if(!m_uiIsActive)
            return;

        readFromGhostBegin(vecArry,dof);

        return;

    }

    void DA::petscReadFromGhostEnd(PetscScalar* vecArry, unsigned int dof) 
    {
        if(!m_uiIsActive)
            return;

        readFromGhostEnd(vecArry,dof);

        return;

    }


    void DA::petscVecTopvtu(const Vec& local, const char * fPrefix,char** nodalVarNames,bool isElemental,bool isGhosted,unsigned int dof) 
    {

        PetscScalar *arry=NULL;
        VecGetArray(local,&arry);

        vecTopvtu(arry,fPrefix,nodalVarNames,isElemental,isGhosted,dof);

        VecRestoreArray(local,&arry);

    }


    
    PetscErrorCode DA::petscChangeVecToMatBased(Vec& v1,bool isElemental,bool isGhosted, unsigned int dof) const
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

    
    
    PetscErrorCode DA::petscChangeVecToMatFree(Vec& v1,bool isElemental,bool isGhosted,unsigned int dof) const
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


    PetscErrorCode DA::petscDestroyVec(Vec & vec)
    {
            VecDestroy(&vec);
            vec=NULL;
            return 0;
    }
    
    
    
#endif

}



