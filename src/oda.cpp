/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando.
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class.
 * @date 04/04/2019
 **/

#include "oda.h"

#include <algorithm>

#define OCT_NO_CHANGE 0u
#define OCT_SPLIT 1u
#define OCT_COARSE 2u

namespace ot
{
    template <unsigned int dim>
    DA<dim>::DA() : m_refel{dim,1} {
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
    DA<dim>::DA(const ot::TreeNode<C,dim> *inTree, unsigned int nEle, MPI_Comm comm, unsigned int order, unsigned int grainSz, double sfc_tol)
        : m_refel{dim, order}
    {
        construct(inTree, nEle, comm, order, grainSz, sfc_tol);
    }

    template <unsigned int dim>
    void DA<dim>::construct(const ot::TreeNode<C,dim> *inTree, unsigned int nEle, MPI_Comm comm, unsigned int order, unsigned int grainSz, double sfc_tol)
    {
        //TODO
        // ???  leftover uninitialized member variables.
        //
        /// unsigned int m_uiLocalElementSz;
        /// unsigned int m_uiTotalElementSz;

        m_uiElementOrder = order;
        m_uiNpE = intPow(order + 1, dim);

        unsigned int intNodesPerEle = intPow(order - 1, dim);

        // TODO take into account grainSz and sfc_tol to set up activeComm.

        int nProc, rProc;

        m_uiGlobalComm = comm;
        MPI_Comm_size(m_uiGlobalComm, &nProc);
        MPI_Comm_rank(m_uiGlobalComm, &rProc);
        m_uiGlobalNpes = nProc;
        m_uiRankGlobal = rProc;

        // A processor is 'active' if it has elements, otherwise 'inactive'.
        m_uiIsActive = (nEle > 0);
        MPI_Comm_split(comm, (m_uiIsActive ? 1 : MPI_UNDEFINED), rProc, &m_uiActiveComm);

        if (m_uiIsActive)
        {
          MPI_Comm_size(m_uiActiveComm, &nProc);
          MPI_Comm_rank(m_uiActiveComm, &rProc);
          m_uiActiveNpes = nProc;
          m_uiRankActive = rProc;

          m_uiCommTag = 0;

          // Splitters for distributed exchanges.
          m_treePartFront = inTree[0];
          m_treePartBack = inTree[nEle-1];

          // Generate nodes from the tree. First, element-exterior nodes.
          std::vector<ot::TNPoint<C,dim>> nodeList;
          for (unsigned int ii = 0; ii < nEle; ii++)
              ot::Element<C,dim>(inTree[ii]).appendExteriorNodes(order, nodeList);

          // Count unique element-exterior nodes.
          unsigned int glbExtNodes = ot::SFC_NodeSort<C,dim>::dist_countCGNodes(nodeList, order, &m_treePartFront, &m_treePartBack, m_uiActiveComm);

          // Finish generating nodes from the tree - element-interior nodes.
          // TODO measure if keeping interior nodes at end of list good/bad for performance.
          for (unsigned int ii = 0; ii < nEle; ii++)
              ot::Element<C,dim>(inTree[ii]).appendInteriorNodes(order, nodeList);

          unsigned int locIntNodes = intNodesPerEle * nEle;
          unsigned int glbIntNodes = 0;
          par::Mpi_Allreduce(&locIntNodes, &glbIntNodes, 1, MPI_SUM, m_uiActiveComm);

          m_uiLocalNodalSz = nodeList.size();
          m_uiGlobalNodeSz = glbExtNodes + glbIntNodes;

          // Create scatter/gather maps. Scatter map reflects whatever ordering is in nodeList.
          m_sm = ot::SFC_NodeSort<C,dim>::computeScattermap(nodeList, &m_treePartFront, m_uiActiveComm);
          m_gm = ot::SFC_NodeSort<C,dim>::scatter2gather(m_sm, m_uiLocalNodalSz, m_uiActiveComm);

          // Export from gm: dividers between local and ghost segments.
          m_uiTotalNodalSz   = m_gm.m_totalCount;
          m_uiPreNodeBegin   = 0;
          m_uiPreNodeEnd     = m_gm.m_locOffset;
          m_uiLocalNodeBegin = m_gm.m_locOffset;
          m_uiLocalNodeEnd   = m_gm.m_locOffset + m_gm.m_locCount;
          m_uiPostNodeBegin  = m_gm.m_locOffset + m_gm.m_locCount;;
          m_uiPostNodeEnd    = m_gm.m_totalCount;

          // Note: We will offset the starting address whenever we copy with scattermap.
          // Otherwise we should build-in the offset to the scattermap here.

          // Find offset into the global array.
          unsigned long locSz = m_uiLocalNodalSz;
          par::Mpi_Scan(&locSz, &m_uiGlobalRankBegin, 1, MPI_SUM, m_uiActiveComm);
          m_uiGlobalRankBegin -= locSz;

          // Create vector of node coordinates, with ghost segments allocated.
          m_tnCoords.resize(m_uiTotalNodalSz);
          for (unsigned int ii = 0; ii < m_uiLocalNodalSz; ii++)
            m_tnCoords[m_uiLocalNodeBegin + ii] = nodeList[ii];
          nodeList.clear();

          // Fill ghost segments of node coordinates vector.
          std::vector<ot::TreeNode<C,dim>> tmpSendBuf(m_sm.m_map.size());
          ot::SFC_NodeSort<C,dim>::template ghostExchange<ot::TreeNode<C,dim>>(
              &(*m_tnCoords.begin()), &(*tmpSendBuf.begin()), m_sm, m_gm, m_uiActiveComm);
          //TODO transfer ghostExchange into this class, then use new method.

          // Identify the (local ids of) domain boundary nodes in local vector.
          // To use the ids in the ghosted vector you need to shift by m_uiLocalNodeBegin.
          m_uiBdyNodeIds.clear();
          for (unsigned int ii = 0; ii < m_uiLocalNodalSz; ii++)
          {
            if (m_tnCoords[ii + m_uiLocalNodeBegin].isOnDomainBoundary())
              m_uiBdyNodeIds.push_back(ii);
          }
        }
        else
        {
          m_uiLocalNodalSz = 0;
          m_uiGlobalNodeSz = 0;

          m_uiTotalNodalSz   = 0;
          m_uiPreNodeBegin   = 0;
          m_uiPreNodeEnd     = 0;
          m_uiLocalNodeBegin = 0;
          m_uiLocalNodeEnd   = 0;
          m_uiPostNodeBegin  = 0;
          m_uiPostNodeEnd    = 0;

          m_uiGlobalRankBegin = 0;
        }
    }


    template <unsigned int dim>
    DA<dim>::~DA()
    {
    }



    template <unsigned int dim>
    ot::DA<dim>* DA<dim>::remesh(const ot::TreeNode<C,dim> * oldTree,
                                 const DA_FLAGS::Refine * flags,
                                 unsigned int sz,
                                 unsigned int grainSz,
                                 double ld_bal,
                                 unsigned int sfK) const
    {
        const unsigned int localElementBegin = 0;
        const unsigned int localElementEnd= sz;
        bool isRemesh=false;
        bool isRemesh_g;
        MPI_Comm commGlobal = m_uiGlobalComm;

        const unsigned int NUM_CHILDREN = 1u << dim;

        std::vector<unsigned int> octflags;
        octflags.resize(sz,OCT_NO_CHANGE);

        const unsigned int levelDiff = log2(m_uiElementOrder);

        if(sz > 0)
        {
          const ot::TreeNode<C,dim>* allElements = oldTree;
          //1. check to see if we need a remesh.
          for(unsigned int i=0;i<sz;i++)
          {
            // We will enforce that all leaf siblings are on the same rank.
            //
            // 1. TODO need to enforce, while generating flags[], that
            //    1a. the root element is not marked for coarsening;
            //    1b. no elements shalower than m_uiMaxDepth-levelDiff are marked for splitting.
            //
            // 2. TODO How can we suppress coarsening of an element whose
            //    siblings are not actually present, but were previously refined?
            //
            unsigned int ele=i+localElementBegin;
            if(flags[i]==DA_FLAGS::Refine::DA_REFINE) {
              if((allElements[ele].getLevel()+levelDiff+1)>=m_uiMaxDepth)
              {
                octflags[i]=OCT_NO_CHANGE;
              }
              else
              {
                octflags[i]=OCT_SPLIT;
                isRemesh=true;
              }
            }
            else if(flags[i]==DA_FLAGS::Refine::DA_COARSEN)
            {
              if(((i+NUM_CHILDREN-1)<sz)  && (allElements[ele].getParent() == allElements[ele+NUM_CHILDREN-1].getParent()) && (allElements[ele].getLevel()>0))
              {
                for(unsigned int child=0;child<NUM_CHILDREN;child++)
                  octflags[i+child]=OCT_COARSE;

                isRemesh= true;
                i+=(NUM_CHILDREN-1);
              }
            }
            else
            {
              octflags[i]=OCT_NO_CHANGE;
            }
          }
        }


        MPI_Allreduce(&isRemesh,&isRemesh_g,1,MPI_CXX_BOOL,MPI_LOR,commGlobal);
        // return null da since no need to remesh
        if(!isRemesh_g)
            return NULL;

        // Build (unbalanced) tree from oldTree, obeying OCT_SPLIT or OCT_COARSE.
        std::vector<ot::TreeNode<C,dim>> newOctants;

        for (unsigned int octIdx = 0; octIdx < sz; octIdx++)
        {
          switch (octflags[octIdx])
          {
            case OCT_SPLIT:
              for (unsigned int c = 0; c < NUM_CHILDREN; c++)
                newOctants.push_back(oldTree[octIdx].getChildMorton(c));
              break;
            case OCT_COARSE:
              newOctants.push_back(oldTree[octIdx].getParent());
              octIdx += NUM_CHILDREN - 1;
              break;
            case OCT_NO_CHANGE:
              newOctants.push_back(oldTree[octIdx]);
              break;
          }
        }






        //TODO  Fill in implementation of ReMesh here.
        /// m_uiMesh->setOctreeRefineFlags(&(*(octflags.begin())),octflags.size());
        /// ot::Mesh* newMesh=m_uiMesh->ReMesh(grainSz,ld_bal,sfK);

        /// ot::DA<dim>* newDA= new DA(newMesh);

        // TODO consider how we will need to overlap elements for the
        // intergrid transfer.

        ot::DA<dim>* newDA = new DA();  //TODO

        return newDA;

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

        const PetscScalar * inArry=NULL;
        PetscScalar * outArry=NULL;

        VecGetArrayRead(in,&inArry);
        VecGetArray(out,&outArry);

        std::copy(inArry, inArry + dof*m_uiLocalNodalSz, outArry + dof*m_uiLocalNodeBegin);

        VecRestoreArrayRead(in,&inArry);
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

        const PetscScalar * gVecArry=NULL;
        PetscScalar * localArry=NULL;

        VecGetArrayRead(gVec,&gVecArry);
        VecGetArray(local,&localArry);

        std::copy(gVecArry + dof*m_uiLocalNodeBegin, gVecArry + dof*m_uiLocalNodeEnd, localArry);

        VecRestoreArrayRead(gVec,&gVecArry);
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
        const PetscScalar *arry=NULL;
        VecGetArrayRead(local,&arry);

        vecTopvtu(arry,fPrefix,nodalVarNames,isElemental,isGhosted,dof);

        VecRestoreArrayRead(local,&arry);
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



