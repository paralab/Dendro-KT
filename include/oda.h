/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando. 
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class. 
 * @date 04/04/2019
 **/

#ifndef DENDRO_KT_ODA_H
#define DENDRO_KT_ODA_H

#include "asyncExchangeContex.h"
#include "dendro.h"
#include "mpi.h"
#include "treeNode.h"
#include "mathUtils.h"
#include "refel.h"
#include "binUtils.h"

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <cstring>


#ifdef BUILD_WITH_PETSC
#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#endif

#define VECType DendroScalar
#define MATType DendroScalar

namespace ot
{
template <unsigned int dim>
class DA
{
    using C = unsigned int;    // Integer coordinate type.

  private:

    /**@brief: dim of the problem*/  
    static constexpr unsigned int m_uiDim = dim; 

    //TODO m_uiBdyNodeIds, getOctreeBoundaryNodeIndices()
   
    /// /**@brief domain boundary node ids*/
    /// std::vector<unsigned int> m_uiBdyNodeIds;

    /**@brief total nodal size (with ghost nodes)*/
    unsigned int m_uiTotalNodalSz;

    /**@brief number of local nodes*/
    unsigned int m_uiLocalNodalSz;

    /**@brief number of local element sz*/ 
    unsigned int m_uiLocalElementSz;

    /**@brief: number of total element sz (local + ghost elements)*/
    unsigned int m_uiTotalElementSz;

    /**@brief pre ghost node begin*/
    unsigned int m_uiPreNodeBegin;

    /**@brief: pre ghost node end */
    unsigned int m_uiPreNodeEnd;

    /**@brief: local nodes begin*/ 
    unsigned int m_uiLocalNodeBegin;
    
    /**@brief: local nodes end*/ 
    unsigned int m_uiLocalNodeEnd;

    /**@brief: post ghost begin*/ 
    unsigned int m_uiPostNodeBegin;

    /**@brief: post ghost end*/ 
    unsigned int m_uiPostNodeEnd;
    
    /**@brief: internal scatter map. */
    ot::ScatterMap m_sm;

    /**@brief: internal gather map. */
    ot::GatherMap m_gm;

    /**@brief contexts for async data transfers*/
    std::vector<ot::AsyncExchangeContex> m_uiMPIContexts;

    /**@brief: mpi tags*/
    unsigned int m_uiCommTag;

    /**@brief: total number of nodes accross all the processes*/
    DendroIntL m_uiGlobalNodeSz;

    /**@brief: Global mpi communicator */ 
    MPI_Comm m_uiGlobalComm; 

    /**@brief: active mpi communicator (subset of the) m_uiGlobalComm*/
    MPI_Comm m_uiActiveComm;
    
    /**@brief: true if current DA is active, part of the active comm.*/
    bool m_uiIsActive;

    /**@brief: element order*/  
    unsigned int m_uiElementOrder;

    /**@brief: number of nodes per element*/ 
    unsigned int m_uiNpE;

    /**@brief: active number of procs */
    unsigned int m_uiActiveNpes;

    /**@brief: global number of procs*/
    unsigned int m_uiGlobalNpes;

    /**@brief: active rank w.r.t. to active comm.*/ 
    unsigned int m_uiRankActive;

    /**@brief: global rank w.r.t. to global comm. */  
    unsigned int m_uiRankGlobal;

    /**@brief: First treeNode in the local partition of the tree.*/
    ot::TreeNode<C,dim> m_treePartFront;

    /**@brief: Last treeNode in the local partition of the tree.*/
    ot::TreeNode<C,dim> m_treePartBack;

    /**@brief: coordinates of nodes in the vector. */
    std::vector<ot::TreeNode<C,dim>> m_tnCoords;

    //TODO I don't think RefElement member belongs in DA (distributed array),
    //  but it has to go somewhere that the polyOrder is known.
    RefElement m_refel;

  public:

        /**@brief: Constructor for the DA data structures
         * @param [in] in : input octree, need to be 2:1 balanced unique sorted octree.
         * @param [in] comm: MPI global communicator for mesh generation.
         * @param [in] order: order of the element.
         * @param [in] grainSz: Number of suggested elements per processor,
         * @param [in] sfc_tol: SFC partitioning tolerance,
         * */
        DA();

        DA(const ot::TreeNode<C,dim> *inTree, unsigned int nEle, MPI_Comm comm, unsigned int order, unsigned int grainSz = 100, double sfc_tol = 0.3);

        /**@biref: Construct a DA from a function
         *
         * */
        //TODO implement this!
        template <typename T>
        DA(std::function<void(T, T, T, T *)> func, unsigned int dofSz, MPI_Comm comm, unsigned int order, double interp_tol, unsigned int grainSz = 100, double sfc_tol = 0.3);

        /**
         * @brief deconstructor for the DA class.
         * */
        ~DA();

        /**@brief returns the local nodal size*/
        inline unsigned int getLocalNodalSz() const { return m_uiLocalNodalSz; }

        /**@brief returns the pre ghost nodal size*/
        inline unsigned int getPreNodalSz() const { return (m_uiPreNodeEnd-m_uiPreNodeBegin); }

        /**@brief returns the post nodal size*/
        inline unsigned int getPostNodalSz() const { return (m_uiPostNodeEnd-m_uiPostNodeBegin); }

        /**@brief returns the total nodal size (this includes the ghosted region as well.)*/
        inline unsigned int getTotalNodalSz() const { return m_uiTotalNodalSz; }

        /**@brief see if the current DA is active*/
        inline bool isActive() { return m_uiIsActive; }

        /**@brief get number of nodes per element*/
        inline unsigned int getNumNodesPerElement() const { return m_uiNpE; }

        /**@brief get element order*/
        inline unsigned int getElementOrder() const { return m_uiElementOrder; }

        /**@brief: returns the global MPI communicator*/
        inline MPI_Comm getGlobalComm() const { return m_uiGlobalComm; }

        /**@brief: returns the active MPI sub com of the global communicator*/
        inline MPI_Comm getCommActive() const
        {
            if (m_uiIsActive)
                return m_uiActiveComm;
            else
                return MPI_COMM_NULL;
        }

        /**@brief: global mpi com. size*/
        inline unsigned int getNpesAll() const { return m_uiGlobalNpes; };

        /**@brief: number of processors active */
        inline unsigned int getNpesActive() const
        {
            if (m_uiIsActive)
                return m_uiActiveNpes;
            else
                return 0;
        }

        /**@brief: rank with respect to the global comm. */
        inline unsigned int getRankAll() const { return m_uiRankGlobal; };

        /**@brief: rank w.r.t active comm.  */
        inline unsigned int getRankActive() const
        {
            if (m_uiIsActive)
                return m_uiRankActive;
            else
                return m_uiRankGlobal;
        }

        /**@brief: get the max depth of the octree*/
        inline unsigned int getMaxDepth() const { return m_uiMaxDepth; };
       
        /**@brief: get the dimensionality of the octree*/
        inline unsigned int getDimension() const { return m_uiDim; };

        /**@brief: get pointer to the (ghosted) array of nodal coordinates. */
        inline const ot::TreeNode<C,dim> * getTNCoords() const { return &(*m_tnCoords.cbegin()); }

        /**@brief: get first treeNode of the local partition of the tree (front splitter). */
        inline const ot::TreeNode<C,dim> * getTreePartFront() const { return &m_treePartFront; }

        /**@brief: get last treeNode of the local partition of the tree (back splitter). */
        inline const ot::TreeNode<C,dim> * getTreePartBack() const { return &m_treePartBack; }

        //TODO again, I don't think RefElement belongs in DA, but it is for now. Maybe it belongs?
        inline const RefElement * getReferenceElement() const { return &m_refel; }

        /**
          * @brief Creates a ODA vector
          * @param [in] local : VecType pointer
          * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
          * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
          * @param [in] dof: degrees of freedoms
          * */
        template <typename T>
        int createVector(T *&local, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**
          * @brief Creates a ODA vector std::vector<T>
          * @param [in] local : VecType pointer
          * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
          * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
          * @param [in] dof: degrees of freedoms
          * */
        template <typename T>
        int createVector(std::vector<T> &local, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**
             * @brief deallocates the memory allocated for a vector
             * @param[in/out] local: pointer to the vector
             * */
        template <typename T>
        void destroyVector(T *&local) const;

        template <typename T>
        void destroyVector(std::vector<T> &local) const;

        /**
          * @brief Initiate the ghost nodal value exchange.
          * @note It is assumed the dofs {A,B,C} are stored ABC ABC ABC ABC.
          * */
        template <typename T>
        void readFromGhostBegin(T *vec, unsigned int dof = 1);

        /**
          * @brief Sync the ghost element exchange
          * @note It is assumed the dofs {A,B,C} are stored ABC ABC ABC ABC.
          * */
        template <typename T>
        void readFromGhostEnd(T *vec, unsigned int dof = 1);

        /**
         * @brief Initiate accumilation across ghost elements
         * @note It is assumed the dofs {A,B,C} are stored ABC ABC ABC ABC.
         */
        template <typename T>
        void writeToGhostsBegin(T *vec, unsigned int dof = 1);

        /**
         * @brief Sync accumilation across ghost elements
         * @note It is assumed the dofs {A,B,C} are stored ABC ABC ABC ABC.
         */
        template <typename T>
        void writeToGhostsEnd(T *vec, unsigned int dof = 1);

        /**
             * @brief convert nodal local vector with ghosted buffer regions.
             * @param[in] in: input vector (should be nodal and non ghosted)
             * @param[out] out: coverted nodal vector with ghost regions.
             * @param[in] isAllocated: true if the out is allocated, false otherwise.
             * @param[in] dof: degrees of freedoms
             * */
        template <typename T>
        void nodalVecToGhostedNodal(const T *in, T *&out, bool isAllocated = false, unsigned int dof = 1) const;

        /**
             * @brief convert ghosted nodal vector to local vector (without ghosting)
             * @param[in] gVec: ghosted vector
             * @param[out] local: local vector (assume an allocated vector)
             * @param[in] isAllocated: true if the out is allocated, false otherwise.
             * @param[in] dof: degrees of freedoms
             * */

        template <typename T>
        void ghostedNodalToNodalVec(const T *gVec, T *&local, bool isAllocated = false, unsigned int dof = 1) const;

        /**
             * @brief initialize a variable vector to a function depends on spatial coords.
             * @param[in/out] local: allocated vector, initialized vector
             * @param[in] func: user specified function
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             *
             * */
        template <typename T>
        void setVectorByFunction(T *local, std::function<void(T, T, T, T *)> func, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**
             * @brief initialize a variable vector to a function depends on spatial coords.
             * @param[in/out] local: allocated vector, initialized vector
             * @param[in] value: user specified scalar values (size should be the  dof size)
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             * Note: Initialize the ghost region as well.
             *
             * */
        template <typename T>
        void setVectorByScalar(T *local, const T *value, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**@brief write the vec to pvtu file
             * @param[in] local: variable vector
             * @param[in] fPrefix: file name prefix
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             * */
        template <typename T>
        void vecTopvtu(T *local, const char *fPrefix, char **nodalVarNames = NULL, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1);

        /**
             * @brief returns a pointer to a dof index,
             * @param [in] in: input vector pointer
             * @param [in] dofInex: dof index which is the pointer is needed, should be less than dof, value the vector created.
             * @param [in] isElemental: true if this is an elemental vector/ false otherwise
             * @param [in] isGhosted: true if this is a ghosted vector
             * @return pointer to dofIndex.
             * */
        template <typename T>
        T *getVecPointerToDof(T *in, unsigned int dofInex, bool isElemental = false, bool isGhosted = false) const;

        /**
             * @brief copy vecotor to sorce to destination, assumes the same number of dof.
             * @param [in/out] dest: destination pointer
             * @param [in] source: source pointer
             * @param [in] isElemental: true if this is an elemental vector/ false otherwise
             * @param [in] isGhosted: true if this is a ghosted vector
             * @param [in] dof: degrees of freedoms
             * */

        template <typename T>
        void copyVectors(T *dest, const T *source, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**
             * @brief more premitive copy, from source pointer to the dest pointer
             * @param [in/out] dest: destination pointer
             * @param [in] source: source pointer
             * @param [in] isElemental: true if this is an elemental vector/ false otherwise
             * @param [in] isGhosted: true if this is a ghosted vector
             * */
        template <typename T>
        void copyVector(T *dest, const T *source, bool isElemental = false, bool isGhosted = false) const;

        // all the petsc functionalities goes below with the pre-processor gards.
        #ifdef BUILD_WITH_PETSC

        /**
             * @brief Creates a PETSC vector
             * @param [in] local : petsc vector
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             * */

        PetscErrorCode petscCreateVector(Vec &local, bool isElemental, bool isGhosted, unsigned int dof) const;

        /**
             @brief Returns a PETSc Matrix of appropriate size of the requested type.
            @param M the matrix
            @param mtype the type of matrix
            @param dof the number of degrees of freedom per node.
            */
        PetscErrorCode createMatrix(Mat &M, MatType mtype, unsigned int dof = 1) const;

        /**
             * @brief convert nodal local vector with ghosted buffer regions.
             * @param[in] in: input vector (should be nodal and non ghosted)
             * @param[out] out: coverted nodal vector with ghost regions.
             * @param[in] isAllocated: true if the out is allocated, false otherwise.
             * @param[in] dof: degrees of freedoms
             * */
        PetscErrorCode petscNodalVecToGhostedNodal(const Vec &in, Vec &out, bool isAllocated = false, unsigned int dof = 1) const;

        /**
            * @brief convert ghosted nodal vector to local vector (without ghosting)
            * @param[in] gVec: ghosted vector
            * @param[out] local: local vector (assume an allocated vector)
            * @param[in] isAllocated: true if the out is allocated, false otherwise.
            * @param[in] dof: degrees of freedoms
            * */

        PetscErrorCode petscGhostedNodalToNodalVec(const Vec &gVec, Vec &local, bool isAllocated = false, unsigned int dof = 1) const;

        /**
             * @brief Initiate the ghost nodal value exchange
             * @param[in] vec: vector in need to perform ghost exchange (Need be ghosted vector)
             * @param[in] vecArry: pointer to from the VecGetArray()
             * @param[in] dof: Degrees of freedoms
             * */

        void petscReadFromGhostBegin(PetscScalar *vecArry, unsigned int dof = 1);

        /**
             * @brief Sync the ghost element exchange
             * @param[in] vec: vector in need to perform ghost exchange (Need be ghosted vector)
             * @param[in] vecArry: pointer to from the VecGetArray()
             * @param[in] dof: Degrees of freedoms
             * */
        void petscReadFromGhostEnd(PetscScalar *vecArry, unsigned int dof = 1);

        /**
             * @brief initialize a variable vector to a function depends on spatial coords.
             * @param[in/out] local: allocated vector, initialized vector
             * @param[in] func: user specified function
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             *
             * */
        template <typename T>
        void petscSetVectorByFunction(Vec &local, std::function<void(T, T, T, T *)> func, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**
             * @brief initialize a variable vector to a function depends on spatial coords.
             * @param[in/out] local: allocated vector, initialized vector
             * @param[in] value: user specified scalar values (size should be the  dof size)
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             * Note: Initialize the ghost region as well.
             *
             * */
        template <typename T>
        void petscSetVectorByScalar(Vec &local, const T *value, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**@brief write the vec to pvtu file
             * @param[in] local: variable vector
             * @param[in] fPrefix: file name prefix
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             * */
        void petscVecTopvtu(const Vec &local, const char *fPrefix, char **nodalVarNames = NULL, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1);

        /**
             * @brief a wrapper for setting values into the Matrix.  This internally calls PETSc's MatSetValues() function.
             * Call PETSc's MatAssembly routines to assemble the matrix after setting the values. It would be more efficient to set values in chunks by
             calling this function multiple times with different sets of values instead of a single call at the end of the loop. One can use the size of 'records' to determine the number of
            such chunks. Calls to this function with the INSERT_VALUES and ADD_VALUES options cannot be mixed without intervening calls to PETSc's MatAssembly routines.
            * @param mat The matrix
            * @param records The values and their indices
            * @param dof the number of degrees of freedom per node
            * @param mode Either INSERT_VALUES or ADD_VALUES
            * @return an error flag
            * @note records will be cleared inside the function
            */
        //PetscErrorCode petscSetValuesInMatrix(Mat mat, std::vector<ot::MatRecord> &records, unsigned int dof, InsertMode mode) const;

        /** 
             * @brief: Dendro 5 vectors with multiple dof values are stored in 00000000,11111111 for multiple dof values, 
             * if the user wants to perform matrix based computations (i.e. not use matrix free option), this will reorder the 
             * vector valules 01,01,01,...  
             * @param [in,out] v1: input and out put vector, 
             * @param [in] dof: number of dof.  
             */
        PetscErrorCode petscChangeVecToMatBased(Vec &v1, bool isElemental, bool isGhosted, unsigned int dof = 1) const;

        /** 
             * @brief: This is the inverse function for the "petscChangeVecToMatBased"
             * @param [in,out] v1: input and out put vector, 
             * @param [in] dof: number of dof.  
             */
        PetscErrorCode petscChangeVecToMatFree(Vec &v1, bool isElemental, bool isGhosted, unsigned int dof = 1) const;
     
        /**@brief: dealloc the petsc vector
         * */  
        PetscErrorCode petscDestroyVec(Vec & vec);       

        #endif
};


template class DA<2u>;
template class DA<3u>;
template class DA<4u>;

} // end of namespace ot.


#include "oda.tcc"


#endif // end of DENDRO_KT_ODA_H
