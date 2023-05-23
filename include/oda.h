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
#include "octUtils.h"
#include "distTree.h"
#include "lazy.hpp"

#include "filterFunction.h"

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

// Based on Dendro-5.0
namespace DA_FLAGS
{
  /**
   * @brief loop flags,
   * ALL : Loop over all the elements (note here we loop only on the local).
   * WRITABLE: Loop over INDEPENDENT U W_DEPENDENT
   * INDEPENDENT : Loop over all the local elements which DOES NOT point to any ghost node region.
   * W_DEPENDENT : Loop over all local elements which has AT LEAST ONE node which point to ghost node region
   * W_BOUNDARY : Loop over all the local elements which is on the domain boundary.
   /// * 
   /// * LOCAL_ELEMENTS : Loop over local elements of the mesh
   /// * PREGHOST_ELEMENTS : Loop over pre ghost elements. 
   /// * POSTGHOST_ELEMENTS : Loop over post ghost elements
   * 
   *
   * */
  /// enum LoopType {ALL,WRITABLE,INDEPENDENT,W_DEPENDENT,W_BOUNDARY,LOCAL_ELEMENTS,PREGHOST_ELEMENTS,POSTGHOST_ELEMENTS};
  enum LoopType {ALL,WRITABLE,INDEPENDENT,W_DEPENDENT,W_BOUNDARY};///,LOCAL_ELEMENTS,PREGHOST_ELEMENTS,POSTGHOST_ELEMENTS};

  /**
   * @brief contains the refine flags.
   * DA_NO_CHANGE : no change needed for the octant
   * DA_REFINE : refine the octant
   * DA_COARSEN: coarsen the octant.
   * **/
  enum Refine { DA_NO_CHANGE = OCT_FLAGS::OCT_NO_CHANGE,
                DA_REFINE    = OCT_FLAGS::OCT_REFINE,
                DA_COARSEN   = OCT_FLAGS::OCT_COARSEN };
}

template <unsigned int dim>
class DA;

template <unsigned int dim>
using MultiDA = std::vector<DA<dim>>;

/**
 * @brief Construct a DA representing the nodes of a hypercuboid with grid
 *         extents pow(2,a) * pow(2,b) * pow(2,c) * pow(2,d) (in 4D).
 * @param level Level of the uniform refinement.
 * @param extentPowers Array of powers {a,b,c,...} depending on the dimension. a,b,c,... <= level.
 * @param eleOrder Elemental order.
 * @param [out] newSubDA The resulting DA object.
 */
template <unsigned int dim>
DendroIntL constructRegularSubdomainDA(DA<dim> &newSubDA,
                                 unsigned int level,
                                 std::array<unsigned int, dim> extentPowers,
                                 unsigned int eleOrder,
                                 MPI_Comm comm,
                                 double sfc_tol = 0.3);

template <unsigned int dim>
DendroIntL constructRegularSubdomainDA(DA<dim> &newSubDA,
                                 std::vector<TreeNode<unsigned int, dim>> &newTreePart,
                                 unsigned int level,
                                 std::array<unsigned int, dim> extentPowers,
                                 unsigned int eleOrder,
                                 MPI_Comm comm,
                                 double sfc_tol = 0.3);


/**
 * @brief Creates a uniform coarse grid at level coarsestLevel,
 *        having the extents determined by extentPowers,
 *        then generates a grid hierarchy by subdividing uniformly
 *        the coarse grid until finestLevel is reached.
 */
template <unsigned int dim>
void constructRegularSubdomainDAHierarchy(
                                 std::vector<DA<dim>> &newMultiSubDA,
                                 std::vector<DA<dim>> &newSurrogateMultiSubDA,
                                 unsigned int coarsestLevel,
                                 unsigned int finestLevel,
                                 std::array<unsigned int, dim> extentPowers,
                                 unsigned int eleOrder,
                                 MPI_Comm comm,
                                 size_t grainSz = 100,
                                 double sfc_tol = 0.3);

/**
 * @brief Transfer data to an identically-structured but differently-partitioned grid, e.g. surrogate.
 *
 * @param srcDA Mesh-free representation of the source grid.
 * @param srcLocal Pointer to local segment of the source vector.
 * @param destDA Mesh-free representation of the destination grid.
 * @param destLocal Pointer to the local segment of the destination vector.
 * @param comm MPI communicator over which the transfer takes place. Need not equal the src/dest DA comms.
 */
template <unsigned int dim, typename DofT>
void distShiftNodes(const DA<dim> &srcDA, const DofT *srcLocal, const DA<dim> &destDA, DofT *destLocal, unsigned int ndofs = 1);


template <unsigned int dim>
class DA
{
  public:
    using C = unsigned int;    // Integer coordinate type.
    static constexpr unsigned template_dim = dim;

    /**
     * Returns IN to discard, OUT to keep, and INTERCEPTED to keep and mark as boundary.
     */
    using DomainDecider = ::ibm::DomainDecider;

    friend DendroIntL constructRegularSubdomainDA<dim>(
        DA<dim> &newSubDA,
        std::vector<TreeNode<unsigned int, dim>> &newTreePart,
        unsigned int level,
        std::array<unsigned int, dim> extentPowers,
        unsigned int eleOrder,
        MPI_Comm comm,
        double sfc_tol);


  private:

    /**@brief: dim of the problem*/  
    static constexpr unsigned int m_uiDim = dim; 

    /**@brief domain boundary node ids*/
    std::vector<size_t> m_uiBdyNodeIds;

    /**@brief total nodal size (with ghost nodes)*/
    size_t m_uiTotalNodalSz;

    /**@brief number of local nodes*/
    size_t m_uiLocalNodalSz;

    /**@brief number of local element sz*/ 
    size_t m_uiLocalElementSz;

    /**@brief: number of total element sz (local + ghost elements)*/
    /// size_t m_uiTotalElementSz;  // Our ghosts are node-based, not element.

    /**@brief pre ghost node begin*/
    size_t m_uiPreNodeBegin;

    /**@brief: pre ghost node end */
    size_t m_uiPreNodeEnd;

    /**@brief: local nodes begin*/ 
    size_t m_uiLocalNodeBegin;
    
    /**@brief: local nodes end*/ 
    size_t m_uiLocalNodeEnd;

    /**@brief: post ghost begin*/ 
    size_t m_uiPostNodeBegin;

    /**@brief: post ghost end*/ 
    size_t m_uiPostNodeEnd;

    /**@brief: position of local nodal segment in the distributed array. */
    DendroIntL m_uiGlobalRankBegin;

    /**@brief: position of local elements segment in the distributed array. */
    DendroIntL m_uiGlobalElementBegin;

    /**@brief: map, size of ghosted node array, gives global node indices.*/
    std::vector<RankI> m_uiLocalToGlobalNodalMap;
    
    /**@brief: internal scatter map. */
    ScatterMap m_sm;

    /**@brief: internal gather map. */
    GatherMap m_gm;

    size_t m_totalSendSz;
    size_t m_totalRecvSz;
    int m_numDestNeighbors;
    int m_numSrcNeighbors;

    /**@brief contexts for async data transfers*/
    mutable std::vector<AsyncExchangeContex> m_uiMPIContexts;

    /**@brief: mpi tags*/
    mutable unsigned int m_uiCommTag;

    /**@brief: total number of nodes accross all the processes*/
    DendroIntL m_uiGlobalNodeSz;

    /**@brief: total number of elements across all processes*/
    DendroIntL m_uiGlobalElementSz;

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
    TreeNode<C,dim> m_treePartFront;

    /**@brief: Last treeNode in the local partition of the tree.*/
    TreeNode<C,dim> m_treePartBack;

    /**@brief: coordinates of nodes in the vector. */
    std::vector<TreeNode<C,dim>> m_tnCoords;

    /**@brief: for each (ghosted) node, the global element id of owning element. */
    std::vector<DendroIntL> m_ghostedNodeOwnerElements;

    /**@brief: Input stratum of grid hierarchy for multigrid.
     * Only needed due to DistTree interface. */
    int m_stratum;

    mutable Lazy<std::vector<int>> m_elements_per_node;  // ghosted, note petsc wants local

    const DistTree<C, dim> *m_dist_tree;
    typename DistTree<C, dim>::LivePtr m_dist_tree_lifetime;

    //TODO I don't think RefElement member belongs in DA (distributed array),
    //  but it has to go somewhere that the polyOrder is known.
    //
    //  --> It is ok here, but it should be made a shared_ptr
    //      so that mutligrid only needs one refel.
    RefElement m_refel;


  private:

        /** @brief The latter part of construct() if already have ownedNodes and scatter/gather maps. */
        void construct(const std::vector<TreeNode<C,dim>> &ownedNodes,
                       const ScatterMap &sm,
                       const GatherMap &gm,
                       unsigned int eleOrder,
                       const TreeNode<C,dim> *treePartFront,
                       const TreeNode<C,dim> *treePartBack,
                       bool isActive,
                       MPI_Comm globalComm,
                       MPI_Comm activeComm);


  public:

        /**@brief: Constructor for the DA data structures
         * @param [in] in : input octree, need to be 2:1 balanced unique sorted octree.
         * @param [in] comm: MPI global communicator for mesh generation.
         * @param [in] order: order of the element.
         * @param [in] grainSz: Number of suggested elements per processor,
         * @param [in] sfc_tol: SFC partitioning tolerance,
         */
        DA(unsigned int order = 1);

        /// DA(std::vector<TreeNode<C,dim>> &inTree, MPI_Comm comm, unsigned int order, size_t grainSz = 100, double sfc_tol = 0.3);
        /// DA(const std::vector<TreeNode<C,dim>> &inTree, MPI_Comm comm, unsigned int order, size_t grainSz = 100, double sfc_tol = 0.3);

        DA(const DistTree<C,dim> &inDistTree, int stratum, MPI_Comm comm, unsigned int order, size_t grainSz /*= 100*/, double sfc_tol /*= 0.3*/);

        /**
         * @brief Construct DA assuming only one level of grid. Delegates to the "stratum" constructor using stratum=0.
         */
        DA(const DistTree<C,dim> &inDistTree, MPI_Comm comm, unsigned int order, size_t grainSz = 100, double sfc_tol = 0.3);

        // Construct multiple DA for multigrid.
        static void multiLevelDA(std::vector<DA> &outDAPerStratum, const DistTree<C, dim> &inDistTree, MPI_Comm comm, unsigned int order, size_t grainSz = 100, double sfc_tol = 0.3);


        /** @brief Construct oda for regular grid. */
        DA(MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol, std::vector<TreeNode<C, dim>> &outTreePart);

        /**@brief: Construct a DA from a function
         * @param [in] func : input function, we will produce a 2:1 balanced unique sorted octree from this.
         * @param [in] comm: MPI global communicator for mesh generation.
         * @param [in] order: order of the element.
         * @param [in] interp_tol: allowable interpolation error against func; controls refinement.
         * @param [in] grainSz: Number of suggested elements per processor.
         * @param [in] sfc_tol: SFC partitioning tolerance,
         */
        template <typename T>
        DA(std::function<void(const T *, T *)> func, unsigned int dofSz, MPI_Comm comm, unsigned int order, double interp_tol, size_t grainSz = 100, double sfc_tol = 0.3);

        /**
         * @brief deconstructor for the DA class.
         * */
        ~DA();


        // move constructor
        DA(DA &&movedDA) = default;


        /**
         * @brief does the work for the constructors.
         */
        /// void construct(const TreeNode<C,dim> *inTree, size_t nEle, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol);
        void constructStratum(const DistTree<C, dim> &distTree, int stratum, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol);
        void construct(const DistTree<C, dim> &distTree, MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol);

        /** @brief The latter part of construct() if already have ownedNodes. */
        void _constructInner(const std::vector<TreeNode<C,dim>> &ownedNodes,
                            const ScatterMap &sm,
                            const GatherMap &gm,
                            unsigned int eleOrder,
                            const TreeNode<C,dim> *treePartFront,
                            const TreeNode<C,dim> *treePartBack,
                            bool isActive,
                            MPI_Comm globalComm,
                            MPI_Comm activeComm);

        /**@brief returns the local element size*/
        inline size_t getLocalElementSz() const { return m_uiLocalElementSz; }

        /**@brief returns the local nodal size*/
        inline size_t getLocalNodalSz() const { return m_uiLocalNodalSz; }

        /**@brief returns the start offset for local node segment in ghosted vector. */
        inline size_t getLocalNodeBegin() const { return m_uiLocalNodeBegin; }

        /**@brief returns the pre ghost nodal size*/
        inline size_t getPreNodalSz() const { return (m_uiPreNodeEnd-m_uiPreNodeBegin); }

        /**@brief returns the post nodal size*/
        inline size_t getPostNodalSz() const { return (m_uiPostNodeEnd-m_uiPostNodeBegin); }

        /**@brief returns the total nodal size (this includes the ghosted region as well.)*/
        inline size_t getTotalNodalSz() const { return m_uiTotalNodalSz; }

        /**@brief returns the global number of nodes across all processors. */
        inline RankI getGlobalNodeSz() const { return m_uiGlobalNodeSz; }

        /**@brief returns the rank of the begining of local segment among all processors. */
        inline RankI getGlobalRankBegin() const { return m_uiGlobalRankBegin; }

        inline DendroIntL getGlobalElementSz() const { return m_uiGlobalElementSz; }

        inline DendroIntL getGlobalElementBegin() const { return m_uiGlobalElementBegin; }

        inline const std::vector<RankI> & getNodeLocalToGlobalMap() const { return m_uiLocalToGlobalNodalMap; }

        /**@brief see if the current DA is active*/
        inline bool isActive() const { return m_uiIsActive; }

        int getNumOutboundRanks() const { return m_sm.m_sendProc.size(); }

        int getNumInboundRanks() const { return m_gm.m_recvProc.size(); }

        size_t getTotalSendSz() const { return m_totalSendSz; }
        size_t getTotalRecvSz() const { return m_totalRecvSz; }
        int getNumDestNeighbors() const { return m_numDestNeighbors; }
        int getNumSrcNeighbors() const { return m_numSrcNeighbors; }

        /**@brief get number of nodes per element*/
        inline unsigned int getNumNodesPerElement() const { return m_uiNpE; }

        /**@brief get element order*/
        inline unsigned int getElementOrder() const { return m_uiElementOrder; }

        /**@brief: returns the global MPI communicator*/
        inline MPI_Comm getGlobalComm() const { return m_uiGlobalComm; }

        /**@brief: returns the active MPI sub com of the global communicator*/
        inline MPI_Comm getCommActive() const
        {
          return m_uiActiveComm;
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

        /**@brief: get pointer to the (ghosted) array of nodal coordinates. */
        inline const TreeNode<C,dim> * getTNCoords() const { return &(*m_tnCoords.cbegin()); }

        /**@brief get pointer to the (ghosted) array of owning elements. */
        inline const DendroIntL * getNodeOwnerElements() const { return &(*m_ghostedNodeOwnerElements.cbegin()); }

        /**@brief: get first treeNode of the local partition of the tree (front splitter). */
        inline const TreeNode<C,dim> * getTreePartFront() const { return (m_uiLocalElementSz > 0 ? &m_treePartFront : nullptr); }

        /**@brief: get last treeNode of the local partition of the tree (back splitter). */
        inline const TreeNode<C,dim> * getTreePartBack() const { return (m_uiLocalElementSz > 0 ? &m_treePartBack : nullptr); }

        //TODO again, I don't think RefElement belongs in DA, but it is for now. Maybe it belongs?
        inline const RefElement * getReferenceElement() const { return &m_refel; }

        /**@brief replaces bdyIndex with a copy of the boundary node indices. */
        inline void getBoundaryNodeIndices(std::vector<size_t> &bdyIndex) const { bdyIndex = m_uiBdyNodeIds; }

        /**@brief returns a const ref to boundary node indices. */
        inline const std::vector<size_t> & getBoundaryNodeIndices() const { return m_uiBdyNodeIds; }

        inline const std::vector<int> & elements_per_node() const;  // ghosted, note petsc wants local

        /**@brief: Pointer to the DistTree used in construction. */
        const DistTree<C, dim> * dist_tree() const
        {
          assert(not m_dist_tree_lifetime.expired());
          return m_dist_tree;
        }

        /**@brief: Stratum (in grid hierarchy) used in construction. */
        inline int stratum() const { return m_stratum; }

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
        void readFromGhostBegin(T *vec, unsigned int dof = 1) const;

        /**
          * @brief Sync the ghost element exchange
          * @note It is assumed the dofs {A,B,C} are stored ABC ABC ABC ABC.
          * */
        template <typename T>
        void readFromGhostEnd(T *vec, unsigned int dof = 1) const;

        /**
         * @brief Initiate accumilation across ghost elements
         * @note It is assumed the dofs {A,B,C} are stored ABC ABC ABC ABC.
         */
        template <typename T>
        void writeToGhostsBegin(T *vec, unsigned int dof = 1, const char * isDirtyOut = nullptr) const;

        /**
         * @brief Sync accumilation across ghost elements
         * @note It is assumed the dofs {A,B,C} are stored ABC ABC ABC ABC.
         */
        template <typename T>
        void writeToGhostsEnd(T *vec, unsigned int dof = 1, bool useAccumulation = true, const char * isDirtyOut = nullptr) const;

        /**
             * @brief convert nodal local vector with ghosted buffer regions.
             * @param[in] in: input vector (should be nodal and non ghosted)
             * @param[out] out: coverted nodal vector with ghost regions.
             * @param[in] isAllocated: true if the out is allocated, false otherwise.
             * @param[in] dof: degrees of freedoms
             * */
        template <typename T>
        void nodalVecToGhostedNodal(const T *in, T *&out, bool isAllocated = false, unsigned int dof = 1) const;

        template <typename T>
        void nodalVecToGhostedNodal(const T *in, T *&&out, unsigned int dof = 1) const; //if temporary pointer then can't return a new pointer; assume already allocated.

        /**
             * @brief convert ghosted nodal vector to local vector (without ghosting)
             * @param[in] gVec: ghosted vector
             * @param[out] local: local vector (assume an allocated vector)
             * @param[in] isAllocated: true if the out is allocated, false otherwise.
             * @param[in] dof: degrees of freedoms
             * */

        template <typename T>
        void ghostedNodalToNodalVec(const T *gVec, T *&local, bool isAllocated = false, unsigned int dof = 1) const;

        template <typename T>
        void ghostedNodalToNodalVec(const T *gVec, T *&&local, unsigned int dof = 1) const; //if temporary pointer then can't return a new pointer; assume already allocated.


        // std::vector versions
        template<typename T>
        void nodalVecToGhostedNodal(const std::vector<T> &in, std::vector<T> &out,bool isAllocated = false,unsigned int dof = 1) const;

        template<typename T>
        void ghostedNodalToNodalVec(const std::vector<T> gVec, std::vector<T> &local,bool isAllocated = false,unsigned int dof = 1) const;


        /**
             * @brief initialize a variable vector to a function depends on spatial coords.
             * @param[in/out] local: allocated vector, initialized vector -- offset by the dofIdx
             * @param[in] func: user specified function
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms -- dofStride
             * @note: Only iterates for a single variable. But, the variables are stored [abc][abc], so your function receives output ptr &[abc].
             *
             * */
        template <typename T>
        void setVectorByFunction(T *local, std::function<void(const T *, T *)> func, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

        /**
             * @brief initialize a variable vector to a function depends on spatial coords.
             * @param[in/out] local: allocated vector, initialized vector -- offset by the dofIdx
             * @param[in] value: user specified scalar values (size should be the  dof size)
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: total degrees of freedoms -- dofStride
             * @param [in] initDof: number of degrees of freedom to initialize
             * Note: Initialize the ghost region as well.
             *
             * */
        template <typename T>
        void setVectorByScalar(T *local, const T *value, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1, unsigned int initDof = 1) const;

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
         * @brief Finds the owner rank for each TreeNode, based on the front splitters.
         * @param[in] pNodes List of TreeNode points. Level will be reassigned as m_uiMaxDepth during search.
         * @param[in] n Number of TreeNode points to search.
         * @param[out] ownerRanks List of mpi ranks such that pNodes[i] belongs to ownerRanks[i].
         */
        void computeTreeNodeOwnerProc(const TreeNode<C, dim> * pNodes, unsigned int n, int* ownerRanks) const;




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
             * @brief initialize a variable vector to a function depends on spatial coords.
             * @param[in/out] local: allocated vector, initialized vector
             * @param[in] func: user specified function
             * @param [in] isElemental: True if creating a elemental vector (cell data vector) false for a nodal vector
             * @param [in] isGhosted: True will allocate ghost nodal values as well, false will only allocate memory for local nodes.
             * @param [in] dof: degrees of freedoms
             *
             * */
        template <typename T>
        void petscSetVectorByFunction(Vec &local, std::function<void(const T *, T *)> func, bool isElemental = false, bool isGhosted = false, unsigned int dof = 1) const;

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
        //PetscErrorCode petscSetValuesInMatrix(Mat mat, std::vector<MatRecord> &records, unsigned int dof, InsertMode mode) const;
     
        /**@brief: dealloc the petsc vector
         * */  
        PetscErrorCode petscDestroyVec(Vec & vec) const;

        #endif
};


} // end of namespace ot.


#include "oda.tcc"


#endif // end of DENDRO_KT_ODA_H
