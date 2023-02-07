/**
 * @brief: contains basic da (distributed array) functionality for the dendro-kt
 * @authors: Masado Ishii, Milinda Fernando.
 * School of Computiing, University of Utah
 * @note: based on dendro5 oda class.
 * @date 04/04/2019
 **/

#include "parUtils.h"
#include <sstream>

namespace ot
{

    namespace util
    {
      template <typename C, unsigned int dim>
      void constructRegularGrid(MPI_Comm comm, size_t grainSz, double sfc_tol, std::vector<ot::TreeNode<C,dim>> &outTree)
      {
        int nProc, rProc;
        MPI_Comm_size(comm, &nProc);
        MPI_Comm_rank(comm, &rProc);

        if (grainSz == 0)
          grainSz = 1;

        // numElements == pow(2, dim*endL); --> endL = roundUp(log(numElements)/dim);
        const size_t endL = (binOp::binLength(nProc*grainSz - 1) + dim - 1) / dim;
        const size_t numElem1D = 1u << endL;
        const size_t globNumElem = 1u << (endL*dim);
        const size_t len = 1u << (m_uiMaxDepth - endL);

        // To make a distributed regular grid (regular implies balanced),
        // follow the lexicographic order and then use distTreeSort().
        const size_t locEleCount = globNumElem / nProc + (rProc < globNumElem % nProc ? 1 : 0);
        const size_t locEleRank = globNumElem / nProc * rProc + (rProc < globNumElem % nProc ? rProc : globNumElem % nProc);

        std::array<C,dim> eleMultiIdx;
        size_t r = locEleRank, q = 0;
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
        for (size_t ii = 0; ii < locEleCount; ii++)
        {
          std::array<C,dim> eleCoords;
          for (int d = 0; d < dim; d++)
            eleCoords[d] = len * eleMultiIdx[d];
          outTree[ii] = ot::TreeNode<C,dim>(eleCoords, endL);

          incrementBaseB<C,dim>(eleMultiIdx, numElem1D);    // Lexicographic advancement.
        }

        SFC_Tree<C,dim>::distTreeSort(outTree, sfc_tol, comm);
        SFC_Tree<C, dim>::distRemoveDuplicates(
            outTree, sfc_tol, SFC_Tree<C, dim>::RM_DUPS_AND_ANC, comm);
        // There would only be duplicates in the periodic case.
      }

    }//namespace ot::util


    template <unsigned int dim>
    void constructRegularSubdomainDAHierarchy(
                                     std::vector<DA<dim>> &newMultiSubDA,
                                     std::vector<DA<dim>> &newSurrogateMultiSubDA,
                                     unsigned int coarsestLevel,
                                     unsigned int finestLevel,
                                     std::array<unsigned int, dim> extentPowers,
                                     unsigned int eleOrder,
                                     MPI_Comm comm,
                                     size_t grainSz,
                                     double sfc_tol)
    {
      using C = typename DA<dim>::C;

      std::array<double, dim> bounds;
      for (int d = 0; d < dim; ++d)
        bounds[d] = (1u << (m_uiMaxDepth - (coarsestLevel - extentPowers[d]))) / (1.0*(1u<<m_uiMaxDepth));

      DistTree<C, dim> dtree =
          DistTree<C, dim>::constructSubdomainDistTree( coarsestLevel,
                                                        DistTree<C, dim>::BoxDecider(bounds),
                                                        comm,
                                                        sfc_tol );

      DistTree<C, dim> surrogate =
          dtree.generateGridHierarchyDown(finestLevel-coarsestLevel+1, sfc_tol, comm);

      DA<dim>::multiLevelDA(newMultiSubDA, dtree, comm, eleOrder, grainSz, sfc_tol);
      DA<dim>::multiLevelDA(newSurrogateMultiSubDA, surrogate, comm, eleOrder, grainSz, sfc_tol);
    }


    template <unsigned int dim>
    DendroIntL constructRegularSubdomainDA(DA<dim> &newSubDA,
                                     std::vector<TreeNode<unsigned int, dim>> &newTreePart,
                                     unsigned int level,
                                     std::array<unsigned int, dim> extentPowers,
                                     unsigned int eleOrder,
                                     MPI_Comm comm,
                                     double sfc_tol)
    {
      using C = typename DA<dim>::C;

      newSubDA.m_refel = RefElement(dim, eleOrder);

      // =========================
      // Outline
      // =========================
      // * Generate tree in Morton order.
      // * Partition tree using sfc_tol
      // * DA construction.
      // =========================

      constexpr unsigned int NUM_CHILDREN = 1u << dim;
      constexpr unsigned int rotOffset = 2*NUM_CHILDREN;  // num columns in rotations[].

      int nProc, rProc;
      MPI_Comm_size(comm, &nProc);
      MPI_Comm_rank(comm, &rProc);

      for (int d = 0; d < dim; d++)
      {
        if (level < extentPowers[d])
        {
          fprintf(stderr, "WARNING [constructRegularSubdomainDA()]: "
                          "level(%u) must be >= extentPowers(%u). "
                          "Increasing level to %u to compensate.\n",
                          level, extentPowers[d], extentPowers[d]);
          level = extentPowers[d];
        }
      }

      std::array<double, dim> bounds;
      for (int d = 0; d < dim; ++d)
        bounds[d] = (1u << (m_uiMaxDepth - (level - extentPowers[d]))) / (1.0*(1u<<m_uiMaxDepth));

      DistTree<C, dim> dtree =
          DistTree<C, dim>::constructSubdomainDistTree( level,
                                                        typename DistTree<C, dim>::BoxDecider(bounds),
                                                        comm,
                                                        sfc_tol );

      newSubDA.construct(dtree, comm, eleOrder, 100, sfc_tol);

      newTreePart = dtree.getTreePartFiltered();

      return newSubDA.getLocalElementSz();
    }


    template <unsigned int dim>
    DendroIntL constructRegularSubdomainDA(DA<dim> &newSubDA,
                                     unsigned int level,
                                     std::array<unsigned int, dim> extentPowers,
                                     unsigned int eleOrder,
                                     MPI_Comm comm,
                                     double sfc_tol)
    {
      std::vector<TreeNode<unsigned int, dim>> unusedTree;
      return constructRegularSubdomainDA<dim>(newSubDA, unusedTree, level, extentPowers, eleOrder, comm, sfc_tol);
    }




    template <unsigned int dim, typename DofT>
    void distShiftNodes(const DA<dim> &srcDA,
                        const DofT *srcLocal,
                        const DA<dim> &dstDA,
                        DofT *dstLocal,
                        unsigned int ndofs)
    {
      constexpr bool printDebug = false;

      MPI_Comm comm = srcDA.getGlobalComm();
      int rProc, nProc;
      MPI_Comm_rank(comm, &rProc);
      MPI_Comm_size(comm, &nProc);

      std::string rankPrefix;
      { std::stringstream ss;
        ss << "[" << rProc << "] ";
        rankPrefix = ss.str();
      }
      if(printDebug) std::cerr << rankPrefix << "Enter. nProc==" << nProc << "\n";

      int compare_comms;
      MPI_Comm_compare(srcDA.getGlobalComm(), dstDA.getGlobalComm(), &compare_comms);
      const bool identical_comms = (compare_comms == MPI_IDENT);
      if (!identical_comms)
      {
        MPI_Comm dstComm = dstDA.getGlobalComm();
        int dst_rProc, dst_nProc;
        MPI_Comm_rank(dstComm, &dst_rProc);
        MPI_Comm_size(dstComm, &dst_nProc);

        std::stringstream ss;
        ss << "Error: srcDA(rank " << rProc << "/" << nProc << ") and "
           << "dstDA(rank " << dst_rProc << "/" << dst_nProc << ") "
           << "have different global comms!";
        std::cerr << ss.str() << "\n";
        assert(false);
      }

      if (nProc == 1)
      {
        if(printDebug) std::cerr << rankPrefix << "Single processor policy.\n";
        std::copy_n(srcLocal, ndofs * dstDA.getLocalNodalSz(), dstLocal);
        return;
      }

      par::shift<DofT>(
          comm,
          srcLocal, srcDA.getLocalNodalSz(), srcDA.getGlobalRankBegin(),
          dstLocal, dstDA.getLocalNodalSz(), dstDA.getGlobalRankBegin(),
          ndofs);
    }




    template <unsigned int dim>
    template <typename T>
    DA<dim>::DA(std::function<void(const T *, T *)> func, unsigned int dofSz, MPI_Comm comm, unsigned int order, double interp_tol, size_t grainSz, double sfc_tol)
        /// : m_refel{dim, order}
    {
      /// std::vector<unsigned int> varIndex(dofSz);
      /// for (unsigned int ii = 0; ii < dofSz; ii++)
      ///   varIndex[ii] = ii;

      /// // Get a complete tree sufficiently granular to represent func with accuracy interp_tol.
      /// std::vector<ot::TreeNode<C,dim>> completeTree;
      /// function2Octree<C,dim>(func, dofSz, &(*varIndex.cbegin()), dofSz, completeTree, m_uiMaxDepth, interp_tol, sfc_tol, order, comm);

      /// // Make the tree balanced, using completeTree as a minimal set of TreeNodes.
      /// // Calling distTreeBalancing() on a complete tree with ptsPerElement==1
      /// // should do exactly what we want.
      /// std::vector<ot::TreeNode<C,dim>> balancedTree;
      /// ot::SFC_Tree<C,dim>::distTreeBalancing(completeTree, balancedTree, 1, sfc_tol, comm);

      /// ot::DistTree<C,dim> distTree(balancedTree, comm);   // Uses default domain decider.

      /// // Create ODA based on balancedTree.
      /// construct(distTree, comm, order, grainSz, sfc_tol);

      throw std::logic_error(
          "This constructor is deprecated. Pass the arguments to an intermediate DistTree and then use the DistTree constructor:\n"
          "    DistTree<unsigned, dim> distTree = DistTree<unsigned, dim>::constructDistTreeByFunc(func, dofSz, comm, order, interp_tol, sfc_tol);\n"
          "    DA<dim> da(distTree, comm, order, grainSz, sfc_tol);\n"
          );
    }

    template <unsigned int dim>
    DA<dim>::DA(MPI_Comm comm, unsigned int order, size_t grainSz, double sfc_tol, std::vector<TreeNode<C,dim>> &outTreePart)
        : m_refel{dim, order}
    {
        // Ignore interp_tol and just pick a uniform refinement level to satisfy grainSz.
        std::vector<ot::TreeNode<C,dim>> tree;
        util::constructRegularGrid<C,dim>(comm, grainSz, sfc_tol, tree);

        ot::DistTree<C,dim> distTree(tree, comm);   // Uses default domain decider.

        outTreePart = distTree.getTreePartFiltered();

        construct(distTree, comm, order, grainSz, sfc_tol);
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
                {
                    throw std::logic_error("Ghosted elemental size not automatically computed.");
                    /// local=new T[dof*m_uiTotalElementSz];
                }
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
                {
                    throw std::logic_error("Ghosted elemental size not automatically computed.");
                    /// local.resize(dof*m_uiTotalElementSz);
                }
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
    void DA<dim>::nodalVecToGhostedNodal(const T* in, T* const & out,unsigned int dof) const
    {
      T *out_ptr = out;
      return this->nodalVecToGhostedNodal(in, out_ptr, true, dof);
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
    template<typename T>
    void DA<dim>::ghostedNodalToNodalVec(const T* gVec,T* const & local,unsigned int dof) const
    {
      T *local_ptr = local;
      return this->ghostedNodalToNodalVec(gVec, local_ptr, true, dof);
    }


    template <unsigned int dim>
    template<typename T>
    void DA<dim>::nodalVecToGhostedNodal(const std::vector<T> &in, std::vector<T> &out,bool isAllocated,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return;

        if(!isAllocated)
            createVector<T>(out,false,true,dof);

        // Assumes layout [abc][abc][...], so just need single shift.
        std::copy(in.cbegin(), in.cend(), out.begin() + dof*m_uiLocalNodeBegin);
    }

    template <unsigned int dim>
    template<typename T>
    void DA<dim>::ghostedNodalToNodalVec(const std::vector<T> gVec, std::vector<T> &local,bool isAllocated,unsigned int dof) const
    {
        if(!(m_uiIsActive))
            return;

        if(!isAllocated)
            createVector(local,false,false,dof);

        // Assumes layout [abc][abc][...], so just need single shift.
        typename std::vector<T>::const_iterator srcStart = gVec.cbegin() + dof*m_uiLocalNodeBegin;
        std::copy(srcStart, srcStart + dof*m_uiLocalNodalSz, local.begin());
    }



    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostBegin(T* vec,unsigned int dof) const
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

          const size_t upstBSz = m_uiTotalNodalSz - m_uiLocalNodalSz;
          const size_t dnstBSz = m_sm.m_map.size();

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
              size_t upstCount = dof*m_gm.m_recvCounts[upstIdx];
              size_t upstProc = m_gm.m_recvProc[upstIdx];
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
            for (size_t k = 0; k < dnstBSz; k++)
            {
              const T *nodeSrc = vec + dof * (m_sm.m_map[k] + m_uiLocalNodeBegin);
              std::copy(nodeSrc, nodeSrc + dof, dnstB + dof * k);
            }

            // 3b. Fire the sends.
            for (unsigned int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
            {
              T *dnstProcStart = dnstB + dof * m_sm.m_sendOffsets[dnstIdx];
              size_t dnstCount = dof*m_sm.m_sendCounts[dnstIdx];
              size_t dnstProc = m_sm.m_sendProc[dnstIdx];
              par::Mpi_Isend(dnstProcStart, dnstCount, dnstProc, m_uiCommTag, m_uiActiveComm, &reql[dnstIdx]);
            }
          }
        }

        m_uiCommTag++;   // inactive procs also advance tag.
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::readFromGhostEnd(T *vec,unsigned int dof) const
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
    void DA<dim>::writeToGhostsBegin(T *vec, unsigned int dof, const char * isDirtyOut) const
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

          char* dnstB_dirty = NULL;
          const char* upstB_dirty = NULL;

          // 1. Prepare asynchronous exchange context.
          const unsigned int nUpstProcs = m_gm.m_recvProc.size();
          const unsigned int nDnstProcs = m_sm.m_sendProc.size();
          m_uiMPIContexts.reserve(m_uiMPIContexts.size() + 2);
          m_uiMPIContexts.emplace_back(vec, typeid(T).hash_code(), nUpstProcs, nDnstProcs);
          AsyncExchangeContex &ctx = m_uiMPIContexts.back();

          AsyncExchangeContex *ctx_dirty = nullptr;
          if (isDirtyOut)
          {
            m_uiMPIContexts.emplace_back(isDirtyOut, typeid(char).hash_code(), nUpstProcs, nDnstProcs);
            ctx_dirty = &m_uiMPIContexts.back();
          }

          const size_t upstBSz = m_uiTotalNodalSz - m_uiLocalNodalSz;
          const size_t dnstBSz = m_sm.m_map.size();

          // 2. Initiate receives. (De-staging done in writeToGhostEnd().)
          if (dnstBSz)
          {
            ctx.allocateSendBuffer(sizeof(T)*dnstBSz*dof);  // Re-use the send buffer for receiving.
            dnstB = (T*) ctx.getSendBuffer();
            MPI_Request *reql = ctx.getDnstRequestList();

            for (unsigned int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
            {
              T *dnstProcStart = dnstB + dof * m_sm.m_sendOffsets[dnstIdx];
              size_t dnstCount = dof*m_sm.m_sendCounts[dnstIdx];
              size_t dnstProc = m_sm.m_sendProc[dnstIdx];
              par::Mpi_Irecv(dnstProcStart, dnstCount, dnstProc, m_uiCommTag, m_uiActiveComm, &reql[dnstIdx]);
            }

            if (isDirtyOut)
            {
              ctx_dirty->allocateSendBuffer(sizeof(char)*dnstBSz*1);  // Re-use the send buffer for receiving.
              dnstB_dirty = (char*) ctx_dirty->getSendBuffer();
              MPI_Request *reql = ctx_dirty->getDnstRequestList();

              for (unsigned int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
              {
                char *dnstProcStart = dnstB_dirty + 1 * m_sm.m_sendOffsets[dnstIdx];
                size_t dnstCount = 1*m_sm.m_sendCounts[dnstIdx];
                size_t dnstProc = m_sm.m_sendProc[dnstIdx];
                par::Mpi_Irecv(dnstProcStart, dnstCount, dnstProc, m_uiCommTag+1, m_uiActiveComm, &reql[dnstIdx]);
              }
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
              size_t upstCount = dof*m_gm.m_recvCounts[upstIdx];
              size_t upstProc = m_gm.m_recvProc[upstIdx];
              par::Mpi_Isend(upstProcStart, upstCount, upstProc, m_uiCommTag, m_uiActiveComm, &reql[upstIdx]);
            }

            if (isDirtyOut)
            {
              upstB_dirty = isDirtyOut;
              MPI_Request *reql = ctx_dirty->getUpstRequestList();

              for (unsigned int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
              {
                const char *upstProcStart = upstB_dirty + 1*m_gm.m_recvOffsets[upstIdx];
                size_t upstCount = 1*m_gm.m_recvCounts[upstIdx];
                size_t upstProc = m_gm.m_recvProc[upstIdx];
                par::Mpi_Isend(upstProcStart, upstCount, upstProc, m_uiCommTag+1, m_uiActiveComm, &reql[upstIdx]);
              }
            }
          }
        }

        m_uiCommTag++;   // inactive procs also advance tag.
        m_uiCommTag += bool(isDirtyOut);
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::writeToGhostsEnd(T *vec, unsigned int dof, bool useAccumulation, const char * isDirtyOut) const
    {
        // The same as readFromGhosts, but roles reversed:
        // Send to upstream, recv from downstream.

        if (m_uiGlobalNpes==1)
            return;

        if (!m_uiIsActive)
            return;

        T* dnstB = NULL;

        char* dnstB_dirty = NULL;

        // 1. Find asynchronous exchange context.
        MPI_Request *reql;
        MPI_Status status;
        auto const ctxPtr = std::find_if(
              m_uiMPIContexts.begin(), m_uiMPIContexts.end(),
              [vec](const AsyncExchangeContex &c){ return ((T*) c.getBuffer()) == vec; });

        // Weak type safety check.
        assert(ctxPtr->getBufferType() == typeid(T).hash_code());

        dnstB = (T*) ctxPtr->getSendBuffer();


        auto ctx_dirty = m_uiMPIContexts.begin();
        if (isDirtyOut)
        {
          ctx_dirty = std::find_if(
              m_uiMPIContexts.begin(), m_uiMPIContexts.end(),
              [isDirtyOut](const AsyncExchangeContex &c){ return ((char*) c.getBuffer()) == isDirtyOut; });
          assert(ctx_dirty->getBufferType() == typeid(char).hash_code());
          dnstB_dirty = (char*) ctx_dirty->getSendBuffer();
        }

        const unsigned int nUpstProcs = m_gm.m_recvProc.size();
        const unsigned int nDnstProcs = m_sm.m_sendProc.size();
        const size_t dnstBSz = m_sm.m_map.size();

        // 2. Wait on recvs.
        reql = ctxPtr->getDnstRequestList();
        for (int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
          MPI_Wait(&reql[dnstIdx], &status);

        if (isDirtyOut)
        {
          reql = ctx_dirty->getDnstRequestList();
          for (int dnstIdx = 0; dnstIdx < nDnstProcs; dnstIdx++)
            MPI_Wait(&reql[dnstIdx], &status);
        }

        // 3. "De-stage" the received downstream data.
        for (size_t k = 0; k < dnstBSz; k++)
        {
          if (isDirtyOut == nullptr || dnstB_dirty[k])
          {
            // Instead of simply copying from the downstream data, we need to accumulate it.
            const T *nodeSrc = dnstB + dof * k;
            /// std::copy(nodeSrc, nodeSrc + dof, vec + dof * m_sm.m_map[k]);
            for (unsigned int v = 0; v < dof; v++)
              if (useAccumulation)
                vec[dof * (m_sm.m_map[k] + m_uiLocalNodeBegin) + v] += nodeSrc[v];
              else
                vec[dof * (m_sm.m_map[k] + m_uiLocalNodeBegin) + v] = nodeSrc[v];
          }
        }

        // 4. Wait on sends.
        reql = ctxPtr->getUpstRequestList();
        for (int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
          MPI_Wait(&reql[upstIdx], &status);

        if (isDirtyOut)
        {
          reql = ctx_dirty->getUpstRequestList();
          for (int upstIdx = 0; upstIdx < nUpstProcs; upstIdx++)
            MPI_Wait(&reql[upstIdx], &status);
        }

        // 5. Release the asynchronous exchange context.
        /// ctxPtr->deAllocateRecvBuffer();
        ctxPtr->deAllocateSendBuffer();
        m_uiMPIContexts.erase(ctxPtr);

        if (isDirtyOut)
        {
          ctx_dirty->deAllocateSendBuffer();
          m_uiMPIContexts.erase(ctx_dirty);
        }
    }

    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByFunction(T* local,std::function<void(const T *, T*)>func,bool isElemental, bool isGhosted, unsigned int dof) const
    {
        constexpr int MAX_DIM = 4;  //TODO move to dendro.h
        T fCoords[MAX_DIM] = {0, 0, 0, 0};  // Supports funcs of 3D/4D even when run in 2D.
        std::array<C,dim> tnCoords;

        const T scale = 1.0 / (1u << m_uiMaxDepth);  // Root domain is unit cube.

        if (!isElemental)
        {
            const size_t nBegin = (isGhosted ? 0 : m_uiLocalNodeBegin);
            const size_t nodalSz = (isGhosted ? m_uiTotalNodalSz : m_uiLocalNodalSz);
            // Assumes interleaved variables, [abc][abc].
            for (size_t k = 0; k < nodalSz; k++)
            {
                m_tnCoords[nBegin + k].getAnchor(tnCoords);
                for (int d = 0; d < dim; d++)
                  fCoords[d] = scale * tnCoords[d];

                func(fCoords, &local[dof*k]);
            }
        }
        else
        {
          throw std::logic_error("Elemental version not implemented!");
        }
    }


    template <unsigned int dim>
    template <typename T>
    void DA<dim>::setVectorByScalar(T* local,const T* value,bool isElemental, bool isGhosted, unsigned int dof, unsigned int initDof) const
    {

        size_t arrSz;

        if(!isElemental)
        {

            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }

            for(size_t node=0;node<arrSz;node++)
                for(unsigned int var = 0; var < initDof; var++)
                    local[dof*node + var] = value[var];

        }else{
            // TODO, haven't considered elemental case.
            throw std::logic_error("Elemental version not implemented!");

            if(isGhosted) {
                throw std::logic_error("Ghosted elemental size not automatically computed.");
                /// arrSz = m_uiTotalElementSz;
            }else{
                arrSz=m_uiLocalElementSz;
            }

            for(unsigned int var=0;var<dof;var++)
            {
                for(size_t ele=0;ele<arrSz;ele++)
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

        size_t arrSz;

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

      throw std::logic_error("Not implemented!");
    }


    template <unsigned int dim>
    template<typename T>
    void DA<dim>::copyVector(T* dest,const T* source,bool isElemental,bool isGhosted) const
    {
        if(!(m_uiIsActive))
            return ;

        size_t arrSz;

        if(!isElemental)
        {
            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }


        }else{

            if(isGhosted) {
                throw std::logic_error("Ghosted elemental size not automatically computed.");
                /// arrSz = m_uiTotalElementSz;
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

        size_t arrSz;

        if(!isElemental)
        {
            if(isGhosted) {
                arrSz = m_uiTotalNodalSz;
            }else{
                arrSz=m_uiLocalNodalSz;
            }


        }else{

            if(isGhosted) {
                throw std::logic_error("Ghosted elemental size not automatically computed.");
                /// arrSz = m_uiTotalElementSz;
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



