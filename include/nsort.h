/**
 * @file:nsort.h
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-02-20
 * @brief: Variations of the TreeSort algorithm (tsort.h) for fem-nodes in 4D+.
 */

#ifndef DENDRO_KT_NSORT_H
#define DENDRO_KT_NSORT_H

#include "tsort.h"
#include "treeNode.h"
#include "hcurvedata.h"
#include "parUtils.h"
#include "binUtils.h"

#include "filterFunction.h"

#include <iostream>
#include <bitset>

#include <mpi.h>
#include <vector>
#include <queue>
#include <unordered_set>
#include <map>
#include <functional>
#include <stdio.h>

namespace ot {

  /**@brief Identification of face-interior/edge-interior etc., for nodes.
   * @description There are OuterDim+1 cell dimensions and pow(2,OuterDim) total cell orientations.
   *     The less-significant bits are used to store orientation, while the
   *     more-significant bits are used as redundant quick-access flags for the dimension.
   *     Orientation is represented as a bitvector, with a bit being set if that axis is part of the cell volume.
   *     A 0-cell (point) has all bits set to 0, while a OuterDim-cell (whole volume) has all bits set to 1.
   *     The cell dimension is how many bits are set to 1.
   * @note If the embedding dimension is OuterDim<=5, then unsigned char suffices.
   * @note The template information is not used right now since OuterDim is assumed to be 4 or less.
   *       However, making this a template means that CellType for higher OuterDims than 5 can be made
   *       as template instantiations later.
   */
  template <unsigned int OuterDim>
  struct CellType
  {
    using FlagType = unsigned char;

    /**@brief Constructs a 0-cell (point) inside the hypercube of dimension OuterDim. */
    CellType() { m_flag = 0; }

    // No checks are performed to make sure that c_orient is consistent with c_dim.
    operator FlagType&() { return m_flag; }
    operator FlagType() { return m_flag; }

    FlagType get_dim_flag() const { return m_flag >> m_shift; }
    FlagType get_orient_flag() const { return m_flag & m_mask; }

    void set_dimFlag(FlagType c_dim) { m_flag = (m_flag & m_mask) | (c_dim << m_shift); }
    void set_orientFlag(FlagType c_orient) { m_flag = (m_flag & (~m_mask)) | (c_orient & m_mask); }

    // TODO void set_flags(FlagType c_orient); // Counts 1s in c_orient and uses count for c_dim.

    static std::array<CellType, (1u<<OuterDim)-1> getExteriorOrientHigh2Low();
    static std::array<CellType, (1u<<OuterDim)-1> getExteriorOrientLow2High();

    // Data members.
    FlagType m_flag;

    private:
      // Usage: prefix=0u, lengthLeft=OuterDim, onesLeft=faceDimension, dest=arrayStart.
      // Recursion depth is equal to onesLeft.
      static void emitCombinations(FlagType prefix, unsigned char lengthLeft, unsigned char onesLeft, CellType * &dest);

      static const FlagType m_shift = 4u;
      static const FlagType m_mask = (1u << m_shift) - 1;
  };


  /**@brief TreeNode + extra attributes to keep track of node uniqueness. */
  template <typename T, unsigned int dim>
  class TNPoint : public TreeNode<T,dim>
  {
    public:
      /**
       * @brief Constructs a node at the extreme "lower-left" corner of the domain.
       */
      TNPoint();

      /**
        @brief Constructs a point.
        @param coords The coordinates of the point.
        @param level The level of the point (i.e. level of the element that spawned it).
        @note Uses the "dummy" overload of TreeNode() so that the coordinates are copied as-is.
        */
      TNPoint (const std::array<T,dim> coords, unsigned int level);

      /**@brief Copy constructor */
      TNPoint (const TNPoint & other);


      /** @brief Assignment operator. No checks for dim or maxD are performed. It's ok to change dim and maxD of the object using the assignment operator.*/
      TNPoint & operator = (TNPoint const  & other);


      /** @brief Compares level and full coordinates, disregarding selection status. */
      bool operator== (TNPoint const &that) const;

      /** @brief Compares level and full coordinates, disregarding selection status. */
      bool operator!= (TNPoint const &that) const;


      int get_owner() const { return m_owner; }
      void set_owner(int owner) { m_owner = owner; }

      /// long get_globId() const { return m_globId; }
      /// void set_globId(long globId) { m_globId = globId; }

      /**@brief Get type of cell to which point is interior, at native level. */
      CellType<dim> get_cellType() const;

      /**@brief Get type of cell to which point is interior, at parent of native level. */
      CellType<dim> get_cellTypeOnParent() const;

      /**@brief Get type of cell to which point is interior, at arbitrary level. */
      CellType<dim> get_cellType(LevI lev) const;

      static CellType<dim> get_cellType(const TreeNode<T, dim> &tnPoint, LevI lev);

      /**@brief Return whether own cell type differs from cell type on parent. */
      bool isCrossing() const;

      bool getIsCancellation() const;

      void setIsCancellation(bool isCancellation);

      /** @brief Assuming nodes on a cell are ordered lexicographically, get the rank of this node on the given cell. */
      unsigned int get_lexNodeRank(const TreeNode<T,dim> &hostCell, unsigned int polyOrder) const;

      /** @brief Assuming nodes on a cell are ordered lexicographically, get the rank of given node on the given cell. */
      static unsigned int get_lexNodeRank(const TreeNode<T,dim> &hostCell, const TreeNode<T, dim> &tnPoint, unsigned int polyOrder);

      static unsigned int get_nodeRank1D(const TreeNode<T, dim> &hostCell, const TreeNode<T, dim> &tnPoint, unsigned int d, unsigned int polyOrder);

      static std::array<unsigned, dim> get_nodeRanks1D(const TreeNode<T, dim> &hostCell, const TreeNode<T, dim> &tnPoint, unsigned int polyOrder);

      static void get_relNodeCoords(const TreeNode<T,dim> &containingSubtree,
                                    const TreeNode<T,dim> &tnPoint,
                                    unsigned int polyOrder,
                                    std::array<unsigned int, dim> &numerators,
                                    unsigned int &denominator);

    protected:
      // Data members.
      int m_owner = -1;
      long m_globId = -1;
      //TODO These members could be overlayed in a union if we are careful.


      bool m_isCancellation = false;
  };

  // The convention to nudge points, including boundary points, into containers.
  template <typename SrcType, typename RsltType>
  inline RsltType KeyFunInboundsContainer(const SrcType &pt)
  {
    using T = typename SrcType::coordType;
    constexpr unsigned int dim = ot::coordDim(&pt);
    std::array<T,dim> coords;
    pt.getAnchor(coords);
    /// const unsigned int lev = pt.getLevel();            // Container.
    const unsigned int lev = m_uiMaxDepth;             // Nearest in-bounds point.
    const unsigned int len = 1u << (m_uiMaxDepth-lev);
    const unsigned int domainUpper = (1u<<m_uiMaxDepth) - 1;

    for (int d = 0; d < dim; d++)
    {
      if (coords[d] > domainUpper)
      {
        /// coords[d] -= len;            // Container.
        coords[d] = domainUpper;      // Nearest in-bounds point.
      }
    }

    return {coords, lev};   // Erases resolution below lev.
  }

  template <typename SrcType, typename RsltType>
  using KeyFunInboundsContainer_t = std::function<RsltType(const SrcType &)>;

  template <typename T, unsigned int dim>
  class Element : public TreeNode<T,dim>
  {
    public:
      // Bring in parent constructors.
      Element () : TreeNode<T,dim> () {}
      Element (const std::array<T,dim> coords, unsigned int level) : TreeNode<T,dim> (coords, level) {}
      Element (const Element & other) : TreeNode<T,dim> (other) {}
      Element (const int dummy, const std::array<T,dim> coords, unsigned int level) :
          TreeNode<T,dim> (dummy, coords, level) {}

      // Constructor from TreeNode to Element.
      Element(const TreeNode<T,dim> & other) : TreeNode<T,dim>(other) {}

      using TreeNode<T,dim>::operator=;

      std::array<T, dim> getNodeX(const std::array<unsigned, dim> &numerators, unsigned polyOrder) const;
      TNPoint<T, dim>    getNode(const std::array<unsigned, dim> &numerators, unsigned polyOrder) const;

      /** @brief Append nodes in lexicographic order. */
      template <typename TN = TNPoint<T,dim>>
      void appendNodes(unsigned int order, std::vector<TN> &nodeList) const;

      void appendInteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const;
      void appendExteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList, const ::ibm::DomainDecider &domainDecider) const;
      void appendCancellationNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const;

      void appendKFaces(CellType<dim> kface, std::vector<TreeNode<T,dim>> &nodeList, std::vector<CellType<dim>> &kkfaces) const;


      /** @brief Maps child (this) hanging nodes to parent nodes
       *         that do not overlap with child nonhanging nodes,
       *         and acts as identity for child non-hanging nodes. */
      std::array<unsigned, dim> hanging2ParentIndicesBijection(
          const std::array<unsigned, dim> &indices, unsigned polyOrder) const;

      /**
       * @returns true if the coordinates lie in the element or on the element boundary.
       * @note similar implementation to TreeNode::isOnDomainBoundary().
       */
      bool isIncident(const ot::TreeNode<T,dim> &pointCoords) const;

      /**
       * @brief Using bit-wise ops, identifies which children are touching a point.
       * @param [in] pointCoords Coordinates of the point incident on 0 or more children.
       * @param [out] incidenceOffset The Morton child # of the first incident child.
       * @param [out] incidenceSubspace A bit string of axes, with a '1'
       *                for each incident child that is adjacent to the first incident child.
       * @param [out] incidenceSubspaceDim The number of set ones in incidenceSubspace.
       *                The number of incident children is pow(2, incidenceSubspaceDim).
       * @note Use with TallBitMatrix to easily iterate over the child numbers of incident children.
       * @note It is ASSUMED that isIncident(pointCoords) is true.
       */
      void  incidentChildren(
          const ot::TreeNode<T,dim> &pointCoords,
          typename ot::CellType<dim>::FlagType &incidenceOffset,
          typename ot::CellType<dim>::FlagType &incidenceSubspace,
          typename ot::CellType<dim>::FlagType &incidenceSubspaceDim) const;
  };


  struct ScatterMap
  {
    std::vector<RankI> m_map;
    std::vector<RankI> m_sendCounts;
    std::vector<RankI> m_sendOffsets;
    std::vector<int> m_sendProc;
  };

  struct GatherMap
  {
    /// static void resizeLocalCounts(GatherMap &gm, RankI newLocalCounts, int rProc)
    /// {
    ///   RankI accum = 0;
    ///   int procIdx = 0;
    ///   while (procIdx < gm.m_recvProc.size() && gm.m_recvProc[procIdx] < rProc)
    ///     accum += gm.m_recvCounts[procIdx++];
    ///   gm.m_locCount = newLocalCounts;
    ///   /// gm.m_locOffset = accum;   // This will be the same.
    ///   accum += newLocalCounts;
    ///   while (procIdx < gm.m_recvProc.size())
    ///   {
    ///     gm.m_recvOffsets[procIdx] = accum;
    ///     accum += gm.m_recvCounts[procIdx++];
    ///   }
    ///   gm.m_totalCount = accum;
    /// }

    std::vector<int> m_recvProc;
    std::vector<RankI> m_recvCounts;
    std::vector<RankI> m_recvOffsets;

    RankI m_totalCount;
    RankI m_locCount;
    RankI m_locOffset;
  };

  inline size_t computeTotalSendSz(const ScatterMap &sm)
  {
    size_t total = 0;
    for (RankI sendCount : sm.m_sendCounts)
      total += sendCount;
    return total;
  }

  inline size_t totalRecvSz(const GatherMap &gm)
  {
    return gm.m_totalCount - gm.m_locCount;
  }



  std::ostream & operator<<(std::ostream &out, const ScatterMap &sm);
  std::ostream & operator<<(std::ostream &out, const GatherMap &gm);

  template <typename T, unsigned int dim>
  struct SFC_NodeSort
  {
    /**
     * @brief Takes distributed sorted lists of owned nodes, uses key generation to compute sufficient scattermap.
     * @note Might produce some nodes that don't need to be exchanged. Hopefully it's not too many surplus.
     */
    static ScatterMap computeScattermap(const std::vector<TNPoint<T,dim>> &ownedNodes, const TreeNode<T,dim> *treePartStart, MPI_Comm comm);

    /**
     * @brief Exchange counts from senders to receivers.
     * @TODO change the name ("gather map" means something else).
     */
    static GatherMap scatter2gather(const ScatterMap &sm, RankI localCount, MPI_Comm comm);


    /** @brief Stage and send our data (using ScatterMap), and receive ghost data into ghost buffers (using GatherMap). */
    //TODO this function doesn't really belong in this class, it doesn't depend on T or dim at all. --> the 'oda' class
    template <typename da>
    static void ghostExchange(da *dataAndGhostBuffers, da *sendBufferSpace, const ScatterMap &sm, const GatherMap &gm, MPI_Comm comm);

    /** @brief Send back contributions to owners (using GatherMap), receive and unstage/accumulate to our data (using ScatterMap). */
    // TODO move this to the 'oda' class as well.
    template <typename da>
    static void ghostReverse(da *dataAndGhostBuffers, da *sendBufferSpace, const ScatterMap &sm, const GatherMap &gm, MPI_Comm comm);

  }; // struct SFC_NodeSort



}//namespace ot

#include "nsort.tcc"


namespace par {

  //Forward Declaration
  template <typename T>
    class Mpi_datatype;

      /**@brief A template specialization of the abstract class "Mpi_datatype" for communicating messages of type "ot::TNPoint".*/
      template <typename T, unsigned int dim>
      class Mpi_datatype< ot::TNPoint<T,dim> > {

      /*@masado Omitted all the comparison/reduction operations, limited to ::value().*/
      public:

      /**@return The MPI_Datatype corresponding to the datatype "ot::TNPoint".*/
      static MPI_Datatype value()
      {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
          first = false;
          MPI_Type_contiguous(sizeof(ot::TNPoint<T,dim>), MPI_BYTE, &datatype);
          MPI_Type_commit(&datatype);
        }

        return datatype;
      }

    };
}//end namespace par



#endif//DENDRO_KT_NSORT_H
