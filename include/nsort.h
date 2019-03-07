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

#include <iostream>
#include <bitset>

#include <mpi.h>
#include <vector>
#include <queue>
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

    FlagType get_dim_flag() { return m_flag >> m_shift; }
    FlagType get_orient_flag() { return m_flag & m_mask; }

    void set_dimFlag(FlagType c_dim) { m_flag = (m_flag & m_mask) | (c_dim << m_shift); }
    void set_orientFlag(FlagType c_orient) { m_flag = (m_flag & (~m_mask)) | (c_orient & m_mask); }

    // TODO void set_flags(FlagType c_orient); // Counts 1s in c_orient and uses count for c_dim.

    FlagType m_flag;

    private:
      static const FlagType m_shift = 4u;
      static const FlagType m_mask = (1u << m_shift) - 1;
  };


  /**@brief TreeNode + extra attributes to keep track of node uniqueness. */
  template <typename T, unsigned int dim>
  class TNPoint : public TreeNode<T,dim>
  {
    public:
      enum IsSelected { No, Maybe, Yes, Base };

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

      /**
        @brief Constructs an octant (without checks).
        @param dummy : not used yet.
        @param coords The coordinates of the point.
        @param level The level of the point (i.e. level of the element that spawned it).
      */
      TNPoint (const int dummy, const std::array<T,dim> coords, unsigned int level);

      /** @brief Assignment operator. No checks for dim or maxD are performed. It's ok to change dim and maxD of the object using the assignment operator.*/
      TNPoint & operator = (TNPoint const  & other);


      /** @brief Compares level and full coordinates, disregarding selection status. */
      bool operator== (TNPoint const &that) const;

      /** @brief Compares level and full coordinates, disregarding selection status. */
      bool operator!= (TNPoint const &that) const;


      IsSelected get_isSelected() const { return m_isSelected; }
      void set_isSelected(IsSelected isSelected) { m_isSelected = isSelected; }

      unsigned char get_numInstances() const { return m_numInstances; }
      void set_numInstances(unsigned char numInstances) { m_numInstances = numInstances; }
      void incrementNumInstances(char delta_numInstances = 1) { m_numInstances += delta_numInstances; }

      int get_owner() const { return m_owner; }
      void set_owner(int owner) { m_owner = owner; }

      /**
       * @brief The point may be incident on one or more grid lines (hyperplanes) at coarseness level `hlev'.
       *   If so, this method returns the smallest-indexed such hyperplane.
       *   A hyperplane is indexed by the normal axis.
       *   If the point is not incident on any of the grid hyperplanes, then the method returns dim.
       */
      unsigned char get_firstIncidentHyperplane(unsigned int hlev) const;

      /**@brief Get type of cell to which point is interior, at native level. */
      CellType<dim> get_cellType() const;

      /**@brief Get type of cell to which point is interior, at parent of native level. */
      CellType<dim> get_cellTypeOnParent() const;

      /**@brief Get type of cell to which point is interior, at arbitrary level. */
      CellType<dim> get_cellType(LevI lev) const;

      /**@brief Get the deepest cell such that the point is not on the boundary of the cell. */
      TreeNode<T,dim> getFinestOpenContainer() const;

      /** @brief Get the cell that generated this point, based on coordinates and level. */
      TreeNode<T,dim> getCell() const;

    protected:
      // Data members.
      IsSelected m_isSelected;
      unsigned char m_numInstances = 1;
      int m_owner = -1;
      //TODO These members could be overlayed in a union if we are careful.
  };


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

      void appendNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const;
      void appendInteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const;
      void appendExteriorNodes(unsigned int order, std::vector<TNPoint<T,dim>> &nodeList) const;
  };


  template <typename T, unsigned int dim>
  struct SFC_NodeSort
  {
    /**
     * @brief Count all unique, nonhanging nodes in/on the domain, when the node list is a distributed array. Also compact node list and compute ``scatter map.''
     */
    //TODO make the input a vector that can change size.
    static RankI dist_countCGNodes(std::vector<TNPoint<T,dim>> &points, unsigned int order, const TreeNode<T,dim> *treePartStart, MPI_Comm comm);


    /**
     * @brief Count all unique, nonhanging nodes in/on the domain.
     * @param classify If true, perform node classification and mark them appropriately. If false,
     *        the points are still sorted, but they are not marked - instead the number of instances
     *        is recorded in the first instance of every duplicate bunch. If false, the return value
     *        is meaningless.
     * @note Assumes none of the points are (dim)-cell interior points.
     *   To achieve this, use Element::appendExteriorNodes() and Element::appendInteriorNodes() separately.
     */
    static RankI countCGNodes(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order, bool classify = true);

    /**
     * @brief Sorts points `as points', meaning by coordinate first. NOTE: Doesn't enforce any ordering among points with identical coordinates.
     */
    static void locTreeSortAsPoints(TNPoint<T,dim> *points, RankI begin, RankI end, LevI sLev, LevI eLev, RotI pRot);

    private:

      /**
       * @brief Count the number of duplicate coordinate locations, if all are at the same level, or yield 0 if there are mixed levels.
       * @param [in] start The start of the scan.
       * @param [in] end Scan won't enter end.
       * @param [out] firstCoarsest The first duplicate, or if there are mixed levels, the first with the coarser level.
       * @param [out] next The next element that was not scanned. Future scans can pick up from here.
       * @param [out] numDups If all same level, the number of duplicates. If mixed levels, 0.
       * @note Assumes that start < end.
       * @note Assumes that the field m_numInstances has been properly initialized for all points.
       */
      static void scanForDuplicates(TNPoint<T,dim> *start, TNPoint<T,dim> *end, TNPoint<T,dim> * &firstCoarsest, TNPoint<T,dim> * &firstFinest, TNPoint<T,dim> * &next, unsigned int &numDups);

      /** @brief Moves all domain boundary points to the end, returning the number of boundary points. */
      static RankI filterDomainBoundary(TNPoint<T,dim> *start, TNPoint<T,dim> *end);

      /** @brief Breaks up an interface into the component hyperplanes. */
      static void bucketByHyperplane(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int hlev, std::array<RankI,dim+1> &hSplitters);

      /**
       * @brief Depth-first traversal: pre-order bucketing, post-order calling resolveInterface (bottom up).
       * @param sLev The level to separate children into sibiling buckets.
       * @param pRot The SFC orientation of the parent (containing) region.
       */
      template<typename ResolverT>
      static RankI countCGNodes_impl(ResolverT resolveInterface, TNPoint<T,dim> *start, TNPoint<T,dim> *end, LevI sLev, RotI pRot, unsigned int order);

      /**
       * @brief For order 1 or 2, alignment of points means we can count duplicates at node site to resolve duplicates/hanging nodes.
       * @note Assumes the points are already sorted -- as points, such that all points with same coordinates appear together, regardless of level.
       * @note Assumes that the field m_numInstances has been properly initialized for all points.
       */
      static RankI resolveInterface_lowOrder(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order);

      /**
       * @brief For order > 2, alignment might not hold. However, we can use the fact that order > 2
       *        to take advantage of locality of k'-face interior nodes of differing levels to
       *        resolve duplicates/hanging nodes using a small buffer.
       * @note Assumes the points are already sorted -- as points, such that all points with same coordinates appear together, regardless of level.
       */
      static RankI resolveInterface_highOrder(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int order);

      /**
       * @brief A ``pseudo-resolver'' method that counts literal duplicate points,
       *        i.e. having both the same coordinates and level. Does not classify points
       *        in any way. Also the return value is not meaningful at all.
       * @note The reason for having this function is for a final preprocessing
       *       stage before finding the processor-boundary nodes.
       */
      static RankI countInstances(TNPoint<T,dim> *start, TNPoint<T,dim> *end, unsigned int unused_order);

      /**
       * @brief Broadcast the first TreeNode from every processor so we have global access to the splitter list.
       */
      static std::vector<TreeNode<T,dim>> dist_bcastSplitters(const TreeNode<T,dim> *start, MPI_Comm comm);

      /**
       * @brief Find which processors upon which the node is incident.
       * @description Generates keys from pt.getDFD() and then calls getContainingBlocks().
       */
      static int getProcNeighbours(TNPoint<T,dim> pt, const TreeNode<T,dim> *splitters, int numSplitters, std::vector<int> &procNbList);
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
