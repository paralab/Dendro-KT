//
// Created by milinda on 10/5/18.
//

/**
 * @author Hari Sundar
 * @author Rahul Sampath
 * @author Milinda Fernando
 * @brief Refactored version of the TreeNode class with the minimum data required.
 * @remarks m_uiMaxDepth is removed from the arttributre list and listed as a global variable.
 * */

#ifndef DENDRO_KT_TREENODE_H
#define DENDRO_KT_TREENODE_H

static const unsigned int MAX_LEVEL=31;
extern unsigned int m_uiMaxDepth;

#include <algorithm>
#include "dendro.h"
#include "pcoord.h"
#include <iostream>
#include <array>
#include <vector>
#include <mpi.h>

namespace ot {

    constexpr bool debug_have_periodic = true;

    // Template forward declarations.
    template <typename T, unsigned int dim> class TreeNode;
    template <typename T, unsigned int dim> std::ostream& operator<<(std::ostream& os, TreeNode<T,dim> const& other);

    // Use 4D: unsigned short
    //     5D: unsigned int
    //     6D: unsigned long long.
    using ExtantCellFlagT = unsigned short;

    /// template <int dim>  constexpr int nchild() { return 1 << dim; }
    constexpr int nchild(int dim) { return 1 << dim; }

    template <template <typename TNT, unsigned TND> typename TN, typename T, unsigned dim>
    constexpr unsigned coordDim(const TN<T, dim> *) { return dim; }

    /**
      @brief A class to manage octants.
      @tparam dim the dimension of the tree
      */
    template<typename T,unsigned int dim>
    class TreeNode {
    protected:

        /**TreeNode coefficients*/
        periodic::PCoord<T, dim> m_coords;

        /**level of the tree node*/
        unsigned int m_uiLevel;

        /**As a point, is this point exposed as part of the tree/domain boundary?
         * As an element, does the element have any exposed exterior points?
         * @note false unless previously set (or copied after setting) setIsOnTreeBdry(). */
        bool m_isOnTreeBdry;

        // weight functionality
        unsigned int m_weight;

        // m_isOnTreeBdry is just a tag.
        // Not computed automatically by TreeNode.

        /**
          @brief Constructs an octant
          @param dummy : not used yet.
          @param coords The coordinates of the anchor of the octant
          @param level The level of the octant
        */
        TreeNode (const int dummy, const std::array<T,dim> coords, unsigned int level);

    public:

        using coordType = T;

      /**
        @brief Constructs a root octant
        @see MAX_LEVEL
        */
      TreeNode ();

      /**
        @brief Constructs an octant
        @param coords The coordinates of the anchor of the octant
        @param level The level of the octant
        @note Coordinates are not masked to level, however periodic is applied.
        */
      TreeNode (const std::array<T,dim> coords, unsigned int level);

      TreeNode (const std::array<T,dim> coords, unsigned int level, unsigned int weight);

      /**@brief Copy constructor */
      TreeNode (const TreeNode & other) = default;


      /** @brief Assignment operator. No checks for dim or maxD are performed. It's ok to change dim and maxD of the current octant using the assignment operator.*/
      TreeNode & operator = (TreeNode const  & other) = default;

      /** @brief Two octants are equal if their respective anchors are equal and their levels are equal. */
      bool  operator == ( TreeNode const  &other) const;

      /** @brief Two octants are equal if their respective anchors are equal and their levels are equal. */
      bool  operator != (TreeNode const  &other) const;

      /** @brief stream operator to output treeNodes with std::cout */
      friend std::ostream & operator << <T,dim> (std::ostream & os,TreeNode<T,dim> const & node) ;

      /**@brief returns the max depth*/
      unsigned int getMaxDepth() const;

      /**@brief return the level of the octant*/
      unsigned int getLevel() const;

      /**@brief return the level of the octant*/
      unsigned int getWeight() const;

      /**@brief set the level of the octant*/
      void setLevel(unsigned int lev);

      /**@brief return the coordinates of the dth dimention*/
      T getX(int d) const;

      const std::array<T, dim> & getX() const { return m_coords.coords(); }

      inline const periodic::PCoord<T, dim> & coords() const { return m_coords; }

      inline const periodic::PRange<T, dim> range() const;

      // This gives write access to coordinates.
      /**@brief set the coordinate in the dth dimension*/
      void setX(int d, T coord);

      void setX(const std::array<T, dim> &coords);

      /**@brief get the coordinates of the octant
       * @param[out]: coppied coordinates*/
      int getAnchor(std::array<T,dim> &xyz) const;

      // These methods don't tell you which element is which neighbor,
      //  look in nsort.h / nsort.tcc for that.

      /**As a point, is this point exposed as part of the tree/domain boundary?
       * As an element, does the element have any exposed exterior points?
       * Before the tree is filtered, this is set assuming the unit hypercube. */
      inline bool getIsOnTreeBdry() const;

      /**@note If you set this yourself you may invalidate the tree. */
      inline void setIsOnTreeBdry(bool isOnTreeBdry);

      /**@brief return the (Morton indexed) child number at specified level.*/
      unsigned char getMortonIndex(T level) const;

      /**@brief return the (Morton indexed) child number at the node's own level.*/
      unsigned char getMortonIndex() const;

      /**@brief set the last digit (at node's level) in each component of the anchor. */
      void setMortonIndex(unsigned char child);

      /**@brief Returns the greatest depth at which the other node shares an ancestor.*/
      unsigned int getCommonAncestorDepth(const TreeNode &other) const;

      /**
       @author Masado Ishii
        */
      std::array<char, MAX_LEVEL+1> getBase32Hex(unsigned int lev = 0) const;

      /**
        @return the parent of this octant
        */
      TreeNode  getParent() const;

      /**
        @author Rahul Sampath
        @param The level of the ancestor
        @return the ancestor of this octant at level 'ancLev'
        */
      TreeNode	getAncestor(unsigned int ancLev) const;

      /**
        @author Rahul Sampath
        @return the Deepest first decendant of this octant
        */
      TreeNode getDFD() const;

      /**
        @author Rahul Sampath
        @return the deepest last decendant of this octant
        */
      TreeNode getDLD() const;


      /**@brief returns true if *(this) octant is root. false otherwise*/
      bool isRoot() const;

      /**
        @author Rahul Sampath
        @brief Checks if this octant is an ancestor of the input
        @return 'true' if this is an ancestor of 'other'
        */
      bool isAncestor(const TreeNode & other) const;

      bool isAncestorInclusive(const TreeNode & other) const;

      /**@brief return the smallest (in morton id) of this octant */
      TreeNode getFirstChildMorton() const;

      /**@brief return a child of this octant with the given child number.*/
      TreeNode getChildMorton(unsigned char child) const;

      /**@brief lower bound of the d dimension*/
      T lowerBound(int d) const;

      /**@brief upper bound of the d dimension, not mapped periodic*/
      T upperBound(int d) const;

      /**
        @brief Append in-bounds neighbors of node to node list.
        @author Masado Ishii
       */
      void appendAllNeighbours(std::vector<TreeNode> &nodeList) const;
      void appendCoarseNeighbours(std::vector<TreeNode> &nodeList) const;
    };

} // end namespace ot

// Template specializations
/*
template class ot::TreeNode<unsigned int, 1>;
template class ot::TreeNode<unsigned int, 2>;
template class ot::TreeNode<unsigned int, 3>;
template class ot::TreeNode<unsigned int, 4>;
*/

#include "treeNodeModified.tcc"


namespace par {

  //Forward Declaration
  template <typename T>
    class Mpi_datatype;

      /**@brief A template specialization of the abstract class "Mpi_datatype" for communicating messages of type "ot::TreeNode".*/
      template <typename T, unsigned int dim>
      class Mpi_datatype< ot::TreeNode<T,dim> > {

      /*@masado Omitted all the comparison/reduction operations, limited to ::value().*/
      public:

      /**@return The MPI_Datatype corresponding to the datatype "ot::TreeNode".*/
      static MPI_Datatype value()
      {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
          first = false;
          MPI_Type_contiguous(sizeof(ot::TreeNode<T,dim>), MPI_BYTE, &datatype);
          MPI_Type_commit(&datatype);
        }

        return datatype;
      }

    };
}//end namespace par





#endif //DENDRO_KT_TREENODE_H


