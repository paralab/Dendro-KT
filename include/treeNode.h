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
#include <iostream>
#include <array>
#include <vector>
#include <mpi.h>

namespace ot {

    // Template forward declarations.
    template <typename T, unsigned int dim> class TreeNode;
    template <typename T, unsigned int dim> std::ostream& operator<<(std::ostream& os, TreeNode<T,dim> const& other);

    /**
      @brief A class to manage octants.
      @tparam dim the dimension of the tree
      */
    template<typename T,unsigned int dim>
    class TreeNode {
    protected:

        /**TreeNode coefficients*/
        std::array<T,dim> m_uiCoords;

        /**level of the tree node*/
        unsigned int m_uiLevel;


    public:

        static constexpr char numChildren = (1u << dim);

        //@masado, can you please fix this to work with any dim. (@masado: Looks like it's not being used.)
        ///using Flag2K = unsigned char;  // Has 8 bits, will work for (dim <= 4).

      /**
        @brief Constructs a root octant
        @see MAX_LEVEL
        */
      TreeNode ();

      /**
        @brief Constructs an octant
        @param coords The coordinates of the anchor of the octant
        @param level The level of the octant
        */
      TreeNode (const std::array<T,dim> coords, unsigned int level);

      /**@brief Copy constructor */
      TreeNode (const TreeNode & other);

      /**
        @brief Constructs an octant
        @param dummy : not used yet.
        @param coords The coordinates of the anchor of the octant
        @param level The level of the octant
      */
      TreeNode (const int dummy, const std::array<T,dim> coords, unsigned int level);

      /** @brief Assignment operator. No checks for dim or maxD are performed. It's ok to change dim and maxD of the current octant using the assignment operator.*/
      TreeNode & operator = (TreeNode const  & other);

      /** @brief Two octants are equal if their respective anchors are equal and their levels are equal. */
      bool  operator == ( TreeNode const  &other) const;

      /** @brief Two octants are equal if their respective anchors are equal and their levels are equal. */
      bool  operator != (TreeNode const  &other) const;

      /**@brief The comparisons are based on the Morton/Hilbert ordering of the octants */
      bool operator < (TreeNode const &other) const;

      /**@brief overloaded comparison operator based on some SFC ordering. */
      bool operator <= (TreeNode const &other) const;

      /** @brief stream operator to output treeNodes with std::cout */
      friend std::ostream & operator << <T,dim> (std::ostream & os,TreeNode<T,dim> const & node) ;

      /**@breif return the tree dimension*/
      unsigned int getDim() const;

      /**@brief returns the max depth*/
      unsigned int getMaxDepth() const;

      /**@brief return the level of the octant*/
      unsigned int getLevel() const;

      /**@brief return the flag set by setFlag function (flag sotored in the m_uiLevel)*/
      unsigned int getFlag() const;

      /**@brief return the coordinates of the dth dimention*/
      T getX(int d) const;

      /**@brief get the coordinates of the octant
       * @param[out]: coppied coordinates*/
      int getAnchor(std::array<T,dim> &xyz) const;

      /**@brief return the (Morton indexed) child number at specified level.*/
      unsigned char getMortonIndex(T level) const;

      /**@brief return the (Morton indexed) child number at the node's own level.*/
      unsigned char getMortonIndex() const;

      /**@brief set the last digit (at node's level) in each component of the anchor. */
      void setMortonIndex(unsigned char child);

      /**@brief set the octant flag*/
      int setFlag(unsigned int w);

      /**@brief update the flag by taking bitwise-or of old value with w.*/
      int orFlag(unsigned int w);

      /**
       @author Masado Ishii
        */
      std::array<char, MAX_LEVEL+1> getBase32Hex() const;

      /**
        @author Rahul Sampath
        @name Pseudo-Getters
       */
      T getParentX(int d) const;

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

      /**@brief return the smallest (in morton id) of this octant */
      TreeNode getFirstChildMorton() const;

      /**@brief max coord of the d dimention*/
      T minX(int d) const;

      /**@brief min coord of the d dimention*/
      T maxX(int d) const;

      /**@brief min coord of the octant (leftmost corner)*/
      std::array<T,dim> minX() const;

      /**@brief max coord of the octant (rightmost corner)*/
      std::array<T,dim> maxX() const;

     /**
       @brief Return neighbor at the same level.
       @author Masado Ishii
       @tparam offsets Specify relative position as (-1,0,+1) for each dimension.
      */
      TreeNode getNeighbour(std::array<signed char,dim> offsets) const;

      /**
        @brief Return adjacent neighbor at the same level.
        @author Masado Ishii
        @tparam offsets Specify dimension of adjacency and relative position \ as (-1,0,+1) for that dimension.
        */
      TreeNode getNeighbour(unsigned int d, signed char offset) const;

      /**
        @brief Append in-bounds neighbors of node to node list.
        @author Masado Ishii
       */
      void appendAllNeighbours(std::vector<TreeNode> &nodeList) const;

    };

} // end namespace ot

// Template specializations
/*
template class ot::TreeNode<unsigned int, 1>;
template class ot::TreeNode<unsigned int, 2>;
template class ot::TreeNode<unsigned int, 3>;
template class ot::TreeNode<unsigned int, 4>;
*/

#include "treeNode.tcc"


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


