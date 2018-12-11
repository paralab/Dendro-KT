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
//#define m_uiMaxDepth 30
extern unsigned int m_uiMaxDepth;

#include <algorithm>
#include "dendro.h"
#include <iostream>
#include <array>

#include <mpi.h>

namespace ot {

    // Template forward declarations.
    template <typename T, unsigned int dim> class TreeNode;
    template <typename T, unsigned int dim>
        std::ostream& operator<<(std::ostream& os, TreeNode<T,dim> const& other);

    /**
      @brief A class to manage octants.
      @author Rahul Sampath
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

      /** @name Templated types.  */
      //@{
        //using Flag2K = typename detail::Flag2K<dim>::type; // I don't know how to do this.
        using Flag2K = unsigned char;  // Has 8 bits, will work for (dim <= 4).
      //@}

      /** @name Constructors */
      //@{

      /**
        @author Rahul Sampath
        @brief Constructs a root octant
        @see MAX_LEVEL
        */
      TreeNode ();

      /**
        @author Rahul Sampath
        @brief Constructs an octant
        @param coords The coordinates of the anchor of the octant
        @param level The level of the octant
        */
      TreeNode (const std::array<T,dim> coords, unsigned int level);

      /**
        @author Rahul Sampath
        @brief Copy constructor
        */
      TreeNode (const TreeNode & other);

      /**
        @brief Constructor without checks: only for faster construction.
       */
      TreeNode (const int dummy, const std::array<T,dim> coords, unsigned int level);
      //@}


      /** @name Overloaded Operators */
      //@{

      /**
        @author Rahul Sampath
        @brief Assignment operator. No checks for dim or maxD are performed.
        It's ok to change dim and maxD of the current octant using the assignment operator.
        */
      TreeNode & operator = (TreeNode const  & other);

      /**
        @author Rahul Sampath
        @brief Two octants are equal if their respective anchors are equal and their levels are
        equal.
        */
      bool  operator == ( TreeNode const  &other) const;

      /**
        @author Rahul Sampath
        @brief Two octants are equal if their respective anchors are equal and their levels are
        equal.
        */
      bool  operator != (TreeNode const  &other) const;

          //@masado I didn't do inequality comparisons yet because I don't \
                    know what space-filling curve will be used.

          //@masado But I will temporarily add `<' with respecto to the Morton order.
      /**
        @author Rahul Sampath
        @author Hari Sundar
        @brief The comparisons are based on the Morton/Hilbert 
               ordering of the octants
        */
      bool operator < (TreeNode const &other) const;

      bool operator <= (TreeNode const &other) const;

      /**
        @author Rahul Sampath
        */
      friend std::ostream & operator << <T,dim> (std::ostream & os,TreeNode<T,dim> const & node) ;
      //@}


      /**
        @author Rahul Sampath
        @name Getters and Setters
        */
      //@{
      unsigned int getDim() const;
      unsigned int getMaxDepth() const;
      //unsigned int getWeight() const;
      unsigned int getLevel() const;
      unsigned int getFlag() const;
      T getX(int d) const;
      int getAnchor(std::array<T,dim> &xyz) const;
      //Point getAnchor() const { return Point(m_uiX, m_uiY, m_uiZ); };
      //unsigned char getChildNumber(bool real=true) const;
      unsigned char getMortonIndex(T level) const;
      unsigned char getMortonIndex() const;
      void setMortonIndex(unsigned char child);
      //int setWeight(unsigned int w);
      //int addWeight(unsigned int w);
      int setFlag(unsigned int w);
      int orFlag(unsigned int w);
      //@}

      /**
       @author Masado Ishii
        */
      std::array<char, MAX_LEVEL+1> getBase32Hex() const;

      /**
        @author Rahul Sampath
        @name Pseudo-Getters
       */
      //@{
      T getParentX(int d) const;

      /**
        @author Rahul Sampath
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

        //@masado Are we using Morton?
      ///     TreeNode getDFDMorton() const;

        //@masado Are we using this? Involves Hilbert curve.
      ////  /*
      ////   * @author Milinda Fernando
      ////   * @return the coarset first decendant of this octant.
      ////   * This is always the min x, min y, min z (the coordinates we store) but if we are
      ////   * using Hilbert Curve this needs to be calculated.
      ////   * */

      ////  TreeNode getCFD() const;


      /**
        @author Rahul Sampath
        @return the deepest last decendant of this octant
        */
      TreeNode getDLD() const;

      //@}


      /**
        @name is-tests
       */
      //@{

      bool isRoot() const;

      /**
        @author Rahul Sampath
        @brief Checks if this octant is an ancestor of the input
        @return 'true' if this is an ancestor of 'other'
        */
      bool isAncestor(const TreeNode & other) const;

        //@masado I need to ask how boundary info will be used.
      ///  /**
      ///    @author Hari Sundar
      ///    @brief flags is a datastructure which will store which boundaries were
      ///    touched. highest 3 bits are for +Z,+y, and +x axes ... and and
      ///    smallest 3 are for -z,-y and -x axes.
      ///    */
      ///  bool isBoundaryOctant(int type=POSITIVE, Flag2K *flags=NULL) const;

      ///  /**
      ///    @author Hari Sundar
      ///    @brief flags is a datastructure which will store which boundaries were
      ///    touched. highest 3 bits are for +Z,+y, and +x axes ... and and
      ///    smallest 3 are for -z,-y and -x axes.
      ///    */
      ///  bool isBoundaryOctant(const TreeNode &block, int type=POSITIVE, Flag2K *flags=NULL) const;
      //@}

      /**
        @name Other Morton-code related methods.
       */
      //@{
      ///    /**
      ///     *@author Dhairya Malhotra
      ///     *@return the next octant in morton order which has the least depth.
      ///     */
      ///    TreeNode getNext() const;

      ///    /**
      ///     *@author Dhairya Malhotra
      ///     *@return the smallest (in morton id) of this octant
      ///     */
      ///    TreeNode getFirstChild() const;

      ///    /**
      ///      @author Rahul Sampath
      ///      @return the Morton encoding for this octant
      ///      */
      ///    std::vector<bool> getMorton() const;
      //@}

          /**
           *@author Dhairya Malhotra
           *@return the smallest (in morton id) of this octant
           */
          TreeNode getFirstChildMorton() const;

      /** @name Mins and maxes */
      //@{
      T minX(int d) const;
      T maxX(int d) const;
      std::array<T,dim> minX() const;
      std::array<T,dim> maxX() const;
      //@}


      /**
        @author Rahul Sampath
        @name Get Neighbours at the same level as the current octant
        */
      //@{

      //@masado An octant that is not on the boundary has 3^dim - 1 neighbors. \
          (The original methods return the root octant for a nonexistent neighbor.) \
          Original methods: The primitives +/-X +/-Y +/-Z are defined by computing len \
          (as in max()) and either adding or subtracting this from current address \
          in specified dimension. Other combinations are defined in terms of those primitives.

        /**
         @brief Return neighbor at the same level.
         @author Masado Ishii
         @tparam offsets Specify relative position as (-1,0,+1) for each dimension.
        */
        TreeNode getNeighbour(std::array<signed char,dim> offsets) const;

        /**
         @brief Return adjacent neighbor at the same level.
         @author Masado Ishii
         @tparam offsets Specify dimension of adjacency and relative position \
                         as (-1,0,+1) for that dimension.
        */
        TreeNode getNeighbour(unsigned int d, signed char offset) const;

///        //@masado I only define the named neighbours for dim==3.
///        /**
///          @author Rahul Sampath
///          */
///        //@{
///        TreeNode  getLeft() const;
///        TreeNode  getRight() const;
///        TreeNode  getTop() const;
///        TreeNode  getBottom() const;
///        TreeNode  getFront() const;
///        TreeNode  getBack() const;
///        TreeNode  getTopLeft() const;
///        TreeNode  getTopRight() const;
///        TreeNode  getBottomLeft() const;
///        TreeNode  getBottomRight() const;
///        TreeNode  getLeftFront() const;
///        TreeNode  getRightFront() const;
///        TreeNode  getTopFront() const;
///        TreeNode  getBottomFront() const;
///        TreeNode  getTopLeftFront() const;
///        TreeNode  getTopRightFront() const;
///        TreeNode  getBottomLeftFront() const;
///        TreeNode  getBottomRightFront() const;
///        TreeNode  getLeftBack() const;
///        TreeNode  getRightBack() const;
///        TreeNode  getTopBack() const;
///        TreeNode  getBottomBack() const;
///        TreeNode  getTopLeftBack() const;
///        TreeNode  getTopRightBack() const;
///        TreeNode  getBottomLeftBack() const;
///        TreeNode  getBottomRightBack() const;

        std::vector<TreeNode> getAllNeighbours() const;
        //@}

      //@}

    };

} // end namespace ot

// Template specializations
template class ot::TreeNode<unsigned int, 1>;
template class ot::TreeNode<unsigned int, 2>;
template class ot::TreeNode<unsigned int, 3>;
template class ot::TreeNode<unsigned int, 4>;

#include "TreeNode.tcc"




namespace par {

  //Forward Declaration
  template <typename T>
    class Mpi_datatype;

/**
@author Rahul Sampath, rahul.sampath@gmail.com
@brief A template specialization of the abstract class "Mpi_datatype" for communicating messages of type "ot::TreeNode".
*/
  template <typename T, unsigned int dim>
    class Mpi_datatype< ot::TreeNode<T,dim> > {

    /* @masado Omitted all the comparison/reduction operations, limited to ::value(). */

      public:

      /**
       @return The MPI_Datatype corresponding to the datatype "ot::TreeNode".
     */
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


