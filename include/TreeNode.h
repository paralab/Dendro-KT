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

static unsigned int MAX_LEVEL=31;
//#define m_uiMaxDepth 30
extern unsigned int m_uiMaxDepth;

#include <algorithm>
#include "dendro.h"
#include <iostream>
#include <array>

namespace ot {

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


     

    };
        
} // end namespace ot

//#include "TreeNode.tcc"

// Template specializations
template class ot::TreeNode<unsigned int, 1>;
template class ot::TreeNode<unsigned int, 2>;
template class ot::TreeNode<unsigned int, 3>;
template class ot::TreeNode<unsigned int, 4>;

#endif //DENDRO_KT_TREENODE_H


