//
// Created by milinda on 10/5/18.
//

/**
 * @author Hari Sundar
 * @author Rahul Sampath
 * @author Milinda Fernando
 * @breif Refactored version of the TreeNode class with the minimum data required.
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
      */
    template<typename T,unsigned int dim>
    class TreeNode {
    protected:
        
        /**TreeNode coefficients*/
        std::array<T,dim> m_uiCoords;

        /**level of the tree node*/
        unsigned int m_uiLevel;


    public:

     

    };
        
}


#endif //DENDRO_KT_TREENODE_H


