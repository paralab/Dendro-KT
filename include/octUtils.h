/**
 * @file:octUtils.h
 * @author: Masado Ishii  -- UofU SoC,
 * @breif contains utility functions for the octree related computations.
 */

#ifndef DENDRO_KT_OCTUTILS_H
#define DENDRO_KT_OCTUTILS_H

#include <vector>
#include <random>
#include <iostream>
#include <stdio.h>
#include "treeNode.h"

namespace ot
{

    /**
     * @author: Masado Ishi
     * @brief: generate random set of treeNodes for a specified dimension
     * @param[in] numPoints: number of treeNodes need to be generated.
     * */
    template <typename T, unsigned int dim>
    inline std::vector<ot::TreeNode<T,dim>> getPts(unsigned int numPoints)
    {
        std::vector<ot::TreeNode<T,dim>> points;
        std::array<T,dim> uiCoords;

        //const T maxCoord = (1u << MAX_LEVEL) - 1;
        const T maxCoord = (1u << m_uiMaxDepth) - 1;
        const T leafLevel = m_uiMaxDepth;

        // Set up random number generator.
        std::random_device rd;
        std::mt19937 gen(rd());
        //gen.seed(1331);             // Uncomment for deterministic testing.
        std::uniform_int_distribution<T> distCoord(0, maxCoord);
        std::uniform_int_distribution<T> distLevel(1, m_uiMaxDepth);

        // Add points sequentially.
        for (int ii = 0; ii < numPoints; ii++)
        {
            for (T &u : uiCoords)
            {
                u = distCoord(gen);
            }
            //ot::TreeNode<T,dim> tn(uiCoords, leafLevel);
            ot::TreeNode<T,dim> tn(uiCoords, distLevel(gen));
            points.push_back(tn);
        }

        return points;
    }

    /**
     * @author Masado Ishii
     * @brief  Separate a list of TreeNodes into separate vectors by level.
     */
    template <typename T, unsigned int dim>
    inline std::vector<std::vector<ot::TreeNode<T,dim>>>
        stratifyTree(const std::vector<ot::TreeNode<T,dim>> &tree)
    {
      std::vector<std::vector<ot::TreeNode<T,dim>>> treeLevels;

      treeLevels.resize(m_uiMaxDepth + 1);

      for (ot::TreeNode<T,dim> tn : tree)
        treeLevels[tn.getLevel()].push_back(tn);

      return treeLevels;
    }

    /**
     * @brief perform slicing operation on k trees.
     * @param[in] in: input k-tree
     * @param[out] out: sliced k-tree
     * @param[in] numNodes: number of input nodes
     * @param[in] sDim: slicing dimention.
     * @param[in] sliceVal: extraction value for the slice
     * @param[in] tolernace: tolerance value for slice extraction.
     * */
     template<typename T, unsigned int dim>
     void sliceKTree(const ot::TreeNode<T,dim> * in,std::vector<ot::TreeNode<T,dim>> & out,unsigned int numNodes, unsigned int sDim, T sliceVal,double tolerance=1e-6)
     {

         out.clear();
         for(unsigned int i=0;i<numNodes;i++)
         {
             if(fabs(in[i].getX(sDim)-sliceVal)<=tolerance)
                 out[i].push_back(in[i]);
         }


     }



}// end of namespace ot




#endif //DENDRO_KT_OCTUTILS_H
