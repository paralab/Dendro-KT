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

}// end of namespace ot




#endif //DENDRO_KT_OCTUTILS_H
