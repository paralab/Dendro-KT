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
        std::mt19937_64 gen(rd());
        //gen.seed(1331);             // Uncomment for deterministic testing.
        /// std::uniform_int_distribution<T> distCoord(0, maxCoord);
        std::normal_distribution<double> distCoord((1u << m_uiMaxDepth) / 2, (1u << m_uiMaxDepth) / 100);
        std::uniform_int_distribution<T> distLevel(m_uiMaxDepth, m_uiMaxDepth);

        double coordClampLow = 0;
        double coordClampHi = (1u << m_uiMaxDepth);

        // Add points sequentially.
        for (int ii = 0; ii < numPoints; ii++)
        {
            for (T &u : uiCoords)
            {
                double dc = distCoord(gen);
                dc = (dc < coordClampLow ? coordClampLow : dc > coordClampHi ? coordClampHi : dc);
                u = (T) dc;
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
     * @author Masado Ishii
     * @brief  Add, remove, or permute dimensions from one TreeNode to another TreeNode.
     */
    template <typename T, unsigned int sdim, unsigned int ddim>
    inline void permuteDims(unsigned int nDims,
        const ot::TreeNode<T,sdim> &srcNode, unsigned int *srcDims,
        ot::TreeNode<T,ddim> &dstNode, unsigned int *dstDims)
    {
      dstNode.setLevel(srcNode.getLevel());
      for (unsigned int dIdx = 0; dIdx < nDims; dIdx++)
        dstNode.setX(dstDims[dIdx], srcNode.getX(srcDims[dIdx]));
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
     void sliceKTree(const ot::TreeNode<T,dim> * in,std::vector<ot::TreeNode<T,dim>> & out,unsigned int numNodes, unsigned int sDim, T sliceVal)
     {

         out.clear();
         for(unsigned int i=0;i<numNodes;i++)
         {
             if(in[i].minX(sDim) <= sliceVal && sliceVal <= in[i].maxX(sDim))
                 out.push_back(in[i]);
         }


     }

     /**
      * @author Masado Ishii
      * @brief perform a slicing operation and also lower the dimension.
      * @pre template parameter dim must be greater than 1.
      */
     template <typename T, unsigned int dim>
     void projectSliceKTree(const ot::TreeNode<T,dim> *in, std::vector<ot::TreeNode<T, dim-1>> &out,
         unsigned int numNodes, unsigned int sliceDim, T sliceVal)
     {
       std::vector<ot::TreeNode<T,dim>> sliceVector;
       sliceKTree(in, sliceVector, numNodes, sliceDim, sliceVal);

       // Lower the dimension.
       unsigned int selectDimSrc[dim-1];
       unsigned int selectDimDst[dim-1];
       #pragma unroll (dim-1)
       for (unsigned int dIdx = 0; dIdx < dim-1; dIdx++)
       {
         selectDimSrc[dIdx] = (dIdx < sliceDim ? dIdx : dIdx+1);
         selectDimDst[dIdx] = dIdx;
       }

       out.clear();
       ot::TreeNode<T, dim-1> tempNode;
       for (const ot::TreeNode<T,dim> &sliceNode : sliceVector)
       {
         permuteDims<T, dim, dim-1>(dim-1, sliceNode, selectDimSrc, tempNode, selectDimDst);
         out.push_back(tempNode);
       }
     }



}// end of namespace ot




#endif //DENDRO_KT_OCTUTILS_H
