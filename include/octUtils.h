/**
 * @file:octUtils.h
 * @author: Masado Ishii  -- UofU SoC,
 * @breif contains utility functions for the octree related computations.
 */

#ifndef DENDRO_KT_OCTUTILS_H
#define DENDRO_KT_OCTUTILS_H

#include <vector>
#include <functional>
#include <random>
#include <iostream>
#include <stdio.h>

#include "refel.h"
#include "nsort.h"
#include "tsort.h"
#include "treeNode.h"

#include "parUtils.h"

namespace ot
{

    /**
     * @author: Masado Ishi
     * @brief: generate random set of treeNodes for a specified dimension
     * @param[in] numPoints: number of treeNodes need to be generated.
     * */
    template <typename T, unsigned int dim>
    inline std::vector<ot::TreeNode<T,dim>> getPts(unsigned int numPoints, unsigned int sLev = m_uiMaxDepth, unsigned int eLev = m_uiMaxDepth)
    {
        std::vector<ot::TreeNode<T,dim>> points;
        std::array<T,dim> uiCoords;

        //const T maxCoord = (1u << MAX_LEVEL) - 1;
        const T maxCoord = (1u << m_uiMaxDepth) - 1;
        const T leafLevel = m_uiMaxDepth;

        // Set up random number generator.
        std::random_device rd;
        std::mt19937_64 gen(rd());    // 1. Use this for random/pseudorandom testing.
        /// std::mt19937_64 gen(1331);    // 2. Use this for deterministic testing.
        /// std::uniform_int_distribution<T> distCoord(0, maxCoord);
        std::normal_distribution<double> distCoord((1u << m_uiMaxDepth) / 2, (1u << m_uiMaxDepth) / 25);
        std::uniform_int_distribution<T> distLevel(sLev, eLev);

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



/**
   * @author  Hari Sundar
   * @author  Milinda Fernando
   * @author  Masado Ishii
   * @brief   Generates an octree based on a function provided by the user based on the Wavelet method to decide on adaptivity.
   * @param[in] fx:        the function that taxes $x,y,z$ coordinates and returns the the value at that point
   * @param[in] numVars: number of total variables computed from fx function.
   * @param[in] varIndex: variable index location that can be used to ascess double * pointer in fx, to determine the refinement of octree.
   * @param[in] numInterpVars: Number of variables considered during the refinement
   * @param[in] maxDepth:  The maximum depth that the octree should be refined to.
   * @param[in] interp_tol: user specified tolerance for the wavelet representation of the function.
   * @param[in] sfc_tol: sfc tree partitioning tolerance to control communication/load balance tradeoff.
   * @param[in] elementOrder order of the element when defining the wavelet representation of the function.
   * @param[in] comm      The MPI communicator to be use for parallel computation.
   *
   * Generates an octree based on a function provided by the user. The function is expected to return the
   * signed distance to the surface that needs to be meshed. The coordinates are expected to be in [0,1]^3.
   *
   * Works on arbitrary dimensionality.
   *
   * Ported from Dendro-5.0.
   */

template <typename T, unsigned int dim>
int function2Octree(std::function<void(const double *, double*)> fx,const unsigned int numVars,const unsigned int* varIndex,const unsigned int numInterpVars, std::vector<ot::TreeNode<T,dim>> & nodes,unsigned int maxDepth, const double & interp_tol, const double sfc_tol, unsigned int elementOrder,MPI_Comm comm );

template <typename T, unsigned int dim>
int function2Octree(std::function<void(const double *, double*)> fx,const unsigned int numVars,const unsigned int* varIndex,const unsigned int numInterpVars, std::vector<ot::TreeNode<T,dim>> & nodes,unsigned int maxDepth, const double & interp_tol, const double sfc_tol, unsigned int elementOrder,MPI_Comm comm )
{
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  constexpr unsigned int NUM_CHILDREN = 1u << dim;

  // "nodes" meaning TreeNodes here.
  nodes.clear();
  std::vector<ot::TreeNode<T,dim>> nodes_new;

  unsigned int depth = 1;
  unsigned int num_intersected=1;
  unsigned int num_intersected_g=1;
  const unsigned int nodesPerElement = intPow(elementOrder+1, dim);

  double* varVal=new double [numVars];
  double* dist_parent=new double[numVars*nodesPerElement];
  double* dist_child=new double[numVars*nodesPerElement];
  double* dist_child_ip=new double[numVars*nodesPerElement];

  // "nodes" meaning element nodes here.
  std::vector<ot::TreeNode<T,dim>> tmpENodes(nodesPerElement);
  tmpENodes.clear();
  double ptCoords[dim];

  const double domScale = 1.0 / (1u << m_uiMaxDepth);
  RefElement refEl(dim, elementOrder);
  double l2_norm=0;
  bool splitOctant=false;


  if (!rank) {
    // root does the initial refinement
    //std::cout<<"initial ref:"<<std::endl;
    ot::TreeNode<T,dim> root;
    for (unsigned int cnum = 0; cnum < NUM_CHILDREN; cnum++)
      nodes.push_back(root.getChildMorton(cnum));

    while ( (num_intersected > 0 ) && (num_intersected < size/**size*/ ) && (depth < maxDepth) ) {
      std::cout << "Depth: " << depth << " n = " << nodes.size() << std::endl;
      num_intersected = 0;

      for (auto elem: nodes ){
        splitOctant=false;
        if ( elem.getLevel() != depth ) {
          nodes_new.push_back(elem);
          continue;
        }

        // check and split

        // Evaluate fx() on positions of (e)nodes of elem.
        tmpENodes.clear();
        ot::Element<T,dim>(elem).appendNodes(elementOrder, tmpENodes);
        for (unsigned int eNodeIdx = 0; eNodeIdx < tmpENodes.size(); eNodeIdx++)
        {
          for (int d = 0; d < dim; d++)
            ptCoords[d] = domScale * tmpENodes[eNodeIdx].getX(d);   // TODO this is what class Point is for.
          fx(ptCoords, varVal);
          for (unsigned int var = 0; var < numInterpVars; var++)
            dist_parent[varIndex[var]*nodesPerElement + eNodeIdx] = varVal[varIndex[var]];
        }
        tmpENodes.clear();  // Yeah this is redundant but it makes clear how 'tmp' the buffer really is.

        // Interpolate each parent->child and check if within error tolerance.
        for(unsigned int cnum=0;cnum<NUM_CHILDREN;cnum++)
        {
          ot::TreeNode<T,dim> elemChild = elem.getChildMorton(cnum);

          // Evaluate fx() on positions of (e)nodes of elemChild.
          tmpENodes.clear();
          ot::Element<T,dim>(elemChild).appendNodes(elementOrder, tmpENodes);
          for (unsigned int eNodeIdx = 0; eNodeIdx < tmpENodes.size(); eNodeIdx++)
          {
            for (int d = 0; d < dim; d++)
              ptCoords[d] = domScale * tmpENodes[eNodeIdx].getX(d);   // TODO this is what class Point is for.
            fx(ptCoords, varVal);
            for (unsigned int var = 0; var < numInterpVars; var++)
              dist_child[varIndex[var]*nodesPerElement + eNodeIdx] = varVal[varIndex[var]];
          }
          tmpENodes.clear();

          for(unsigned int var=0;var<numInterpVars;var++)
          {
            refEl.IKD_Parent2Child<dim>(dist_parent+varIndex[var]*nodesPerElement, dist_child_ip+varIndex[var]*nodesPerElement, cnum);
            l2_norm=normLInfty(dist_child+varIndex[var]*nodesPerElement, dist_child_ip+varIndex[var]*nodesPerElement, nodesPerElement);
            if(l2_norm>interp_tol)
            {
              splitOctant=true;
              break;
            }
          }

          if(splitOctant) break;
        }

        if (!splitOctant) {
          nodes_new.push_back(elem);
        }else {
          for (unsigned int cnum = 0; cnum < NUM_CHILDREN; cnum++)
            nodes_new.push_back(elem.getChildMorton(cnum));
          num_intersected++;
        }
      }
      depth++;
      std::swap(nodes, nodes_new);
      nodes_new.clear();
    }
  } // !rank

  // now scatter the elements.
  DendroIntL totalNumOcts = nodes.size(), numOcts;

  par::Mpi_Bcast<DendroIntL>(&totalNumOcts, 1, 0, comm);

  // TODO do proper load balancing.
  numOcts = totalNumOcts/size + (rank < totalNumOcts%size);
  par::scatterValues<ot::TreeNode<T,dim>>(nodes, nodes_new, numOcts, comm);
  std::swap(nodes, nodes_new);
  nodes_new.clear();


  // now refine in parallel.
  par::Mpi_Bcast(&depth, 1, 0, comm);
  num_intersected=1;

  ot::TreeNode<T,dim> root;

  while ( (num_intersected > 0 ) && (depth < maxDepth) ) {
    if(!rank)std::cout << "Depth: " << depth << " n = " << nodes.size() << std::endl;
    num_intersected = 0;

    for (auto elem: nodes ){
      splitOctant=false;
      if ( elem.getLevel() != depth ) {
        nodes_new.push_back(elem);
        continue;
      }

      // Evaluate fx() on positions of (e)nodes of elem.
      tmpENodes.clear();
      ot::Element<T,dim>(elem).appendNodes(elementOrder, tmpENodes);
      for (unsigned int eNodeIdx = 0; eNodeIdx < tmpENodes.size(); eNodeIdx++)
      {
        for (int d = 0; d < dim; d++)
          ptCoords[d] = domScale * tmpENodes[eNodeIdx].getX(d);   // TODO this is what class Point is for.
        fx(ptCoords, varVal);
        for (unsigned int var = 0; var < numInterpVars; var++)
          dist_parent[varIndex[var]*nodesPerElement + eNodeIdx] = varVal[varIndex[var]];
      }
      tmpENodes.clear(); 

      // check and split

      // Interpolate each parent->child and check if within error tolerance.
      for(unsigned int cnum=0;cnum<NUM_CHILDREN;cnum++)
      {
        ot::TreeNode<T,dim> elemChild = elem.getChildMorton(cnum);

        // Evaluate fx() on positions of (e)nodes of elemChild.
        tmpENodes.clear();
        ot::Element<T,dim>(elemChild).appendNodes(elementOrder, tmpENodes);
        for (unsigned int eNodeIdx = 0; eNodeIdx < tmpENodes.size(); eNodeIdx++)
        {
          for (int d = 0; d < dim; d++)
            ptCoords[d] = domScale * tmpENodes[eNodeIdx].getX(d);   // TODO this is what class Point is for.
          fx(ptCoords, varVal);
          for (unsigned int var = 0; var < numInterpVars; var++)
            dist_child[varIndex[var]*nodesPerElement + eNodeIdx] = varVal[varIndex[var]];
        }
        tmpENodes.clear();

        for(unsigned int var=0;var<numInterpVars;var++)
        {
          refEl.IKD_Parent2Child<dim>(dist_parent+varIndex[var]*nodesPerElement, dist_child_ip+varIndex[var]*nodesPerElement, cnum);
          l2_norm=normLInfty(dist_child+varIndex[var]*nodesPerElement, dist_child_ip+varIndex[var]*nodesPerElement, nodesPerElement);
          //std::cout<<"rank: "<<rank<<" node: "<<elem<<" l2 norm : "<<l2_norm<<" var: "<<varIndex[var]<<std::endl;
          if(l2_norm>interp_tol)
          {
            splitOctant=true;
            break;
          }
        }

        if(splitOctant) break;
      }

      if (!splitOctant) {
        nodes_new.push_back(elem);
      }else {
        for (unsigned int cnum = 0; cnum < NUM_CHILDREN; cnum++)
          nodes_new.push_back(elem.getChildMorton(cnum));
        num_intersected++;
      }
    }
    depth++;
    std::swap(nodes, nodes_new);
    nodes_new.clear();

    // The tree is already a complete tree, just need to re-partition and remove dups.
    // Dendro-KT distTreeSort() doesn't remove duplicates automatically;
    // however, distTreeConstruction() does. Calling distTreeConstruction()
    // on an already complete tree should do exactly what we want.
    ot::SFC_Tree<T,dim>::distRemoveDuplicates(nodes, sfc_tol, false, comm);

    // This is buggy because distTreeConstruction doesn't respect maxPtsPerRegion,
    // because distTreePartition() doesn't respect noSplitThresh.
    /// ot::SFC_Tree<T,dim>::distTreeConstruction(nodes, nodes_new, 1, sfc_tol, comm);

    par::Mpi_Allreduce(&num_intersected,&num_intersected_g,1,MPI_MAX,comm);
    num_intersected=num_intersected_g;
  }

  delete[] dist_child;
  delete[] dist_child_ip;
  delete[] dist_parent;
  delete[] varVal;
}



template <typename T, unsigned int dim, typename NodeT>
std::ostream & printNodes(const ot::TreeNode<T, dim> *coordBegin,
                          const ot::TreeNode<T, dim> *coordEnd,
                          const NodeT *valBegin,
                          std::ostream & out = std::cout)
{
  ot::TreeNode<T, dim> subdomain;
  unsigned int deepestLev = 0;
  const unsigned int numNodes = coordEnd - coordBegin;
  using YXV = std::pair<std::pair<T,T>, NodeT>;
  const T top = 1u << m_uiMaxDepth;
  std::vector<YXV> zipped;
  /// if (numNodes)
  ///   subdomain = *coordBegin;
  for (unsigned int ii = 0; ii < numNodes; ii++)
  {
    /// while (!(Element<T,dim>(subdomain).isIncident(coordBegin[ii])))
    ///   subdomain = subdomain.getParent();

    if (coordBegin[ii].getLevel() > deepestLev)
      deepestLev = coordBegin[ii].getLevel();

    zipped.push_back(YXV{{top - coordBegin[ii].getX(1), coordBegin[ii].getX(0)}, valBegin[ii]});
  }
  subdomain = ot::TreeNode<T, dim>();
  const T origin[2] = {subdomain.getX(0), top - subdomain.getX(1)};

  std::sort(zipped.begin(), zipped.end());

  const unsigned int numTiles1D = (1u << int(deepestLev) - int(subdomain.getLevel())) + 1;
  const unsigned int charBound = (numTiles1D * 10 + 4)*numTiles1D + 2;
  std::vector<char> charBuffer(charBound + 10, '\0');
  char * s = charBuffer.data();
  const char * bufEnd = &(*charBuffer.end()) - 10;

  T cursorY = 0, cursorX = 0;
  cursorY = origin[1];
  for (unsigned int ii = 0; ii < numNodes;)
  {
    cursorY = zipped[ii].first.first;
    cursorX = origin[0];

    while (ii < numNodes && zipped[ii].first.first == cursorY)
    {
      T x = zipped[ii].first.second;
      NodeT val = zipped[ii].second;

      while (cursorX < x)
      {
        s += snprintf(s, bufEnd-s, "    \t");
        cursorX += (1u << m_uiMaxDepth - deepestLev);
      }
      s += snprintf(s, bufEnd-s, "%01.2f\t", val);
      cursorX += (1u << m_uiMaxDepth - deepestLev);
      ii++;
    }

    if (ii < numNodes)
    {
      T nextY = zipped[ii].first.first;
      while (cursorY < nextY)
      {
        s += snprintf(s, bufEnd-s, "\n\n\n");
        cursorY += (1u << m_uiMaxDepth - deepestLev);
      }
    }
  }
  s += snprintf(s, bufEnd-s, "\n");

  out << charBuffer.data();

  return out;
}




}// end of namespace ot




#endif //DENDRO_KT_OCTUTILS_H
