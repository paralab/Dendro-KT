/**
 * @author: Milinda Fernando
 * School of Computing, University of Utah
 * @brief: Contains utility function to traverse the k-tree in the SFC order, 
 * These utility functions will be used to implement the mesh free matvec. 
 * 
 * 
*/

#ifndef DENDRO_KT_TRAVERSE_H
#define DENDRO_KT_TRAVERSE_H

#include<iostream>
#include<functional>


namespace fem
{
    /**
     * @brief: top_down bucket function
     * @param [in] coords_in: input points
     * @param [out] coords_out: output points 
     * @param [in] vec_in: input vector 
     * @param [out] vec_out: output vector
     * @param [in] sz: number of points (in points)
     * @param [out] out: bucktet points with duplications on the element boundary. 
     * @param [out] offsets: offsets for bucketed array size would be 1u<<dim
     * @param [out] counts: bucket counts
     * @param [in] refElement: Reference element
     * @param [out] map: bucketting scatter map from parent to child
     * @return: Return true when we hit a leaf element. 
     */
    template<typename T,typename TN, typename RE, unsigned int dim>
    bool top_down(const TN* coords_in, TN* coords_out, const T* vec_in, T* vec_out, unsigned int sz, unsigned int* offsets, unsigned int* counts, const RE* refElement, unsigned int * scattermap)
    {

    }

    /**
     * 
     * @brief: top_down bucket function
     * @param [in] in: input points
     * @param [in] sz: number of points (in points)
     * @param [out] out: bucktet points with duplications on the element boundary. 
     * @param [out] offsets: offsets for bucketed array size would be 1u<<dim
     * @param [out] counts: bucket counts
     * @param [in] refElement: Reference element
     * @return: Return true when we hit the root element. 
     */
    template<typename T,typename TN, typename RE, unsigned int dim>
    bool bottom_up(const TN* coords_in, TN* coords_out, const T* vec_in, T* vec_out, unsigned int sz, const unsigned int* offsets, const unsigned int* counts,const RE* refElement,unsigned int* gathermap)
    {


    }

    /**
     * @brief : mesh-free matvec
     * @param [in] vecIn: input vector (local vector)
     * @param [out] vecOut: output vector (local vector) 
     * @param [in] coords: coordinate points for the partition
     * @param [in] sz: number of points
     * @param [in] eleOp: Elemental operator (i.e. elemental matvec)
     * @param [in] refElement: reference element.
     */
    template<typename T,typename TN, typename RE,  unsigned int dim>
    void matvec(const T* vecIn, T* vecOut, const TN* coords, unsigned int sz, std::function<void(const T*, T*, TN* coords)> eleOp, const RE* refElement)
    {

        // 1. initialize the output vector to zero. 
        for(unsigned int i=0;i<sz;i++)
            vecOut[i] = (T)0;

        const unsigned int num_children=1u<<dim;

        //todo: if possible move these memory allocations out of the function. 
        // One option would be to do a dummy matvec (traversal) and figure out exact memory needed and use pre allocation for
        // subsequent matvecs. 

        unsigned int * offset = new unsigned int[num_children];
        unsigned int * counts = new unsigned int[num_children];
        
        // change this ot the correct pointer. 
        TN* coord_outs=NULL;
        T* vec_out = NULL;
        T* smap = NULL;
        T* gmap = NULL;

        bool isLeaf= top_down<TN,dim>(coords, coord_outs, vecIn, vec_out, sz, offset, counts,smap);
        
        if(!isLeaf)
        {
            // input points counts[i] > nPe assert();
            for(unsigned int c=0;c<num_children;c++)
            {
                matvec<T,TN,RE,dim>(vecOut+offset[c],vec_out,coord_outs+[offset[c]],counts[c],eleOp,refElement);
            }
        
        }else
        {
            // call eleOp function
            // Note you might need to identify the parent nodes for parent to child interpolation. 
            // (use reference element class for interpolation)

        }

        // call for bottom up;
        bool isRoot= bottom_up<T,TN,RE,dim> bottom_up(coords,coord_outs,vecOut,vec_out,offset,counts,refElement,gmap);
        
        delete [] offset;
        delete [] counts;

    }
   

} // end of namespace fem


#endif