//
// Created by milinda on 7/12/17.
// Updated by Masado on 6/5/20.
/**
*@author Milinda Fernando
*School of Computing, University of Utah
*@brief Contans utility functions to save/load octree and variable list defined on the otree.
*/
//

#ifndef DENDRO_KT_CHECKPOINT_H
#define DENDRO_KT_CHECKPOINT_H

#include "treeNode.h"
#include "oda.h"
#include <iostream>
#include <fstream>

namespace io
{

  namespace checkpoint
  {

      /**
      *@brief Write an given octree to a file using binary file format.
      *@param[in] fName: file name
      *@param[in] pNodes: pointer to the begin location of the octree
      *@param[in] num: size of pNodes array or number of elements to write.
      * @note than this will write the elements in the order that they have in pNodes array.
       * binary file format:
       * <number of elements><list of elements ...>
      **/

      template <typename TNT, unsigned int dim>
      int writeOctToFile(const char * fName,const ot::TreeNode<TNT, dim>* pNodes,const size_t num);

      /**
      *@brief Reads an octree speficied by the file name.
      *param[in] fName: input file name
      *param[out] pNodes: TreeNodes read from the specified file.
      **/

      template <typename TNT, unsigned int dim>
      int readOctFromFile(const char * fName,std::vector<ot::TreeNode<TNT, dim>> & pNodes);


      /**
       * @breif: writes the variable vector to a file in binary format.
       * @param[in] fName: input file name.
       * @param[in] da: pointer to the DA which the variable defined on
       * @param [in] vec: pointer to the begin location of the variable vector.
       * binary file format:
       * <totalNodalSz><localBegin><localEnd><ndofs><values..>
       *
       * */
      template <typename T, unsigned int dim>
      int writeVecToFile(const char * fName, const ot::DA<dim>* da,const T* vec, int ndofs, bool isGhosted = false);

      /**
       * @breif: reads variable vector from the binary file.
       * @param[in] fName: input file name.
       * @param[in] da: pointer to the DA which the variable defined on
       * @param [out] vec: pointer to the begin location of the variable vector.(assumes memory allocated)
       * */
      template <typename T, unsigned int dim>
      int readVecFromFile(const char * fName, const ot::DA<dim>* da, T* vec, int ndofs, bool isGhosted = false);

  } // end of namespace checkpoint.

}// end of namespace io




// templated implementations
namespace io
{
  namespace checkpoint
  {
      template <typename TNT, unsigned int dim>
      int writeOctToFile(FILE* outfile,const ot::TreeNode<TNT, dim>* pNodes,const size_t num)
      {
          if(outfile==NULL) {std::cout<<"Output file invalid "<<std::endl; return 1;}
          fwrite(&num,sizeof(unsigned int),1,outfile); // write out the number of elements.

          if(num>0)
            fwrite(pNodes,sizeof(ot::TreeNode<TNT, dim>),num,outfile);

          return 0;
      }

      template <typename TNT, unsigned int dim>
      int writeOctToFile(const char * fName,const ot::TreeNode<TNT, dim>* pNodes,const size_t num)
      {
          FILE* outfile = fopen(fName,"w");
          if(outfile==NULL) {std::cout<<fName<<" file open failed "<<std::endl; return 1;}
          writeOctToFile(outfile, pNodes, num);
          fclose(outfile);
          return 0;
      }


      template <typename TNT, unsigned int dim>
      int readOctFromFile(FILE* inpfile,std::vector<ot::TreeNode<TNT, dim>> & pNodes)
      {
          if(inpfile==NULL) {std::cout<<"Input file invalid "<<std::endl; return 1;}
          unsigned int num=0;
          fread(&num,sizeof(unsigned int ),1,inpfile);

          pNodes.resize(0);
          if(num>0)
          {
              pNodes.resize(num);
              fread(&(*(pNodes.begin())),(sizeof(ot::TreeNode<TNT, dim>)),num,inpfile);
          }

          return 0;
      }

      template <typename TNT, unsigned int dim>
      int readOctFromFile(const char * fName,std::vector<ot::TreeNode<TNT, dim>> & pNodes)
      {
          FILE* inpfile = fopen(fName,"r");
          if(inpfile==NULL) {std::cout<<fName<<" file open failed "<<std::endl; return 1;}
          readOctFromFile(inpfile, pNodes);
          fclose(inpfile);
          return 0;
      }



      template <typename T, unsigned int dim>
      int writeVecToFile(const char * fName, const ot::DA<dim>* da, const T* vec, int ndofs, bool isGhosted)
      {
          const size_t daTotalNodalSz   = da->getTotalNodalSz();
          const size_t daLocalNodalSz   = da->getLocalNodalSz();
          const size_t daLocalNodeBegin = da->getLocalNodeBegin();
          const size_t daLocalNodeEnd   = daLocalNodalSz + daLocalNodeBegin;

          FILE * outfile=fopen(fName,"w");
          if (outfile == NULL)
          {
            std::cout << fName << " file open failed " << std::endl;
            return  1;
          }

          fwrite(&daTotalNodalSz,   sizeof(size_t), 1, outfile);
          fwrite(&daLocalNodeBegin, sizeof(size_t), 1, outfile);
          fwrite(&daLocalNodeEnd,   sizeof(size_t), 1, outfile);
          fwrite(&ndofs,          sizeof(int),    1, outfile);

          const size_t offset = (isGhosted ? daLocalNodeBegin*ndofs : 0);

          if(daLocalNodalSz > 0)
            fwrite((vec + offset), sizeof(T), daLocalNodalSz*ndofs, outfile);

          fclose(outfile);
          return 0;
      }



      template <typename T, unsigned int dim>
      int readVecFromFile(const char * fName, const ot::DA<dim>* da, T* vec, int ndofs, bool isGhosted)
      {
          const size_t daTotalNodalSz   = da->getTotalNodalSz();
          const size_t daLocalNodalSz   = da->getLocalNodalSz();
          const size_t daLocalNodeBegin = da->getLocalNodeBegin();
          const size_t daLocalNodeEnd   = daLocalNodalSz + daLocalNodeBegin;

          size_t fTotalNodalSz;
          size_t fLocalNodalSz;
          size_t fLocalNodeBegin;
          size_t fLocalNodeEnd;
          int fNdofs;

          FILE * infile = fopen(fName,"r");
          if (infile == NULL)
          {
            std::cout << fName << " file open failed " << std::endl;
            return  1;
          }


          fread(&fTotalNodalSz,   sizeof(size_t), 1, infile);
          fread(&fLocalNodeBegin, sizeof(size_t), 1, infile);
          fread(&fLocalNodeEnd,   sizeof(size_t), 1, infile);
          fread(&fNdofs,          sizeof(int),    1, infile);

          if (fTotalNodalSz != daTotalNodalSz)
            std::cout << fName << " file number of total node mismatched with da." << "\n";

          if (fLocalNodeBegin != daLocalNodeBegin)
            std::cout << fName << " file local node begin location mismatched with da." << "\n";

          if (fLocalNodeEnd != daLocalNodeEnd)
            std::cout << fName << " file local node end location mismatched with da." << "\n";

          if (fNdofs != ndofs)
            std::cout << fName << " file ndofs mismatched with function argument ndofs." << "\n";

          if (fTotalNodalSz   != daTotalNodalSz    ||
              fLocalNodeBegin != daLocalNodeBegin  ||
              fLocalNodeEnd   != daLocalNodeEnd    ||
              fNdofs          != ndofs)
            return 1;

          const size_t offset = (isGhosted ? daLocalNodeBegin*ndofs : 0);

          if(daLocalNodalSz > 0)
            fread((vec + offset), sizeof(T), daLocalNodalSz*ndofs, infile);

          fclose(infile);
          return 0;
      }


  } // end of namespace checkpoint

} // end of namespace io




#endif //DENDRO_KT_CHECKPOINT_H
