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
#include <algorithm>

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
       * @brief Write and read ghosted nodal coordinates, to verify loaded checkpoints across API changes.
       */
      template <unsigned int dim>
      int writeDACoordsToFile(const char * fName, const ot::DA<dim> *da);  //write file

      template <unsigned int dim>
      int verifyDACoordsVsFile(const char * fName, const ot::DA<dim> *da, bool &match);  //read file and compare DA

      template <unsigned int dim>
      int readDACoordsFromFile(const char * fName, std::vector<ot::TreeNode<unsigned, dim>> &coords);  //read file


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
          int err = 1 != fwrite(&num,sizeof(unsigned int),1,outfile); // write out the number of elements.
          if (err)
            return err;

          if(num>0)
            err = num != fwrite(pNodes,sizeof(ot::TreeNode<TNT, dim>),num,outfile);
            // ok if TreeNodes are TriviallyCopyable

          return err;
      }

      template <typename TNT, unsigned int dim>
      int writeOctToFile(const char * fName,const ot::TreeNode<TNT, dim>* pNodes,const size_t num)
      {
          FILE* outfile = fopen(fName,"w");
          if(outfile==NULL) {std::cout<<fName<<" file open failed "<<std::endl; return 1;}
          int err = writeOctToFile(outfile, pNodes, num);
          fclose(outfile);
          return err;
      }


      template <typename TNT, unsigned int dim>
      int readOctFromFile(FILE* inpfile,std::vector<ot::TreeNode<TNT, dim>> & pNodes)
      {
          if(inpfile==NULL) {std::cout<<"Input file invalid "<<std::endl; return 1;}
          unsigned int num=0;
          pNodes.resize(0);

          int err = 1 != fread(&num,sizeof(unsigned int ),1,inpfile);
          if (err)
            return err;

          if(num>0)
          {
              pNodes.resize(num);
              err = num != fread(&(*(pNodes.begin())),(sizeof(ot::TreeNode<TNT, dim>)),num,inpfile);
              // ok if TreeNodes are TriviallyCopyable
          }

          return err;
      }

      template <typename TNT, unsigned int dim>
      int readOctFromFile(const char * fName,std::vector<ot::TreeNode<TNT, dim>> & pNodes)
      {
          FILE* inpfile = fopen(fName,"r");
          if(inpfile==NULL) {std::cout<<fName<<" file open failed "<<std::endl; return 1;}
          int err = readOctFromFile(inpfile, pNodes);
          fclose(inpfile);
          return err;
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

          int err = 0;
          if (!err)  err = 1 != fwrite(&daTotalNodalSz,   sizeof(size_t), 1, outfile);
          if (!err)  err = 1 != fwrite(&daLocalNodeBegin, sizeof(size_t), 1, outfile);
          if (!err)  err = 1 != fwrite(&daLocalNodeEnd,   sizeof(size_t), 1, outfile);
          if (!err)  err = 1 != fwrite(&ndofs,          sizeof(int),    1, outfile);

          const size_t offset = (isGhosted ? daLocalNodeBegin*ndofs : 0);

          if(daLocalNodalSz > 0)
            if (!err)
              err = daLocalNodalSz*ndofs != fwrite((vec + offset), sizeof(T), daLocalNodalSz*ndofs, outfile);

          fclose(outfile);
          return err;
      }



      template <typename T, unsigned int dim>
      int readVecFromFile(const char * fName, const ot::DA<dim>* da, T* vec, int ndofs, bool isGhosted)
      {
          const size_t daTotalNodalSz   = da->getTotalNodalSz();
          const size_t daLocalNodalSz   = da->getLocalNodalSz();
          const size_t daLocalNodeBegin = da->getLocalNodeBegin();
          const size_t daLocalNodeEnd   = daLocalNodalSz + daLocalNodeBegin;

          size_t fTotalNodalSz = 0;
          size_t fLocalNodalSz = 0;
          size_t fLocalNodeBegin = 0;
          size_t fLocalNodeEnd = 0;
          int fNdofs = 0;

          FILE * infile = fopen(fName,"r");
          if (infile == NULL)
          {
            std::cout << fName << " file open failed " << std::endl;
            return  1;
          }


          int err = 0;
          if (!err)  err = 1 != fread(&fTotalNodalSz,   sizeof(size_t), 1, infile);
          if (!err)  err = 1 != fread(&fLocalNodeBegin, sizeof(size_t), 1, infile);
          if (!err)  err = 1 != fread(&fLocalNodeEnd,   sizeof(size_t), 1, infile);
          if (!err)  err = 1 != fread(&fNdofs,          sizeof(int),    1, infile);

          if (fTotalNodalSz != daTotalNodalSz)
            std::cout << fName << " file number of total node mismatched with da." << "\n";

          if (fLocalNodeBegin != daLocalNodeBegin)
            std::cout << fName << " file local node begin location mismatched with da." << "\n";

          if (fLocalNodeEnd != daLocalNodeEnd)
            std::cout << fName << " file local node end location mismatched with da." << "\n";

          if (fNdofs != ndofs)
            std::cout << fName << " file ndofs mismatched with function argument ndofs." << "\n";

          if (!err)
            if (fTotalNodalSz   != daTotalNodalSz    ||
                fLocalNodeBegin != daLocalNodeBegin  ||
                fLocalNodeEnd   != daLocalNodeEnd    ||
                fNdofs          != ndofs)
              err = 1;

          const size_t offset = (isGhosted ? daLocalNodeBegin*ndofs : 0);

          if(daLocalNodalSz > 0)
            if (!err)  err = daLocalNodalSz*ndofs != fread((vec + offset), sizeof(T), daLocalNodalSz*ndofs, infile);

          fclose(infile);
          return err;
      }


      // writeDACoordsToFile()
      template <unsigned int dim>
      int writeDACoordsToFile(const char * fName, const ot::DA<dim> *da)
      {
        return writeOctToFile(fName, da->getTNCoords(), da->getTotalNodalSz());
      }

      // verifyDACoordsVsFile()
      template <unsigned int dim>
      int verifyDACoordsVsFile(const char * fName, const ot::DA<dim> *da, bool &match)
      {
        match = false;
        std::vector<ot::TreeNode<unsigned, dim>> fileNodeCoords;
        int err = readDACoordsFromFile(fName, fileNodeCoords);
        if (err)
          return err;
        match = (fileNodeCoords.size() == da->getTotalNodalSz()) and
            std::equal(fileNodeCoords.begin(), fileNodeCoords.end(), da->getTNCoords());
        return err;
      }

      // readDACoordsFromFile()
      template <unsigned int dim>
      int readDACoordsFromFile(const char * fName, std::vector<ot::TreeNode<unsigned, dim>> &coords)
      {
        return readOctFromFile(fName, coords);
      }





  } // end of namespace checkpoint

} // end of namespace io




#endif //DENDRO_KT_CHECKPOINT_H
