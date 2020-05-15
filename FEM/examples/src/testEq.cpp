//
// Created by maksbh on 5/10/20.
//
#include "hcurvedata.h"
#include "distTree.h"
#include "meshLoop.h"
#include "tsort.h"

#include <mpi.h>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <oda.h>
#include <testMat.h>


int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv,NULL,NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);
  int eleOrder = 1;
  typedef  unsigned int UINT;
  const int DIM = 3;
  _InitializeHcurve(DIM);

  bool doRefine = false;
  doRefine  = static_cast<bool>(std::atoi(argv[1]));

  std::vector<ot::TreeNode<UINT, DIM>> newTree;
  std::vector<ot::TreeNode<UINT, DIM>> surrTree;
  std::vector<ot::TreeNode<unsigned int, DIM>> treePart;
  ot::DistTree<UINT , DIM> origDTree = ot::DistTree<UINT , DIM>::constructSubdomainDistTree(1, MPI_COMM_WORLD);
  std::vector<ot::TreeNode<UINT, DIM>> srcTree = origDTree.getTreePartFiltered();
  std::vector<ot::OCT_FLAGS::Refine> octFlags(srcTree.size(),ot::OCT_FLAGS::Refine::OCT_NO_CHANGE);
  /*** Just refine the very first element **/
  if(doRefine){
    octFlags[0] = ot::OCT_FLAGS::Refine::OCT_REFINE;
  }
  ot::SFC_Tree<UINT , DIM>::distRemesh(srcTree, octFlags, newTree, surrTree, 0.3, comm);
  const UINT numElement = newTree.size();
  ot::DA<DIM> * octDA = new ot::DA<DIM>(newTree,MPI_COMM_WORLD,eleOrder);
  const ot::RankI numNodes = octDA->getGlobalNodeSz();
  std::cout << "Number of elements in DA = " << numElement << "\n";
  std::cout << "Number of nodes in DA = " << numNodes << "\n";
  std::cout << "Local to global:\n";
  const std::vector<ot::RankI> &ghostedGlobalNodeId = octDA->getNodeLocalToGlobalMap();
  assert(octDA->getGlobalNodeSz() == octDA->getTotalNodalSz());
  for (ot::RankI nIdx = 0; nIdx < numNodes; ++nIdx)
  {
    fprintf(stdout, "[%2lu]->%2lu  ", nIdx, ghostedGlobalNodeId[nIdx]);
    if ((5*(nIdx+1))%numNodes < (5*nIdx)%numNodes)
      fprintf(stdout, "\n");
  }
  std::cout << "Coordinates:\n";
  for (ot::RankI gnIdx = 0; gnIdx < numNodes; ++gnIdx)
  {
    std::cout << "id_" << std::setw(2) << gnIdx << "   ";
    ot::printtn(octDA->getTNCoords()[gnIdx], 2, std::cout);
    std::cout << "\n";
  }

  std::cout << "\n\nPrint nonhanging for all elements.\n";


  std::cout << "----------------------------------------------Checking overall assembly---------------------------------\n";
  /** Overall Assembly **/

  {
    testEq::testMat<DIM> matrix(octDA, 1, testEq::AssemblyCheck::Overall);

    Vec in;
    Vec out;

    octDA->petscCreateVector(in, false, false, 1);
    VecSet(in, 1.0);
    octDA->petscCreateVector(out, false, false, 1);

    /** Perform MatVec **/
    matrix.matVec(in, out, 1.0);


    /** Perform Matrix Assembly **/
    Mat J;
    octDA->createMatrix(J, MATAIJ, 1);
    MatZeroEntries(J);
    ot::MatCompactRows matCompactRows = matrix.collectMatrixEntries();

    for (int r = 0; r < matCompactRows.getNumRows(); r++) {
      for (int c = 0; c < matCompactRows.getChunkSize(); c++) {
        MatSetValue(J,
                    matCompactRows.getRowIdxs()[r],
                    matCompactRows.getColIdxs()[r * matCompactRows.getChunkSize() + c],
                    matCompactRows.getColVals()[r * matCompactRows.getChunkSize() + c],
                    ADD_VALUES);
      }
    }
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

    Vec matTimesVec;
    octDA->petscCreateVector(matTimesVec, false, false, 1);
    MatMult(J, in, matTimesVec);

    const double *arrayOut, *arrayMatTimesVec;
    VecGetArrayRead(matTimesVec, &arrayMatTimesVec);
    VecGetArrayRead(out, &arrayOut);
    int lsize;
    VecGetSize(matTimesVec, &lsize);
    bool testPassed = true;
    for (int i = 0; i < lsize; i++) {
      double diff = arrayMatTimesVec[i] - arrayOut[i];
      if (fabs(diff) > 1E-9) {
        std::cout << "Difference of " << YLW << diff << NRM << " found at loc = " << YLW << i << NRM << "\t\t";
        std::cout << "matrix-based: " << YLW << arrayMatTimesVec[i] << NRM << "\t  matrix-free: " << YLW << arrayOut[i] << NRM << "\n";
        testPassed = false;
      }
    }
    if (!testPassed) {
      std::cout << RED << "Test Failed" << NRM << "\n";
    } else {
      std::cout << GRN << "Test Passed " << NRM << "\n";
    }
    VecDestroy(&in);
    VecDestroy(&out);
    MatDestroy(&J);
    VecDestroy(&matTimesVec);
  }
  std::cout << "--------------------------------Checking each element assembly-------------------------------------------\n";
  {
    for(int elemID = 0; elemID < numElement; elemID++){
      testEq::testMat<DIM> matrix(octDA, 1, testEq::AssemblyCheck::ElementByElement,elemID);
      std::cout << "Testing for elemID = " << elemID << "\n";
      Vec in;
      Vec out;
      octDA->petscCreateVector(in, false, false, 1);
      VecSet(in, 1.0);
      octDA->petscCreateVector(out, false, false, 1);

      /** Perform MatVec **/
      matrix.matVec(in, out, 1.0);


      /** Perform Matrix Assembly **/
      Mat J;
      octDA->createMatrix(J, MATAIJ, 1);
      MatZeroEntries(J);
      matrix.preMat();
      ot::MatCompactRows matCompactRows = matrix.collectMatrixEntries();

      for (int r = 0; r < matCompactRows.getNumRows(); r++) {
        for (int c = 0; c < matCompactRows.getChunkSize(); c++) {
          MatSetValue(J,
                      matCompactRows.getRowIdxs()[r],
                      matCompactRows.getColIdxs()[r * matCompactRows.getChunkSize() + c],
                      matCompactRows.getColVals()[r * matCompactRows.getChunkSize() + c],
                      ADD_VALUES);
        }
      }
      MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

      Vec matTimesVec;
      octDA->petscCreateVector(matTimesVec, false, false, 1);
      MatMult(J, in, matTimesVec);

      const double *arrayOut, *arrayMatTimesVec;
      VecGetArrayRead(matTimesVec, &arrayMatTimesVec);
      VecGetArrayRead(out, &arrayOut);
      int lsize;
      VecGetSize(matTimesVec, &lsize);
      bool testPassed = true;
      for (int i = 0; i < lsize; i++) {
        double diff = arrayMatTimesVec[i] - arrayOut[i];
        if (fabs(diff) > 1E-9) {
          /// std::cout << YLW << "Difference of " << diff << " found at loc = " << i << NRM << "\n";

          std::cout << "Difference of " << YLW << diff << NRM << " found at loc = " << YLW << i << NRM << "\t\t";
          std::cout << "matrix-based: " << YLW << arrayMatTimesVec[i] << NRM << "\t  matrix-free: " << YLW << arrayOut[i] << NRM << "\n";

          testPassed = false;
        }
      }
      if (!testPassed) {
        std::cout << RED << "Test Failed for elem = " << elemID <<  NRM << "\n";
      } else {
        std::cout << GRN << "Test Passed for elem = " << elemID <<  NRM << "\n";
      }
      VecDestroy(&in);
      VecDestroy(&out);
      MatDestroy(&J);
      VecDestroy(&matTimesVec);
    }
  }




 PetscFinalize();
}
