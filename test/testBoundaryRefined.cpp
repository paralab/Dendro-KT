#ifdef BUILD_WITH_PETSC
#include "petsc.h"
#endif

#include <iostream>
#include <distTree.h>
#include <oda.h>
#include <point.h>
#include <sfcTreeLoop_matvec_io.h>
#include <octUtils.h>

constexpr unsigned int DIM = 3;
constexpr unsigned int nchild = 1u << DIM;

template <unsigned int dim>
void printTree(const std::vector<ot::TreeNode<unsigned int, dim>> &treePart, int elev)
{
  std::cout << "Tree\n";
  for (const ot::TreeNode<unsigned int, dim> &tn : treePart)
  {
    ot::printtn(tn, elev, std::cout);
    std::cout << "\n";
  }
  std::cout << "\n";
}

void printMaxCoords(ot::DA<DIM> & octDA)
{
  const size_t sz = octDA.getTotalNodalSz();
  auto partFront = octDA.getTreePartFront();
  auto partBack = octDA.getTreePartBack();
  const auto tnCoords = octDA.getTNCoords();

  {
    std::vector<double> maxCoords(4, 0.0);

    const int eleOrder = 1;
    const bool visitEmpty = false;
    const unsigned int padLevel = 0;
    const unsigned int npe = octDA.getNumNodesPerElement();

    ot::MatvecBaseCoords<DIM> loop(sz, eleOrder, visitEmpty, padLevel, tnCoords, *partFront, *partBack);
    while (!loop.isFinished())
    {
      if (loop.isPre() && loop.subtreeInfo().isLeaf())
      {
        const double *nodeCoordsFlat = loop.subtreeInfo().getNodeCoords();
        for (int i = 0; i < npe; i++)
        {
          for (int d = 0; d < DIM; d++)
          {
            if (maxCoords[d] < nodeCoordsFlat[i * DIM + d])
            {
              maxCoords[d] = nodeCoordsFlat[i * DIM + d];
            }
          }
        }
        loop.next();
      }
      else
      {
        loop.step();
      }
    }
    std::cout << "maxCoords == " << maxCoords[0] << " " << maxCoords[1] << " " << maxCoords[2] << "\n";
  }
}

void getBoundaryElements(const ot::DA<DIM>* octDA, unsigned  int eleOrder, const std::string fname){
  const size_t sz = octDA->getTotalNodalSz();
  auto partFront = octDA->getTreePartFront();
  auto partBack = octDA->getTreePartBack();
  const auto tnCoords = octDA->getTNCoords();
  const unsigned int npe = octDA->getNumNodesPerElement();
  int counter = 0;
  std::string boundaryFname = fname+"_boundary.txt";
  std::string nonBoundaryFname = fname+"_nonboundary.txt";
  std::ofstream foutBoundary(boundaryFname.c_str());
  std::ofstream foutNonBoundary(nonBoundaryFname.c_str());
  ot::MatvecBaseCoords <DIM> loop(sz,eleOrder, false,0,tnCoords,*partFront,*partBack);
  while(!loop.isFinished()){
    if (loop.isPre() && loop.subtreeInfo().isLeaf()) {
      const double *nodeCoordsFlat = loop.subtreeInfo().getNodeCoords();
      if(loop.subtreeInfo().isElementBoundary()){
        counter++;
        for(int i = 0; i < npe; i++){
          foutBoundary << nodeCoordsFlat[3*i+0] << " "<< nodeCoordsFlat[3*i+1] << " " << nodeCoordsFlat[3*i+2] << "\n";
        }
      }
      else{
        for(int i = 0; i < npe; i++) {
          foutNonBoundary << nodeCoordsFlat[3 * i + 0] << " " << nodeCoordsFlat[3 * i + 1] << " "
                          << nodeCoordsFlat[3 * i + 2] << "\n";
        }
      }
      loop.next();
    }
    else{
      loop.step();
    }
  }
  foutBoundary.close();
  foutNonBoundary.close();
  std::cout << "Number of boundary elements = " << counter << "\n";
}

bool DomainDecider(const double * physCoords, double physSize)
{
  bool isInside = true;
  if (   physCoords[0] < 0.0 || physCoords[0] + physSize > 0.5
      || physCoords[1] < 0.0 || physCoords[1] + physSize > 1.0
      || physCoords[2] < 0.0 || physCoords[2] + physSize > 1.0)
  {
    isInside = false;
  }

//    if ((physCoords[0] > 0.2 and physCoords[1] > 0.2 and physCoords[2] > 0.2 ) and
//    ((physCoords[0] + physSize < 0.4) and (physCoords[1] + physSize < 0.4) and (physCoords[2] + physSize < 0.4))){
//      isInside = false;
//    }

  return isInside;
}


/**
 * main()
 */
int main(int argc, char * argv[]){
  typedef unsigned int DENDRITE_UINT;
  PetscInitialize(&argc, &argv, NULL, NULL);
  _InitializeHcurve(DIM);

  int eleOrder = 1;
  int ndof = 1;
  m_uiMaxDepth = 10;
  int level = 2;

  MPI_Comm comm = MPI_COMM_WORLD;

  constexpr bool printTreeOn = false;  // Can print the contents of the tree vectors.
  unsigned int extents[] = {1,2,1};
  std::array<unsigned int,DIM> a;
  for (int d = 0; d < DIM; ++d)
    a[d] = extents[d];

  using DTree = ot::DistTree<unsigned int, DIM>;
  DTree distTree = DTree::constructSubdomainDistTree( level,DomainDecider,
                                                      comm);
  ot::DA<DIM> *octDA = new ot::DA<DIM>(distTree, comm, eleOrder);
  /// printMaxCoords(*octDA);

  size_t oldTreeSize = 0;
  size_t refinedTreeSize = 0;

  // Access the original tree as a list of tree nodes.
  {
    const std::vector<ot::TreeNode<unsigned int, DIM>> &treePart = distTree.getTreePartFiltered();
    oldTreeSize = treePart.size();

    /// if (printTreeOn)
    ///   printTree(treePart, level+1);
  }

  std::cout << "Old Tree \n";
  std::cout << "Num elements: " << oldTreeSize << "\n";
  getBoundaryElements(octDA,eleOrder, "Old");

  std::vector<ot::OCT_FLAGS::Refine> refineFlags(oldTreeSize, ot::OCT_FLAGS::Refine::OCT_REFINE);

  // distRemeshSubdomain()
  ot::DistTree<unsigned int, DIM> newDistTree, surrDistTree;
  ot::DistTree<unsigned int, DIM>::distRemeshSubdomain(distTree, refineFlags, newDistTree, surrDistTree, 0.3);

  // =======================================================================
  // Currently the boundary is detected from applying the carving function.
  // The boundary is not detected based solely on the existence of TreeNodes
  // in the tree. So DA() constructor needs DistTree, i.e.,
  // newDistTree instead of newTree. Otherwise, DA() constructor uses
  // default boundary definition of the unit cube.
  // =======================================================================
  ot::DA<DIM> * newDA = new ot::DA<DIM>(newDistTree, comm,eleOrder,100,0.3); //DistTree overload
  /// printMaxCoords(*newDA);

  // Access the refined tree as a list of tree nodes.
  {
    const std::vector<ot::TreeNode<DENDRITE_UINT, DIM>> &newTree = newDistTree.getTreePartFiltered();
    /// const std::vector<ot::TreeNode<DENDRITE_UINT, DIM>> &surrTree = surrDistTree.getTreePartFiltered();

    refinedTreeSize = newTree.size();

    /// if (printTreeOn)
    ///   printTree(newTree, level+1);
  }


  std::swap(octDA, newDA);

  std::cout << "New tree \n";
  std::cout << "Num elements: " << refinedTreeSize << "\n";
  getBoundaryElements(octDA, eleOrder, "Refined");

  delete octDA;
  delete newDA;

  _DestroyHcurve();
  PetscFinalize();
}
