#include <iostream>
#include <oda.h>
#include <point.h>
#include <sfcTreeLoop_matvec_io.h>

#include <octUtils.h>


constexpr unsigned int DIM = 2;
constexpr unsigned int nchild = 1 << DIM;

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



void printMaxCoords(ot::DA<DIM> & octDA){
  const size_t sz = octDA.getTotalNodalSz();
  auto partFront = octDA.getTreePartFront();
  auto partBack = octDA.getTreePartBack();
  const auto tnCoords = octDA.getTNCoords();
  {
    std::vector<double> maxCoords(4, 0.0);
    ot::MatvecBaseCoords<DIM> loop(sz, 1, false, 0, tnCoords, *partFront, *partBack);
    while (!loop.isFinished()) {
      if (loop.isPre() && loop.subtreeInfo().isLeaf()) {
        const double *nodeCoordsFlat = loop.subtreeInfo().getNodeCoords();
        for (int i = 0; i < nchild; i++) {
          for (int d = 0; d < DIM; d++) {
            if (maxCoords[d] < nodeCoordsFlat[i * DIM + d]) {
              maxCoords[d] = nodeCoordsFlat[i * DIM + d];
            }
          }
        }
        loop.next();
      } else {
        loop.step();
      }
    }
    std::cout << "maxCoords == " << maxCoords[0] << " " << maxCoords[1] << " " << maxCoords[2] << "\n";
  }
}



int main(int argc, char * argv[]){
  typedef unsigned int DENDRITE_UINT;
  PetscInitialize(&argc, &argv, NULL, NULL);
  _InitializeHcurve(DIM);
  int eleOrder = 1;
  int ndof = 1;
  m_uiMaxDepth = 10;
  int level = 3;
  std::vector<ot::TreeNode<unsigned int, DIM>> treePart;
  ot::DA<DIM> *octDA = new ot::DA<DIM>();

  unsigned int extents[] = {1,2,1};
  std::array<unsigned int,DIM> a;
  for (int d = 0; d < DIM; ++d)
    a[d] = extents[d];

  ot::constructRegularSubdomainDA<DIM>(*octDA,treePart,level,a,eleOrder,MPI_COMM_WORLD);
  printTree(treePart, level+1);
  printMaxCoords(*octDA);

  std::vector<ot::OCT_FLAGS::Refine> refineFlags(treePart.size(),ot::OCT_FLAGS::Refine::OCT_REFINE);

  std::vector<ot::TreeNode<DENDRITE_UINT, DIM>>  newTree;
  std::vector<ot::TreeNode<DENDRITE_UINT, DIM>>  surrTree;
  ot::SFC_Tree<DENDRITE_UINT , DIM>::distRemesh(treePart, refineFlags, newTree, surrTree, 0.3, MPI_COMM_WORLD);

  std::cout << "\n-------\n";

  printTree(newTree, level+1);
        ///DA(DistTree<C,dim> &inDistTree, int stratum, MPI_Comm comm, unsigned int order, size_t grainSz = 100, double sfc_tol = 0.3);
  ot::DA<DIM> * newDA =new ot::DA<DIM>(newTree,MPI_COMM_WORLD,eleOrder,100,0.3);
  printMaxCoords(*newDA);

  /// How to delete octDa and swap with newDA. Below is not working.
  std::swap(octDA, newDA);
  delete newDA;
  PetscFinalize();
}
