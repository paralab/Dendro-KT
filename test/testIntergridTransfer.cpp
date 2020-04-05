#include "hcurvedata.h"
#include "octUtils.h"
#include "intergridTransfer.h"

#include "distTree.h"
#include "oda.h"

#include <stdio.h>
#include <iostream>



template <int dim>
bool testNull();

template <int dim>
bool testMultiDA();


/**
 * main()
 */
int main(int argc, char *argv[])
{
  constexpr unsigned int dim = 2;

  MPI_Init(&argc, &argv);
  _InitializeHcurve(dim);

  /// bool success = testNull<dim>();
  bool success = testMultiDA<dim>();

  _DestroyHcurve();
  MPI_Finalize();

  return !success;
}


template <int dim>
bool testMultiDA()
{
  using C = unsigned int;
  using DofT = float;
  const unsigned int eleOrder = 1;
  const unsigned int eLev = 2;
  unsigned int ndofs = 1;

  MPI_Comm comm = MPI_COMM_WORLD;

  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  using TN = ot::TreeNode<C, dim>;

  // Complete regular grid.
  std::vector<ot::TreeNode<C, dim>> tnlist;
  ot::createRegularOctree(tnlist, eLev, comm);

  // Give DistTree ownership of the octree.
  ot::DistTree<C,dim> dtree(tnlist);

  // Create grid hierarchy.
  ot::DistTree<C, dim> surrogateDTree = dtree.generateGridHierarchyDown(2, 0.1, comm);

  std::cerr << "surrogateDTree.getFilteredTreePartSz(1)==" << surrogateDTree.getFilteredTreePartSz(1) << "\n"
            << "surrogateDTree.getFilteredTreePartSz(1)==" << surrogateDTree.getFilteredTreePartSz(1) << "\n";

  // Create DA for all levels.
  std::vector<ot::DA<dim>> multiDA, surrogateMultiDA;
  ot::DA<dim>::multiLevelDA(multiDA, dtree, comm, eleOrder);
  ot::DA<dim>::multiLevelDA(surrogateMultiDA, surrogateDTree, comm, eleOrder);

  std::vector<DofT> fineVec, coarseVec;
  multiDA[0].createVector(fineVec, false, false, ndofs);
  surrogateMultiDA[1].createVector(coarseVec, false, false, ndofs);
  std::fill(fineVec.begin(), fineVec.end(), 1.0);
  std::fill(coarseVec.begin(), coarseVec.end(), 0.0);

  std::cerr << "surrogateMultiDA[1].getLocalNodalSz()==" << surrogateMultiDA[1].getLocalNodalSz() << "\n"
            << "surrogateMultiDA[1].getLocalNodalSz()==" << surrogateMultiDA[1].getLocalNodalSz() << "\n";

  // TODO read and write ghost


  fem::MeshFreeInputContext<DofT, TN> igtIn{
      fineVec.data(),
      multiDA[0].getTNCoords(),
      multiDA[0].getLocalNodalSz(),
      *multiDA[0].getTreePartFront(),
      *multiDA[0].getTreePartBack()
  };

  fem::MeshFreeOutputContext<DofT, TN> igtOut{
      coarseVec.data(),
      surrogateMultiDA[1].getTNCoords(),
      surrogateMultiDA[1].getLocalNodalSz(),
      *surrogateMultiDA[1].getTreePartFront(),
      *surrogateMultiDA[1].getTreePartBack()
  };

  const RefElement * refel = multiDA[0].getReferenceElement();


  std::cerr << "---- BEFORE ----\n";
  std::cerr << "In vector\n";
  ot::printNodes(igtIn.coords, igtIn.coords + igtIn.sz, fineVec.data(), eleOrder, std::cerr) << "\n";

  std::cerr << "Out vector\n";
  ot::printNodes(igtOut.coords, igtOut.coords + igtOut.sz, coarseVec.data(), eleOrder, std::cerr) << "\n";


  fem::intergridTransfer(igtIn, igtOut, ndofs, refel);



  std::cerr << "\n\n";
  std::cerr << "---- AFTER ----\n";
  std::cerr << "In vector\n";
  ot::printNodes(igtIn.coords, igtIn.coords + igtIn.sz, fineVec.data(), eleOrder, std::cerr) << "\n";

  std::cerr << "Out vector\n";
  ot::printNodes(igtOut.coords, igtOut.coords + igtOut.sz, coarseVec.data(), eleOrder, std::cerr) << "\n";


  return true;
}


template <int dim>
bool testNull()
{
  using C = unsigned int;
  using T = float;
  const unsigned int eleOrder = 1;

  using TN = ot::TreeNode<C, dim>;

  // empty, just for compiling
  std::vector<T> vecIn, vecOut;
  std::vector<ot::TreeNode<C, dim>> coordsIn, coordsOut;
  ot::TreeNode<C, dim> frontIn, frontOut, backIn, backOut;

  unsigned int sz = 0;
  unsigned int ndofs = 1;

  RefElement refel(dim, eleOrder);


  fem::intergridTransfer(
      fem::MeshFreeInputContext<T, TN>{vecIn.data(), coordsIn.data(), sz, frontIn, backIn},
      fem::MeshFreeOutputContext<T, TN>{vecOut.data(), coordsOut.data(), sz, frontOut, backOut},
      ndofs,
      &refel);

  return true;
}
