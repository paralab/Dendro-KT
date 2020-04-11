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

  const bool reportSize = false;
  const bool reportEmpty = true;
  const bool reportContext = false;

  MPI_Comm comm = MPI_COMM_WORLD;

  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  std::string rankPrefix;
  { std::stringstream ss;
    ss << "[" << std::setfill('0') << std::setw(2) << rProc << "] ";
    rankPrefix = ss.str();
  }

  using TN = ot::TreeNode<C, dim>;

  // Complete regular grid.
  std::vector<ot::TreeNode<C, dim>> tnlist;
  ot::createRegularOctree(tnlist, eLev, comm);

  // Give DistTree ownership of the octree.
  ot::DistTree<C,dim> dtree(tnlist);

  /// for (int turn = 0; turn < nProc; ++turn)
  /// {
  ///   if (turn == rProc)
  ///     std::cerr << "-----------------------------------------------------------------------------" << rankPrefix << "Here!\n";
  ///   MPI_Barrier(comm);
  /// }

  // Create grid hierarchy.
  ot::DistTree<C, dim> surrogateDTree = dtree.generateGridHierarchyDown(2, 0.1, comm);

  if (reportSize)
  {
    fprintf(stderr, "%s \t SIZE \t %s%lu\n", rankPrefix.c_str(), "dtree.getFilteredTreePartSz(1)==", dtree.getFilteredTreePartSz(1));
    fprintf(stderr, "%s \t SIZE \t %s%lu\n", rankPrefix.c_str(), "surrogateDTree.getFilteredTreePartSz(1)==", surrogateDTree.getFilteredTreePartSz(1));
  }

  if (reportEmpty)
  {
    if (dtree.getFilteredTreePartSz(0) == 0)
      fprintf(stderr, "%s \t EMPTY \t Fine tree is empty!\n", rankPrefix.c_str());
    if (dtree.getFilteredTreePartSz(1) == 0)
      fprintf(stderr, "%s \t EMPTY \t Coarse tree is empty!\n", rankPrefix.c_str());
    if (surrogateDTree.getFilteredTreePartSz(1) == 0)
      fprintf(stderr, "%s \t EMPTY \t Surrogate tree is empty!\n", rankPrefix.c_str());
  }

  // Create DA for all levels.
  std::vector<ot::DA<dim>> multiDA, surrogateMultiDA;
  ot::DA<dim>::multiLevelDA(multiDA, dtree, comm, eleOrder);
  ot::DA<dim>::multiLevelDA(surrogateMultiDA, surrogateDTree, comm, eleOrder);

  std::vector<DofT> fineVec, coarseVec, surrogateVec;
  multiDA[0].createVector(fineVec, false, false, ndofs);
  multiDA[1].createVector(coarseVec, false, false, ndofs);
  surrogateMultiDA[1].createVector(surrogateVec, false, false, ndofs);
  std::fill(fineVec.begin(), fineVec.end(), 1.0);
  std::fill(coarseVec.begin(), coarseVec.end(), -1.0);
  std::fill(surrogateVec.begin(), surrogateVec.end(), 0.0);

  if (reportSize)
  {
    fprintf(stderr, "%s \t SIZE \t %s%lu\n", rankPrefix.c_str(), "multiDA[1].getLocalNodalSz()==", multiDA[1].getLocalNodalSz());
    fprintf(stderr, "%s \t SIZE \t %s%lu\n", rankPrefix.c_str(), "surrogateMultiDA[1].getLocalNodalSz()==", surrogateMultiDA[1].getLocalNodalSz());
  }

  if (reportEmpty)
  {
    if (multiDA[0].getLocalNodalSz() == 0)
      fprintf(stderr, "%s \t EMPTY \t Fine ODA is empty!\n", rankPrefix.c_str());
    if (multiDA[1].getLocalNodalSz() == 0)
      fprintf(stderr, "%s \t EMPTY \t Coarse ODA is empty!\n", rankPrefix.c_str());
    if (surrogateMultiDA[1].getLocalNodalSz() == 0)
      fprintf(stderr, "%s \t EMPTY \t Surrogate ODA is empty!\n", rankPrefix.c_str());
  }

  fem::MeshFreeInputContext<DofT, TN> igtIn{
      fineVec.data(),
      multiDA[0].getTNCoords(),
      multiDA[0].getLocalNodalSz(),
      *multiDA[0].getTreePartFront(),
      *multiDA[0].getTreePartBack()
  };

  fem::MeshFreeOutputContext<DofT, TN> igtOut{
      surrogateVec.data(),
      surrogateMultiDA[1].getTNCoords(),
      surrogateMultiDA[1].getLocalNodalSz(),
      *surrogateMultiDA[1].getTreePartFront(),
      *surrogateMultiDA[1].getTreePartBack()
  };

  const RefElement * refel = multiDA[0].getReferenceElement();


  /// std::cerr << rankPrefix
  ///           << "---- BEFORE ----\n";
  /// std::cerr << rankPrefix
  ///           << "In vector\n";
  /// if (dim == 2)
  ///   ot::printNodes(igtIn.coords, igtIn.coords + igtIn.sz, fineVec.data(), eleOrder, std::cerr) << "\n";
  /// else
  ///   std::cerr << rankPrefix
  ///             << "Can't print high-dimensional grid.\n";

  /// std::cerr << rankPrefix
  ///           << "Out vector (surrogate)\n";
  /// if (dim == 2)
  ///   ot::printNodes(igtOut.coords, igtOut.coords + igtOut.sz, surrogateVec.data(), eleOrder, std::cerr) << "\n";
  /// else
  ///   std::cerr << rankPrefix
  ///             << "Can't print high-dimensional grid.\n";

  /// std::cerr << rankPrefix
  ///           << "Out vector (coarse)\n";
  /// if (dim == 2)
  ///   ot::printNodes(multiDA[1].getTNCoords(), multiDA[1].getTNCoords() + multiDA[1].getLocalNodalSz(), coarseVec.data(), eleOrder, std::cerr) << "\n";
  /// else
  ///   std::cerr << rankPrefix
  ///             << "Can't print high-dimensional grid.\n";

  if (reportContext)
  {
    std::string igtInStr, igtOutStr;
    { std::stringstream ss;
      ss << " Front: ";
      ot::printtn(igtIn.partFront, 3, ss);
      ss << " Back: ";
      ot::printtn(igtIn.partBack, 3, ss);
      ss << "  Sz==" << igtIn.sz;
      for (int i = 0; i < igtIn.sz; ++i)
      {
        ss << "  ";
        ot::printtn(igtIn.coords[i], 3, ss);
      }
      igtInStr = ss.str();
    }
    { std::stringstream ss;
      ss << " Front: ";
      ot::printtn(igtOut.partFront, 3, ss);
      ss << " Back: ";
      ot::printtn(igtOut.partBack, 3, ss);
      ss << "  Sz==" << igtOut.sz;
      for (int i = 0; i < igtOut.sz; ++i)
      {
        ss << "  ";
        ot::printtn(igtOut.coords[i], 3, ss);
      }
      igtOutStr = ss.str();
    }

    fprintf(stderr, "%s \t IN_INFO \t %s\n", rankPrefix.c_str(), igtInStr.c_str());
    fprintf(stderr, "%s \t OUT_INFO \t %s\n", rankPrefix.c_str(), igtOutStr.c_str());
  }

  fprintf(stderr, "%s \t PHASE \t locIntergridTransfer\n", rankPrefix.c_str());
  fem::locIntergridTransfer(igtIn, igtOut, ndofs, refel);


  /// std::cerr << "\n\n";
  /// std::cerr << rankPrefix
  ///           << "---- AFTER ----\n";

//   std::cerr << "In vector\n";
//   if (dim == 2)
//     ot::printNodes(igtIn.coords, igtIn.coords + igtIn.sz, fineVec.data(), eleOrder, std::cerr) << "\n";
//   else
//     std::cerr << "Can't print high-dimensional grid.\n";

  /// std::cerr << rankPrefix
  ///           << "Out vector (surrogate)\n";
  /// if (dim == 2)
  ///   ot::printNodes(igtOut.coords, igtOut.coords + igtOut.sz, surrogateVec.data(), eleOrder, std::cerr) << "\n";
  /// else
  ///   std::cerr << rankPrefix
  ///             << "Can't print high-dimensional grid.\n";


  fprintf(stderr, "%s \t PHASE \t Shift nodes\n", rankPrefix.c_str());
  ot::distShiftNodes(surrogateMultiDA[1], surrogateVec.data(), multiDA[1], coarseVec.data());


  /// std::cerr << rankPrefix
  ///           << "Out vector (coarse)\n";
  /// if (dim == 2)
  ///   ot::printNodes(multiDA[1].getTNCoords(), multiDA[1].getTNCoords() + multiDA[1].getLocalNodalSz(), coarseVec.data(), eleOrder, std::cerr) << "\n";
  /// else
  ///   std::cerr << rankPrefix
  ///             << "Can't print high-dimensional grid.\n";


  std::vector<TN> coarseTNVerify;
  multiDA[1].createVector(coarseTNVerify, false, false, ndofs);
  ot::distShiftNodes(surrogateMultiDA[1], surrogateMultiDA[1].getTNCoords(), multiDA[1], coarseTNVerify.data());
  bool verifyPartition = true;
  int misses = 0;
  for (int i = 0; i < multiDA[1].getLocalNodalSz(); ++i)
    if (coarseTNVerify[i] != multiDA[1].getTNCoords()[i])
    {
      verifyPartition = false;
      misses++;
    }

  if (verifyPartition)
    fprintf(stderr, "%s \t RESULT \t Partition verified.\n", rankPrefix.c_str());
  else
    fprintf(stderr, "%s \t RESULT \t %s.\n", rankPrefix.c_str(), "&&&&&&&&&&&&&&&&&&& Partition failed &&&&&&&&&&&&&&&&&&&&&&\n");


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


  fem::locIntergridTransfer(
      fem::MeshFreeInputContext<T, TN>{vecIn.data(), coordsIn.data(), sz, frontIn, backIn},
      fem::MeshFreeOutputContext<T, TN>{vecOut.data(), coordsOut.data(), sz, frontOut, backOut},
      ndofs,
      &refel);

  return true;
}
