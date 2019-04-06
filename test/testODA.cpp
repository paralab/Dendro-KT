/*
 * testCountODA.cpp
 *   Test using the ODA class
 *
 * Masado Ishii  --  UofU SoC, 2019-04-05
 */

#include "testAdaptiveExamples.h"   // Also creates alias like using T = unsigned int;
#include "treeNode.h"
#include "hcurvedata.h"
#include "mathUtils.h"

#include "oda.h"
#include "feMatrix.h"
#include "feVector.h"

#include "mpi.h"

#include <iostream>
#include <vector>


// ---------------------------------------------------------------------
template<typename X>
void distPrune(std::vector<X> &list, MPI_Comm comm);

template<unsigned int dim, unsigned int endL, unsigned int order>
void testMatvec(MPI_Comm comm);

template <typename T, unsigned int dim>
class myConcreteFeMatrix;

// ---------------------------------------------------------------------


//
// main()
//
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int nProc, rProc;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  constexpr unsigned int dim = 2;
  const unsigned int endL = 3;
  const unsigned int order = 2;

  testMatvec<dim,endL,order>(comm);

  MPI_Finalize();

  return 0;
}
// end main()

// ====================================================================

//
// myConcreteFeMatrix
//
template <typename T, unsigned int dim>
class myConcreteFeMatrix : public feMatrix<T, dim>
{
  public:
    static constexpr unsigned int order = 1;   // Only support a static order for now.  //TODO add order paramter to elementalMatVec()
    using feMatrix<T,dim>::feMatrix;
    virtual void elementalMatVec(const VECType *in, VECType *out, double *coords, double scale) override;
};

template <typename T, unsigned int dim>
void myConcreteFeMatrix<T,dim>::elementalMatVec(const VECType *in, VECType *out, double *coords, double scale)
{
  // Dummy identity.
  const unsigned int nPe = intPow(order + 1, dim);
  for (int ii = 0; ii < nPe; ii++)
    for (int d = 0; d < dim; d++)
      out[dim*ii + d] = in[dim*ii + d];
}


//
// testMatvec()
//
template<unsigned int dim, unsigned int endL, unsigned int order>
void testMatvec(MPI_Comm comm)
{
  _InitializeHcurve(dim);

  double sfc_tol = 0.3;

  // Example tree. Already known to be balanced, otherwise call distTreeBalancing().
  std::vector<ot::TreeNode<T, dim>> tree;
  // Example1<dim>::fill_tree(endL, tree);     // Refined core.
  // Example2<dim>::fill_tree(endL, tree);     // Uniform grid.
  Example3<dim>::fill_tree(endL, tree);      // Refined fringe.
  distPrune(tree, comm);
  ot::SFC_Tree<T,dim>::distTreeSort(tree, sfc_tol, comm);

  // Make a DA (distributed array), which distributes and coordinates element nodes across procs.
  ot::DA<dim> oda(&(*tree.cbegin()), tree.size(), comm, order);

  unsigned int dof = 1;

  // Make data vectors that are aligned with oda.
  std::vector<double> inVec, outVec;
  oda.template createVector<double>(inVec, false, false, dof);
  oda.template createVector<double>(outVec, false, false, dof);

  for (unsigned int ii = 0; ii < inVec.size(); ii++)
    inVec[ii] = ii % 5;

  // Define some (data vector and) elemental matrix operator.
  // The matrix operator is defined above in myConcreteFeMatrix.
  //TODO What should be the template parameter T for feMatrix?
  myConcreteFeMatrix<std::nullptr_t, dim> feMtx(&oda, dof);

  // Perform the matvec.
  feMtx.matVec(&(*inVec.cbegin()), &(*outVec.begin()));

  // Show results.
  printf("Input\n");
  unsigned int ii = 0;
  for (auto &&x : inVec)
  {
    printf("\t%6.3f", x);
    if (!((++ii) % 15))
      printf("\n");
  }
  printf("\n\nOutput\n");
  ii = 0;
  for (auto &&x : outVec)
  {
    printf("\t%6.3f", x);
    if (!((++ii) % 15))
      printf("\n");
  }
  printf("\n");

  oda.template destroyVector<double>(inVec);
  oda.template destroyVector<double>(outVec);

  _DestroyHcurve();
}


//
// distPrune()
//
template<typename X>
void distPrune(std::vector<X> &list, MPI_Comm comm)
{
  int nProc, rProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  const int listSize = list.size();
  const int baseSeg = listSize / nProc;
  const int remainder = listSize - baseSeg * nProc;
  const int myStart = rProc * baseSeg + (rProc < remainder ? rProc : remainder);
  const int mySeg = baseSeg + (rProc < remainder ? 1 : 0);

  /// fprintf(stderr, "[%d] listSize==%d, myStart==%d, mySeg==%d\n", rProc, listSize, myStart, mySeg);

  list.erase(list.begin(), list.begin() + myStart);
  list.resize(mySeg);
}

