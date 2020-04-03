#include "hcurvedata.h"
#include "octUtils.h"
#include "intergridTransfer.h"

#include <stdio.h>
#include <iostream>



template <int dim>
bool testNull();


/**
 * main()
 */
int main(int argc, char *argv[])
{
  constexpr unsigned int dim = 2;

  MPI_Init(&argc, &argv);
  _InitializeHcurve(dim);

  bool success = testNull<dim>();

  _DestroyHcurve();
  MPI_Finalize();

  return !success;
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
