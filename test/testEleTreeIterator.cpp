
#include "eleTreeIterator.h"
#include "hcurvedata.h"

#include <stdio.h>

int main(int argc, char *argv[])
{
  constexpr unsigned int dim = 3;
  using C = unsigned int;
  using T = float;

  _InitializeHcurve(dim);

  fprintf(stdout, "Hello world!\n");

  ot::TreeNode<C,dim> someNodeCoords[80];
  T someNodeVals[80];
  EleTreeIterator<C, dim, T> it(80, someNodeCoords, someNodeVals, 1, ot::TreeNode<C,dim>(), ot::TreeNode<C,dim>());

  _DestroyHcurve();

  return 0;
}
