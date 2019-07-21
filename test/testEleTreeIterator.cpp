
#include "eleTreeIterator.h"
#include "hcurvedata.h"

#include <stdio.h>

int main(int argc, char *argv[])
{
  constexpr unsigned int dim = 2;
  using C = unsigned int;
  using T = float;

  m_uiMaxDepth = 3;

  _InitializeHcurve(dim);

  fprintf(stdout, "Hello world!\n");

  const C s1 = (1u << m_uiMaxDepth) / (1u << 1);
  const unsigned int nodeLev = 1;

  std::vector<ot::TreeNode<C,dim>> nineCoords;
  {
    using Coord = std::array<C,dim>;
    nineCoords.emplace_back(1, Coord{0,    0}, nodeLev);
    nineCoords.emplace_back(1, Coord{s1,   0}, nodeLev);
    nineCoords.emplace_back(1, Coord{2*s1, 0}, nodeLev);

    nineCoords.emplace_back(1, Coord{0,    s1}, nodeLev);
    nineCoords.emplace_back(1, Coord{s1,   s1}, nodeLev);
    nineCoords.emplace_back(1, Coord{2*s1, s1}, nodeLev);

    nineCoords.emplace_back(1, Coord{0,    2*s1}, nodeLev);
    nineCoords.emplace_back(1, Coord{s1,   2*s1}, nodeLev);
    nineCoords.emplace_back(1, Coord{2*s1, 2*s1}, nodeLev);
  }

  std::vector<T> nineVals;
  nineVals.emplace_back(0);
  nineVals.emplace_back(1);
  nineVals.emplace_back(2);

  nineVals.emplace_back(3);
  nineVals.emplace_back(4);
  nineVals.emplace_back(5);

  nineVals.emplace_back(6);
  nineVals.emplace_back(7);
  nineVals.emplace_back(8);


  const bool assumeHilbert = false;
  int lastChild = (assumeHilbert ? 2 : 3);
  ElementLoop<C, dim, T> loop(
      9,
      &(*nineCoords.begin()),
      1,
      ot::TreeNode<C,dim>().getChildMorton(0),
      ot::TreeNode<C,dim>().getChildMorton(lastChild));
  loop.initialize(&(*nineVals.begin()));

  int eleCounter = 0;

  while (!loop.isExhausted())
  {
    std::cerr << "Not exhausted, eleCounter == "
              << eleCounter++ << " "
              << loop.m_curTreeAddr << ".\n";
    loop.next();
  }
  std::cerr << "Finally exhausted.\n";

  loop.finalize(&(*nineVals.begin()));

  _DestroyHcurve();

  return 0;
}
