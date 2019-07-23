
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

  const C s2 = (1u << m_uiMaxDepth) / (1u << 2);
  const unsigned int nodeLev = 1;

  std::vector<ot::TreeNode<C,dim>> elevenCoords;
  {
    using Coord = std::array<C,dim>;
    elevenCoords.emplace_back(1, Coord{0,    0}, nodeLev);
    elevenCoords.emplace_back(1, Coord{2*s2, 0}, nodeLev);
    elevenCoords.emplace_back(1, Coord{4*s2, 0}, nodeLev);

    elevenCoords.emplace_back(1, Coord{3*s2, s2}, nodeLev+1);

    elevenCoords.emplace_back(1, Coord{0,    2*s2}, nodeLev);
    elevenCoords.emplace_back(1, Coord{2*s2, 2*s2}, nodeLev);
    elevenCoords.emplace_back(1, Coord{4*s2, 2*s2}, nodeLev);

    elevenCoords.emplace_back(1, Coord{s2, 3*s2}, nodeLev+1);

    elevenCoords.emplace_back(1, Coord{0,    4*s2}, nodeLev);
    elevenCoords.emplace_back(1, Coord{2*s2, 4*s2}, nodeLev);
    elevenCoords.emplace_back(1, Coord{4*s2, 4*s2}, nodeLev);
  }

  std::vector<T> elevenVals;
  elevenVals.emplace_back(0);
  elevenVals.emplace_back(1);
  elevenVals.emplace_back(2);

  elevenVals.emplace_back(3);

  elevenVals.emplace_back(4);
  elevenVals.emplace_back(5);
  elevenVals.emplace_back(6);

  elevenVals.emplace_back(7);

  elevenVals.emplace_back(8);
  elevenVals.emplace_back(9);
  elevenVals.emplace_back(10);



  int lastChild = 3;
  ElementLoop<C, dim, T> loop(
      11,
      &(*elevenCoords.begin()),
      1,
      ot::TreeNode<C,dim>().getChildMorton(0),
      ot::TreeNode<C,dim>().getChildMorton(lastChild));
  loop.initialize(&(*elevenVals.begin()));

  int eleCounter = 0;

  while (!loop.isExhausted())
  {
    std::cerr << "Not exhausted, eleCounter == "
              << eleCounter++ << " "
              << loop.m_curTreeAddr << ".\n";
    loop.next();
  }
  std::cerr << "Finally exhausted.\n";

  loop.finalize(&(*elevenVals.begin()));

  _DestroyHcurve();

  return 0;
}
