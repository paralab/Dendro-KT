
#include "sendUtilization.h"


namespace bench
{
  template <unsigned int dim>
  long long unsigned computeSendRequired(ot::DA<dim> *octDA, const ot::DistTree<unsigned, dim> &distTree)
  {
    const unsigned ndof = 1;

    double *ghostedArray;
    octDA->createVector(ghostedArray, false, true, ndof);
    std::fill(ghostedArray, ghostedArray + octDA->getTotalNodalSz(), 0.0);

    const size_t sz = octDA->getTotalNodalSz();
    auto partFront = octDA->getTreePartFront();
    auto partBack = octDA->getTreePartBack();
    const auto tnCoords = octDA->getTNCoords();
    const unsigned int nPe = octDA->getNumNodesPerElement();

    std::vector<double> leafBuffer(nPe, 0.0);

    const bool visitEmpty = false;
    const unsigned padlevel = 0;
    ot::MatvecBaseOut<dim, double, true> treeloop(sz,
                                                  ndof,
                                                  octDA->getElementOrder(),
                                                  visitEmpty,
                                                  padlevel,
                                                  tnCoords,
                                                  &(*distTree.getTreePartFiltered().cbegin()),
                                                  distTree.getTreePartFiltered().size(),
                                                  *partFront,
                                                  *partBack);

    while (!treeloop.isFinished())
    {
      if (treeloop.isPre() && treeloop.subtreeInfo().isLeaf())
      {
        for(int nIdx = 0; nIdx < nPe; nIdx++)
          leafBuffer[nIdx] = 1.0;
        treeloop.subtreeInfo().overwriteNodeValsOut(&leafBuffer[0]);
        treeloop.next();
      }
      else
        treeloop.step();
    }

    treeloop.finalize(ghostedArray);

    for (size_t ii = octDA->getLocalNodeBegin(); ii < octDA->getLocalNodeBegin() + octDA->getLocalNodalSz(); ++ii)
      ghostedArray[ii] = 0.0;

    octDA->writeToGhostsBegin(ghostedArray, ndof);
    octDA->writeToGhostsEnd(ghostedArray, ndof);

    long long unsigned countNonzero = 0;

    for (size_t ii = octDA->getLocalNodeBegin(); ii < octDA->getLocalNodeBegin() + octDA->getLocalNodalSz(); ++ii)
      if (ghostedArray[ii] > 0.0)
        countNonzero++;

    octDA->destroyVector(ghostedArray);

    return countNonzero;
  }



  template long long unsigned computeSendRequired<2>(ot::DA<2> *octDA, const ot::DistTree<unsigned, 2> &distTree);
  template long long unsigned computeSendRequired<3>(ot::DA<3> *octDA, const ot::DistTree<unsigned, 3> &distTree);
  template long long unsigned computeSendRequired<4>(ot::DA<4> *octDA, const ot::DistTree<unsigned, 4> &distTree);

}

