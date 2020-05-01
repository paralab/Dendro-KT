
#include "dendro.h"
#include "treeNode.h"
#include "nsort.h"
#include "poissonMat.h"
#include "hcurvedata.h"



template <typename UICoordT, typename FCoordT, unsigned int dim>
void convertToFlatCoords(const ot::TreeNode<UICoordT, dim> &element,
                         unsigned int eleOrder, unsigned int nPe,
                         const ot::TreeNode<UICoordT, dim> *tncoords,
                         FCoordT *fcoords);



int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  constexpr unsigned int dim = 2;
  _InitializeHcurve(dim);

  const unsigned int eleOrder = 2;
  const unsigned int nPe=intPow(eleOrder+1, dim);



  using TNT = unsigned int;
  using TN = ot::TreeNode<TNT, dim>;

  TN treeRoot;
  std::vector<TN> treeNodes;
  /// treeNodes.push_back(treeRoot);
  for (int c = 0; c < (1u << dim); c++)
    treeNodes.push_back(treeRoot.getChildMorton(c));
  ot::DA<dim> daRoot(treeNodes, comm, eleOrder, 1, 0);
  // treeNodes is emptied.

  std::cout << "(1u<<dim) == " << (1u<<dim) << "\n";
  std::cout << "daRoot.getLocalNodalSz() == " << daRoot.getLocalNodalSz() << "\n";
  const size_t numNodes = daRoot.getLocalNodalSz();


  const RefElement refel(dim, eleOrder);
  const RefElement &daRefel = *daRoot.getReferenceElement();
  std::cout << "refel.getOrder(),  getDim() == " << refel.getOrder() << "  " << refel.getDim() << "\n";
  std::cout << "daRefel.getOrder(),  getDim() == " << daRefel.getOrder() << "  " << daRefel.getDim() << "\n";
  std::cout << "Reference element stored data:\n";
  std::cout << "----DgT1d_hadm2[]\n";
  for (int na = 0; na <= eleOrder; na++)
  {
    for (int nb = 0; nb <= eleOrder; nb++)
    {
      std::cout << "  " << std::setw(12) << std::setprecision(3)
                << refel.getDgT1d_hadm2()[na*(eleOrder+1) + nb];
    }
    std::cout << "\n";
    for (int nb = 0; nb <= eleOrder; nb++)
    {
      std::cout << " _" << std::setw(12) << std::setprecision(3)
                << daRefel.getDgT1d_hadm2()[na*(eleOrder+1) + nb];
    }
    std::cout << "\n";
  }
  std::cout << "----QT1d_hadm2[]\n";
  for (int na = 0; na <= eleOrder; na++)
  {
    for (int nb = 0; nb <= eleOrder; nb++)
    {
      std::cout << "  " << std::setw(12) << std::setprecision(3)
                << refel.getQT1d_hadm2()[na*(eleOrder+1) + nb];
    }
    std::cout << "\n";
    for (int nb = 0; nb <= eleOrder; nb++)
    {
      std::cout << " _" << std::setw(12) << std::setprecision(3)
                << daRefel.getQT1d_hadm2()[na*(eleOrder+1) + nb];
    }
    std::cout << "\n";
  }





  PoissonEq::PoissonMat<dim> pmat(&daRoot, 1);

  const TN *coords = daRoot.getTNCoords();
  std::vector<double> flatcoords(dim*nPe);
  convertToFlatCoords(*daRoot.getTreePartFront(), eleOrder, nPe, coords, flatcoords.data());


  /// // Print coords.
  /// for (int nIdx = 0; nIdx < nPe; nIdx++)
  /// {
  ///   ot::printtn(coords[nIdx], 2, std::cout);
  ///   std::cout << "    \t(";
  ///   for (int d = 0; d < dim; d++)
  ///     std::cout << flatcoords[nIdx * dim + d] << ",\t";
  ///   std::cout << ")\n";
  /// }


  std::vector<double> npeBufferIn(nPe, 0.0);
  std::vector<double> npeBufferOut(nPe, 0.0);

  // Two ways to get the diagonal.
  //   diag1 : Use elementalMatVec() with each standard basis vector
  //           and get the ith component of the result.
  //
  //   diag2 : Use elementalSetDiag() to assemble diagonal directly.
  //
  // They should be the same in principle.
  //
  std::vector<double> diag1(nPe, 0.0);
  std::vector<double> diag2(nPe, 0.0);


  constexpr bool noVisitEmpty = false;
  ot::MatvecBaseOut<dim, double, true> treeLoopOut(numNodes, 1, eleOrder, noVisitEmpty, 0, coords, *daRoot.getTreePartFront(), *daRoot.getTreePartBack());

  int eidx = 0;

  while (!treeLoopOut.isFinished())
  {
    if (treeLoopOut.isPre() && treeLoopOut.subtreeInfo().isLeaf())
    {
      std::cout << "\n\n";
      std::cout << "Element " << eidx++ << "\n";

      const double * nodeCoordsFlat = treeLoopOut.subtreeInfo().getNodeCoords();
      const TN *hereCoords = treeLoopOut.subtreeInfo().readNodeCoordsIn();
      for (int nIdx = 0; nIdx < nPe; nIdx++)
      {
        ot::printtn(hereCoords[nIdx], 2, std::cout);
        std::cout << "    \t(";
        for (int d = 0; d < dim; d++)
          std::cout << nodeCoordsFlat[nIdx * dim + d] << ",\t";
        std::cout << ")\n";
      }

      // ---------------------------------------------

      // diag1: elementalMatVec()
      for (int i = 0; i < nPe; i++)
      {
        npeBufferIn.clear();
        npeBufferIn.resize(nPe, 0.0);
        npeBufferOut.clear();
        npeBufferOut.resize(nPe, 0.0);

        npeBufferIn[i] = 1.0;

        pmat.elementalMatVec(npeBufferIn.data(), npeBufferOut.data(), 1, nodeCoordsFlat, 1.0);
        diag1[i] = npeBufferOut[i];
      }
      std::cout << "diag1: Many elementalMatVec()...\n";
      for (double x : diag1)
        std::cout << "  " << x;
      std::cout << "\n\n";

      // ---------------------------------------------

      // diag2: elementalSetDiag()
      pmat.elementalSetDiag(diag2.data(), 1, nodeCoordsFlat, 1.0);
      std::cout << "diag2: Single elementalSetDiag()...\n";
      for (double x : diag2)
        std::cout << "  " << x;
      std::cout << "\n\n";

      // ---------------------------------------------

      // Compare
      double absdiff = 0.0f;
      for (int nIdx = 0; nIdx < nPe; nIdx++)
      {
        double diff = diag1[nIdx] - diag2[nIdx];
        if (diff < 0.0)
          diff = -diff;

        if (absdiff < diff)
          absdiff = diff;
      }

      std::cout << "Linf difference is " << absdiff << "\n";


      /// eleSet(&(*leafResult.begin()), ndofs, nodeCoordsFlat, scale);
      /// treeLoopOut.subtreeInfo().overwriteNodeValsOut(&(*leafResult.begin()));

      treeLoopOut.next();
    }
    else
      treeLoopOut.step();
  }





  _DestroyHcurve();

  MPI_Finalize();
}


template <typename UICoordT, typename FCoordT, unsigned int dim>
void convertToFlatCoords(const ot::TreeNode<UICoordT, dim> &element,
                         unsigned int eleOrder, unsigned int nPe,
                         const ot::TreeNode<UICoordT, dim> *tncoords,
                         FCoordT *fcoords)
{
  const unsigned int curLev = element.getLevel();

  const double domainScale = 1.0 / double(1u << m_uiMaxDepth);
  const double elemSz = double(1u << m_uiMaxDepth - curLev) / double(1u << m_uiMaxDepth);
  double translate[dim];
  for (int d = 0; d < dim; d++)
    translate[d] = domainScale * element.getX(d);

  std::array<unsigned int, dim> numerators;
  unsigned int denominator;

  for (size_t nIdx = 0; nIdx < nPe; nIdx++)
  {
    ot::TNPoint<unsigned int, dim>::get_relNodeCoords(
        element, tncoords[nIdx], eleOrder,
        numerators, denominator);

    for (int d = 0; d < dim; ++d)
      fcoords[nIdx * dim + d] =
          translate[d] + elemSz * numerators[d] / denominator;
  }
}






