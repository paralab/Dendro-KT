
#include "distTree.h"
#include "oda.h"

#include "intergridTransfer.h"

#include "point.h"
#include "refel.h"
#include "tensor.h"

#include <iostream>


//
// ConstMeshPointers  (wrapper)
//
template <unsigned int dim>
class ConstMeshPointers
{
  private:
    const ot::DistTree<unsigned int, dim> *m_distTree;
    const ot::DA<dim> *m_da;

  public:
    // ConstMeshPointers constructor
    ConstMeshPointers(const ot::DistTree<unsigned int, dim> *distTree,
                 const ot::DA<dim> *da)
      :
        m_distTree(distTree),
        m_da(da)
    {}

    // Copy constructor and copy assignment.
    ConstMeshPointers(const ConstMeshPointers &other) = default;
    ConstMeshPointers & operator=(const ConstMeshPointers &other) = default;

    // distTree()
    const ot::DistTree<unsigned int, dim> * distTree() const { return m_distTree; }

    // numElements()
    size_t numElements() const
    {
      return m_distTree->getTreePartFiltered().size();
    }

    // da()
    const ot::DA<dim> * da() const { return m_da; }

    // printSummary()
    std::ostream & printSummary(std::ostream & out, const std::string &pre = "", const std::string &post = "") const;
};

//
// Vector  (wrapper)
//
template <typename ValT>
class Vector
{
  private:
    std::vector<ValT> m_data;
    bool m_isGhosted;
    size_t m_ndofs = 1;
    unsigned long long m_globalNodeRankBegin = 0;

  public:
    // Vector constructor
    template <unsigned int dim>
    Vector(const ConstMeshPointers<dim> &mesh, bool isGhosted, size_t ndofs)
    : m_isGhosted(isGhosted), m_ndofs(ndofs)
    {
      mesh.da()->createVector(m_data, false, isGhosted, ndofs);
      m_globalNodeRankBegin = mesh.da()->getGlobalRankBegin();
    }

    // data()
    std::vector<ValT> & data() { return m_data; }
    const std::vector<ValT> & data() const { return m_data; }

    // isGhosted()
    bool isGhosted() const { return m_isGhosted; }

    // ndofs()
    size_t ndofs() const { return m_ndofs; }

    // ptr()
    ValT * ptr() { return m_data.data(); }
    const ValT * ptr() const { return m_data.data(); }

    // size()
    size_t size() const { return m_data.size(); }

    // globalRankBegin();
    unsigned long long globalRankBegin() const { return m_globalNodeRankBegin; }
};



template <unsigned int dim>
static std::vector<ot::OCT_FLAGS::Refine> flagRefineBoundary(
    const ConstMeshPointers<dim> &mesh);

template <typename ValT>
static void initialize(Vector<ValT> &vec, ValT begin);

template <unsigned int dim, typename ValT>
static void prolongation(const ConstMeshPointers<dim> &coarseMesh,
                         const Vector<ValT> &coarseIn,
                         const ConstMeshPointers<dim> &surrogateMesh,
                         const ConstMeshPointers<dim> &fineMesh,
                         Vector<ValT> &fineOut);

template <unsigned int dim, typename ValT>
static void restriction(const ConstMeshPointers<dim> &fineMesh,
                        const Vector<ValT> &fineIn,
                        const ConstMeshPointers<dim> &surrogateMesh,
                        const ConstMeshPointers<dim> &coarseMesh,
                        Vector<ValT> &coarseOut);


template <unsigned int dim>
class InnerProduct
{
  public:
    InnerProduct(const ConstMeshPointers<dim> &mesh)
      : m_mesh(mesh)
    {
      setProblemDimensions(Point<dim>(0.0), Point<dim>(1.0));
    }

    // setProblemDimensions()
    inline void setProblemDimensions(const Point<dim>& pt_min, const Point<dim>& pt_max)
    {
      m_uiPtMin = pt_min;
      m_uiPtMax = pt_max;
    }

    // compute()
    template <typename ValT>
    ValT compute(const Vector<ValT> &vecA,
                 const Vector<ValT> &vecB);

  protected:
    // elementalInnerProduct()
    template <typename ValT>
    ValT elementalInnerProduct(const ValT *vecA,
                               const ValT *vecB,
                               unsigned int ndofs,
                               const double *coords,
                               double scale,
                               bool isElementBoundary);

    // doubleBufferPipeline()
    template <typename ValT>
    static void doubleBufferPipeline(const ValT * fromPtrs[], ValT * toPtrs[], const ValT *in, ValT *out, ValT * tmp1, ValT * tmp2 = nullptr);


    // gridX_to_X()  (single axis)
    double gridX_to_X(unsigned int d, double x) const
    {
      double Rg=1.0;
      return (((x)/(Rg))*((m_uiPtMax.x(d)-m_uiPtMin.x(d)))+m_uiPtMin.x(d));
    }

    // gridX_to_X()  (multi-dimensional point)
    Point<dim> gridX_to_X(Point<dim> x) const
    {
      double newCoords[dim];
      for (unsigned int d = 0; d < dim; d++)
        newCoords[d] = gridX_to_X(d, x.x(d));
      return Point<dim>(newCoords);
    }

  protected:
    const ConstMeshPointers<dim> m_mesh;
    Point<dim> m_uiPtMin;
    Point<dim> m_uiPtMax;
};



template <unsigned int dim, typename ValT>
ValT coordinateProduct(const ConstMeshPointers<dim> &mesh,
                       const Vector<ValT> &vecA,
                       const Vector<ValT> &vecB)
{
  const size_t localSize = vecA.ndofs() * mesh.da()->getLocalNodalSz();

  ValT localProduct = 0.0f;
  for (size_t ii = 0; ii < localSize; ++ii)
    localProduct += vecA.ptr()[ii] * vecB.ptr()[ii];

  MPI_Comm comm = mesh.da()->getGlobalComm();
  ValT globalProduct = 0.0f;
  par::Mpi_Allreduce(&localProduct, &globalProduct, 1, MPI_SUM, comm);

  return globalProduct;
}


//
// ElementLoopIn  (wrapper)
//
template <unsigned int dim, typename ValT>
class ElementLoopIn
{
  private:
    ot::MatvecBaseIn<dim, ValT> m_loop;

  public:
    ot::MatvecBaseIn<dim, ValT> & loop() { return m_loop; }

    ElementLoopIn(const ConstMeshPointers<dim> &mesh,
                  const Vector<ValT> &ghostedVec)
      :
        m_loop(mesh.da()->getTotalNodalSz(),
               ghostedVec.ndofs(),
               mesh.da()->getElementOrder(),
               false,
               0,
               mesh.da()->getTNCoords(),
               ghostedVec.ptr(),
               mesh.distTree()->getTreePartFiltered().data(),
               mesh.distTree()->getTreePartFiltered().size(),
               *mesh.da()->getTreePartFront(),
               *mesh.da()->getTreePartBack())
    { }
};



constexpr unsigned int dim = 2;
using uint = unsigned int;
using val_t = double;

//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  _InitializeHcurve(dim);

  MPI_Comm comm = MPI_COMM_WORLD;
  const int eleOrder = 1;
  const unsigned int ndofs = 1;
  const double sfc_tol = 0.3;

  using DTree_t = ot::DistTree<uint, dim>;
  using DA_t = ot::DA<dim>;

  //
  // Define coarse grid.
  //
  const int coarseLev = 1;
  const DTree_t coarseDTree = DTree_t::constructSubdomainDistTree(
      coarseLev, comm, sfc_tol);
  const DA_t * coarseDA = new DA_t(coarseDTree, comm, eleOrder);
  const ConstMeshPointers<dim> coarseMesh(&coarseDTree, coarseDA);
  
  //
  // Define fine grid.
  //
  std::vector<ot::OCT_FLAGS::Refine> octFlags = flagRefineBoundary(coarseMesh);
  DTree_t fineDTree;
  DTree_t surrogateDTree;
  DTree_t::distRemeshSubdomain(coarseDTree, octFlags, fineDTree, surrogateDTree, sfc_tol);
  const DA_t * fineDA = new DA_t(fineDTree, comm, eleOrder);
  const ConstMeshPointers<dim> fineMesh(&fineDTree, fineDA);
  const DA_t * surrogateDA = new DA_t(surrogateDTree, comm, eleOrder);
  const ConstMeshPointers<dim> surrogateMesh(&surrogateDTree, surrogateDA);

  // Print mesh summaries.
  coarseMesh.printSummary(std::cout,    "Coarse grid:    ", "\n");
  fineMesh.printSummary(std::cout,      "Fine grid:      ", "\n");
  surrogateMesh.printSummary(std::cout, "Surrogate grid: ", "\n");

  //
  // Define coarse and fine vectors.
  //
  const bool isGhosted = false;
  Vector<val_t> coarseU(coarseMesh, isGhosted, ndofs);
  Vector<val_t> fineV(fineMesh, isGhosted, ndofs);
  Vector<val_t> Pu(fineMesh, isGhosted, ndofs);
  Vector<val_t> Rv(coarseMesh, isGhosted, ndofs);

  /// initialize(coarseU, val_t(10));
  /// initialize(fineV, val_t(101));
  initialize(coarseU, val_t(1));    // coordinate of function
  initialize(fineV, val_t(1));      // coordinate of functional

  // Compute prolongation and restriction.
  prolongation(coarseMesh, coarseU, surrogateMesh, fineMesh, Pu);
  restriction(fineMesh, fineV, surrogateMesh, coarseMesh, Rv);

  // Assert equality of inner products.
  /// const val_t inner_product_Pu_v = InnerProduct<dim>(fineMesh).compute(Pu, fineV);
  /// const val_t inner_product_u_Rv = InnerProduct<dim>(coarseMesh).compute(coarseU, Rv);

  // These aren't really inner products, but
  // the Natural Pairings of function spaces with their duals.
  // The basis used in the dual space happens to be the
  // set of coordinate-extraction functionals.
  // Therefore, to evaluate the natural pairing of a
  // function and a functional, simply take the
  // dot product of their coordinate vectors.
  const val_t inner_product_Pu_v = coordinateProduct(fineMesh, Pu, fineV);
  const val_t inner_product_u_Rv = coordinateProduct(coarseMesh, coarseU, Rv);

  const bool matching = fabs(inner_product_Pu_v - inner_product_u_Rv) < 1e-5;
  fprintf(stdout, "%s%s%s: [Pu, v]: %f %s%s [u, Rv]: %f%s\n",
      (matching ? GRN : RED),
      (matching ? "success" : "failure"),
      NRM,
      inner_product_Pu_v,
      (matching ? "==" : "!="),
      (matching ? GRN : RED),
      inner_product_u_Rv,
      NRM);

  delete coarseDA;
  delete fineDA;
  delete surrogateDA;

  _DestroyHcurve();
  PetscFinalize();

  return 0;
}


//
// flagRefineBoundary()
//
template <unsigned int dim>
static std::vector<ot::OCT_FLAGS::Refine> flagRefineBoundary(
    const ConstMeshPointers<dim> &mesh)
{
  const size_t numElements = mesh.numElements();
  const std::vector<ot::TreeNode<unsigned int, dim>> &elements =
      mesh.distTree()->getTreePartFiltered();

  using RefineFlag = ot::OCT_FLAGS::Refine;
  std::vector<RefineFlag> flags(numElements, RefineFlag::OCT_NO_CHANGE);

  for (size_t ii = 0; ii < numElements; ++ii)
  {
    if (elements[ii].getIsOnTreeBdry())
      flags[ii] = RefineFlag::OCT_REFINE;
  }

  return flags;
}


//
// initialize()
//
template <typename ValT>
static void initialize(Vector<ValT> &vec, ValT begin)
{
  /// const long long m = 100;
  const size_t size = vec.data().size();
  begin += vec.ndofs() * vec.globalRankBegin();
  for (size_t ii = 0; ii < size; ++ii)
  {
    /// vec.data()[ii] = ((long long)(begin++)) % m;
    vec.data()[ii] = 1.0f;
  }
}



//
// prolongation()
//
template <unsigned int dim, typename ValT>
static void prolongation(const ConstMeshPointers<dim> &coarseMesh,
                         const Vector<ValT> &coarseIn,
                         const ConstMeshPointers<dim> &surrogateMesh,
                         const ConstMeshPointers<dim> &fineMesh,
                         Vector<ValT> &fineOut)
{
  //
  // Intergrid Transfer with Injection.
  //

  std::cout << "prolongation()\n";

  const size_t ndofs = coarseIn.ndofs();

  // Ghosted array for output.
  static Vector<ValT> fineOutGhosted(fineMesh, true, fineOut.ndofs());

  // Temporary surrogate array (also ghosted).
  static Vector<ValT> surrogateGhosted(surrogateMesh, true, ndofs);
  const size_t surrogateLocalOffset = ndofs * surrogateMesh.da()->getLocalNodeBegin();

  // Align local nodes to the fine grid partition local nodes.
  ot::distShiftNodes(*coarseMesh.da(),
                     coarseIn.data().data(),
                     *surrogateMesh.da(),
                     surrogateGhosted.data().data() + surrogateLocalOffset,
                     ndofs);
  surrogateMesh.da()->readFromGhostBegin(surrogateGhosted.data().data(), ndofs);
  surrogateMesh.da()->readFromGhostEnd(surrogateGhosted.data().data(), ndofs);
  // Note that depending on whether the fine grid was generated from the
  // coarse grid, or whether the coarse grid was generated from the
  // fine grid, will change the relative ordering of
  // distShiftNodes, readFromGhost, and writeToGhost.
  // -   If fine=generateFrom(coarse), then
  //     *   |surrogate_nodes| == |coarse_nodes|, and
  //     *   partition_{by_surrogate} == partition_{by_fine}.
  //
  // -   If coarse=generateFrom(fine), then
  //     *   |surrogate_nodes| == |fine_nodes|, and
  //     *   partition_{by_surrogate} == partition_{by_coarse}.

  // Local intergrid transfer.
  using TreeNodeT = ot::TreeNode<unsigned int, dim>;
  fem::MeshFreeInputContext<ValT, TreeNodeT>
      inctx{ surrogateGhosted.data().data(),
             surrogateMesh.da()->getTNCoords(),
             (unsigned) surrogateMesh.da()->getTotalNodalSz(),
             &(*surrogateMesh.distTree()->getTreePartFiltered().cbegin()),
             surrogateMesh.numElements(),
             *surrogateMesh.da()->getTreePartFront(),
             *surrogateMesh.da()->getTreePartBack() };
  fem::MeshFreeOutputContext<ValT, TreeNodeT>
      outctx{ fineOutGhosted.data().data(),
              fineMesh.da()->getTNCoords(),
              (unsigned) fineMesh.da()->getTotalNodalSz(),
              &(*fineMesh.distTree()->getTreePartFiltered().cbegin()),
              fineMesh.numElements(),
              *fineMesh.da()->getTreePartFront(),
              *fineMesh.da()->getTreePartBack() };
  const RefElement * refel = fineMesh.da()->getReferenceElement();
  std::vector<char> outDirty(fineMesh.da()->getTotalNodalSz(), 0);
  fem::locIntergridTransfer(inctx, outctx, ndofs, refel, &(*outDirty.begin()));
  // The outDirty array is needed when useAccumulation==false (hack).

  // Finish distributed intergrid transfer.
  fineMesh.da()->writeToGhostsBegin(fineOutGhosted.data().data(), ndofs, &(*outDirty.cbegin()));
  fineMesh.da()->writeToGhostsEnd(fineOutGhosted.data().data(), ndofs, false, &(*outDirty.cbegin()));
  fineMesh.da()->ghostedNodalToNodalVec(fineOutGhosted.data(), fineOut.data(), true, ndofs);
}


//
// restriction()
//
template <unsigned int dim, typename ValT>
static void restriction(const ConstMeshPointers<dim> &fineMesh,
                        const Vector<ValT> &fineIn,
                        const ConstMeshPointers<dim> &surrogateMesh,
                        const ConstMeshPointers<dim> &coarseMesh,
                        Vector<ValT> &coarseOut)
{
  const size_t ndofs = fineIn.ndofs();
  const unsigned int eleOrder = fineMesh.da()->getElementOrder();

  // Input fine, output coarse.
  // Depending how grids were generated, fine may need to be moved or not.
  // (If fine==generateFrom(coarse), then surrogate is partitioned by fine.)

  std::cout << "restriction()\n";

  // Ghosted array for input.
  static Vector<ValT> fineInGhosted(fineMesh, true, ndofs);
  fineMesh.da()->nodalVecToGhostedNodal(fineIn.data(), fineInGhosted.data(), true, ndofs);

  // Temporary surrogate array (also ghosted).
  static Vector<ValT> surrogateGhosted(surrogateMesh, true, ndofs);
  std::fill(surrogateGhosted.data().begin(), surrogateGhosted.data().end(), 0);

  // Basically intergrid transfer. Main things are:
  // - Coarse UseAccumulation=true;
  // - Fine hanging nodes are wiped out before transfer.

  if (fineMesh.da()->getLocalNodalSz() > 0)
  {
    ot::MatvecBaseIn<dim, ValT> loopFine(fineMesh.da()->getTotalNodalSz(),
                                         ndofs,
                                         eleOrder,
                                         false,
                                         0,
                                         fineMesh.da()->getTNCoords(),
                                         fineInGhosted.ptr(),
                                         fineMesh.distTree()->getTreePartFiltered().data(),
                                         fineMesh.distTree()->getTreePartFiltered().size(),
                                         *fineMesh.da()->getTreePartFront(),
                                         *fineMesh.da()->getTreePartBack());
    const bool acc = true;
    ot::MatvecBaseOut<dim, ValT, acc> loopCoarse(surrogateMesh.da()->getTotalNodalSz(),
                                                 ndofs,
                                                 eleOrder,
                                                 true,
                                                 1,
                                                 surrogateMesh.da()->getTNCoords(),
                                                 surrogateMesh.distTree()->getTreePartFiltered().data(),
                                                 surrogateMesh.distTree()->getTreePartFiltered().size(),
                                                 *surrogateMesh.da()->getTreePartFront(),
                                                 *surrogateMesh.da()->getTreePartBack());

    const unsigned int nPe = intPow(eleOrder+1, dim);
    std::vector<ValT> leafBuffer(ndofs * nPe, 0.0);

    while (!loopFine.isFinished())
    {
      // Depth controlled by fine.
      if (loopFine.isPre() && loopFine.subtreeInfo().isLeaf())
      {
        const ValT * fineLeafIn = loopFine.subtreeInfo().readNodeValsIn();
        for (size_t nIdx = 0; nIdx < nPe; ++nIdx)
        {
          if (loopFine.subtreeInfo().readNodeNonhangingIn()[nIdx])
            for (int dof = 0; dof < ndofs; ++dof)
              leafBuffer[ndofs * nIdx + dof] = fineLeafIn[ndofs * nIdx + dof];
          else
            for (int dof = 0; dof < ndofs; ++dof)
              leafBuffer[ndofs * nIdx + dof] = 0.0f;
        }

        loopCoarse.subtreeInfo().overwriteNodeValsOut(leafBuffer.data());

        loopFine.next();
        loopCoarse.next();
      }
      else
      {
        loopFine.step();
        loopCoarse.step();
      }
    }
    const size_t writtenSz = loopCoarse.finalize(surrogateGhosted.ptr());
  }

  // writeToGhost before distShiftNodes(),
  // because distShiftNodes() ignores information in ghost segments.
  surrogateMesh.da()->writeToGhostsBegin(surrogateGhosted.ptr(), ndofs);
  surrogateMesh.da()->writeToGhostsEnd(surrogateGhosted.ptr(), ndofs);

  // Align from the fine grid partition local nodes to coarse local nodes.
  const size_t surrogateLocalOffset = ndofs * surrogateMesh.da()->getLocalNodeBegin();
  ot::distShiftNodes(*surrogateMesh.da(),
                     surrogateGhosted.ptr() + surrogateLocalOffset,
                     *coarseMesh.da(),
                     coarseOut.ptr(),
                     ndofs);
}



//
// doubleBufferPipeline()
//
template <unsigned int dim>
template <typename ValT>
void InnerProduct<dim>::doubleBufferPipeline(const ValT * fromPtrs[], ValT * toPtrs[], const ValT *in, ValT *out, ValT * tmp1, ValT * tmp2)
{
  // There are (dim) operations, hence (dim) edges.
  //   in -> ... -> tmp1 -> ... -> tmp2 -> out.
  // Make tmp2 the final buffer before out.

  if (tmp2 == nullptr)
  {
    tmp2 = tmp1;
    tmp1 = out;
  }

  // Create edges last to first. Connect edges by swapping "to" and "from".
  fromPtrs[dim-1] = tmp2;
  toPtrs[dim-1] = tmp1;
  for (int d = dim-1; d > 0; --d)
  {
    fromPtrs[d-1] = toPtrs[d];
    toPtrs[d-1] = (fromPtrs[d-1] == tmp1 ? tmp2 : tmp1);
  }

  // Override "from" of first edge and "to" of last edge.
  fromPtrs[0] = in;
  toPtrs[dim-1] = out;
}


//
// elementalInnerProduct
//
template <unsigned int dim>
template <typename ValT>
ValT InnerProduct<dim>::elementalInnerProduct(const ValT *vecA, const ValT *vecB, unsigned int ndofs, const double *coords, double scale, bool isElementBoundary)
{
  // Compute the following from right to left:
  //   vecB^T Q^T W Q vecA

  const RefElement* refEl = this->m_mesh.da()->getReferenceElement();
  const unsigned int eleOrder = refEl->getOrder();
  const unsigned int nrp = eleOrder+1;
  const unsigned int nPe = intPow(eleOrder+1, dim);

  static std::vector<ValT> intraOpBuffer;
  intraOpBuffer.clear();
  intraOpBuffer.resize(ndofs * nPe, 0.0f);

  static std::vector<ValT> interOpBuffer;
  interOpBuffer.clear();
  interOpBuffer.resize(ndofs * nPe, 0.0f);

  // Fill with chain of matrices acting on 1 axis at a time.
  const double * mat1dPtrs[dim];

  // Fill with chain of intermediate variables.
  const ValT * imFromPtrs[dim];
  ValT * imToPtrs[dim];

  // Interpolate vecA to quadrature points.  (LMult by Q)
  doubleBufferPipeline(imFromPtrs, imToPtrs, vecA, interOpBuffer.data(), intraOpBuffer.data());
  for (unsigned int d = 0; d < dim; d++)
    mat1dPtrs[d] = refEl->getQ1d();
  KroneckerProduct<dim, ValT, true>(nrp, mat1dPtrs, imFromPtrs, imToPtrs, ndofs);

  // Integration weights and Jacobian.  (LMult by W)
  // To integrate in reference space must multiply by Jacobian.
  const Point<dim> eleMin(&coords[0 * dim]);
  const Point<dim> eleMax(&coords[(nPe-1) * dim]);
  const Point<dim> sz = gridX_to_X(eleMax) - gridX_to_X(eleMin);
  const double refElSz=refEl->getElementSz();
  const Point<dim> J = sz * (1.0 / refElSz);
  double J_product = 1.0;
  for (unsigned int d = 0; d < dim; d++)
    J_product *= J.x(d);
  SymmetricOuterProduct<double, dim>::applyHadamardProduct(eleOrder+1, interOpBuffer.data(), refEl->getWgq(), J_product);

  // Transpose of interpolation-to-quadrature-points.  (LMult by Q^T)
  doubleBufferPipeline(imFromPtrs, imToPtrs, interOpBuffer.data(), interOpBuffer.data(), intraOpBuffer.data());
  for (unsigned int d = 0; d < dim; d++)
    mat1dPtrs[d] = refEl->getQT1d();  // transpose
  KroneckerProduct<dim, ValT, true>(nrp, mat1dPtrs, imFromPtrs, imToPtrs, ndofs);

  // Product with vecB.  (LMult by vecB^T)
  ValT product = 0;
  for (size_t node_dof = 0; node_dof < ndofs * nPe; ++node_dof)
    product += vecB[node_dof] * interOpBuffer[node_dof];

  return product;
}


//
// innerProduct()
//
template <unsigned int dim>
template <typename ValT>
ValT InnerProduct<dim>::compute(const Vector<ValT> &vecA,
                                const Vector<ValT> &vecB)
{
  const size_t ndofs = vecA.ndofs();

  // Read from ghosts.
  static Vector<ValT> ghostedA(this->m_mesh, true, ndofs);
  static Vector<ValT> ghostedB(this->m_mesh, true, ndofs);

  this->m_mesh.da()->nodalVecToGhostedNodal(vecA.data(), ghostedA.data(), true, ndofs);
  this->m_mesh.da()->readFromGhostBegin(ghostedA.data().data(), ndofs);
  this->m_mesh.da()->readFromGhostEnd(ghostedA.data().data(), ndofs);

  this->m_mesh.da()->nodalVecToGhostedNodal(vecB.data(), ghostedB.data(), true, ndofs);
  this->m_mesh.da()->readFromGhostBegin(ghostedB.data().data(), ndofs);
  this->m_mesh.da()->readFromGhostEnd(ghostedB.data().data(), ndofs);

  // Integrate in local element loop.
  ValT localIntegration = 0;
  if (this->m_mesh.da()->getLocalNodalSz() > 0)
  {
    // TODO what determines the scale parameter?
    const double scale = 1.0;

    ElementLoopIn<dim, ValT> loopA(this->m_mesh, ghostedA);
    ElementLoopIn<dim, ValT> loopB(this->m_mesh, ghostedB);

    while (!loopA.loop().isFinished())
    {
      if (loopA.loop().isPre() && loopA.loop().subtreeInfo().isLeaf())
      {
        const double * nodeCoordsFlat = loopA.loop().subtreeInfo().getNodeCoords();
        const ValT * valsAFlat = loopA.loop().subtreeInfo().readNodeValsIn();
        const ValT * valsBFlat = loopB.loop().subtreeInfo().readNodeValsIn();

        localIntegration += this->elementalInnerProduct(valsAFlat,
                                                        valsBFlat,
                                                        ndofs,
                                                        nodeCoordsFlat,
                                                        scale,
                               loopA.loop().subtreeInfo().isElementBoundary());
        loopA.loop().next();
        loopB.loop().next();
      }
      else
      {
        loopA.loop().step();
        loopB.loop().step();
      }
    }
  }

  // Allreduce.
  ValT globalIntegration = 0;
  MPI_Comm comm = this->m_mesh.da()->getGlobalComm();
  par::Mpi_Allreduce(&localIntegration, &globalIntegration, 1, MPI_SUM, comm);

  return globalIntegration;
}



template <unsigned int dim>
std::ostream & ConstMeshPointers<dim>::printSummary(std::ostream &out, const std::string &pre, const std::string &post) const
{
  MPI_Comm comm = this->da()->getGlobalComm();
  int commSize, commRank;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  const size_t myNumElements = this->numElements();
  const size_t myNumNodes = this->da()->getLocalNodalSz();

  std::vector<size_t> allNumElements;
  std::vector<size_t> allNumNodes;
  if (commRank == 0)
  {
    allNumElements.resize(commSize);
    allNumNodes.resize(commSize);
  }

  par::Mpi_Gather(&myNumElements, allNumElements.data(), 1, 0, comm);
  par::Mpi_Gather(&myNumNodes, allNumNodes.data(), 1, 0, comm);

  if (commRank == 0)
  {
    size_t globalNumElements = 0;
    for (size_t n : allNumElements)
      globalNumElements += n;

    const size_t globalNumNodes = this->da()->getGlobalNodeSz();

    out << pre;

    out << "number_of_elements == "
        << globalNumElements << " ("
        << allNumElements[0];
    for (int r = 1; r < commSize; ++r)
      out << "+" << allNumElements[r];
    out << "); ";

    out << "number_of_nodes == "
        << globalNumNodes << " ("
        << allNumNodes[0];
    for (int r = 1; r < commSize; ++r)
      out << "+" << allNumNodes[r];
    out << ");";

    out << post;
  }

  return out;
}