
#include "distTree.h"
#include "oda.h"

#include "intergridTransfer.h"

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

  public:
    // Vector constructor
    template <unsigned int dim>
    Vector(const ConstMeshPointers<dim> &mesh, bool isGhosted, size_t ndofs)
    : m_isGhosted(isGhosted), m_ndofs(ndofs)
    {
      mesh.da()->createVector(m_data, false, isGhosted, ndofs);
    }

    // data()
    std::vector<ValT> & data() { return m_data; }
    const std::vector<ValT> & data() const { return m_data; }

    // isGhosted()
    bool isGhosted() const { return m_isGhosted; }

    // ndofs()
    size_t ndofs() const { return m_ndofs; }
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

template <unsigned int dim, typename ValT>
static ValT innerProduct(const ConstMeshPointers<dim> &mesh,
                         const Vector<ValT> &vecA,
                         const Vector<ValT> &vecB);


constexpr unsigned int dim = 4;
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
  const int coarseLev = 2;
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

  initialize(coarseU, val_t(10));
  initialize(fineV, val_t(101));

  // Compute prolongation and restriction.
  prolongation(coarseMesh, coarseU, surrogateMesh, fineMesh, Pu);
  restriction(fineMesh, fineV, surrogateMesh, coarseMesh, Rv);

  // Assert equality of inner products.
  const val_t inner_product_Pu_v = innerProduct(fineMesh, Pu, fineV);
  const val_t inner_product_u_Rv = innerProduct(coarseMesh, coarseU, Rv);
  const bool matching = (inner_product_Pu_v == inner_product_u_Rv);
  fprintf(stdout, "%s%s: %f %s %f%s\n",
      (matching ? GRN : RED),
      (matching ? "success" : "failure"),
      inner_product_Pu_v,
      (matching ? "==" : "!="),
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
  const size_t size = vec.data().size();
  for (size_t ii = 0; ii < size; ++ii)
    vec.data()[ii] = (begin++);
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
  throw std::logic_error("restriction() not implemented!");
}


//
// innerProduct()
//
template <unsigned int dim, typename ValT>
static ValT innerProduct(const ConstMeshPointers<dim> &mesh,
                         const Vector<ValT> &vecA,
                         const Vector<ValT> &vecB)
{
  throw std::logic_error("innerProduct() not implemented!");
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
