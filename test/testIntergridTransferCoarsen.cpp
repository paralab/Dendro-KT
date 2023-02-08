//
// Created by maksbh on 11/9/20.
//

#include <iostream>
#include <oda.h>
#include <point.h>
#include <sfcTreeLoop_matvec_io.h>
#include <intergridTransfer.h>
static constexpr unsigned int DIM = 2;
typedef ot::DA<DIM> DA;
typedef unsigned int DENDRITE_UINT;

typedef ot::DistTree<unsigned int, DIM> DistTREE;
typedef ot::TreeNode<DENDRITE_UINT, DIM> TREENODE;

// intergrid_coarse_to_fine()
void intergrid_coarse_to_fine(
    const double * const input, const DA *oldDA,  const DistTREE & oldDistTree,  // coarse by coarse
                                const DA *surrDA, const DistTREE & surrDistTree, // coarse by fine
         double * const output, const DA *newDA,  const DistTREE & newDistTree,  // fine by fine
         int ndof = 1)
{
  static std::vector<VECType> fineGhosted, surrGhosted;
  surrDA->template createVector<VECType>(surrGhosted, false, true, ndof);
  newDA->template createVector<VECType>(fineGhosted, false, true, ndof);
  std::fill(fineGhosted.begin(), fineGhosted.end(), 0);

  // Surrogate partition matches fine partition. Shift coarse data onto surrogate.
  ot::distShiftNodes(
      *oldDA, input, *surrDA, surrGhosted.data() + ndof * surrDA->getLocalNodeBegin(), ndof);
  surrDA->template readFromGhostBegin<VECType>(surrGhosted.data(), ndof);
  surrDA->template readFromGhostEnd<VECType>(surrGhosted.data(), ndof);

  fem::MeshFreeInputContext<VECType, TREENODE>
    inctx = fem::mesh_free_input_context(surrGhosted.data(), surrDA, surrDistTree);
   
  fem::MeshFreeOutputContext<VECType, TREENODE>
    outctx = fem::mesh_free_output_context(fineGhosted.data(), newDA, newDistTree);

  // Hack: Only to writeToGhosts, when not all owned nodes touch owned cells.
  // (When fixed, i.e. owned nodes touch owned cells, can use readFromGhosts).
  static std::vector<char> outDirty;
  const char zero = 0;
  newDA->template createVector<char>(outDirty, false, true, 1);
  newDA->setVectorByScalar(outDirty.data(), &zero, false, true, 1);

  const RefElement *refel = newDA->getReferenceElement();
  fem::locIntergridTransfer(inctx, outctx, ndof, refel, outDirty.data());

  newDA->template writeToGhostsBegin<VECType>(fineGhosted.data(), ndof, outDirty.data());
  newDA->template writeToGhostsEnd<VECType>(fineGhosted.data(), ndof, false, outDirty.data());
  newDA->template ghostedNodalToNodalVec<VECType>(fineGhosted.data(), output, ndof);
}


// intergrid_fine_to_coarse()
void intergrid_fine_to_coarse(
    const double * const input, const DA *oldDA,  const DistTREE & oldDistTree,  // fine by fine
                                const DA *surrDA, const DistTREE & surrDistTree, // coarse by fine
         double * const output, const DA *newDA,  const DistTREE & newDistTree,  // coarse by coarse
         int ndof = 1)
{
  // fine -> intergrid surrogate -> shift coarse

  static std::vector<VECType> fineGhosted;
  static std::vector<VECType> surrGhosted;
  oldDA->template createVector(fineGhosted, false, true, ndof);
  surrDA->template createVector<VECType>(surrGhosted, false, true, ndof);
  oldDA->nodalVecToGhostedNodal(input, fineGhosted.data(), ndof);
  oldDA->readFromGhostBegin(fineGhosted.data(), ndof);
  oldDA->readFromGhostEnd(fineGhosted.data(), ndof);
  std::fill(surrGhosted.begin(), surrGhosted.end(), 0);

  fem::MeshFreeInputContext<VECType, TREENODE>
    inctx = fem::mesh_free_input_context(fineGhosted.data(), oldDA, oldDistTree);
   
  fem::MeshFreeOutputContext<VECType, TREENODE>
    outctx = fem::mesh_free_output_context(surrGhosted.data(), surrDA, surrDistTree);

  // Hack: Only to writeToGhosts, when not all owned nodes touch owned cells.
  // (When fixed, i.e. owned nodes touch owned cells, can use readFromGhosts).
  static std::vector<char> outDirty;
  const char zero = 0;
  surrDA->template createVector<char>(outDirty, false, true, 1);
  surrDA->setVectorByScalar(outDirty.data(), &zero, false, true, 1);

  const RefElement *refel = newDA->getReferenceElement();
  fem::locIntergridTransfer(inctx, outctx, ndof, refel, &(*outDirty.begin()));

  surrDA->template writeToGhostsBegin<VECType>(surrGhosted.data(), ndof, &(*outDirty.cbegin()));
  surrDA->template writeToGhostsEnd<VECType>(surrGhosted.data(), ndof, false, &(*outDirty.cbegin()));

  ot::distShiftNodes(*surrDA, surrGhosted.data() + ndof * surrDA->getLocalNodeBegin(),
                     *newDA, output,
                     ndof);
}



void checkIntergridTransfer(
        const double * const array,
        const ot::DA<DIM> * octDA,
        const ot::DistTree<unsigned int, DIM> &distTree,
        const unsigned int ndof)
{
    double *ghostedArray;
    octDA->nodalVecToGhostedNodal(array,ghostedArray, false,ndof);
    octDA->readFromGhostBegin(ghostedArray,ndof);
    octDA->readFromGhostEnd(ghostedArray,ndof);
    const size_t sz = octDA->getTotalNodalSz();
    auto partFront = octDA->getTreePartFront();
    auto partBack = octDA->getTreePartBack();
    const auto tnCoords = octDA->getTNCoords();
    const unsigned int nPe = octDA->getNumNodesPerElement();
    ot::MatvecBase<DIM, PetscScalar> treeloop(sz, ndof, octDA->getElementOrder(), tnCoords, ghostedArray, &(*distTree.getTreePartFiltered().cbegin()), distTree.getTreePartFiltered().size(), *partFront, *partBack);
    bool testPassed = true;
    while (!treeloop.isFinished())
    {
        if (treeloop.isPre() && treeloop.subtreeInfo().isLeaf())
        {
            const double * nodeCoordsFlat = treeloop.subtreeInfo().getNodeCoords();
            const PetscScalar * nodeValsFlat = treeloop.subtreeInfo().readNodeValsIn();
            for(int i = 0; i < nPe; i++){
                double correctValue = 0;
                for(int dim = 0; dim < DIM; dim++){
                    correctValue += nodeCoordsFlat[DIM*i+dim];
                }
                for(int dof = 0; dof < ndof; dof++) {
                    double interpolatedValue = nodeValsFlat[i * ndof + dof];
                    if (fabs(interpolatedValue - correctValue) > 1E-6) {
                        std::cout << "Value at (" << nodeCoordsFlat[DIM * i + 0] << " ," << nodeCoordsFlat[DIM * i + 1]
                                  << ") = " << interpolatedValue << "\n";
                        testPassed = false;
                    }
                }

            }
            treeloop.next();
        }
        else
            treeloop.step();
    }
    bool gtestPassed;
    MPI_Reduce(&testPassed,&gtestPassed,1,MPI_CXX_BOOL,MPI_LAND,0,MPI_COMM_WORLD);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(!rank) {
      if (gtestPassed) {
        std::cout << GRN << "TEST passed" << NRM << "\n";
      } else {
        std::cout << RED << "TEST failed" << NRM << "\n";
      }
    }
    delete [] ghostedArray;
    MPI_Barrier(MPI_COMM_WORLD);
}


template <typename T>
auto readonly(T && object) -> std::remove_reference_t<T>
{
  return std::move<T>(object);
}

template <typename T>
const T & readonly(T & object)
{
  return object;
}


int main(int argc, char * argv[]){
  static const char *varname[]{"u"};

  using DENDRITE_UINT = unsigned  int;
    using TREENODE = ot::TreeNode<DENDRITE_UINT, DIM>;
    PetscInitialize(&argc, &argv, NULL, NULL);
    _InitializeHcurve(DIM);
    int eleOrder = 1;
    unsigned int ndof = 1;
    MPI_Comm comm = MPI_COMM_WORLD;

    ot::DistTree<unsigned int, DIM> fineDistTree;
    {
        std::vector<ot::TreeNode<unsigned int, DIM>> treePart;
        ot::createRegularOctree(treePart, 5, comm);
        fineDistTree = ot::DistTree<unsigned int, DIM>(treePart, comm);
    }
    ot::DA<DIM> *fineDA = new ot::DA<DIM>(fineDistTree, comm, eleOrder);

    // Set Vector by a function
    double * fineVec;
    fineDA->template createVector<VECType>(fineVec,false,false,ndof);
    std::function<void(const double *, double *)> functionPointer = [&](const double *x, double *var) {
        double sum = 0.0;
        for (int d = 0; d < DIM; ++d)
            sum += x[d];
        var[0] = sum;
    };
    fineDA->setVectorByFunction(fineVec,functionPointer,false,false,ndof);

    // Refinement Flags (attempt to coarsen everything)
    std::vector<ot::OCT_FLAGS::Refine> octFlags(fineDistTree.getTreePartFiltered().size(), ot::OCT_FLAGS::Refine::OCT_COARSEN);
    ot::DistTree<unsigned int, DIM> coarseDistTree;
    ot::DistTree<unsigned int, DIM> surrDistTree;
    {
        std::vector<ot::TreeNode<DENDRITE_UINT, DIM>> coarseTree;
        std::vector<ot::TreeNode<DENDRITE_UINT, DIM>> surrTree;
        ot::SFC_Tree<DENDRITE_UINT , DIM>::distRemeshWholeDomain(fineDistTree.getTreePartFiltered(), octFlags, coarseTree, 0.3, comm);
        surrTree = ot::SFC_Tree<DENDRITE_UINT , DIM>::getSurrogateGrid(ot::RemeshPartition::SurrogateOutByIn, fineDistTree.getTreePartFiltered(), coarseTree, comm); // Line 156
        coarseDistTree = ot::DistTree<unsigned int, DIM>(coarseTree, comm);
        surrDistTree = ot::DistTree<unsigned int, DIM>(surrTree, comm,ot::DistTree<unsigned int,DIM>::NoCoalesce);
    }
    ot::DA<DIM> *coarseDA = new ot::DA<DIM>(coarseDistTree, comm, eleOrder);
    ot::DA<DIM> *surrDA = new ot::DA<DIM>(surrDistTree, comm, eleOrder);
    std::cout << "Number of elements in fineDA " << fineDA->getLocalElementSz() << "\n";
    std::cout << "Number of elements in coarseDA " << coarseDA->getLocalElementSz() << "\n";
    std::cout << "Number of elements in surrDA " << surrDA->getLocalElementSz() << "\n";

    double * coarseVec;
    coarseDA->template createVector<VECType>(coarseVec,false,false,ndof);

    intergrid_fine_to_coarse(fineVec, fineDA, fineDistTree, surrDA, surrDistTree, coarseVec, coarseDA, coarseDistTree, 1);

    checkIntergridTransfer(coarseVec, coarseDA, coarseDistTree, 1);

    delete coarseDA;
    delete fineDA;
    delete surrDA;
    delete coarseVec;
    delete fineVec;
    PetscFinalize();
}
