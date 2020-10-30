//
// Created by maksbh on 10/30/20.
//
#include <matrix.h>
#include <oda.h>
static constexpr unsigned int DIM = 3;
typedef ot::TreeNode<unsigned int, DIM> TREENODE;
void generateRefinementFlags(const ot::DA<DIM> *octDA, const std::vector<TREENODE> &treePart,
                             std::vector<ot::OCT_FLAGS::Refine> &refineFlags) {
    const size_t sz = octDA->getTotalNodalSz();
    auto partFront = octDA->getTreePartFront();
    auto partBack = octDA->getTreePartBack();
    const auto tnCoords = octDA->getTNCoords();
    const unsigned int eleOrder = octDA->getElementOrder();
    refineFlags.resize(octDA->getLocalElementSz());
    std::fill(refineFlags.begin(), refineFlags.end(), ot::OCT_FLAGS::Refine::OCT_NO_CHANGE);
    const unsigned int npe = octDA->getNumNodesPerElement();
    int counter = 0;

    ot::MatvecBaseCoords<DIM> loop(sz, eleOrder, false, 0, tnCoords, &(*treePart.cbegin()), treePart.size(), *partFront,
                                   *partBack);
    while (!loop.isFinished()) {
        if (loop.isPre() && loop.subtreeInfo().isLeaf()) {
            if (loop.subtreeInfo().isElementBoundary()) {
                refineFlags[counter] =  ot::OCT_FLAGS::Refine::OCT_REFINE;

            }
            counter++;
            loop.next();

        } else {
            loop.step();

        }
    }

}

int main(int argc, char *argv[]) {

    PetscInitialize(&argc, &argv, NULL, NULL);
    _InitializeHcurve(DIM);
    int eleOrder = 2;
    typedef unsigned int DENDRITE_UINT;
    std::vector<ot::TreeNode<DENDRITE_UINT, DIM>> newTree;
    std::vector<ot::TreeNode<DENDRITE_UINT, DIM>> surrTree;
    std::vector<ot::TreeNode<unsigned int, DIM>> treePart;
    ot::createRegularOctree(treePart, 3, MPI_COMM_WORLD);

    ot::DA<DIM> *newDA = new ot::DA<DIM>(treePart, MPI_COMM_WORLD, eleOrder);
    {
        std::vector<ot::OCT_FLAGS::Refine> octFlags;
        generateRefinementFlags(newDA, treePart, octFlags);
        ot::SFC_Tree<DENDRITE_UINT, DIM>::distRemeshWholeDomain(treePart, octFlags, newTree, surrTree, 0.3,
                                                                MPI_COMM_WORLD);
        ot::DA<DIM> *octDA = new ot::DA<DIM>(newTree, MPI_COMM_WORLD, eleOrder, 100, 0.3);
        std::swap(octDA, newDA);
        std::swap(newTree,treePart);
        delete octDA;
    }

    Vec funcVec;
    newDA->petscCreateVector(funcVec, false, false, 1);
    std::function<void(const double *, double *)> functionPointer = [&](const double *x, double *var) {
        var[0] = sin(M_PI * x[0]) * sin(M_PI * x[1]) * sin(M_PI * x[2]);
    };
    newDA->petscSetVectorByFunction(funcVec, functionPointer, false, false, 1);



    Mat J;

    Vec result, matVecResult;
    newDA->petscCreateVector(result, false, false, 1);
    newDA->petscCreateVector(matVecResult, false, false, 1);
    for(int counter = 14; counter < 15; counter++) {
        newDA->createMatrix(J, MATAIJ);
        matrix<DIM> mat(newDA, treePart, counter);
        mat.getAssembledMatrix(&J, MATAIJ);
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

        MatMult(J, funcVec, result);
        const double *value, *value1;
        VecGetArrayRead(result, &value);

        matrix<DIM> matVec(newDA, treePart, counter);
        VecSet(matVecResult, 0.0);
        matVec.matVec(funcVec, matVecResult);
        VecGetArrayRead(matVecResult, &value1);
        std::vector<double> maxDiff(newDA->getLocalNodalSz(), 0.0);
        for (int i = 0; i < newDA->getLocalNodalSz(); i++) {
            maxDiff[i] = fabs(value[i] - value1[i]);
            if (maxDiff[i] > 1E-5){
                std::cout << value[i] << " " << value1[i] << "\n";
            }
        }
        double diff = *std::max_element(maxDiff.begin(), maxDiff.end());
        if(diff > 1E-5){
            std::cout << "Failed for " << counter << " " << diff << "\n";
        }
        else{
            std::cout << "Passed for " << counter << "\n";
        }
        VecRestoreArrayRead(matVecResult, &value1);
        VecRestoreArrayRead(result, &value);
        MatDestroy(&J);
    }


//
    PetscFinalize();
}