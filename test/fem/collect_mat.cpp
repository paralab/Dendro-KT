#include <dollar.hpp>
#include "dollar_stat.h"

#include "treeNode.h"
#include "distTree.h"
#include "oda.h"
#include "octUtils.h"
#include "FEM/include/feMatrix.h"

#include <vector>
#include <array>
#include <petsc.h>

#include "test/octree/multisphere.h"

constexpr int DIM = 3;
using uint = unsigned int;
using DofT = double;

using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;
using DistTree = ot::DistTree<uint, DIM>;
using DA = ot::DA<DIM>;

// octlist()
const OctList & octlist(const DistTree &dtree) { return dtree.getTreePartFiltered(); }

// make_dist_tree()
ot::DistTree<uint, DIM> make_dist_tree(size_t grain, double sfc_tol, MPI_Comm comm);

// print_dollars()
void print_dollars(MPI_Comm comm);


class AllOnes : public feMatrix<AllOnes, DIM>
{
  private:
    int m_degree = 1;
    int m_npe = (1<<DIM);
  public:
    AllOnes(const DA *da, const OctList &octlist, int ndofs)
      : feMatrix<AllOnes, DIM>(da, &octlist, ndofs),
        m_degree(da->getElementOrder()),
        m_npe(da->getNumNodesPerElement())
    { }

    virtual void elementalMatVec(
        const VECType *in,
        VECType *out,
        unsigned int ndofs,
        const double *coords,
        double scale,
        bool isElementBoundary)
    {
      const VECType sum = std::accumulate(in, in + ndofs * m_npe, VECType(0));
      std::fill(out, out + ndofs * m_npe, sum);
    }

    void getElementalMatrix(
        std::vector<ot::MatRecord> &records,
        const double *coords,
        bool isElementBoundary)
    {
      const int ndofs = this->ndofs();
      const int npe = this->m_npe;

      int x = 0.0;
      for (int i = 0; i < npe; ++i)
        for (int id = 0; id < ndofs; ++id)
          for (int j = 0; j < npe; ++j)
            for (int jd = 0; jd < ndofs; ++jd)
              records.push_back(ot::MatRecord(i, j, id, jd, double(x++ & 255)));
    }
};


void usage(const char *prog)
{
  fprintf(stderr, "Usage: %s ndofs iterations save_mats\n", prog);
}


//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank, comm_size;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  const double sfc_tol = 0.1;
  const size_t grain = 4e4;
  const int degree = 1;

  if (argc < 4)
    return (usage(argv[0]), 1);
  const int ndofs = atol(argv[1]);
  const int iterations = atol(argv[2]);
  const bool save_mats = atol(argv[3]);

  if (comm_rank == 0)
    printf("ndofs=%d  iterations=%d\n", ndofs, iterations);

  DistTree dtree;
  DA *da;

  {DOLLAR("construct.octree")
    dtree = make_dist_tree(grain, sfc_tol, comm);
    /// dtree = DistTree::constructSubdomainDistTree(3, comm, sfc_tol);
  }
  {DOLLAR("construct.da")
    da = new DA(dtree, comm, degree, int{}, sfc_tol);
  }
  printf("[%d] e:%lu/%llu n:%lu/%llu\n",
      comm_rank,
      da->getLocalElementSz(), da->getGlobalElementSz(),
      da->getLocalNodalSz(), da->getGlobalNodeSz());

  AllOnes matrix(da, octlist(dtree), ndofs);
  Mat petsc_mat;
  da->createMatrix(petsc_mat, MATAIJ, ndofs);
  for (int iteration = 0; iteration < iterations; ++iteration)
    matrix.getAssembledMatrix(&petsc_mat, MATAIJ);

  MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);

  print_dollars(comm);

  if (save_mats)
  {
    PetscViewer viewer;
    PetscViewerASCIIOpen(
        comm,
        ("matrix_d" + std::to_string(ndofs) + "_i" + std::to_string(iterations)).c_str(),
        &viewer);
    MatView(petsc_mat, viewer);
  }

  _DestroyHcurve();
  DendroScopeEnd();
  PetscFinalize();
  return 0;
}


// make_dist_tree()
ot::DistTree<uint, DIM> make_dist_tree(size_t grain, double sfc_tol, MPI_Comm comm)
{
  test::SphereSet<DIM> sphereSet;
  sphereSet.carveSphere(0.10, {0.15, 0.6, 0.5});
  sphereSet.carveSphere(0.10, {0.7, 0.4, 0.5});
  sphereSet.carveSphere(0.210, {0.45, 0.55, 0.5});
  ot::DistTree<uint, DIM> distTree =
      ot::DistTree<uint, DIM>::minimalSubdomainDistTreeGrain(
          grain, sphereSet, comm, sfc_tol);
  return distTree;
}

// print_dollars()
void print_dollars(MPI_Comm comm)
{
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  dollar::DollarStat dollar_stat(comm);
  dollar::clear();
  dollar::DollarStat dollar_mean = dollar_stat.mpi_reduce_mean();
  if (comm_rank == 0)
  {
    std::cout << "\n" << "[Mean np=" << comm_size << "]\n";
    dollar_mean.text(std::cout);
  }
}
