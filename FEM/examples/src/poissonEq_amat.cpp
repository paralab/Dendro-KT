#include <iostream>
#include <mpi.h>
#include <stdio.h>

// Dendro
#include "dendro.h"
#include "oda.h"
#include "octUtils.h"
#include "point.h"
#include "poissonMat.h"
#include "poissonGMG.h"
// bug: if petsc is included (with logging)
//  before dendro parUtils then it triggers
//  dendro macros that use undefined variables.

#ifdef BUILD_WITH_PETSC
#include <petsc.h>
#endif

#include "aMat.hpp"
#include "aMatBased.hpp"
#include "aMatFree.hpp"
#include "aVec.hpp"
#include "constraintRecord.hpp"
#include "enums.hpp"
#include "fe_vector.hpp"
#include "integration.hpp"
#include "ke_matrix.hpp"
#include "maps.hpp"
#include "solve.hpp"

#include <Eigen/Dense>

using Eigen::Matrix;

constexpr int DIM = 2;




template <typename Functor, typename Iterator>
struct MapRange
{
  Functor m_functor;
  Iterator m_begin;
  Iterator m_end;

  template <typename InRange>
  MapRange(const Functor &functor, InRange &&in_range)
    : m_functor(functor),
      m_begin(std::begin(in_range)),
      m_end(std::end(in_range))
  {}

  struct Token
  {
    Iterator it;
    auto operator*() -> decltype(m_functor(*it)) { return m_functor(*it); }
    bool operator!=(const Token &that) const { return this->it != that.it; }
  };

  Token begin() const { return Token{m_begin}; }
  Token end() const { return Token{m_end}; }

  MapRange(const MapRange &) = default;
  MapRange(MapRange &&) = default;
  MapRange & operator=(const MapRange &) = default;
  MapRange & operator=(MapRange &&) = default;
};

template <typename Functor, typename InRange>
MapRange<Functor, InRange> map(
    const Functor &functor, InRange &&in_range)
{
  return MapRange<Functor, decltype(std::begin(in_range))>(
      functor, std::forward<InRange>(in_range));
}




class LocalIdx
{
  public:
    size_t m_idx;
    explicit LocalIdx(size_t idx) : m_idx(idx) {}
    operator size_t() const { return m_idx; }
};

class GhostedIdx
{
  public:
    size_t m_idx;
    explicit GhostedIdx(size_t idx) : m_idx(idx) {}
    operator size_t() const { return m_idx; }
};


template <int dim>
class AABB
{
  protected:
    Point<dim> m_min;
    Point<dim> m_max;

  public:
    AABB(const Point<dim> &min, const Point<dim> &max) : m_min(min), m_max(max) { }
    const Point<dim> & min() const { return m_min; }
    const Point<dim> & max() const { return m_max; }
};


//
// ConstMeshPointers  (wrapper)
//
template <unsigned int dim>
class ConstMeshPointers
{
  private:
    const ot::DistTree<unsigned int, dim> *m_distTree;
    const ot::MultiDA<dim> *m_multiDA;
    const unsigned m_stratum;

  public:
    // ConstMeshPointers constructor
    ConstMeshPointers(const ot::DistTree<unsigned int, dim> *distTree,
                      const ot::MultiDA<dim> *multiDA,
                      unsigned stratum = 0)
      : m_distTree(distTree), m_multiDA(multiDA), m_stratum(stratum)
    {}

    // Copy constructor and copy assignment.
    ConstMeshPointers(const ConstMeshPointers &other) = default;
    ConstMeshPointers & operator=(const ConstMeshPointers &other) = default;

    // distTree()
    const ot::DistTree<unsigned int, dim> * distTree() const { return m_distTree; }

    // numElements()
    size_t numElements() const
    {
      return m_distTree->getTreePartFiltered(m_stratum).size();
    }

    // da()
    const ot::DA<dim> * da() const { return &((*m_multiDA)[m_stratum]); }

    // multiDA()
    const ot::MultiDA<dim> * multiDA() const { return m_multiDA; }

    unsigned stratum() const { return m_stratum; }

    // printSummary()
    std::ostream & printSummary(std::ostream & out, const std::string &pre = "", const std::string &post = "") const;

    // --------------------------------------------------------------------

    // local2ghosted
    GhostedIdx local2ghosted(const LocalIdx &local) const
    {
      return GhostedIdx(da()->getLocalNodeBegin() + local);
    }

    std::array<double, dim> nodeCoord(const LocalIdx &local, const AABB<dim> &aabb) const
    {
      // Floating point coordinates in the unit cube.
      std::array<double, dim> coord;
      ot::treeNode2Physical( da()->getTNCoords()[local2ghosted(local)],
                             da()->getElementOrder(),
                             coord.data() );

      // Coordinates in the box represented by aabb.
      for (int d = 0; d < dim; ++d)
        coord[d] = coord[d] * (aabb.max().x(d) - aabb.min().x(d)) + aabb.min().x(d);

      return coord;
    }

    std::array<double, dim> nodeCoord(const GhostedIdx &ghosted, const AABB<dim> &aabb) const
    {
      // Floating point coordinates in the unit cube.
      std::array<double, dim> coord;
      ot::treeNode2Physical( da()->getTNCoords()[ghosted],
                             da()->getElementOrder(),
                             coord.data() );

      // Coordinates in the box represented by aabb.
      for (int d = 0; d < dim; ++d)
        coord[d] = coord[d] * (aabb.max().x(d) - aabb.min().x(d)) + aabb.min().x(d);

      return coord;
    }
};

//
// Vector  (wrapper)
//
template <typename ValT>
class Vector
{
  private:
    std::vector<ValT> m_data;
    bool m_isGhosted = false;
    size_t m_ndofs = 1;
    unsigned long long m_globalNodeRankBegin = 0;

  protected:
    Vector(size_t size, bool isGhosted, size_t ndofs, unsigned long long globalRankBegin)
      : m_data(size), m_isGhosted(isGhosted), m_ndofs(ndofs), m_globalNodeRankBegin(globalRankBegin)
    { }

  public:
    // Vector constructor
    template <unsigned int dim>
    Vector(const ConstMeshPointers<dim> &mesh, bool isGhosted, size_t ndofs, const ValT *input = nullptr)
    : m_isGhosted(isGhosted), m_ndofs(ndofs)
    {
      mesh.da()->createVector(m_data, false, isGhosted, ndofs);
      if (input != nullptr)
        std::copy_n(input, m_data.size(), m_data.begin());
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

template <typename ValT>
class LocalVector : public Vector<ValT>
{
  protected:
    LocalVector(size_t size, size_t ndofs, unsigned long long globalRankBegin)
      : Vector<ValT>::Vector(size, false, ndofs, globalRankBegin)
    { }

  public:
    template <unsigned int dim>
    LocalVector(const ConstMeshPointers<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
    : Vector<ValT>(mesh, false, ndofs, input)
    {}

    ValT & operator()(const LocalIdx &local, size_t dof = 0) {
      return this->ptr()[local * this->ndofs() + dof];
    }
    const ValT & operator()(const LocalIdx &local, size_t dof = 0) const {
      return this->ptr()[local * this->ndofs() + dof];
    }

    ValT & operator[](const LocalIdx &local) { return (*this)(local, 0); }
    const ValT & operator[](const LocalIdx &local) const { return (*this)(local, 0); }

    LocalVector operator+(const LocalVector &b) const {
      const LocalVector &a = *this;
      LocalVector c(this->size(), this->ndofs(), this->globalRankBegin());
      assert(sizes_match(a, b));
      for (size_t i = 0; i < this->size(); ++i)
        c[LocalIdx(i)] = a[LocalIdx(i)] + b[LocalIdx(i)];
      return c;
    }

    LocalVector operator-(const LocalVector &b) const {
      const LocalVector &a = *this;
      LocalVector c(this->size(), this->ndofs(), this->globalRankBegin());
      assert(sizes_match(a, b));
      for (size_t i = 0; i < this->size(); ++i)
        c[LocalIdx(i)] = a[LocalIdx(i)] - b[LocalIdx(i)];
      return c;
    }

    static bool sizes_match(const LocalVector &a, const LocalVector &b)
    {
      return a.size() == b.size();
    }
};

template <typename ValT>
class GhostedVector : public Vector<ValT>
{
  public:
    template <unsigned int dim>
    GhostedVector(const ConstMeshPointers<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
    : Vector<ValT>(mesh, true, ndofs, input)
    {}

    ValT & operator[](const GhostedIdx &ghosted) { return this->ptr()[ghosted]; }
    const ValT & operator[](const GhostedIdx &ghosted) const { return this->ptr()[ghosted]; }
};


//
// PetscVector  (wrapper)
//
template <typename ValT>
class PetscVector
{
  private:
    Vec m_data;
    size_t m_size;
    bool m_isGhosted;
    size_t m_ndofs = 1;
    unsigned long long m_globalNodeRankBegin = 0;

  public:
    // PetscVector constructor
    template <unsigned int dim>
    PetscVector(const ConstMeshPointers<dim> &mesh, bool isGhosted, size_t ndofs, const ValT *input = nullptr)
    : m_size(
        isGhosted ?
        mesh.da()->getTotalNodalSz() * ndofs : mesh.da()->getLocalNodalSz() * ndofs),
      m_isGhosted(isGhosted),
      m_ndofs(ndofs)
    {
      mesh.da()->petscCreateVector(m_data, false, isGhosted, ndofs);
      if (input != nullptr)
      {
        const size_t localSz = ndofs * mesh.da()->getLocalNodalSz();
        const size_t localBegin = ndofs * mesh.da()->getLocalNodeBegin();
        if (isGhosted)
          input += localBegin;

        ValT *array;
        VecGetArray(m_data, &array);
        std::copy_n(input, localSz, array);
        VecRestoreArray(m_data, &array);
      }
      m_globalNodeRankBegin = mesh.da()->getGlobalRankBegin();
    }

    // vec()
    Vec vec() { return m_data; }
    const Vec vec() const { return m_data; }  // doesn't really enforce const

    // isGhosted()
    bool isGhosted() const { return m_isGhosted; }

    // ndofs()
    size_t ndofs() const { return m_ndofs; }

    // size()
    size_t size() const { return m_size; }

    // globalRankBegin();
    unsigned long long globalRankBegin() const { return m_globalNodeRankBegin; }
};

template <typename ValT>
class LocalPetscVector : public PetscVector<ValT>
{
  public:
    template <unsigned int dim>
    LocalPetscVector(const ConstMeshPointers<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
    : PetscVector<ValT>(mesh, false, ndofs, input)
    {}
};


template <typename ValT>
void copyToPetscVector(LocalPetscVector<ValT> &petscVec, const LocalVector<ValT> &vector)
{
  ValT *array;
  VecGetArray(petscVec.vec(), &array);
  std::copy_n(vector.ptr(), petscVec.size(), array);
  VecRestoreArray(petscVec.vec(), &array);
}

template <typename ValT>
void copyFromPetscVector(LocalVector<ValT> &vector, const LocalPetscVector<ValT> &petscVec)
{
  ValT *array;
  VecGetArray(petscVec.vec(), &array);
  std::copy_n(array, vector.size(), vector.ptr());
  VecRestoreArray(petscVec.vec(), &array);
}


template <unsigned int dim, typename NodeT>
std::ostream & print(const ConstMeshPointers<dim> &mesh,
                     const LocalVector<NodeT> &vec,
                     std::ostream & out = std::cout);


//
// ElementLoopIn  (wrapper)
//
template <unsigned int dim, typename ValT>
class ElementLoopIn
{
  private:
    ot::MatvecBaseIn<dim, ValT> m_loop;
  public:
    ElementLoopIn(const ConstMeshPointers<dim> &mesh,
                  const Vector<ValT> &ghostedVec);
    ot::MatvecBaseIn<dim, ValT> & loop() { return m_loop; }
};

//
// ElementLoopOut  (wrapper)
//
template <unsigned int dim, typename ValT>
class ElementLoopOut
{
  private:
    ot::MatvecBaseOut<dim, ValT, true> m_loop;
  public:
    ElementLoopOut(const ConstMeshPointers<dim> &mesh, unsigned int ndofs);
    ot::MatvecBaseOut<dim, ValT, true> & loop() { return m_loop; }
};

//
// ElementLoopOutOverwrite  (wrapper)
//
template <unsigned int dim, typename ValT>
class ElementLoopOutOverwrite
{
  private:
    ot::MatvecBaseOut<dim, ValT, false> m_loop;
  public:
    ElementLoopOutOverwrite(const ConstMeshPointers<dim> &mesh, unsigned int ndofs);
    ot::MatvecBaseOut<dim, ValT, false> & loop() { return m_loop; }
};


struct AMatWrapper
{
  typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>,
                    double, unsigned long, unsigned int>  aMatFree;
  typedef par::Maps<double, unsigned long, unsigned int>  aMatMaps;

  aMatFree* stMatFree = NULL;
  aMatMaps* meshMaps = NULL;

  // ------------------------------------

  template <unsigned int dim>
  AMatWrapper(const ConstMeshPointers<dim> &mesh,
              PoissonEq::PoissonMat<dim> &poissonMat,
              const double * boundaryValues,
              int ndofs = 1)
  {
    mesh.da()->allocAMatMaps(meshMaps, mesh.distTree()->getTreePartFiltered(), boundaryValues, ndofs);
    mesh.da()->createAMat(stMatFree, meshMaps);
    stMatFree->set_matfree_type((par::MATFREE_TYPE)1);
    poissonMat.getAssembledAMat(stMatFree);
    stMatFree->finalize();
  }

  ~AMatWrapper()
  {
    // TODO DA destroy amat stuff???
  }
};


template <unsigned int dim>
struct Equation
{
  public:
    typedef void * DistTreeAddrT;
    typedef int StratumT;
    typedef std::pair<DistTreeAddrT, StratumT> Key;

  protected:
    Point<dim> m_min;
    Point<dim> m_max;

  public:
    Equation(const AABB<dim> &aabb) : m_min(aabb.min()), m_max(aabb.max()) { }

    template <typename ValT>
    void matvec(const ConstMeshPointers<dim> &mesh, const LocalVector<ValT> &in, LocalVector<ValT> &out);

    template <typename ValT>
    void rhsvec(const ConstMeshPointers<dim> &mesh, const LocalVector<ValT> &in, LocalVector<ValT> &out);

    template <typename ValT>
    void assembleDiag(const ConstMeshPointers<dim> &mesh, LocalVector<ValT> &diag_out);

    void dirichlet(const ConstMeshPointers<dim> &mesh, const double *prescribed_vals);

    // TODO make private again...
    AMatWrapper & atOrInsertAMat(const ConstMeshPointers<dim> &mesh) const;

  private:
    //temporary
    static int ndofs() { return 1; }

    const Key key(const ConstMeshPointers<dim> &mesh) const;
    PoissonEq::PoissonMat<dim> & atOrInsertPoissonMat(
        const ConstMeshPointers<dim> &mesh) const;
    PoissonEq::PoissonVec<dim> & atOrInsertPoissonVec(
        const ConstMeshPointers<dim> &mesh) const;

    std::vector<double> & atOrInsertDirichletVec(
        const ConstMeshPointers<dim> &mesh) const;

    mutable std::map<Key, PoissonEq::PoissonMat<dim>> m_poissonMats;
    mutable std::map<Key, PoissonEq::PoissonVec<dim>> m_poissonVecs;
    mutable std::map<Key, std::shared_ptr<AMatWrapper>> m_amats;
    mutable std::map<Key, std::vector<double>> m_dirichletVecs;
};


template <typename T>
class ReduceSeq
{
  private:
    bool m_nonempty = false;
    T m_min = 0;
    T m_max = 0;
    T m_sum = 0;
    T m_sum2 = 0;

    void first(const T & val)
    {
      m_min = val;
      m_max = val;
      m_sum = val;
      m_sum2 = val * val;
    }

    void next(const T & val)
    {
      if (val < m_min)  m_min = val;
      if (val > m_max)  m_max = val;
      m_sum += val;
      m_sum2 += val*val;
    }

  public:
    ReduceSeq()
    {}

    void include(const T & val)
    {
      if (m_nonempty)
        this->next(val);
      else
      {
        this->first(val);
        this->m_nonempty = true;
      }
    }

    const T & min() const { return m_min; }
    const T & max() const { return m_max; }
    const T & sum() const { return m_sum; }
    const T & sum2() const { return m_sum2; }
};


template <typename C, unsigned int dim>
std::vector<bool> boundaryFlags(
    const ot::DistTree<C, dim> &distTree,
    const int stratum,
    const ot::DA<dim> &da);




//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = PETSC_COMM_WORLD;
  const int eleOrder = 1;
  const unsigned int ndofs = 1;
  const double sfc_tol = 0.3;
  using uint = unsigned int;
  using DTree_t = ot::DistTree<uint, DIM>;
  using DA_t = ot::DA<DIM>;
  using Mesh_t = ConstMeshPointers<DIM>;

  const uint fineLevel = 3;
  const size_t dummyInt = 100;
  const size_t singleDof = 1;

  enum Method { matrixFreeJacobi, aMatAssembly, hybridJacobi };
  const Method method = matrixFreeJacobi;

  // Mesh
  DTree_t dtree = DTree_t::constructSubdomainDistTree(
      fineLevel, comm, sfc_tol);
  std::vector<DA_t> das(1);
  das[0].constructStratum(dtree, 0, comm, eleOrder, dummyInt, sfc_tol);
  Mesh_t mesh(&dtree, &das, 0);
  printf("localElements=%lu\n", mesh.numElements());
  /// printf("boundaryNodes=%lu\n", mesh.da()->getBoundaryNodeIndices().size());

  // Indicate boundary elements should be 'explicit'.
  if (method == hybridJacobi)
  {
    std::vector<bool> bdryFlags = boundaryFlags(dtree, 0, das[0]);
    das[0].explicitFlags(bdryFlags);
    /// das[0].explicitFlags(std::vector<bool>(mesh.numElements(), false));  // ignore
  }

  AABB<DIM> bounds(Point<DIM>(-1.0), Point<DIM>(1.0));

  // Vector
  LocalVector<double> u_vec(mesh, singleDof);
  LocalVector<double> v_vec(mesh, singleDof);
  LocalVector<double> f_vec(mesh, singleDof);
  LocalVector<double> rhs_vec(mesh, singleDof);

  const double coefficient[] = {1, 2, 5, 3};
  const double sum_coeff = std::accumulate(coefficient, coefficient + DIM, 0);

  // u_exact function
  const auto u_exact = [=] (const double *x) {
    double expression = 1;
    for (int d = 0; d < DIM; ++d)
      expression += coefficient[d] * x[d] * x[d];
    return expression;
  };
  // ... is the solution to -div(grad(u)) = f, where f is
  const auto f = [=] (const double *x) {
    return -2*sum_coeff;
  };
  // ... and boundary is prescribed (matching u_exact)
  const auto u_bdry = [=] (const double *x) {
    return u_exact(x);
  };

  // Initialize  u=Dirichlet(0)  and  f={f function}
  for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
    u_vec[LocalIdx(ii)] = ((ii % 43) * (ii % 97)) % 10 + 1;  // arbitrary func
  for (size_t bdyIdx : mesh.da()->getBoundaryNodeIndices())
    u_vec[LocalIdx(bdyIdx)] = u_bdry(mesh.nodeCoord(LocalIdx(bdyIdx), bounds).data());
  for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
    f_vec[LocalIdx(ii)] = f(mesh.nodeCoord(LocalIdx(ii), bounds).data());

  /// print(mesh, u_vec);  // 2D grid of values in the terminal

  Equation<DIM> equation(bounds);

  // Set dirichlet before setting up other matrix abstractions
  std::vector<double> prescribed_bdry(mesh.da()->getBoundaryNodeIndices().size());
  for (size_t bii = 0; bii < mesh.da()->getBoundaryNodeIndices().size(); ++bii)
  {
    size_t bdyIdx = mesh.da()->getBoundaryNodeIndices()[bii];
    prescribed_bdry[bii] = u_bdry(mesh.nodeCoord(LocalIdx(bdyIdx), bounds).data());
  }
  equation.dirichlet(mesh, prescribed_bdry.data());

  // Compute r.h.s. of weak formulation.
  equation.rhsvec(mesh, f_vec, rhs_vec);

  // for comparison
  LocalVector<double> u_exact_vec(mesh, singleDof);
  for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
    u_exact_vec.ptr()[ii] = u_exact(mesh.nodeCoord(LocalIdx(ii), bounds).data());

  // Function to check solution error.
  const auto sol_err_max = [&]() {
      double err_max = 0.0;
      for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
      {
        const double err_ii = abs(
            u_vec[LocalIdx(ii)] -
            u_exact_vec[LocalIdx(ii)] );
        err_max = fmax(err_max, err_ii);
      }
      return err_max;
  };

  const double tol=1e-12;
  const unsigned int max_iter=1500;

  switch (method)
  {
    case matrixFreeJacobi:
    {
      // Jacobi method:
      LocalVector<double> diag_vec(mesh, singleDof);
      equation.assembleDiag(mesh, diag_vec);

      fprintf(stdout, "[%3d] solution err_max==%e\n", 0, sol_err_max());
      double check_res = std::numeric_limits<double>::infinity();
      for (int iter = 0; iter < max_iter && check_res > tol; ++iter)
      {
        ReduceSeq<double> iter_diff;
        ReduceSeq<double> residual;

        // matvec: overwrites v = Au
        equation.matvec(mesh, u_vec, v_vec);

        // Jacobi update: x -= D^-1 (Ax-b)
        for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
        {
          const LocalIdx lii(ii);
          const double res = (v_vec[lii] - rhs_vec[lii]);
          const double update = (v_vec[lii] - rhs_vec[lii]) / diag_vec[lii];
          u_vec[lii] -= update;
          iter_diff.include(fabs(update));
          residual.include(fabs(res));
        }

        // Check solution error
        if ((iter + 1) % 50 == 0 || (iter + 1 == 1))
        {
          fprintf(stdout, "[%3d] solution err_max==%e", iter+1, sol_err_max());
          /// fprintf(stdout, "\n");
          fprintf(stdout, "\t\t max_change==%e", iter_diff.max());
          /// fprintf(stdout, "\n");
          fprintf(stdout, "\t res==%e", residual.max());
          fprintf(stdout, "\n");

          check_res = residual.max();
        }
      }
      break;
     }

    case aMatAssembly:
    {
      const AMatWrapper & amatWrapper = equation.atOrInsertAMat(mesh);
      Mat petscMat = amatWrapper.stMatFree->get_matrix();

      // PETSc solver context: Create and set KSP
      KSP ksp;
      KSPCreate(comm, &ksp);
      KSPSetType(ksp, KSPCG);
      KSPSetFromOptions(ksp);

      // Set tolerances.
      KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_iter);

      // Set operators.
      KSPSetOperators(ksp, petscMat, petscMat);

      // Set preconditioner.
      PC pc;
      KSPGetPC(ksp, &pc);
      PCSetType(pc, PCJACOBI);
      PCSetFromOptions(pc);

      // Compensate r.h.s. of weak formulation. Amat subtracts boundary from rhs.
      LocalPetscVector<double> Mfrhs(mesh, ndofs, rhs_vec.ptr());
      amatWrapper.stMatFree->apply_bc(Mfrhs.vec());

      // Copy vectors to Petsc vectors.
      LocalPetscVector<double> ux(mesh, ndofs, u_vec.ptr());
      /// print(mesh, u_vec);  // 2D grid of values in the terminal

      // Solve the system.
      KSPSolve(ksp, Mfrhs.vec(), ux.vec());

      PetscInt numIterations;
      KSPGetIterationNumber(ksp, &numIterations);

      // Copy back from Petsc vector.
      copyFromPetscVector(u_vec, ux);
      fprintf(stdout, "[%3d] solution err_max==%e\n", numIterations, sol_err_max());

      KSPDestroy(&ksp);
      break;
    }

    case hybridJacobi:
    {
      // [ ] equation.hybrid_matvec()
      // [ ] equation.hybrid_assembleDiag()
      // [ ] equation.hybrid_rhsvec()
      // [ ] equation.hybrid_dirichlet()

      // Jacobi method:
      LocalVector<double> diag_vec(mesh, singleDof);
      equation.assembleDiag(mesh, diag_vec);

      fprintf(stdout, "[%3d] solution err_max==%e\n", 0, sol_err_max());
      double check_res = std::numeric_limits<double>::infinity();
      for (int iter = 0; iter < max_iter && check_res > tol; ++iter)
      {
        ReduceSeq<double> iter_diff;
        ReduceSeq<double> residual;

        // matvec: overwrites v = Au
        equation.matvec(mesh, u_vec, v_vec);

        // Jacobi update: x -= D^-1 (Ax-b)
        for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
        {
          const LocalIdx lii(ii);
          const double res = (v_vec[lii] - rhs_vec[lii]);
          const double update = (v_vec[lii] - rhs_vec[lii]) / diag_vec[lii];
          u_vec[lii] -= update;
          iter_diff.include(fabs(update));
          residual.include(fabs(res));
        }

        // Check solution error
        if ((iter + 1) % 50 == 0 || (iter + 1 == 1))
        {
          fprintf(stdout, "[%3d] solution err_max==%e", iter+1, sol_err_max());
          /// fprintf(stdout, "\n");
          fprintf(stdout, "\t\t max_change==%e", iter_diff.max());
          /// fprintf(stdout, "\n");
          fprintf(stdout, "\t res==%e", residual.max());
          fprintf(stdout, "\n");

          check_res = residual.max();
        }
      }
      break;
    }
  }

  /// print(mesh, u_vec);  // 2D grid of values in the terminal

  /// printf("\n-------------------------------------------------------\n");
  /// print(mesh, u_vec - u_exact_vec);  // 2D grid of values in the terminal

  DendroScopeEnd();
  PetscFinalize();
}


template <typename C, unsigned int dim>
std::vector<bool> boundaryFlags(
    const ot::DistTree<C, dim> &distTree,
    const int stratum,
    const ot::DA<dim> &da)
{
  const std::vector<ot::TreeNode<C, dim>> &elements =
      distTree.getTreePartFiltered(stratum);
  const size_t size = elements.size();

  std::vector<bool> boundaryFlags(size, false);

  for (size_t ii = 0; ii < size; ++ii)
    boundaryFlags[ii] = elements[ii].getIsOnTreeBdry();

  return boundaryFlags;
}



template <unsigned int dim>
const typename Equation<dim>::Key Equation<dim>::key(const ConstMeshPointers<dim> &mesh) const
{
  return Key{DistTreeAddrT(mesh.distTree()), StratumT(mesh.stratum())};
}

template <unsigned int dim>
PoissonEq::PoissonMat<dim> & Equation<dim>::atOrInsertPoissonMat(
    const ConstMeshPointers<dim> &mesh) const
{
  const std::vector<double> &dirichletVec = this->atOrInsertDirichletVec(mesh);

  const Key key = this->key(mesh);
  if (m_poissonMats.find(key) == m_poissonMats.end())
  {
    m_poissonMats.insert(
        std::make_pair(key,
        PoissonEq::PoissonMat<dim>(
          mesh.da(),
          &mesh.distTree()->getTreePartFiltered(),
          this->ndofs(),
          dirichletVec.data())));
    m_poissonMats.at(key).setProblemDimensions(m_min, m_max);
  }
  return m_poissonMats.at(key);
}

template <unsigned int dim>
PoissonEq::PoissonVec<dim> & Equation<dim>::atOrInsertPoissonVec(
    const ConstMeshPointers<dim> &mesh) const
{
  const Key key = this->key(mesh);
  if (m_poissonVecs.find(key) == m_poissonVecs.end())
  {
    m_poissonVecs.insert(
        std::make_pair(key,
        PoissonEq::PoissonVec<dim>(
          const_cast<ot::DA<dim> *>(mesh.da()),   // violate const for weird feVec non-const necessity
          &mesh.distTree()->getTreePartFiltered(),
          this->ndofs())));
    m_poissonVecs.at(key).setProblemDimensions(m_min, m_max);
  }
  return m_poissonVecs.at(key);
}

template <unsigned int dim>
AMatWrapper & Equation<dim>::atOrInsertAMat(const ConstMeshPointers<dim> &mesh) const
{
  // Dirichlet boundary conditions, assume dirichlet() already called, else 0.
  std::vector<double> &dirichletVec = this->atOrInsertDirichletVec(mesh);

  const Key key = this->key(mesh);
  if (m_amats.find(key) == m_amats.end())
  {
    m_amats.insert(
        std::make_pair(key, std::make_shared<AMatWrapper>(
            mesh, this->atOrInsertPoissonMat(mesh), dirichletVec.data(), this->ndofs())) );
  }
  return *m_amats.at(key);
}

template <unsigned int dim>
std::vector<double> & Equation<dim>::atOrInsertDirichletVec(
        const ConstMeshPointers<dim> &mesh) const
{
  const Key key = this->key(mesh);
  if (m_dirichletVecs.find(key) == m_dirichletVecs.end())
  {
    m_dirichletVecs.insert(std::make_pair(key, std::vector<double>()));
  }

  const size_t numBdryNodes = mesh.da()->getBoundaryNodeIndices().size();
  m_dirichletVecs.at(key).resize(numBdryNodes * this->ndofs(), 0.0);
  return m_dirichletVecs.at(key);
}


template <unsigned int dim>
template <typename ValT>
void Equation<dim>::matvec(
    const ConstMeshPointers<dim> &mesh,
    const LocalVector<ValT> &in,
    LocalVector<ValT> &out)
{
  if (in.ndofs() != this->ndofs())
    throw std::logic_error("ndofs not supported");

  PoissonEq::PoissonMat<dim> &poissonMat =
      this->atOrInsertPoissonMat(mesh);

  poissonMat.matVec(in.ptr(), out.ptr());
}


template <unsigned int dim>
template <typename ValT>
void Equation<dim>::rhsvec(
    const ConstMeshPointers<dim> &mesh,
    const LocalVector<ValT> &in,
    LocalVector<ValT> &out)
{
  if (in.ndofs() != this->ndofs())
    throw std::logic_error("ndofs not supported");

  PoissonEq::PoissonVec<dim> &poissonVec =
      this->atOrInsertPoissonVec(mesh);

  poissonVec.computeVec(in.ptr(), out.ptr());
}


template <unsigned int dim>
template <typename ValT>
void Equation<dim>::assembleDiag(const ConstMeshPointers<dim> &mesh, LocalVector<ValT> &diag_out)
{
  if (diag_out.ndofs() != this->ndofs())
    throw std::logic_error("ndofs not supported");

  PoissonEq::PoissonMat<dim> &poissonMat =
      this->atOrInsertPoissonMat(mesh);

  poissonMat.setDiag(diag_out.ptr());
}

template <unsigned int dim>
void Equation<dim>::dirichlet(const ConstMeshPointers<dim> &mesh, const double *prescribed_vals)
{
  std::vector<double> &dirichletVec = this->atOrInsertDirichletVec(mesh);

  const size_t numBdryNodes = mesh.da()->getBoundaryNodeIndices().size();
  for (size_t bvii = 0; bvii < numBdryNodes * this->ndofs(); ++bvii)
    dirichletVec[bvii] = prescribed_vals[bvii];

  // Note that dirichlet() should be called before making PoissonMat and aMat.
}



template <unsigned int dim, typename NodeT>
std::ostream & print(const ConstMeshPointers<dim> &mesh,
                     const LocalVector<NodeT> &vec,
                     std::ostream & out)
{
  const ot::DA<dim> *da = mesh.da();
  return ot::printNodes(
      da->getTNCoords() + da->getLocalNodeBegin(),
      da->getTNCoords() + da->getLocalNodeBegin() + da->getLocalNodalSz(),
      vec.ptr(),
      da->getElementOrder(),
      out);
}







// ElementLoopIn::ElementLoopIn()
template <unsigned int dim, typename ValT>
ElementLoopIn<dim, ValT>::ElementLoopIn(
    const ConstMeshPointers<dim> &mesh,
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
           mesh.distTree()->getTreePartFiltered().size())
{ }

// ElementLoopOut::ElementLoopOut()
template <unsigned int dim, typename ValT>
ElementLoopOut<dim, ValT>::ElementLoopOut(
    const ConstMeshPointers<dim> &mesh, unsigned int ndofs)
  :
    m_loop(mesh.da()->getTotalNodalSz(),
           ndofs,
           mesh.da()->getElementOrder(),
           false,
           0,
           mesh.da()->getTNCoords(),
           mesh.distTree()->getTreePartFiltered().data(),
           mesh.distTree()->getTreePartFiltered().size())
{ }

// ElementLoopOutOverwrite::ElementLoopOutOverwrite()
template <unsigned int dim, typename ValT>
ElementLoopOutOverwrite<dim, ValT>::ElementLoopOutOverwrite(
    const ConstMeshPointers<dim> &mesh, unsigned int ndofs)
  :
    m_loop(mesh.da()->getTotalNodalSz(),
           ndofs,
           mesh.da()->getElementOrder(),
           false,
           0,
           mesh.da()->getTNCoords(),
           mesh.distTree()->getTreePartFiltered().data(),
           mesh.distTree()->getTreePartFiltered().size())
{ }




