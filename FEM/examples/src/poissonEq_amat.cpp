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
    bool m_isGhosted;
    size_t m_ndofs = 1;
    unsigned long long m_globalNodeRankBegin = 0;

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
  public:
    template <unsigned int dim>
    LocalVector(const ConstMeshPointers<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
    : Vector<ValT>(mesh, false, ndofs, input)
    {}

    ValT & operator[](const LocalIdx &local) { return this->data()[local]; }
    const ValT & operator[](const LocalIdx &local) const { return this->data()[local]; }
};

template <typename ValT>
class GhostedVector : public Vector<ValT>
{
  public:
    template <unsigned int dim>
    GhostedVector(const ConstMeshPointers<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
    : Vector<ValT>(mesh, true, ndofs, input)
    {}

    ValT & operator[](const GhostedIdx &ghosted) { return this->data()[ghosted]; }
    const ValT & operator[](const GhostedIdx &ghosted) const { return this->data()[ghosted]; }
};


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

    std::map<Key, PoissonEq::PoissonMat<dim>> m_poissonMats;
    std::map<Key, PoissonEq::PoissonVec<dim>> m_poissonVecs;

  public:
    Equation(const AABB<dim> &aabb) : m_min(aabb.min()), m_max(aabb.max()) { }

    template <typename ValT>
    void matvec(const ConstMeshPointers<dim> &mesh, const LocalVector<ValT> &in, LocalVector<ValT> &out);

    template <typename ValT>
    void rhsvec(const ConstMeshPointers<dim> &mesh, const LocalVector<ValT> &in, LocalVector<ValT> &out);

    template <typename ValT>
    void assembleDiag(const ConstMeshPointers<dim> &mesh, LocalVector<ValT> &diag_out);

  private:
    const Key key(const ConstMeshPointers<dim> &mesh);
};



//
// main()
//
int main(int argc, char * argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  /// DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = PETSC_COMM_WORLD;
  const int eleOrder = 1;
  const unsigned int ndofs = 1;
  const double sfc_tol = 0.3;
  using uint = unsigned int;
  using DTree_t = ot::DistTree<uint, DIM>;
  using DA_t = ot::DA<DIM>;
  using Mesh_t = ConstMeshPointers<DIM>;

  const uint fineLevel = 5;
  const size_t dummyInt = 100;
  const size_t singleDof = 1;

  // Mesh
  DTree_t dtree = DTree_t::constructSubdomainDistTree(
      fineLevel, comm, sfc_tol);
  std::vector<DA_t> das(1);
  das[0].constructStratum(dtree, 0, comm, eleOrder, dummyInt, sfc_tol);
  Mesh_t mesh(&dtree, &das, 0);
  printf("localElements=%lu\n", mesh.numElements());
  /// printf("boundaryNodes=%lu\n", mesh.da()->getBoundaryNodeIndices().size());

  AABB<DIM> bounds(Point<DIM>(-1.0), Point<DIM>(1.0));

  // Vector
  LocalVector<double> u_vec(mesh, singleDof);
  LocalVector<double> v_vec(mesh, singleDof);
  LocalVector<double> f_vec(mesh, singleDof);
  LocalVector<double> rhs_vec(mesh, singleDof);

  // u_exact function  : product of sines
  const auto u_exact = [&] (const double *x) {
    double result = 1.0;
    for (int d = 0; d < DIM; ++d)
      result *= sin(2 * M_PI * x[d]);
    return result;
  };
  // ... is the solution to div(grad(u)) = f, where f is
  const auto f = [&] (const double *x) {
    return 4 * M_PI * M_PI * DIM * u_exact(x);
  };

  // Initialize  u=Dirichlet(0)  and  f={f function}
  for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
    u_vec[LocalIdx(ii)] = ((ii % 43) * (ii % 97)) % 10;
  for (size_t bdyIdx : mesh.da()->getBoundaryNodeIndices())
    u_vec.data()[bdyIdx] = 0.0;
  for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
    f_vec.data()[ii] = f(mesh.nodeCoord(LocalIdx(ii), bounds).data());

  Equation<DIM> equation(bounds);

  // Compute r.h.s. of weak formulation.
  equation.rhsvec(mesh, f_vec, rhs_vec);

  LocalVector<double> u_exact_vec(mesh, singleDof);
  for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
    u_exact_vec.data()[ii] = u_exact(mesh.nodeCoord(LocalIdx(ii), bounds).data());

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


  // Jacobi method:
  const int iter_max = 1500;
  LocalVector<double> diag_vec(mesh, singleDof);
  equation.assembleDiag(mesh, diag_vec);

  fprintf(stdout, "[%3d] solution err_max==%e\n", 0, sol_err_max());
  for (int iter = 0; iter < iter_max; ++iter)
  {
    // matvec: overwrites v = Au
    equation.matvec(mesh, u_vec, v_vec);

    // Jacobi update: x -= D^-1 (Ax-b)
    for (size_t ii = 0; ii < mesh.da()->getLocalNodalSz(); ++ii)
    {
      const LocalIdx lii(ii);
      u_vec[lii] -= (v_vec[lii] - rhs_vec[lii]) / diag_vec[lii];
    }

    // Restore boundary condition
    for (size_t bdyIdx : mesh.da()->getBoundaryNodeIndices())
      u_vec[LocalIdx(bdyIdx)] = 0.0;

    // Check solution error
    if ((iter + 1) % 50 == 0)
      fprintf(stdout, "[%3d] solution err_max==%e\n", iter+1, sol_err_max());
  }

  /// print(mesh, u_vec);  // 2D grid of values in the terminal

  _DestroyHcurve();
  /// DendroScopeEnd();
  PetscFinalize();
}




template <unsigned int dim>
const typename Equation<dim>::Key Equation<dim>::key(const ConstMeshPointers<dim> &mesh)
{
  return Key{DistTreeAddrT(mesh.distTree()), StratumT(mesh.stratum())};
}


template <unsigned int dim>
template <typename ValT>
void Equation<dim>::matvec(
    const ConstMeshPointers<dim> &mesh,
    const LocalVector<ValT> &in,
    LocalVector<ValT> &out)
{
  // TODO replace the feMatrix structure with something
  // easier to read, partially implement, and extend
  // Discretized Poisson operator. TODO

  const size_t ndofs = in.ndofs();
  /// static GhostedVector<ValT> in_ghosted(mesh, ndofs), out_ghosted(mesh, ndofs);

  const Key key = this->key(mesh);
  if (m_poissonMats.find(key) == m_poissonMats.end())
  {
    m_poissonMats.insert(
        std::make_pair(key,
        PoissonEq::PoissonMat<dim>(mesh.da(), &mesh.distTree()->getTreePartFiltered(), ndofs)));
    m_poissonMats.at(key).setProblemDimensions(m_min, m_max);
  }

  m_poissonMats.at(key).matVec(in.data().data(), out.data().data());
}


template <unsigned int dim>
template <typename ValT>
void Equation<dim>::rhsvec(
    const ConstMeshPointers<dim> &mesh,
    const LocalVector<ValT> &in,
    LocalVector<ValT> &out)
{
  // TODO replace the feVector structure with something
  // easier to read, partially implement, and extend
  // Discretized Poisson rhs operator. TODO

  const size_t ndofs = in.ndofs();

  const Key key = this->key(mesh);
  if (m_poissonVecs.find(key) == m_poissonVecs.end())
  {
    m_poissonVecs.insert(
        std::make_pair(key,
        PoissonEq::PoissonVec<dim>(
          const_cast<ot::DA<dim> *>(mesh.da()),   // violate const for weird feVec non-const necessity
          &mesh.distTree()->getTreePartFiltered(), ndofs)));
    m_poissonVecs.at(key).setProblemDimensions(m_min, m_max);
  }

  m_poissonVecs.at(key).computeVec(in.data().data(), out.data().data());
}


template <unsigned int dim>
template <typename ValT>
void Equation<dim>::assembleDiag(const ConstMeshPointers<dim> &mesh, LocalVector<ValT> &diag_out)
{
  const size_t ndofs = diag_out.ndofs();

  const Key key = this->key(mesh);
  if (m_poissonMats.find(key) == m_poissonMats.end())
  {
    m_poissonMats.insert(
        std::make_pair(key,
        PoissonEq::PoissonMat<dim>(mesh.da(), &mesh.distTree()->getTreePartFiltered(), ndofs)));
    m_poissonMats.at(key).setProblemDimensions(m_min, m_max);
  }

  m_poissonMats.at(key).setDiag(diag_out.data().data());
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
      vec.data().data(),
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
           mesh.distTree()->getTreePartFiltered().size(),
           *mesh.da()->getTreePartFront(),
           *mesh.da()->getTreePartBack())
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
           mesh.distTree()->getTreePartFiltered().size(),
           *mesh.da()->getTreePartFront(),
           *mesh.da()->getTreePartBack())
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
           mesh.distTree()->getTreePartFiltered().size(),
           *mesh.da()->getTreePartFront(),
           *mesh.da()->getTreePartBack())
{ }




