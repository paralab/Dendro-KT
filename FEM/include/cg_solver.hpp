
#ifndef DENDRO_KT_CG_SOLVER
#define DENDRO_KT_CG_SOLVER

#include "include/oda.h"
#include "FEM/include/solver_utils.hpp"

#include <vector>

namespace solve
{

  // cgSolver()
  template <unsigned int dim, typename MatMult>
  int cgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, VECType * u, const VECType * rhs, int max_iter, double relResErr, bool print_progress);

  // pcgSolver()
  template <unsigned int dim, typename MatMult, typename PCSolver>
  int pcgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, const PCSolver &pc_solve, VECType * u, const VECType * rhs, int max_iter, double relResErr, bool print_progress);


  // cgSolver()
  template <unsigned int dim, typename MatMult>
  int cgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, VECType * u, const VECType * rhs, int max_iter, double relResErr, bool print_progress)
  {
    // only single dof per node supported here

    // Vector norm and dot
    const auto vec_norm_linf = [da](const VECType *v) -> VECType {
      return normLInfty(v, da->getLocalNodalSz(), da->getCommActive());
    };
    const auto vec_dot = [da](const VECType *u, const VECType *v) -> VECType {
      return dot(u, v, da->getLocalNodalSz(), da->getCommActive());
    };

    // residual
    const auto residual = [da](auto &&matrix, VECType *r, const VECType *u, const VECType *rhs) -> void {
      matrix(u, r);
      subt(rhs, r, da->getLocalNodalSz(), r);  // r = rhs - r
    };

    //future: use reduction rate as an additional exit criterion
    /// util::ConvergenceRate residual_convergence(3);
    ///  residual_convergence.observe_step(res);
    ///     residual_convergence.observe_step(res);
    ///     const double res_rate = residual_convergence.rate();
    ///     if (res_rate > 0.95)
    ///       break;

    const double normb = vec_norm_linf(rhs);
    const double thresh = relResErr * normb;

    static std::vector<VECType> r;
    static std::vector<VECType> p;
    static std::vector<VECType> Ap;
    const size_t localSz = da->getLocalNodalSz();
    r.resize(localSz);
    /// p.resize(localSz);
    Ap.resize(localSz);

    int step = 0;
    residual(mat_mult, &r[0], u, rhs);
    VECType rmag = vec_norm_linf(&r[0]);
    const VECType rmag0 = rmag;
    fprintf(stdout, "step==%d  normb==%e  res==%e \n", step, normb, rmag);
    if (rmag <= thresh)
      return step;
    VECType rProd = vec_dot(&r[0], &r[0]);

    VECType iterLInf = 0.0f;

    p = r;
    while (step < max_iter)
    {
      mat_mult(&p[0], &Ap[0]);  // Ap
      const VECType pProd = vec_dot(&p[0], &Ap[0]);

      const VECType alpha = rProd / pProd;
      iterLInf = alpha * vec_norm_linf(&p[0]);
      for (size_t ii = 0; ii < localSz; ++ii)
        u[ii] += alpha * p[ii];
      ++step;

      const VECType rProdPrev = rProd;

      // Explicitly re-calculate the residual once every ~20 steps.
      if (step % 20 == 0)
        residual(mat_mult, &r[0], u, rhs);
      else
        for (size_t ii = 0; ii < localSz; ++ii)
          r[ii] -= alpha * Ap[ii];

      rmag = vec_norm_linf(&r[0]);
      if (rmag <= thresh)
        break;
      rProd = vec_dot(&r[0], &r[0]);

      const VECType beta = rProd / rProdPrev;
      for (size_t ii = 0; ii < localSz; ++ii)
        p[ii] = r[ii] + beta * p[ii];

      if (print_progress and step % 10 == 0)
        fprintf(stdout, "step==%d  res==%e  reduce==%e  diff==%e  rProd==%e  pProd==%e  a==%e  b==%e\n", step, rmag, rmag/rmag0, iterLInf, rProd, pProd, alpha, beta);
    }
    fprintf(stdout, "step==%d  normb==%e  res==%e  reduce==%e\n", step, normb, rmag, rmag/rmag0);

    return step;
  }


  // pcgSolver()
  template <unsigned int dim, typename MatMult, typename PCSolver, typename Monitor>
  int pcgSolver(const ot::DA<dim> *da, const MatMult &mat_mult, const PCSolver &pc_solve, VECType * u, const VECType * rhs, int max_iter, double relResErr, bool print_progress, const Monitor &monitor)
  {
    // https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    // only single dof per node supported here

    // Vector norm and dot
    const auto vec_norm_linf = [da](const VECType *v) -> VECType {
      return normLInfty(v, da->getLocalNodalSz(), da->getCommActive());
    };
    const auto vec_norm_l2 = [da](const VECType *v) -> VECType {
      return normL2(v, da->getLocalNodalSz(), da->getCommActive());
    };
    const auto vec_dot = [da](const VECType *u, const VECType *v) -> VECType {
      return dot(u, v, da->getLocalNodalSz(), da->getCommActive());
    };

    // residual
    const auto residual = [da](auto &&matrix, VECType *r, const VECType *u, const VECType *rhs) -> void {
      matrix(u, r);
      subt(rhs, r, da->getLocalNodalSz(), r);  // r = rhs - r
    };

    //future: use reduction rate as an additional exit criterion
    /// util::ConvergenceRate residual_convergence(3);
    ///  residual_convergence.observe_step(res);
    ///     residual_convergence.observe_step(res);
    ///     const double res_rate = residual_convergence.rate();
    ///     if (res_rate > 0.95)
    ///       break;

    const double normb = vec_norm_linf(rhs);
    const double thresh = relResErr * normb;

    static std::vector<VECType> r;
    static std::vector<VECType> p;
    static std::vector<VECType> z;
    static std::vector<VECType> Ap;
    const size_t localSz = da->getLocalNodalSz();
    r.resize(localSz);
    /// p.resize(localSz);
    z.resize(localSz);
    Ap.resize(localSz);

    int step = 0;
    residual(mat_mult, &r[0], u, rhs);
    VECType rmag = vec_norm_linf(&r[0]);
    VECType rmag_l2 = vec_norm_l2(&r[0]);
    monitor(rmag, rmag_l2);
    const VECType rmag0 = rmag;
    fprintf(stdout, "step==%d  normb==%e  res==%e \n", step, normb, rmag);
    if (rmag <= thresh)
      return step;

    VECType iterLInf = 0.0f;

    std::fill(z.begin(), z.end(), 0.0);
    pc_solve(z.data(), r.data());

    VECType rProd = vec_dot(&r[0], &z[0]);

    p = z;
    while (step < max_iter)
    {
      mat_mult(&p[0], &Ap[0]);  // Ap
      const VECType pProd = vec_dot(&p[0], &Ap[0]);

      const VECType alpha = rProd / pProd;
      iterLInf = alpha * vec_norm_linf(&p[0]);
      for (size_t ii = 0; ii < localSz; ++ii)
        u[ii] += alpha * p[ii];
      ++step;

      const VECType rProdPrev = rProd;

      // Explicitly re-calculate the residual once every ~20 steps.
      if (step % 20 == 0)
        residual(mat_mult, &r[0], u, rhs);
      else
        for (size_t ii = 0; ii < localSz; ++ii)
          r[ii] -= alpha * Ap[ii];

      rmag = vec_norm_linf(&r[0]);
      VECType rmag_l2 = vec_norm_l2(&r[0]);
      monitor(rmag, rmag_l2);
      if (rmag <= thresh)
        break;

      std::fill(z.begin(), z.end(), 0.0);
      pc_solve(z.data(), r.data());

      rProd = vec_dot(&r[0], &z[0]);

      const VECType beta = rProd / rProdPrev;
      for (size_t ii = 0; ii < localSz; ++ii)
        p[ii] = z[ii] + beta * p[ii];

      if (print_progress and step % 10 == 0)
        fprintf(stdout, "step==%d  res==%e  reduce==%e  diff==%e  rProd==%e  pProd==%e  a==%e  b==%e\n", step, rmag, rmag/rmag0, iterLInf, rProd, pProd, alpha, beta);
    }
    fprintf(stdout, "step==%d  normb==%e  res==%e  reduce==%e\n", step, normb, rmag, rmag/rmag0);

    return step;
  }





}

#endif//DENDRO_KT_CG_SOLVER
