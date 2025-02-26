//
// Created by milinda on 1/19/17.
//

/**
 *
 * @author Milinda Fernando
 * School of Computing University of Utah.
 * @brief Constains lapack routines such as linear system solve, eigen solve to build the interpolation matrices.
 *
 *
 * */


#ifndef SFCSORTBENCH_LAPAC_H
#define SFCSORTBENCH_LAPAC_H

#include <cstring>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace lapack
{

/**
 *  @brief: Wrapper for LAPACK DGESV solver for AX=B. Parameters are given below.
 *  @param[in] n : number of rows or columns of linear system
 *  @param[in] nrhs: number of right hand sides.
 *  @param[in] A: matrix A (row major)
 *  @param[in] B: matrix B (row major)
 *  @param[out] X:  matrix X (solution)
 */
inline void eigen_DGESV(int n, int nrhs, const double * A, const double * B, double * X)
{
  using Eigen::Dynamic;
  using Eigen::RowMajor;
  using RowMajorMatrix = Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>;
  using MatrixWrapper = Eigen::Map<RowMajorMatrix>;
  using ConstMatrixWrapper = Eigen::Map<const RowMajorMatrix>;

  ConstMatrixWrapper A_wrap(A, n, n);
  ConstMatrixWrapper B_wrap(B, n, nrhs);
  MatrixWrapper X_wrap(X, n, nrhs);

  X_wrap = A_wrap.lu().solve(B_wrap);
}




/**
 *  @brief: Wrapper for LAPACK DGESV compute eigen values of a square matrix of A. Parameters are given below.
 *  @param[in] n : number of rows or columns of linear system
 *  @param[in] A: matrix A (row major)
 *  @param[out] wr: real part of eigen values
 *  @param[out] vs eigen vectors (row major)
 */
inline void eigen_DSYEV(int n, const double * A, double * wr, double * vs)
{
  using Eigen::Dynamic;
  using Eigen::RowMajor;
  using RowMajorMatrix = Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>;
  using MatrixWrapper = Eigen::Map<RowMajorMatrix>;
  using ConstMatrixWrapper = Eigen::Map<const RowMajorMatrix>;

  ConstMatrixWrapper A_wrap(A, n, n);
  MatrixWrapper wr_wrap(wr, n, 1);
  MatrixWrapper vs_wrap(vs, n, n);

  Eigen::SelfAdjointEigenSolver<RowMajorMatrix>
      evd(A_wrap, Eigen::ComputeEigenvectors);

  const bool success = (evd.info() == Eigen::Success);

  wr_wrap = evd.eigenvalues();
  vs_wrap = evd.eigenvectors();
}

}// end of namespace


#endif //SFCSORTBENCH_LAPAC_H
