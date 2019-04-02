//
// Created by milinda on 1/15/17.
//

/**
 *
 * @author Milinda Fernando
 * @author Masado Ishii
 * @breif contains the utilities for tensor kronecker products for interpolations.
 *
 * */

#ifndef SFCSORTBENCH_DENDROTENSOR_H
#define SFCSORTBENCH_DENDROTENSOR_H



#include <arraySlice.h>

template <typename da>
struct MatKernelAssign
{
  const da *X;
  da *Y;
  void operator()(da d, unsigned int x_ii, unsigned int y_ii) const { Y[y_ii] = d * X[x_ii]; }
};

template <typename da>
struct MatKernelAccum
{
  const da *X;
  da *Y;
  void operator()(da d, unsigned int x_ii, unsigned int y_ii) const { Y[y_ii] += d * X[x_ii]; }
};


// Local matrix multiplication over 1D of a general slice of tensor.
template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent>
struct IterateBindMatrix;

// Specialization of IterateBindMatrix to the full tensor.
template <unsigned int dim, unsigned int tangent>
using IterateTensorBindMatrix = IterateBindMatrix<dim, (1u<<dim)-1, tangent>;

// Specialization of IterateBindMatrix to a (K-1)-face of the tensor.
template <unsigned int dim, unsigned int face, unsigned int tangent>
using IterateFacetBindMatrix = IterateBindMatrix<dim, ((1u<<dim)-1) - (1u<<face), tangent>;


//
// IterateBindMatrix
//
// Usage: IterateBindMatrix<dim, sliceFlag, tangent>::template iterate_bind_matrix<da>(M, A, X, Y);
//
template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent>
struct IterateBindMatrix
{
  static constexpr unsigned int upperOrientFlag = sliceFlag & (- (1u<<(tangent+1)));
  static constexpr unsigned int lowerOrientFlag = sliceFlag & ((1u<<tangent) - 1);

  using OuterLoop = PowerLoopSlice<dim, upperOrientFlag>;
  using InnerLoop = PowerLoopSlice<dim, lowerOrientFlag>;

  template <typename da>
  struct InnerKernel
  {
    MatKernelAssign<da> &m_kernel1;
    MatKernelAccum<da> &m_kernel2;
    struct SAssign { InnerKernel &p; void operator()(unsigned int innerIdx) { p.actuate(p.m_kernel1, innerIdx); } } Assign;
    struct SAccum { InnerKernel &p; void operator()(unsigned int innerIdx) { p.actuate(p.m_kernel2, innerIdx); } } Accum;
    unsigned int m_M;
    InnerKernel(MatKernelAssign<da> &kernel1, MatKernelAccum<da> &kernel2, unsigned int M)
        : m_kernel1(kernel1), m_kernel2(kernel2), Assign{*this}, Accum{*this}, m_M(M) {}
    InnerKernel() = delete;

    da m_d;
    unsigned int m_row;
    unsigned int m_col;
    unsigned int m_outerIdx;
    template <typename Kernel>
    inline void actuate(Kernel &kernel, unsigned int innerIdx);
  };

  template <typename da>
  struct OuterKernel
  {
    InnerKernel<da> &m_kernel;
    unsigned int m_M;
    const da *m_APtr;
    OuterKernel(InnerKernel<da> &kernel, unsigned int M, da *A) : m_kernel(kernel), m_M(M), m_APtr(A) {}
    OuterKernel() = delete;

    unsigned int m_row;
    inline void operator()(unsigned int outerIdx);
  };

  template <typename da>
  inline static void iterate_bind_matrix(const unsigned int M, da *A, da *X, da *Y);
};


  //
  // iterate_bind_matrix()
  //
  template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent>
  template <typename da>
  inline void IterateBindMatrix<dim,sliceFlag,tangent>::iterate_bind_matrix(const unsigned int M, da *A, da *X, da *Y)
  {
    // For each row of the matrix, iterate through the hyperplane
    // of the index space that is normal to the axis `face'.
    // `dim' specifies the dimension of the index space.
    // `tangent' specifies which axis should be bound to the matrix row.

    MatKernelAssign<da> matKernelAssign{X, Y};
    MatKernelAccum<da> matKernelAccum{X, Y};

    InnerKernel<da> inner{matKernelAssign, matKernelAccum, M};
    OuterKernel<da> outer{inner, M, A};

    unsigned int &row = outer.m_row;
    for (row = 0; row < M; row++)
      OuterLoop::template loop(M, outer);
  }
  //
  // OuterKernel::operator()()
  //
  template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent>
  template <typename da>
  inline void IterateBindMatrix<dim,sliceFlag,tangent>::
       OuterKernel<da>::
       operator()(unsigned int outerIdx)
  {
    // Iterate down tangent axis (input and matrix).
    m_kernel.m_outerIdx = outerIdx;
    m_kernel.m_row = m_row;
    unsigned int &col = m_kernel.m_col;
    col = 0;
    m_kernel.m_d = m_APtr[m_row];
    InnerLoop::template loop(m_M, m_kernel.Assign);
    for (col = 1; col < m_M; col++)
    {
      m_kernel.m_d = m_APtr[col * m_M + m_row];
      InnerLoop::template loop(m_M, m_kernel.Accum);
    }
  }
  //
  // InnerKernel::actuate()
  //
  template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent>
  template <typename da>
  template <typename Kernel>
  inline void IterateBindMatrix<dim,sliceFlag,tangent>::
       InnerKernel<da>::
       actuate(Kernel &kernel, unsigned int innerIdx)
  {
    // Combine indices into x_ii and y_ii;
    const unsigned int tangentStride = intPow(m_M, tangent);
    unsigned int x_ii = m_outerIdx + m_col * tangentStride + innerIdx;
    unsigned int y_ii = m_outerIdx + m_row * tangentStride + innerIdx;
    kernel(m_d, x_ii, y_ii);
  }




/** Apply the 1D interpolation for the input vector x and output the interpolated values in the vector Y.
 *
 *
 *
 * \param [in]  M  size of the vector
 * \param [in]  A  interpolation matrix
 * \param [in]  X  input data for the interpolation
 * \param [out] Y  interpolated values.
 *

 */
void DENDRO_TENSOR_AIIX_APPLY_ELEM (const int M, const double*  A, const double*  X, double*  Y);



/** Apply the 1D interpolation for the input vector x and output the interpolated values in the vector Y.
 *
 *
 *
 * \param [in]  M  size of the vector
 * \param [in]  A  interpolation matrix
 * \param [in]  X  input data for the interpolation
 * \param [out] Y  interpolated values.
 *

 */
void DENDRO_TENSOR_IIAX_APPLY_ELEM(const int M, const double*  A, const double*  X, double*  Y);


/** Apply the 1D interpolation for the input vector x and output the interpolated values in the vector Y.
 *
 *
 *
 * \param [in]  M  size of the vector
 * \param [in]  A  interpolation matrix
 * \param [in]  X  input data for the interpolation
 * \param [out] Y  interpolated values.
 *

 */
void DENDRO_TENSOR_IAIX_APPLY_ELEM (const int M, const double*  A, const double*  X, double*  Y);




/** Apply the 1D interpolation for the input vector x and output the interpolated values in the vector Y.
 *
 *
 *
 * \param [in]  M  size of the vector
 * \param [in]  A  interpolation matrix
 * \param [in]  X  input data for the interpolation
 * \param [out] Y  interpolated values.
 *

 */
void DENDRO_TENSOR_IAX_APPLY_ELEM_2D(const int M, const double*  A, const double*  X, double*  Y);



/** Apply the 1D interpolation for the input vector x and output the interpolated values in the vector Y.
 *
 *
 *
 * \param [in]  M  size of the vector
 * \param [in]  A  interpolation matrix
 * \param [in]  X  input data for the interpolation
 * \param [out] Y  interpolated values.
 *

 */
void DENDRO_TENSOR_AIX_APPLY_ELEM_2D (const int M, const double*  A, const double*  X, double*  Y);


#endif //SFCSORTBENCH_DENDROTENSOR_H
