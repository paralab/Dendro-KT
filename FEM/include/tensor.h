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

// TODO make a new namespace



/**
 * @tparam dim Dimension of element, i.e. order of tensor.
 * @tparam da  Datatype of vectors.
 * @tparam forward If true, axes are evaluated in increasing order.
 * @param [in] M Size of tensor in 1D.
 * @param [in] A Array of pointers to interpolation matrices, ordered by axis.
 * @param [in] in Array of pointers to input buffers, ordered with source in position 0.
 * @param [in] out Array of pointers to output buffers, ordered with destination in position (dim-1).
 * @param ndofs Number of degrees of freedom in the vector, e.g. 3 for xyzxyz.
 */
template <unsigned int dim, typename da, bool forward>
void KroneckerProduct(unsigned M, const da **A, const da **in, da **out, unsigned int ndofs);


/**
 * @tparam dim Dimension of element, i.e. order of tensor.
 * @tparam da  Datatype of vectors.
 * @tparam forward If true, axes are evaluated in increasing order.
 * @tparam ndofs Number of degrees of freedom in the vector, e.g. 3 for xyzxyz.
 * @param [in] M Size of tensor in 1D.
 * @param [in] A Array of pointers to interpolation matrices, ordered by axis.
 * @param [in] in Array of pointers to input buffers, ordered with source in position 0.
 * @param [in] out Array of pointers to output buffers, ordered with destination in position (dim-1).
 */
template <unsigned int dim, typename da, bool forward, unsigned int ndofs>
void KroneckerProductFixedDof(unsigned M, const da **A, const da **in, da **out);


/**
 * @brief Forall (i,j,k), Q_ijk *= A * P_ijk, where P_ijk == W_i * W_j * W_k;
 * @tparam dim Order of the tensor.
 * @tparam T Component type.
 *
 * Generalizes the following nested for-loop to any nesting level:
 *   int idx = 0;
 *   for (int k = 0; k < l; k++)
 *   {
 *     const double a2 = A*W[k];
 *     for (int j = 0; j < l; j++)
 *     {
 *       const double a1 = a2*W[j];
 *       for (int i = 0; i < l; i++)
 *       {
 *         const double a0 = a1*W[i];
 *         Q[idx++] *= a0;
 *       }
 *     }
 *   }
 */
template <typename T, unsigned int dim>
struct SymmetricOuterProduct
{
  inline static void applyHadamardProduct(unsigned int length1d, T *Q, const T *W1d, const T premult = 1);
};

template <typename T, unsigned int dim>
inline void SymmetricOuterProduct<T,dim>::applyHadamardProduct(
    unsigned int length1d, T *Q, const T *W1d, const T premult)
{
  const unsigned int stride = intPow(length1d, dim-1);
  for (unsigned int ii = 0; ii < length1d; ii++)
    SymmetricOuterProduct<T, dim-1>::applyHadamardProduct(length1d, &Q[stride * ii], W1d, premult * W1d[ii]);
}

template <typename T>
struct SymmetricOuterProduct<T, 1>
{
  inline static void applyHadamardProduct(unsigned int length1d, T *Q, const T *W1d, const T premult = 1)
  {
    for (unsigned int ii = 0; ii < length1d; ii++)
      Q[ii] *= premult * W1d[ii];
  }
};



template <typename da, unsigned int ndofs>
struct MatKernelAssign
{
  const da *X;
  da *Y;
  void operator()(da d, unsigned int x_ii, unsigned int y_ii) const
  {
    for (int dof = 0; dof < ndofs; dof++)
      Y[ndofs*y_ii + dof] = d * X[ndofs*x_ii + dof];
  }
};

template <typename da, unsigned int ndofs>
struct MatKernelAccum
{
  const da *X;
  da *Y;
  void operator()(da d, unsigned int x_ii, unsigned int y_ii) const
  {
    for (int dof = 0; dof < ndofs; dof++)
      Y[ndofs*y_ii + dof] += d * X[ndofs*x_ii + dof];
  }
};

//TODO there is a way to incorporate ndofs as a new axis of the tensor
//  instead of modifying MatKernelAssign and MatKernelAccum.
//  I have chosen the latter for now for simplicity.


// Local matrix multiplication over 1D of a general slice of tensor.
template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent, unsigned int ndofs>
struct IterateBindMatrix;

// Specialization of IterateBindMatrix to the full tensor.
template <unsigned int dim, unsigned int tangent, unsigned int ndofs>
using IterateTensorBindMatrix = IterateBindMatrix<dim, (1u<<dim)-1, tangent, ndofs>;

// Specialization of IterateBindMatrix to a (K-1)-face of the tensor.
template <unsigned int dim, unsigned int face, unsigned int tangent, unsigned int ndofs>
using IterateFacetBindMatrix = IterateBindMatrix<dim, ((1u<<dim)-1) - (1u<<face), tangent, ndofs>;


//
// IterateBindMatrix
//
// Usage: IterateBindMatrix<dim, sliceFlag, tangent, ndofs>::template iterate_bind_matrix<da>(M, A, X, Y);
//
template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent, unsigned int ndofs>
struct IterateBindMatrix
{
  static constexpr unsigned int upperOrientFlag = sliceFlag & (- (1u<<(tangent+1)));
  static constexpr unsigned int lowerOrientFlag = sliceFlag & ((1u<<tangent) - 1);

  using OuterLoop = PowerLoopSlice<dim, upperOrientFlag>;
  using InnerLoop = PowerLoopSlice<dim, lowerOrientFlag>;

  template <typename da>
  struct InnerKernel
  {
    MatKernelAssign<da, ndofs> &m_kernel1;
    MatKernelAccum<da, ndofs> &m_kernel2;
    struct SAssign { InnerKernel &p; void operator()(unsigned int innerIdx) { p.actuate(p.m_kernel1, innerIdx); } } Assign;
    struct SAccum { InnerKernel &p; void operator()(unsigned int innerIdx) { p.actuate(p.m_kernel2, innerIdx); } } Accum;
    unsigned int m_M;
    InnerKernel(MatKernelAssign<da, ndofs> &kernel1, MatKernelAccum<da, ndofs> &kernel2, unsigned int M)
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
    OuterKernel(InnerKernel<da> &kernel, unsigned int M, const da *A) : m_kernel(kernel), m_M(M), m_APtr(A) {}
    OuterKernel() = delete;

    unsigned int m_row;
    inline void operator()(unsigned int outerIdx);
  };

  template <typename da>
  inline static void iterate_bind_matrix(const unsigned int M, const da *A, const da *X, da *Y);
};


  //
  // iterate_bind_matrix()
  //
  template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent, unsigned int ndofs>
  template <typename da>
  inline void IterateBindMatrix<dim,sliceFlag,tangent,ndofs>::iterate_bind_matrix(const unsigned int M, const da *A, const da *X, da *Y)
  {
    // For each row of the matrix, iterate through the hyperplane
    // of the index space that is normal to the axis `face'.
    // `dim' specifies the dimension of the index space.
    // `tangent' specifies which axis should be bound to the matrix row.

    MatKernelAssign<da, ndofs> matKernelAssign{X, Y};
    MatKernelAccum<da, ndofs> matKernelAccum{X, Y};

    InnerKernel<da> inner{matKernelAssign, matKernelAccum, M};
    OuterKernel<da> outer{inner, M, A};

    unsigned int &row = outer.m_row;
    for (row = 0; row < M; row++)
      OuterLoop::template loop(M, outer);
  }
  //
  // OuterKernel::operator()()
  //
  template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent, unsigned int ndofs>
  template <typename da>
  inline void IterateBindMatrix<dim,sliceFlag,tangent,ndofs>::
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
  template <unsigned int dim, unsigned int sliceFlag, unsigned int tangent, unsigned int ndofs>
  template <typename da>
  template <typename Kernel>
  inline void IterateBindMatrix<dim,sliceFlag,tangent,ndofs>::
       InnerKernel<da>::
       actuate(Kernel &kernel, unsigned int innerIdx)
  {
    // Combine indices into x_ii and y_ii;
    const unsigned int tangentStride = intPow(m_M, tangent);
    unsigned int x_ii = m_outerIdx + m_col * tangentStride + innerIdx;
    unsigned int y_ii = m_outerIdx + m_row * tangentStride + innerIdx;
    kernel(m_d, x_ii, y_ii);
  }



template <unsigned int dim, unsigned int d, bool forward, unsigned int ndofs>
struct KroneckerProduct_loop { template <typename da> static void body(unsigned M, const da **A, const da **in, da **out) {
  constexpr unsigned int ii = (forward ? dim-1 - d : d);

  if (forward)
    IterateTensorBindMatrix<dim, d, ndofs>::template iterate_bind_matrix<da>(M, A[d], in[ii], out[ii]);

  KroneckerProduct_loop<dim, d-1, forward, ndofs>::template body<da>(M,A,in,out);

  if (!forward)
    IterateTensorBindMatrix<dim, d, ndofs>::template iterate_bind_matrix<da>(M, A[d], in[ii], out[ii]);
}};
template <unsigned int dim, bool forward, unsigned int ndofs>
struct KroneckerProduct_loop<dim, 0, forward, ndofs> { template <typename da> static void body(unsigned M, const da **A, const da **in, da **out) {
  constexpr unsigned int ii = (forward ? dim-1 : 0);
  IterateTensorBindMatrix<dim, 0, ndofs>::template iterate_bind_matrix<da>(M, A[0], in[ii], out[ii]);
}};

template <unsigned int dim, typename da, bool forward, unsigned int ndofs>
void KroneckerProductFixedDof(unsigned M, const da **A, const da **in, da **out)
{
  KroneckerProduct_loop<dim, dim-1, forward, ndofs>::template body<da>(M,A,in,out);
}

template <unsigned int dim, typename da, bool forward>
void KroneckerProduct(unsigned M, const da **A, const da **in, da **out, unsigned int ndofs)
{
  // Convert runtime argument to template argument.
  switch (ndofs)
  {
    // TODO add CMake options and #if macros for greater number of dofs.
    case 1:
      KroneckerProductFixedDof<dim, da, forward, 1>(M, A, in, out);
      break;
    case 2:
      KroneckerProductFixedDof<dim, da, forward, 2>(M, A, in, out);
      break;
    case 3:
      KroneckerProductFixedDof<dim, da, forward, 3>(M, A, in, out);
      break;
    default:
      const bool isNumberOfDegreesOfFreedomSupported = false;
      assert(isNumberOfDegreesOfFreedomSupported);  // Need to add more cases.
      break;
  }
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
