/*
 * RotationTableHilbert.cpp
 *
 * Masado Ishii  --  UofU SoC, 2018-11-27
 *
 * Based on
 *   Haverkort, 2012 ("Harmonious Hilbert Curves and Other Extradimensional Space-filling Curves");
 *   Fernando and Sundar, 2018 ("Comparison Free Computations on Octree Based Adaptive Meshes");
 *   Campbell, et al, 2003 ("Dynamic Octree Load Balancing Using Space-filling Curves").
 *
 * Space-filling curves can lead to efficient partitioning in distributed adaptive
 * meshing codes. Fernando and Sundar's paper outlines a novel partitioning
 * algorithm for adaptive meshing, and space-filling curves are fundamental to
 * the algorithm. The algorithm uses an abstract representation of
 * space-filling curves, in the form of rotation tables.
 *
 * Readers are referred to the Campbell paper for the details of how such
 * a rotation table works. The Campbell paper gives specific examples for
 * the Morton ordering and Hilbert ordering, in 2D and 3D.
 *
 * There are many possible generalizations of Hilbert's curve from 2D to
 * 3D and beyond, and it is not obvious how to pick out one of these
 * generalizations--let alone use one in software. Haverkort's paper provides both
 * 1) a property (that of being 'harmonious') that distinguishes a single
 *    Hilbert-like curve in any dimension from the many other possibilities, and
 * 2) a description of the refinement operator in a format that maps well to
 *    computer software.
 *
 * The methods in this source file produce rotation tables for K-dimensional
 * (harmonious) Hilbert curves, analogous to the tables detailed by Campbell, et al.
 * The procedures to generate the tables are based on Haverkort's description.
 * The generated tables will be statically available to the main program
 * (Dendro), for use as in Fernando and Sundar's algorithm.
 */

#define __DEBUG__ 1


#include <array>
#include <iostream>
#include <assert.h>

namespace hilbert
{


/*
 * First, don't worrry about making statically-available tables. That's a
 * C++ issue. Focus on making sure the implementation works. Just output
 * to stdout. Later can use some magical constexpr or something.
 */

/*
 * Need:
 *   - Representation of a region's geometric orientation.
 *   - Implementation of Haverkort's refinement operator.
 *   - Methods to `apply' a parent's orientation to the results of refinement.
 *   - Method to fill out the table.
 */

// Enough bits for the number of dimensions.
// If more than 8 dimensions are needed, change this to something bigger.
using AxBits = unsigned char;

//
// PhysOrient:
//   Representation of the physical distinctions between a given orientation
//   and the root orientation. Namely, permutation and reflection of axes.
//
template <int K>
struct PhysOrient
{
  std::array<int, K> a;    // Haverkort uses 'a' for permutation of axes. i gets a[i].
  AxBits m;                // Haverkort uses 'm' for reflection vector.

  std::array<int, K> a_() const  // Inverse permutation.
  {
    std::array<int, K> A;
    for (int ii = 0; ii < K; ii++)
    {
#if __DEBUG__
      assert(0 <= a[ii] && a[ii] < K);
#endif
      A[a[ii]] = ii;
    }
    return A;
  }

  static PhysOrient identity()
  {
    PhysOrient p;
    for (int ii = 0; ii < K; ii++)
      p.a[ii] = ii;
    p.m = 0;
    return p;
  }

  // To produce children from parents, parent orientation should be
  // applied to the results of refinement.
  //TODO
  AxBits apply(AxBits location) const;        // Group action.
  PhysOrient apply(PhysOrient orient) const;  // Group multiplication.
};


//
// refinement_operator():
//   Haverkort's refinement operator for K-dimensional harmonious Hilbert curve.
//
template <int K>
void refinement_operator(int rank, AxBits &out_loc, PhysOrient<K> &out_orient)
{
  // Note that in Haverkort's notation, index 0 corresponds to the leftmost bit.

  // out_loc is defined to be `c`, and also `m' is defined in terms of `c'.
  //
  // `c' is the reflected Gray code for rank, in base 2, dimension K.
  // The Gray code may be expressed as (where :: is concatenation)
  //   c^d(r) = (r >= 1<<(d-1))  ?  1::c^{d-1}(2d-r)  :   0::c^{d-1}(r).
  // That is, if the head digit is 0, c(.) is evaluated on the tail;
  // if the head digit is 1, each bit in the tail is flipped, and then
  // c(.) is evaluated on the tail.
  //
  // E.g. a,b,c,d,e
  //      ->  a, b + a, c + a + b+a, d + a + b+a + c+a+b+a, e + a + b+a + c+a+b+a + d+a+b+a+c+a+b+a;
  //      ==  a, b + a, c + b + 2a,  d + c + 2b + 4a,       e + d + 2c + 4b + 8a;
  //
  // Now it's clear that each bit receives some multiple of each of the bits to
  // its left. That multiple is a power of two, which is even except for
  // the bit immediately to the left. In other words, each bit in the result
  // is equal to the same bit in the source, plus (XOR) the bit to its left.
  // Therefore,
  //   c(r) = (r >> 1) ^ r;
  //
  int c = (rank >> 1) ^ rank;
  out_loc = c;

  //
  // Reflection (p.23).
  if (rank == 0)
    out_orient.m = c;  // Should be 0.
  else
  {
    int cm1 = ((rank-1) >> 1) ^ (rank-1);
    out_orient.m = cm1;
    // Correct it by setting the rightmost bit to the opposite of the
    // rightmost bit of c.
    out_orient.m = (out_orient.m & -2) | ((~c) & 1);
  }

  //
  // Permutation (p.23)
  //
  // Descriptions of both the permutation and its inverse are given.
  // The two descriptions are equivalent. Below we compute a,
  // and the inline comments show the correspondence with a_inverse.
  //
  int endr = rank & 1;
  int offset = 0;
  for (int ii = 0; ii < K; ii++)
    offset += (((rank >> (K-1 - ii)) & 1) != endr);
  int L = offset-1, R = K-1;
  for (int ii = 0; ii < K; ii++)
  {
    if (((rank >> (K-1 - ii)) & 1) != endr)   // Case one: Goes to the front section.
      out_orient.a[ii] = L--;                // <--> a_inverse[L] = ii
    else                        // Case two: Goes to the back section.
      out_orient.a[ii] = R--;                // <--> a_inverse[R] = ii
  }
}


}  // namespace hilbert.



//
// binary_string():
//   Convert a binary number to a string of characters.
// 
// The buffer c must have at least (K+1) space,
// for K characters and the null byte.
//
void binary_string(unsigned char b, char *c, int K)
{
  c[K] = '\0';
  for (int ii = K-1; ii >= 0; ii--, b >>= 1)
    c[ii] = '0' + (b & 1);
}

template <int K>
void hexadecimal_string(std::array<int, K> h, char *c)
{
  c[K] = '\0';
  for (int ii = 0; ii < K; ii++)
    c[ii] = (h[ii] < 10 ? '0' + h[ii] : 'a' + h[ii] - 10);
}


//
// main():
//   Test the correctness of our methods.
//
int main(int arc, char* argv[])
{
  const int K = 5;
  const int N = 1<<K;

  char s[K+1];

  hilbert::AxBits loc[N];
  hilbert::PhysOrient<K> orient[N];

  for (int r = 0; r < N; r++)
  {
    hilbert::refinement_operator<K>(r, loc[r], orient[r]);

    binary_string(r, s, K);
    std::cout << s << " ";

    binary_string(loc[r], s, K);
    std::cout << s << " ";

    hexadecimal_string<K>(orient[r].a, s);
    std::cout << s << " ";

    hexadecimal_string<K>(orient[r].a_(), s);
    std::cout << s << " ";

    binary_string(orient[r].m, s, K);
    std::cout << s << " ";

    std::cout << '\n';
  }


  return 0;
}





// Compare with 5D table, given by Haverkort:
// rank   loc.   permutation  inv. permutation  refl.
// 00000  00000  43210        43210             00000  
// 00001  00001  32104        32104             00000  
// 00010  00011  43201        34210             00000  
// 00011  00010  21043        21043             00011  
// 00100  00110  43021        24310             00011  
// 00101  00111  21403        31042             00110  
// 00110  00101  43102        32410             00110  
// 00111  00100  10432        10432             00101  
// 01000  01100  40321        14320             00101  
// 01001  01101  24103        32041             01100  
// 01010  01111  41302        31420             01100  
// 01011  01110  14032        20431             01111  
// 01100  01010  41032        21430             01111  
// 01101  01011  14302        30421             01010  
// 01110  01001  42103        32140             01010  
// 01111  01000  04321        04321             01001  
// 10000  11000  04321        04321             01001 
// 10001  11001  42103        32140             11000 
// 10010  11011  14302        30421             11000 
// 10011  11010  41032        21430             11011 
// 10100  11110  14032        20431             11011 
// 10101  11111  41302        31420             11110 
// 10110  11101  24103        32041             11110 
// 10111  11100  40321        14320             11101 
// 11000  10100  10432        10432             11101 
// 11001  10101  43102        32410             10100 
// 11010  10111  21403        31042             10100 
// 11011  10110  43021        24310             10111 
// 11100  10010  21043        21043             10111 
// 11101  10011  43201        34210             10010 
// 11110  10001  32104        32104             10010 
// 11111  10000  43210        43210             10001 
