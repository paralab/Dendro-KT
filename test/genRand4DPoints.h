/*
 * genRand4DPoints.h
 *   Generate a cloud of 4D points as input to later test
 *   our 4D sorting/constructing/balancing.
 *
 * Masado Ishii  --  UofU SoC, 2018-12-03
 */

#ifndef DENDRO_KT_TEST_GEN_4D_POINTS_H
#define DENDRO_KT_TEST_GEN_4D_POINTS_H

#include "TreeNode.h"
#include <vector>
#include <random>
#include <iostream>
#include <stdio.h>


template <typename T, unsigned int dim>
inline std::vector<ot::TreeNode<T,dim>> genRand4DPoints(int numPoints)
{
  std::vector<ot::TreeNode<T,dim>> points;
  std::array<T,dim> uiCoords;

  //const T maxCoord = (1u << MAX_LEVEL) - 1;
  const T maxCoord = (1u << m_uiMaxDepth) - 1;
  const T leafLevel = m_uiMaxDepth;

  // Set up random number generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(14641);             // Uncomment for deterministic testing.
  std::uniform_int_distribution<T> distCoord(0, maxCoord);
  std::uniform_int_distribution<T> distLevel(1, m_uiMaxDepth);

  // Add points sequentially.
  for (int ii = 0; ii < numPoints; ii++)
  {
    for (T &u : uiCoords)
    {
      u = distCoord(gen);
    }
    //ot::TreeNode<T,dim> tn(0, uiCoords, leafLevel);
    ot::TreeNode<T,dim> tn(0, uiCoords, distLevel(gen));
    /// std::cout << tn << '\n';
    points.push_back(tn);
  }

  /// std::cout << '\n';
  return points;
}

#endif // DENDRO_KT_TEST_GEN_4D_POINTS_H
