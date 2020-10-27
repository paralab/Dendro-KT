#include "hcurvedata.h"
#include "treeNode.h"
#include "oda.h"

#include <petsc.h>
#include <array>
#include <vector>

int main(int argc, char * argv[])
{
  constexpr int DIM = 3;

  PetscInitialize(&argc, &argv, NULL, NULL);
  _InitializeHcurve(DIM);

  int eleOrder = 2;
  int ndof = 1;
  m_uiMaxDepth = 10;
  /// int level = 2;
  int level = 5;

  /// std::vector<ot::TreeNode<unsigned int, DIM>> treePart;

  ot::DA<DIM> octDA;
  std::array<unsigned int, DIM> a = {1,1,1};
  ot::constructRegularSubdomainDA<DIM>(octDA, level, a, eleOrder, MPI_COMM_WORLD);

  std::vector<size_t> bdyIndex;
  octDA.getBoundaryNodeIndices(bdyIndex);

  std::cout << "octDA local num elements == " << octDA.getLocalElementSz() << "\n";
  std::cout << "octDA local num nodes == " << octDA.getLocalNodalSz() << "\n";
  std::cout << "octDA local bdyIndex.size() == " << bdyIndex.size() << "\n";

  PetscFinalize();

  return 0;
}
