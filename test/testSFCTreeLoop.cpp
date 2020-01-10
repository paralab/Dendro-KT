
#include "sfcTreeLoop.h"
#include "hcurvedata.h"
#include "octUtils.h"

#include "sfcTreeLoop_matvec.h"

#include <stdio.h>
#include <iostream>
#include <bitset>



bool testNull();
bool testDummySubclass();
bool testTopDownSubclass();
bool testMatvecSubclass();


/**
 * main()
 */
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  /// bool success = testNull();
  /// bool success = testDummySubclass();
  /// bool success = testTopDownSubclass();
  /// bool success = testDummySubclass() && testTopDownSubclass();
  bool success = testMatvecSubclass();
  std::cout << "Result: " << (success ? "success" : "failure") << "\n";

  MPI_Finalize();

  return !success;
}


/**
 * testNull()
 */
bool testNull()
{
  constexpr unsigned int dim = 2;
  using C = unsigned int;
  using T = float;

  const unsigned int eleOrder = 1;

  _InitializeHcurve(dim);

  _DestroyHcurve();

  return true;
}



template <unsigned int dim>
class DummySubclass : public ot::SFC_TreeLoop<dim, ot::Inputs<double>, ot::Outputs<double>, ot::DefaultSummary, DummySubclass<dim>>
{
  using FrameT = ot::Frame<dim, ot::Inputs<double>, ot::Outputs<double>, ot::DefaultSummary, DummySubclass<dim>>;
  public:
    void topDownNodes(FrameT &parentFrame, ot::ExtantCellFlagT *extantChildren)
    {
      if (this->getCurrentSubtree().getLevel() < 2)
        *extantChildren = (1 << (1u << dim)) - 1;  // All children.
      else
        *extantChildren = 0u;

      std::cout << "Top-down nodes on \t" << this->getCurrentSubtree()
                << "      extantChildren==" << std::bitset<4>(*extantChildren).to_string()
                << "\n";
    }

    void bottomUpNodes(FrameT &parentFrame, ot::ExtantCellFlagT extantChildren)
    {
      std::cout << "Bottom-up nodes on \t" << this->getCurrentSubtree() << "\n";
    }

    // If these are commented out we should get a runtime NotImplemented warning.
    void parent2Child(FrameT &parentFrame, FrameT &childFrame)
    {
    }
    void child2Parent(FrameT &parentFrame, FrameT &childFrame)
    {
    }
};


/**
 * testDummySubclass()
 */
bool testDummySubclass()
{
  constexpr unsigned int dim = 2;
  using C = unsigned int;
  using T = float;

  const unsigned int eleOrder = 1;

  m_uiMaxDepth = 3;

  _InitializeHcurve(dim);

  DummySubclass<dim> dummy;
  while (!dummy.isFinished())
  {
    if (!dummy.isPre())
    {
      std::cout << "Returned to subtree \t" << dummy.getSubtreeInfo().getCurrentSubtree() << "\n";
      dummy.next();
    }
    else
    {
      std::cout << "Inspecting subtree \t" << dummy.getSubtreeInfo().getCurrentSubtree() << "\n";
      dummy.step();
    }
  }

  _DestroyHcurve();

  std::cout << "Ignore the message about this test passing, we always return true.\n";
  return true;
}






template <unsigned int dim>
class TopDownSubclass : public ot::SFC_TreeLoop<dim, ot::Inputs<ot::TreeNode<unsigned int, dim>, double>, ot::Outputs<double>, ot::DefaultSummary, TopDownSubclass<dim>>
{
  using FrameT = ot::Frame<dim, ot::Inputs<ot::TreeNode<unsigned int, dim>, double>, ot::Outputs<double>, ot::DefaultSummary, TopDownSubclass<dim>>;
  public:
    void topDownNodes(FrameT &parentFrame, ot::ExtantCellFlagT *extantChildren)
    {
      ot::sfc_tree_utils::topDownNodes(parentFrame, extantChildren);

      if (this->getCurrentSubtree().getLevel() < 2)
        *extantChildren = (1 << (1u << dim)) - 1;  // All children.
      else
        *extantChildren = 0u;
    }

    void bottomUpNodes(FrameT &parentFrame, ot::ExtantCellFlagT extantChildren)
    {
    }

    void parent2Child(FrameT &parentFrame, FrameT &childFrame)
    {
    }
    void child2Parent(FrameT &parentFrame, FrameT &childFrame)
    {
    }
};


/**
 * testTopDownSubclass()
 */
bool testTopDownSubclass()
{
  constexpr unsigned int dim = 2;
  using C = unsigned int;
  using T = float;

  const unsigned int eleOrder = 1;

  m_uiMaxDepth = 3;

  _InitializeHcurve(dim);

  TopDownSubclass<dim> topdown;
  while (!topdown.isFinished())
  {
    if (!topdown.isPre())
    {
      std::cout << "Returned to subtree \t" << topdown.getSubtreeInfo().getCurrentSubtree() << "\n";
      topdown.next();
    }
    else
    {
      std::cout << "Inspecting subtree \t" << topdown.getSubtreeInfo().getCurrentSubtree() << "\n";
      topdown.step();
    }
  }

  _DestroyHcurve();

  std::cout << "Ignore the message about this test passing, we always return true.\n";
  return true;
}




/**
 * testMatvecSubclass()
 */
bool testMatvecSubclass()
{
  constexpr unsigned int dim = 2;
  using C = unsigned int;
  using T = float;

  const unsigned int eleOrder = 1;

  m_uiMaxDepth = 3;

  _InitializeHcurve(dim);

  ot::MatvecBase<dim, T> treeloop_mvec;

  while (!treeloop_mvec.isFinished())
  {
    if (!treeloop_mvec.isPre())
    {
      std::cout << "Returned to subtree \t" << treeloop_mvec.getSubtreeInfo().getCurrentSubtree() << "\n";
      treeloop_mvec.next();
    }
    else
    {
      std::cout << "Inspecting subtree \t" << treeloop_mvec.getSubtreeInfo().getCurrentSubtree() << "\n";
      treeloop_mvec.step();
    }
  }

  _DestroyHcurve();

  std::cout << "Ignore the message about this test passing, we always return true.\n";
  return true;
}


