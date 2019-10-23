/**
 * @file:sfcTreeLoop.h
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-10-23
 * @brief: Stateful const iterator over implicit mesh, giving access to element nodes.
 *         Similar in spirit to eleTreeIterator.h, but hopefully an improvement.
 *
 *         My aim is to make the element loop more flexible and easier to reason about.
 */



/*
 * The recursive structure that is being mimicked:
 *
 * Traverse(subtree, parentData, nodeCoords[], input[], output[])
 * {
 *   // Vectors for children to read/write.
 *   stage_nodeCoords[NumChildren][];
 *   stage_input[NumChildren][];
 *   stage_output[NumChildren][];
 *   childSummaries[NumChildren];
 *
 *   parent2Child(parentData, subtree, nodeCoords, input, output);
 *
 *   UserPreAction(subtree, nodeCoords, input, output);
 *
 *   if (needToDescendFurther)
 *   {
 *     topDownNodes(subtree, nodeCoords,       input,       output,
 *                           stage_nodeCoords, stage_input, stage_output,
 *                           childSummaries);
 *
 *     for (child_sfc = 0; child_sfc < NumChildren; child_sfc++)
 *     {
 *       Traverse(subtree.getChildSFC(child_sfc),
 *                currentData,
 *                stage_nodeCoords[child_sfc],
 *                stage_input[child_sfc],
 *                stage_output[child_sfc]);
 *     }
 *
 *     bottomUpNodes(subtree, nodeCoords,       input,       output,
 *                            stage_nodeCoords, stage_input, stage_output,
 *                            childSummaries);
 *   }
 *
 *   UserPostAction(subtree, nodeCoords, input, output);
 *
 *   child2Parent(parentData, subtree, nodeCoords, input, output);
 * }
 *
 *
 * In the iterative version, the above function will be turned inside out.
 *
 *   - The call stack will be encapsulated inside a stateful iterator.
 *
 *   - The callbacks to UserPreAction() and UserPostAction() will be replaced
 *     by two pairs of entry-exit points whereupon program control is
 *     surrendered and regained at every level.
 *
 *   - The user can decide whether to descend or skip descending, at any level,
 *     by calling step() or next(), respectively.
 */


#ifndef DENDRO_KT_SFC_TREE_LOOP_H
#define DENDRO_KT_SFC_TREE_LOOP_H

#include "nsort.h"
#include "tsort.h"
#include "treeNode.h"
#include "mathUtils.h"
#include "binUtils.h"

/// #include "refel.h"
/// #include "tensor.h"


#include <vector>


namespace ot
{

  // ------------------------------
  // Class declarations
  // ------------------------------
  template <unsigned int dim>
  class SFC_TreeLoop;
  
  template <unsigned int dim>
  class Frame;
  
  class SubtreeInfo;
  
  // ------------------------------
  // Class definitions
  // ------------------------------
  
  //
  // Frame
  //
  template <unsigned int dim>
  class Frame
  {
    using C = unsigned int;
    constexpr unsigned int NumChildren = 1u << dim;

    template <unsigned int dim>
    friend SFC_TreeLoop;

    public:
      /// std::vector<int> &myDataRef;
      /// std::vector<int> childDataStore[NumChildren];  //TODO stubs
      std::vector<TreeNode<C,dim>> &myIncidentNodes;
      std::vector<TreeNode<C,dim>> childIncidentNodes[NumChildren];
      //input
      //output
  
    private:
      Frame *m_parentFrame;
      TreeNode<C,dim> m_currentSubtree;
      bool m_isPre;
      unsigned int m_numExtantChildren;
  };


  //
  // SubtreeInfo
  //
  class SubtreeInfo
  {

  };


  //
  // SFC_TreeLoop
  //
  template <unsigned int dim>
  class SFC_TreeLoop
  {
    constexpr unsigned int NumChildren = 1u << dim;

    protected:
      Frame m_rootFrame;            // Should be init'd on construction, used for resetting.
      std::vector<Frame> m_stack;

      // More stack-like things.

    public:
      // reset()
      void reset()
      {
        // stack.clear();
        m_stack.push_back(m_rootFrame);
        m_stack.back().m_isPre = true;
      }

      // getSubtreeInfo()
      SubtreeInfo getSubtreeInfo()
      {
        return SubtreeInfo();  //TODO
      }

      // step()
      void step()
      {
        if (stack.back().m_isPre)
        {
          stack.back().m_isPre = false;
          Frame &parentFrame = stack.back();
          parentFrame.m_extantChildren = (1u << (1u << dim)) - 1;
          // TODO push child frames in reverse order, use rotation table.
          // Set m_parentFrame, m_currentSubtree, m_isPre.
          // Figure out which children exist in the domain.
          // childFrame = Frame{parentFrame.childDataStore[child_sfc], ...};
          topDownNodes(parentFrame, parentFrame.m_extantChildren);  // Free to resize children buffers.

          if (numExtantChildren > 0)
            // Enter the new top frame, which represents the 0th child.
            parent2Child(*stack.back().m_parentFrame, stack.back());
          else
            bottomUpNodes(parentFrame);
        }
        else         // After a recursive call, can't step immediately.
          next();
      }

      // next()
      void next()
      {
        child2Parent(*stack.back().m_parentFrame, stack.back());
        stack.pop_back();
        // Return to the parent level.

        if (stack.back().m_isPre)
          // Enter the new top frame, which represents some other child.
          parent2Child(*stack.back().m_parentFrame, stack.back());
        else
          bottomUpNodes(stack.back(), stack.back().m_extantChildren);
      }

      //TODO somehow get the set of extant children so we don't traverse empty space.
      virtual void topDownNodes(Frame &parentFrame, unsigned long extantChildren) = 0;
      virtual void bottomUpNodes(Frame &parentFrame, unsigned long extantChildren) = 0;
      virtual void parent2Child(Frame &parentFrame, Frame &childFrame) = 0;
      virtual void child2Parent(Frame &parentFrame, Frame &childFrame) = 0;
  };

}


#endif//DENDRO_KT_SFC_TREE_LOOP_H
