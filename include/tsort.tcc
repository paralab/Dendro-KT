/**
 * @file:tsort.tcc
 * @author: Masado Ishii  --  UofU SoC,
 * @date: 2019-01-11
 */

namespace ot
{

//
// dualTraversal()
//
template <typename T, unsigned int D>
template <typename VisitAction>
void
SFC_Tree<T,D>::dualTraversal(const TreeNode<T,D> *tree1, RankI b1, RankI e1,
                             const TreeNode<T,D> *tree2, RankI b2, RankI e2,
                             LevI sLev, LevI eLev,
                             RotI pRot,
                             VisitAction &visit)
{
  constexpr char numChildren = TreeNode<T,D>::numChildren;
  constexpr unsigned int rotOffset = 2*numChildren;  // num columns in rotations[].

  // Lookup tables to apply rotations.
  const ChildI * const rot_perm = &rotations[pRot*rotOffset + 0*numChildren];
  const RotI * const orientLookup = &HILBERT_TABLE[pRot*numChildren];

  // Scan range to locate children.
  std::array<RankI, numChildren+2> split1, split2;
  SFC_locateBuckets(tree1, b1, e1, sLev, pRot, split1);
  SFC_locateBuckets(tree2, b2, e2, sLev, pRot, split2);
  // The splitter arrays each have numChildren+2 slots, which includes the
  // beginning, middles, and end of the range of children, and ancestors at front.

  bool continueTraversal = visit(tree1, split1, tree2, split2, sLev);
  if (continueTraversal && sLev < eLev)
  {
    // Recurse on non-empty, non-singleton buckets (combining sizes from both trees).
    // Use the splitters to specify ranges for the next level of recursion.
    for (char child_sfc = 0; child_sfc < numChildren; child_sfc++)
    {
      // Columns of HILBERT_TABLE are indexed by the Morton rank.
      // According to Dendro4 TreeNode.tcc:199 they are.
      // (There are possibly inconsistencies in the old code...?
      // Don't worry, we can regenerate the table later.)
      ChildI child = rot_perm[child_sfc] - '0';     // Decode from human-readable ASCII.
      RotI cRot = orientLookup[child];

      // Skip singleton buckets.
      if (split1[child_sfc+2]-split1[child_sfc+1] < 1  ||
          split2[child_sfc+2]-split2[child_sfc+1] < 1 )
        continue;

      dualTraversal(tree1, split1[child_sfc+1], split1[child_sfc+2],
                    tree2, split2[child_sfc+1], split2[child_sfc+2],
                    sLev+1, eLev,
                    cRot,
                    visit);
    }
  }
  
}  // end function()

} // end namespace ot
