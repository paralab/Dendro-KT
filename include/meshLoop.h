
#ifndef DENDRO_KT_MESH_LOOP_H
#define DENDRO_KT_MESH_LOOP_H

#include "treeNode.h"
#include "tsort.h"

#include <vector>


namespace ot
{


template <typename T, unsigned int dim>
class MeshLoopImpl;

// Note: Do NOT set (visitEmpty=true) and (visitPre=false) in the same loop,
// or you will descend to m_uiMaxDepth regardless of the contents of tnlist.
template <typename T, unsigned int dim, bool visitEmpty, bool visitPre, bool visitPost>
class MeshLoopInterface;

template <typename T, unsigned int dim>
using MeshLoopPreSkipEmpty = MeshLoopInterface<T, dim, false, true, false>;

template <typename T, unsigned int dim>
class MeshLoopFrame;


// If (visitPre=false), this means skip the leading side of every subtree.
// If (visitPost=false), this means skip the following side of every subtree.
// If (visitEmpty=true), this means descend down empty subtrees. It is up to the user to define a leaf by calling next().


/**
 * @brief Interface for MeshLoop templated on the type of iteration.
 */
template <typename T, unsigned int dim, bool visitEmpty, bool visitPre, bool visitPost>
class MeshLoopInterface : public MeshLoopImpl<T, dim>
{
  using BaseT = MeshLoopImpl<T,dim>;
  public:
    MeshLoopInterface(TreeNode<T, dim> *tnlist, size_t sz)
      : MeshLoopImpl<T,dim>(tnlist, sz, visitEmpty, visitPre, visitPost)
    {
      if (!visitPre)
        while (!BaseT::isFinished() && BaseT::isPre())
          BaseT::step(visitEmpty, visitPre, visitPost);
    }

    bool step()
    {
      assert((visitPre || visitPost));

      BaseT::step(visitEmpty, visitPre, visitPost);

      if (!visitPre)
        gotoNextPost();

      if (!visitPost)
        gotoNextPre();
    }


    bool next()
    {
      assert((visitPre || visitPost));

      BaseT::next(visitEmpty, visitPre, visitPost);

      if (!visitPre)
        gotoNextPost();

      if (!visitPost)
        gotoNextPre();
    }


    struct Iterator
    {
      MeshLoopInterface &m_ref;
      bool m_markEnd;

      Iterator & operator++() { m_ref.step(); return *this; }

      bool operator!=(const Iterator & other)
      {
        return !((m_ref.isFinished() || m_markEnd) &&
                  (other.m_ref.isFinished() || other.m_markEnd));
      }

      const MeshLoopFrame<T, dim> & operator*() const { return getTopConst(); }

      const MeshLoopFrame<T, dim> & getTopConst() const { return m_ref.getTopConst(); }
    };

    Iterator begin() { return Iterator{*this, false}; }
    Iterator end() { return Iterator{*this, true}; }

  protected:
    bool gotoNextPre()
    {
      while(!BaseT::isFinished() && !BaseT::isPre())
        BaseT::step(visitEmpty, visitPre, visitPost);
      return BaseT::isPre();
    }

    bool gotoNextPost()
    {
      while(!BaseT::isFinished() && BaseT::isPre())
        BaseT::step(visitEmpty, visitPre, visitPost);
      return BaseT::isPre();
    }
};



/**
 * @brief Iterator over TreeNodes (cells in the mesh) with in-place bucketing.
 */
template <typename T, unsigned int dim>
class MeshLoopImpl
{
  public:
    MeshLoopImpl(TreeNode<T, dim> *tnlist, size_t sz, bool vEmpty, bool vPre, bool vPost);
    MeshLoopImpl() = delete;
    bool isPre();
    bool isFinished();
    bool step(bool vEmpty, bool vPre, bool vPost);
    bool next(bool vEmpty, bool vPre, bool vPost);

    const MeshLoopFrame<T, dim> & getTopConst() const { return m_stack.back(); }

  protected:
    using TN = TreeNode<T, dim>;
    static constexpr unsigned int NumChildren = 1u << dim;

    MeshLoopFrame<T, dim> & getTop() { return m_stack.back(); }
    void bucketAndPush(RankI begin, RankI end, LevI lev, RotI pRot);

    std::vector<MeshLoopFrame<T, dim>> m_stack;
    TreeNode<T, dim> *m_ptr;
    size_t m_sz;
};


template <typename T, unsigned int dim>
MeshLoopImpl<T, dim>::MeshLoopImpl(TreeNode<T, dim> *tnlist, size_t sz, bool vEmpty, bool vPre, bool vPost)
  :
    m_ptr(tnlist),
    m_sz(sz)
{
  if ((vPre || vPost) && (sz > 0 || vEmpty))
  {
    RankI begin = 0, end = sz;
    LevI lev = 0;
    RotI pRot = 0;
    bucketAndPush(begin, end, lev, pRot);
  }
}


template <typename T, unsigned int dim>
void MeshLoopImpl<T, dim>::bucketAndPush(RankI begin, RankI end, LevI lev, RotI pRot)
{
  std::array<RankI, NumChildren+1> childSplitters;
  RankI ancStart, ancEnd;

  SFC_Tree<T, dim>::SFC_bucketing(m_ptr, begin, end, lev+1, pRot, childSplitters, ancStart, ancEnd); 

  m_stack.emplace_back(true, begin, end, lev, pRot, std::move(childSplitters), ancStart, ancEnd);
}



template <typename T, unsigned int dim>
bool MeshLoopImpl<T, dim>::isPre()
{
  return (m_stack.size() > 0 && m_stack.back().m_is_pre);
}


template <typename T, unsigned int dim>
bool MeshLoopImpl<T, dim>::isFinished()
{
  return (m_stack.size() == 0);
}



template <typename T, unsigned int dim>
bool MeshLoopImpl<T, dim>::step(bool vEmpty, bool vPre, bool vPost)
{
  if (m_stack.size() == 0)
    throw;  //TODO more specific exception about past the end.
  if (!isPre())
    return next(vEmpty, vPre, vPost);

  m_stack.reserve(m_stack.size() + NumChildren);

  MeshLoopFrame<T, dim> &parentFrame = getTop();
  parentFrame.m_is_pre = false;

  if (parentFrame.m_lev < m_uiMaxDepth)
    for (ChildI child_sfc_rev = 0; child_sfc_rev < NumChildren; ++child_sfc_rev)
    {
      // Figure out child_m and cRot.
      const ChildI child_sfc = NumChildren - 1 - child_sfc_rev;
      const RotI pRot = parentFrame.m_pRot;
      const ChildI * const rot_perm = &rotations[pRot*2 * NumChildren + 0 * NumChildren];
      const ChildI child_m = rot_perm[child_sfc];
      const ChildI cRot = HILBERT_TABLE[pRot * NumChildren + child_m];

      RankI ch_begin = parentFrame.m_child_splitters[child_sfc];
      RankI ch_end = parentFrame.m_child_splitters[child_sfc+1];
      LevI ch_lev = parentFrame.m_lev + 1;

      if ((vPre || vPost) && (ch_end > ch_begin || vEmpty))
      {
        bucketAndPush(ch_begin, ch_end, ch_lev, cRot);
      }
    }

  return isPre();
}

template <typename T, unsigned int dim>
bool MeshLoopImpl<T, dim>::next(bool vEmpty, bool vPre, bool vPost)
{
  m_stack.pop_back();
  return isPre();
}


template <typename T, unsigned int dim>
class MeshLoopFrame
{
  friend MeshLoopImpl<T, dim>;
  using SplitterT = std::array<RankI, (1u<<dim)+1>;

  public:
    MeshLoopFrame() = delete;

    MeshLoopFrame(size_t begin_idx, size_t end_idx, LevI lev, RotI pRot, SplitterT && splitters, RankI anc_begin, RankI anc_end)
      : MeshLoopFrame(true, begin_idx, end_idx, lev, pRot, splitters, anc_begin, anc_end) {}

    MeshLoopFrame(bool is_pre, size_t begin_idx, size_t end_idx, LevI lev, RotI pRot, SplitterT && splitters, RankI anc_begin, RankI anc_end)
      :
        m_is_pre(is_pre),
        m_begin_idx(begin_idx),
        m_end_idx(end_idx),
        m_lev(lev),
        m_pRot(pRot),
        m_child_splitters(splitters),
        m_anc_begin(anc_begin),
        m_anc_end(anc_end)
    { }

    bool get_isPre() const { return m_is_pre; }
    size_t get_begin_idx() const { return m_begin_idx; }
    size_t get_end_idx() const { return m_end_idx; }
    LevI get_lev() const { return m_lev; }
    RotI get_pRot() const { return m_pRot; }
    const SplitterT & get_child_splitters() const { return m_child_splitters; }
    RankI get_anc_begin() const { return m_anc_begin; }
    RankI get_anc_end() const { return m_anc_end; }

    bool is_leaf() const { return m_begin_idx == m_anc_begin && 
                                  m_end_idx == m_anc_end; }


  protected:
    bool m_is_pre;
    size_t m_begin_idx;
    size_t m_end_idx;
    LevI m_lev;
    RotI m_pRot;
    
    SplitterT m_child_splitters;
    RankI m_anc_begin;
    RankI m_anc_end;
};


}//namespace ot



#endif//DENDRO_KT_MESH_LOOP_H
