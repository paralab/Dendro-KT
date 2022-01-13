#ifndef DENDRO_KT_IDX_H
#define DENDRO_KT_IDX_H

namespace idx
{
  class LocalIdx
  {
    public:
      size_t m_idx;
      explicit LocalIdx(size_t idx) : m_idx(idx) {}
      operator size_t() const { return m_idx; }
  };

  class GhostedIdx
  {
    public:
      size_t m_idx;
      explicit GhostedIdx(size_t idx) : m_idx(idx) {}
      operator size_t() const { return m_idx; }
  };

  class Fine
  {
    public:
      size_t m_idx;
      explicit Fine(size_t idx) : m_idx(idx) {}
      operator size_t() const { return m_idx; }
  };
}

#endif//DENDRO_KT_IDX_H
