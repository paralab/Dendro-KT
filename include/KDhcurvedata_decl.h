/**
 * @author Masado Ishii
 * @date   2019-01-18
 */

#ifndef DENDRO_KT_HCURVEDATA_DECL_H
#define DENDRO_KT_HCURVEDATA_DECL_H

#include <cstring>

template <int pDim>
struct HilbertData
{
  static void copyData(char *rotations, int *HILBERT_TABLE);
  static const char m_rotations[];
  static const int m_HILBERT_TABLE[];
};

template <int pDim>
void HilbertData<pDim>::copyData(char *rotations, int *HILBERT_TABLE)
{
  memcpy(rotations, m_rotations, sizeof(m_rotations));
  memcpy(HILBERT_TABLE, m_HILBERT_TABLE, sizeof(m_HILBERT_TABLE));
}

#endif // DENDRO_KT_HCURVEDATA_DECL_H
