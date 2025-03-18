#include "zonelog.h"

namespace zonelog
{
#ifdef ZONELOG_PREALLOCATION
  constexpr size_t preallocation = (ZONELOG_PREALLOCATION);
#else
  constexpr size_t preallocation = 4u << 10; // 4 KiB
#endif//ZONELOG_PREALLOCATION

  Log & global_log() { static Log log; return log; }
}
