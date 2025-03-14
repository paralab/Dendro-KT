#include "zonelog.h"

namespace zonelog
{
  Log & global_log() { static Log log; return log; }
}
