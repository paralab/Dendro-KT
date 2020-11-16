#ifndef DENDRO_KT_SEND_UTILIZATION
#define DENDRO_KT_SEND_UTILIZATION

#include "oda.h"
#include "sfcTreeLoop_matvec_io.h"

namespace bench
{
  template <unsigned int dim>
  long long unsigned computeSendRequired(const ot::DA<dim> *octDA);
}


#endif//DENDRO_KT_SEND_UTILIZATION
