#ifndef DENDRO_KT_SEND_UTILIZATION
#define DENDRO_KT_SEND_UTILIZATION

#include "oda.h"
#include "distTree.h"
#include "sfcTreeLoop_matvec_io.h"

namespace bench
{
  template <unsigned int dim>
  long long unsigned computeSendRequired(ot::DA<dim> *octDA, const ot::DistTree<unsigned, dim> &distTree);

  template <unsigned int dim>
  long long unsigned computeLocNonorigin(ot::DA<dim> *octDA, const ot::DistTree<unsigned, dim> &distTree);
}


#endif//DENDRO_KT_SEND_UTILIZATION
