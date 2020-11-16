
#include "sendUtilization.h"


namespace bench
{
  template <unsigned int dim>
  long long unsigned computeSendRequired(const ot::DA<dim> *octDA)
  {
    throw std::logic_error("Not implemented!");
    return 0;
  }



  template long long unsigned computeSendRequired<2>(const ot::DA<2> *octDA);
  template long long unsigned computeSendRequired<3>(const ot::DA<3> *octDA);
  template long long unsigned computeSendRequired<4>(const ot::DA<4> *octDA);

}

