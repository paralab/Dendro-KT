#ifndef DENDRO_KT_PASSIVE_SHARED_PTR
#define DENDRO_KT_PASSIVE_SHARED_PTR

#include <memory>

namespace ownership
{
  const auto no_op = [](auto){};  // future: constexpr

  template <typename T>
  inline std::shared_ptr<T> passive_shared_ptr(T *ptr)  //note: beware const
  {
    return std::shared_ptr<T>(ptr, no_op);  // passive because deleter is no-op
  }
}

#endif//DENDRO_KT_PASSIVE_SHARED_PTR
