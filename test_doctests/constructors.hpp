#ifndef DENDRO_KT_TEST_CONSTRUCTORS_HPP
#define DENDRO_KT_TEST_CONSTRUCTORS_HPP

namespace test
{
  template <class Class>
  struct Constructors       // Callable object wrapping constructor overloads
  {
    template <typename ... T>
    Class operator()(T && ... ts) const {
      return Class(std::forward<T>(ts)...);
    }
  };
}

#endif//DENDRO_KT_TEST_CONSTRUCTORS_HPP
