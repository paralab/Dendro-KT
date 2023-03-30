/**
 * @author Masado Ishii
 * @date Feb 3, 2022
 * @brief Nested for loop template, e.g. one loop level per dimension.
 */

#ifndef DENDRO_KT_NESTED_FOR_HPP
#define DENDRO_KT_NESTED_FOR_HPP

namespace tmp
{
  /** nested_for()
   *
   *     Each level has range [begin, end).
   *     Functor inner must takes `levels' arguments.
   *     First argument moves fastest, final argument moves slowest.
   */
  template <int levels, typename ItBegin, typename ItEnd, typename Inner>
  constexpr void nested_for(ItBegin begin, ItEnd end, Inner inner);
}


// implemenations
namespace tmp
{
  namespace detail
  {
    template <int I>  struct IntTag{};

    template <int levels, typename ItBegin, typename ItEnd, typename Inner, typename...Args>
    constexpr void nested_for(IntTag<levels>, ItBegin begin, ItEnd end, Inner inner, Args...args)
    {
      for (ItBegin it = begin; it != end; ++it)
        nested_for(IntTag<levels-1>(), begin, end, inner, it, args...);
    }

    template <typename ItBegin, typename ItEnd, typename Inner, typename ...Args>
    constexpr void nested_for(IntTag<0>, ItBegin, ItEnd, Inner inner, Args...args)
    {
      inner(args...);
    }
  }

  // nested_for()
  template <int levels, typename ItBegin, typename ItEnd, typename Inner>
  constexpr void nested_for(ItBegin begin, ItEnd end, Inner inner)
  {
    detail::nested_for(
        detail::IntTag<levels>(),
        begin,
        end,
        inner);
  }
}

#endif//DENDRO_KT_NESTED_FOR_HPP
