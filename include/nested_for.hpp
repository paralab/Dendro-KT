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

  /** nested_for_rect()
   *
   *     Each level i has range [begin[i], end[i]).
   *     Functor inner must takes `levels' arguments.
   *     First argument (level 0) moves fastest, final argument moves slowest.
   */
  template <int levels, typename BeginArray, typename EndArray, typename Inner>
  constexpr void nested_for_rect(const BeginArray &begin, const EndArray &end, Inner inner);
}


// implemenations
namespace tmp
{
  namespace detail
  {
    template <int I>  struct IntTag{};

    // nested_for()
    template <int levels, typename ItBegin, typename ItEnd, typename Inner, typename...Args>
    constexpr void nested_for(IntTag<levels>, ItBegin begin, ItEnd end, Inner inner, const Args &...args)
    {
      for (ItBegin it = begin; it != end; ++it)
        nested_for(IntTag<levels-1>(), begin, end, inner, it, args...);
    }

    // nested_for() (base)
    template <typename ItBegin, typename ItEnd, typename Inner, typename ...Args>
    constexpr void nested_for(IntTag<0>, ItBegin, ItEnd, Inner inner, const Args &...args)
    {
      inner(args...);
    }


    // nested_for_rect()
    template <int levels, typename BeginArray, typename EndArray, typename Inner, typename...Args>
    constexpr void nested_for_rect(IntTag<levels>, const BeginArray &begin, const EndArray &end, Inner inner, const Args &...args)
    {
      auto end_it = end[levels - 1];
      for (auto it = begin[levels - 1]; it != end_it; ++it)
        nested_for_rect(IntTag<levels-1>(), begin, end, inner, it, args...);
    }

    // nested_for_rect() (base)
    template <typename BeginArray, typename EndArray, typename Inner, typename...Args>
    constexpr void nested_for_rect(IntTag<0>, const BeginArray &, const EndArray &, Inner inner, const Args &...args)
    {
      inner(args...);
    }


    //future: A tuple of ranges is also possible in this style, using std::get.
    //        Not recommended where an array would suffice.
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

  // nested_for_rect()
  template <int levels, typename BeginArray, typename EndArray, typename Inner>
  constexpr void nested_for_rect(const BeginArray &begin, const EndArray &end, Inner inner)
  {
    detail::nested_for_rect(
        detail::IntTag<levels>(),
        begin,
        end,
        inner);
  }

}

#endif//DENDRO_KT_NESTED_FOR_HPP
