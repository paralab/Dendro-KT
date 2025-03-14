
#include <iostream>
#include <iomanip>

#include <locale>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "zonelog.hpp"


int fib(int n)
{
  ZONELOG_SCOPE_DATA(static_cast<uint64_t>(n));

  if (n < 2)
    return 1;

  return fib(n - 1) + fib(n - 2);
}

int main()
{
  const int n = 5;

  //warmup
  volatile int result;
  result = fib(n);
  zonelog::global_log().erase_to_front();

  /// zonelog::offline::SumTopDown log_stats;
  zonelog::offline::SumCalls log_stats;
  do
  {
    ZONELOG_NAMED_SCOPE_DATA("outer", 2u);
    ZONELOG_NAMED_SCOPE_DATA("outer", 1u);
    const int repetitions = 2000;
    for (int repeat = 0; repeat < repetitions; ++repeat)
    {
      ZONELOG_NAMED_SCOPE_DATA("inner", 2u);
      ZONELOG_NAMED_SCOPE_DATA("inner", 1u);
      result = fib(n);
    }
  }
  while (false);
  zonelog::offline::flush_aggregate(zonelog::global_log(), log_stats);
  log_stats.print_results();

  std::cout << "\n";
  std::cout << "Log size = " << zonelog::global_log().size() << " bytes.\n";
  zonelog::offline::SumCalls alloc_log_stats;
  zonelog::offline::flush_aggregate(zonelog::internal::debug_log(), alloc_log_stats);
  alloc_log_stats.print_results();

  return 0;
}
