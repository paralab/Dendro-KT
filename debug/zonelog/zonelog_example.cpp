
#include <iostream>
#include <iomanip>

#include <locale>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "zonelog.hpp"

int fib_inner(int n);

int fib(int n)
{
  ZONELOG_SCOPE_DATA(zonelog::EventData(n, n));

  if (n <= 6)
    return fib_inner(n);

  if (n < 2)
    return 1;

  const volatile int zero = 0;
  return fib(n - 1) + fib(n - 2) + zero;
}

int fib_inner(int n)
{
  ZONELOG_SCOPE();

  if (n < 2)
    return 1;

  const volatile int zero = 0;
  return fib_inner(n - 1) + fib_inner(n - 2) + zero;
}

int main()
{
  const int n = 10;

  //warmup
  volatile int result;
  result = fib(n);
  zonelog::global_log().clear();

  /// zonelog::offline::SumTopDown log_stats;
  zonelog::offline::SumCalls log_stats;
  do
  {
    ZONELOG_NAMED_SCOPE("outer_2");
    ZONELOG_NAMED_SCOPE("outer_1");
    const int repetitions = 10000;
    for (int repeat = 0; repeat < repetitions; ++repeat)
    {
      ZONELOG_NAMED_SCOPE("inner_2");
      ZONELOG_NAMED_SCOPE("inner_1");
      result = fib(n);

      zonelog::offline::flush_aggregate(zonelog::global_log(), log_stats);
    }
  }
  while (false);
  zonelog::offline::flush_aggregate(zonelog::global_log(), log_stats);
  log_stats.print_results();
  std::cout << "global_log().max_size() = " << zonelog::global_log().max_size() << "\n";

  return 0;
}
