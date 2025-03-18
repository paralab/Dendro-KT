#include "zonelog.h"
#include "zonelog_call_trie.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <locale>
#include <iostream>

namespace zonelog
{
  //future: Make new class for this secondary aggregation (call trie -> calls).

  std::ostream & operator<<(std::ostream &out, const SumCalls &aggregator)
  {
    const std::map<size_t, clock::duration>
    self_durations =
        [&call_trie = std::as_const(aggregator.call_trie),
         &code_path_properties = std::as_const(aggregator.code_path_properties)]()
    {
      // Self time = (total time) - (total time of all children) >= 0.

      // Initialize self time to total time.
      // This should happen before subtraction to prevent underflows.
      std::map<size_t, clock::duration> self_durations;
      for (auto [code_path, code_path_property]: code_path_properties)
      {
        self_durations[code_path] = code_path_property.duration;
      }

      // Subtract each total time from parent self time.
      // (Yes, this revisits parents, but avoids depth-first-search or sorting.)
      for (auto [code_path, code_path_property]: code_path_properties)
      {
        const size_t parent = call_trie.parent(code_path);
        if (not call_trie.is_root(parent))
          self_durations[parent] -= code_path_property.duration;
      }
      return self_durations;
    }();

    // Aggregate. Code path context is unneeded after extracting self time.
    using CallProperty = SumCalls::CallProperty;
    using CodePathProperty = SumCalls::CodePathProperty;
    using Key = std::tuple<uintptr_t, RawEventData>;
    std::map<Key, CallProperty> call_properties;
    for (auto [code_path, zone_w_data] : aggregator.call_trie.view())
    {
      const CodePathProperty property = aggregator.code_path_properties.at(code_path);
      const long int count = property.count;
      const clock::duration duration = property.duration;
      const clock::duration self_duration = self_durations.at(code_path);
      const Key key = {hash_bytes(zone_w_data.zone_code), zone_w_data.data};
      call_properties[key] +=
          CallProperty{ zone_w_data, count, duration, self_duration };
    }

    // Print out (in unspecified order)
    for (const CallProperty property : std::views::values(call_properties))
    {
      const ZonePtrWData zone_w_data = property.zone_w_data;
      const Zone *zone = zone_decode(zone_w_data.zone_code);
      const bool has_data = zone_w_data.has_data;
      const EventData zone_data {zone_w_data.data};

      const long int count = property.count;

      const double microseconds =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
          property.duration).count() / 1000.0;

      const double self_microseconds = 
          std::chrono::duration_cast<std::chrono::nanoseconds>(
          property.self_duration).count() / 1000.0;

      out << fmt::format(
          std::locale("en_US.UTF-8"), //thousands separators
          "{:>10.2Lf}\t: {:>10.2Lf}\t/ {:>10L}\t= {:>10.2Lf}\t: {:>10.2Lf}\t",
          microseconds,
          self_microseconds,
          count,
          microseconds / count,
          self_microseconds / count);
      if (has_data)
      {
        out << fmt::format("{}.{}({}-{})\t[{}]\n",
            zone->function,
            zone->name,
            zone_data.array[0],
            zone_data.array[1],
            zone->pretty_function);
      }
      else
      {
        out << fmt::format("{}.{}\t[{}]\n",
            zone->function,
            zone->name,
            zone->pretty_function);
      }
    }

    return out;
  }
}
