#include "zonelog.h"
#include "zonelog_call_trie.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <locale>
#include <ostream>
#include <stack>
#include <vector>
#include <tuple>
#include <assert.h>

namespace zonelog
{
  //future: Make new class for this secondary aggregation (call trie -> calls).


  auto CallTrie::self_durations(
      const Trie<ZonePtrWData> &zone_trie,
      const std::map<size_t, CodePathProperty> &code_path_properties)
    -> std::map<size_t, clock::duration>
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
      const size_t parent = zone_trie.parent(code_path);
      if (not zone_trie.is_root(parent))
        self_durations[parent] -= code_path_property.duration;
    }
    return self_durations;
  }


  std::ostream & CallTrie::print_zone_list(std::ostream &out) const
  {
    const std::map<size_t, clock::duration>
    self_durations = CallTrie::self_durations(
        this->zone_trie, this->code_path_properties);

    struct CallProperty {
      ZonePtrWData zone_w_data = {};
      long int count = 0;
      clock::duration duration = {};
      clock::duration self_duration = {};
      clock::time_point last_exit = clock::time_point::min();

      CallProperty & operator+=(const CallProperty &y)
      {
        zone_w_data    = y.zone_w_data;
        count         += y.count;
        duration      += y.duration;
        self_duration += y.self_duration;
        last_exit      = std::max(last_exit, y.last_exit);
        return *this;
      }
    };

    // Aggregate. Code path context is unneeded after extracting self time.
    using CodePathProperty = CallTrie::CodePathProperty;
    using Key = std::tuple<uintptr_t, RawEventData>;
    std::map<Key, CallProperty> call_properties;
    for (auto [code_path, zone_w_data] : this->zone_trie.hash_key_view())
    {
      const CodePathProperty property = this->code_path_properties.at(code_path);
      const long int count = property.count;
      const clock::duration duration = property.duration;
      const clock::duration self_duration = self_durations.at(code_path);
      const clock::time_point last_exit = property.last_exit;
      const Key key = {hash_bytes(zone_w_data.zone_code), zone_w_data.data};
      call_properties[key] +=
          CallProperty{ zone_w_data, count, duration, self_duration, last_exit };
    }

    // Sort by last exit.
    std::vector<CallProperty> sorted_properties = [&](){
        auto && r = std::views::values(call_properties);
        return std::vector(r.begin(), r.end());
    }();
    std::ranges::sort(sorted_properties, {}, &CallProperty::last_exit);

    // Header
    out << fmt::format(
        "{:^13s}\t{:^13s}\t{:^10s}\t{:^13s}\t{:^13s}\t",
        "Time(μs)",
        "Self(μs)",
        "Count",
        "Time/1(μs)",
        "Self/1(μs)");
    out << fmt::format("{:^24s}\t[{}]:{}\n",
        "Zone", "Function", "Line");

    // Print out (in unspecified order)
    for (const CallProperty property : sorted_properties)
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
          "{:>13.2Lf}\t{:>13.2Lf}\t{:>10L}\t{:>13.2Lf}\t{:>13.2Lf}\t",
          microseconds,
          self_microseconds,
          count,
          microseconds / count,
          self_microseconds / count);
      if (has_data)
      {
        out << fmt::format("{:<24s}\t",
            fmt::format("{}.{}({}:{})",
                zone->function,
                zone->name,
                zone_data.array[0],
                zone_data.array[1]));
      }
      else
      {
        out << fmt::format("{:<24s}\t",
            fmt::format("{}.{}",
            zone->function,
            zone->name));
      }
      out << fmt::format("[{}]:{}\n",
          zone->pretty_function,
          zone->line);
    }

    return out;
  }



  enum JointComp : uint8_t {
    empty = 0u, up = 1u<<0, down = 1u<<1, left = 1u<<2, right = 1u<<3
  };

  class TreeLines
  {
    static constexpr std::array<const char *, 16> joints = []() constexpr
    {
      std::array<const char *, 16> joints = {};
      joints.fill("@");
      joints[empty]               = " ";
      joints[right | left]        = "─";
      joints[right | up]          = "└";
      joints[right | down]        = "┌";
      joints[right | left | up]   = "┴";
      joints[right | left | down] = "┬";
      joints[right | up | down]   = "├";
      joints[up | down]           = "│";
      return joints;
    }();

    public:
      std::string node(int depth, bool is_parent, bool is_furthest = false);
    private:
      static std::string render(const std::vector<uint8_t> &row);
    private:
      std::vector<uint8_t> row;
  };


  std::string TreeLines::node(int depth, bool is_nonleaf, bool is_furthest)
  {
    row.resize(depth + 1, empty);

    constexpr bool parent_first = true;
    constexpr bool parent_last  = not parent_first;

    uint8_t dummy;
    uint8_t & parent_lane = (depth > 0 ? row[depth - 1] : dummy);
    uint8_t & child_lane = row[depth];

    std::string node_string;

    // Lines are 'sibling buses' with joints at nodes.
    // Components of joints depend on the type and sequence of the node.
    //             [ bus ]                                  [ node joint ]
    //   right   any node                                 any node
    //   left    (never)                                  any child (or root)
    //   up      any child (except first, if furthest)    parent (if last)
    //   down    any child (except last, if furthest)     parent (if first)

    if (parent_first)
    {
      // Bus: a nonleaf --> the parent lane of its furthest child.
      child_lane = left | right | (is_nonleaf ? down : empty);
      parent_lane =       right | up    | (not is_furthest ? down : empty);

      node_string = render(this->row);

      parent_lane = (not is_furthest ? (up | down) : empty);
    }
    else
    {
      // Bus: a nonleaf <-- the parent lane of its furthest child.
      parent_lane =       right | down  | (not is_furthest ? up : empty);
      child_lane = left | right | (is_nonleaf ? up : empty);

      node_string = render(this->row);

      parent_lane = empty;
    }

    return node_string;
  }

  std::string TreeLines::render(const std::vector<uint8_t> &row)
  {
    std::string string;
    for (uint8_t j: row)
      string += joints[j];
    return string;
  }


  std::ostream & CallTrie::print_call_trie(std::ostream &out) const
  {
    const std::map<size_t, clock::duration>
    self_durations = CallTrie::self_durations(
        this->zone_trie, this->code_path_properties);

    const auto node_last_exit = [this](size_t node) -> clock::time_point
    {
      return this->code_path_properties.at(node).last_exit;
    };

    // Recover top-down parent->child structure.
    // (Entry/exit timestamps would be unreliable due to dynamic control flow.)
    std::multimap<size_t, size_t> c2p;
    for (auto [hash, parent] : this->zone_trie.hash_parent_view())
    {
      c2p.insert({parent, hash});
    }

    // Depth-first traversal.
    // [parents first]  [siblings by last exit]
    struct Node { size_t hash; size_t parent_hash; };
    std::vector<Node> node_list;
    {
      std::stack<Node, std::vector<Node>> stack;

      std::vector<size_t> siblings;
      siblings.reserve(64);

      const auto push_children = [&](auto node, const auto &tree, auto &stack)
      {
        siblings.clear();
        for (auto [b, e] = tree.equal_range(node);
            auto [parent, child] : std::ranges::subrange(b, e))
        {
          assert(parent == node);
          siblings.push_back(child);
        }

        std::ranges::sort(siblings, {}, node_last_exit);

        for (size_t child : siblings | std::views::reverse)
          stack.push(Node{child, node});
      };

      push_children(this->zone_trie.root(), c2p, stack);
      while (not stack.empty())
      {
        const Node node = stack.top();
        stack.pop();
        node_list.push_back(node);
        push_children(node.hash, c2p, stack);
      }
    };

    // The print list is depth-first, parent-first.
    // Compute depths, nonleafs, furthest children.
    // In case parent-last is chosen, traverse in the opposite orders.

    // Depths top-down.
    int max_depth = 0;
    std::vector<int> depths;
    {
      std::stack<size_t, std::vector<size_t>> lineage;
      lineage.push(this->zone_trie.root());
      for (Node node : node_list)
      {
        while (node.parent_hash != lineage.top())
          lineage.pop();
        const int depth = lineage.size() - 1;
        lineage.push(node.hash);

        depths.push_back(depth);
        max_depth = std::max(max_depth, depth);
      }
    };

    // Furthest children and nonleafs bottom-up.
    std::vector<uint8_t> is_furthest;
    std::vector<uint8_t> is_nonleaf;
    {
      std::stack<size_t, std::vector<size_t>> parents;
      for (Node node : node_list | std::views::reverse)
      {
        bool had_children = false;
        while (not parents.empty() and parents.top() == node.hash)
        {
          had_children = true;
          parents.pop();
        }

        const bool furthest = parents.empty() or parents.top() != node.parent_hash;

        parents.push(node.parent_hash);

        is_furthest.push_back(furthest);
        is_nonleaf.push_back(had_children);
      }
      std::ranges::reverse(is_furthest);
      std::ranges::reverse(is_nonleaf);
    }

    // Header
    out << fmt::format(
        "{:^13s}\t{:^13s}\t{:^10s}\t{:^13s}\t{:^13s}\t",
        "Time(μs)",
        "Self(μs)",
        "Count",
        "Time/1(μs)",
        "Self/1(μs)");
    out << fmt::format("{:^{}s}\t", "🌲", max_depth + 1);
    out << fmt::format("{:^24s}\t[{}]:{}\n",
        "Zone", "Function", "Line");

    TreeLines hierarchy_state;
    for (size_t i = 0; Node node : node_list)
    {
      const std::string hierarchy_slice = hierarchy_state.node(
          depths[i], is_nonleaf[i], is_furthest[i]);

      const ZonePtrWData zone_w_data = this->zone_trie.key(node.hash);
      const Zone *zone = zone_decode(zone_w_data.zone_code);
      const bool has_data = zone_w_data.has_data;
      const EventData zone_data {zone_w_data.data};

      const CodePathProperty property = this->code_path_properties.at(node.hash);
      const clock::duration self_duration = self_durations.at(node.hash);

      const long int count = property.count;

      const double microseconds =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
          property.duration).count() / 1000.0;

      const double self_microseconds = 
          std::chrono::duration_cast<std::chrono::nanoseconds>(
          self_duration).count() / 1000.0;

      out << fmt::format(
          std::locale("en_US.UTF-8"), //thousands separators
          "{:>13.2Lf}\t{:>13.2Lf}\t{:>10L}\t{:>13.2Lf}\t{:>13.2Lf}\t",
          microseconds,
          self_microseconds,
          count,
          microseconds / count,
          self_microseconds / count);

      out << fmt::format("{:<{}s}\t", hierarchy_slice, max_depth + 1);

      if (has_data)
      {
        out << fmt::format("{:<24s}\t",
            fmt::format("{}.{}({}:{})",
                zone->function,
                zone->name,
                zone_data.array[0],
                zone_data.array[1]));
      }
      else
      {
        out << fmt::format("{:<24s}\t",
            fmt::format("{}.{}",
            zone->function,
            zone->name));
      }

      out << fmt::format("[{}]:{}\n",
          zone->pretty_function,
          zone->line);

      ++i;
    }

    return out;
  }

}
