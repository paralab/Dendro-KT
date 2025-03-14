#ifndef ZONELOG_HPP
#define ZONELOG_HPP

#include <type_traits>
#include <chrono>
#include <sstream>
#include <array>
#include <map>
#include <string_view>
#include <stdexcept>
#include <ranges>

#define ZONELOG_SCOPE()          ZONELOG_NAMED_SCOPE("")
#define ZONELOG_SCOPE_DATA(data) ZONELOG_NAMED_SCOPE_DATA("", data)

#define ZONELOG_NAMED_SCOPE(name) \
  ZONELOG_NAMED_SCOPE_impl(name, __COUNTER__)

#define ZONELOG_NAMED_SCOPE_DATA(name, data) \
  ZONELOG_NAMED_SCOPE_DATA_impl(name, __COUNTER__, data)

#define ZONELOG_NAMED_SCOPE_impl(name, anon) \
  static constexpr zonelog::Zone  ZONELOG_ZONE(anon) = \
      { name, __func__, __FILE__, __LINE__ }; \
  zonelog::online::ScopeGuard     ZONELOG_GUARD(anon) = \
      { & ZONELOG_ZONE(anon) };

#define ZONELOG_NAMED_SCOPE_DATA_impl(name, anon, data) \
  static constexpr zonelog::Zone  ZONELOG_ZONE(anon) = \
      { name, __func__, __FILE__, __LINE__ }; \
  zonelog::online::ScopeGuard     ZONELOG_GUARD(anon) = \
      { & ZONELOG_ZONE(anon), zonelog::EventData(data) };

#define ZONELOG_ZONE(anon) ZONELOG_CAT(zl_zone_, anon)
#define ZONELOG_GUARD(anon) ZONELOG_CAT(zl_guard_, anon)

#define ZONELOG_CAT(X, Y) ZONELOG_CAT_impl(X, Y)
#define ZONELOG_CAT_impl(X, Y) X ## Y

namespace zonelog
{
  struct alignas(8) Zone
  {
    char const *name;
    char const *function;
    char const *file;
    long int line;
  };

  auto zone_encode(const Zone * zone) -> uintptr_t { return reinterpret_cast<uintptr_t>(zone); }
  auto zone_decode(uintptr_t code) -> const Zone * { return reinterpret_cast<const Zone *>(code); }

  struct ZoneAction
  {
    uintptr_t zone_code : sizeof(uintptr_t) * CHAR_BIT - 2;
    bool pop : 1;
    bool with_data: 1;
  };
  ZoneAction zone_push(const Zone *zone, bool with_data = false) { return { zone_encode(zone), false, with_data}; }
  ZoneAction zone_pop(const Zone *zone, bool with_data = false)  { return { zone_encode(zone), true, with_data }; }
  uintptr_t zone_code(ZoneAction za) { return za.zone_code; }
  const Zone * zone(ZoneAction za) { return zone_decode(zone_code(za)); }
  bool action_is_pop(ZoneAction za) { return za.pop; }
  bool is_push(ZoneAction za) { return not za.pop; }
  bool has_data(ZoneAction za) { return za.with_data; }

  using clock = std::chrono::steady_clock;

  struct Event
  {
    struct Mark {
      ZoneAction zone_action;
      clock::time_point time_stamp;
    } mark = {};

    std::array<uint64_t, 2> data = {};
  };

  struct EventData
  {
    EventData() = default;
    explicit EventData(uint64_t a0)              : array{{ a0 }} { }
    explicit EventData(uint64_t a0, uint64_t a1) : array{{ a0, a1 }} { }
    explicit EventData(std::array<uint64_t, 2> a) : array(a) { }

    operator std::array<uint64_t, 2>() const { return array; }

    std::array<uint64_t, 2> array = {};
  };

  using RawEventData = decltype(EventData::array);

  struct ZonePtrWData
  {
    uintptr_t zone_code = zone_encode(nullptr);
    EventData data = {};
    uint64_t has_data = false;
  };

#ifdef ZONELOG_PREALLOCATION
  static constexpr size_t preallocation = (ZONELOG_PREALLOCATION);
#else
  static constexpr size_t preallocation = 4u << 10; // 8 KiB
#endif//ZONELOG_PREALLOCATION

  class Log
  {
    public:
      Log()
      {
        m_stream.str(std::string(preallocation, '\0'));
        m_stream.seekg(0);
        m_stream.seekp(0);
      }

      void write(Event event)
      {
        m_stream.write(reinterpret_cast<const char *>(&event.mark), sizeof(event.mark));
        if (has_data(event.mark.zone_action))
          m_stream.write(reinterpret_cast<const char *>(&event.data), sizeof(event.data));
      }

      Event read()
      {
        Event event = {};
        m_stream.read(reinterpret_cast<char *>(&event.mark), sizeof(event.mark));
        if (has_data(event.mark.zone_action))
          m_stream.read(reinterpret_cast<char *>(&event.data), sizeof(event.data));
        return event;
      }

      bool has_unread() const
      {
        return m_stream.tellg() < m_stream.tellp();
      }

      void clear()
      {
        m_max_size = std::max(m_max_size, m_stream.view().size());
        m_stream.seekg(0);
        m_stream.seekp(0);
      }

      size_t max_size() const
      {
        return m_max_size;
      }

    private:
      mutable std::stringstream m_stream;
      size_t m_max_size;
  };

  //future: shared pointer or reference counting so libs cooperate w/o fiasco
  Log & global_log() { static Log log; return log; }

  namespace online
  {
    static constexpr Zone alloc_zone = { "alloc", "", __FILE__, __LINE__ };

    namespace internal
    {
      void log_now(Log &log, ZoneAction zone_action, EventData data)
      {
        log.write(Event{{zone_action, clock::now()}, data});
      }
    }

    void log_push(Log &log, const Zone *zone, EventData data)
    {
      internal::log_now(log, zone_push(zone, true), data);
    }

    void log_pop(Log &log, const Zone *zone, EventData data)
    {
      internal::log_now(log, zone_pop(zone, true), data);
    }

    void log_push(Log &log, const Zone *zone)
    {
      internal::log_now(log, zone_push(zone, false), EventData{});
    }

    void log_pop(Log &log, const Zone *zone)
    {
      internal::log_now(log, zone_pop(zone, false), EventData{});
    }

    class ScopeGuard
    {
      public:
        ScopeGuard(const Zone *zone)
          : zone(zone)
        {
          log_push(global_log(), zone);
        }
        ScopeGuard(const Zone *zone, EventData input_data)
          : zone(zone)
        {
          log_push(global_log(), zone, input_data);
        }
        ~ScopeGuard()
        {
          log_pop(global_log(), zone);
        }
      private:
        const Zone * const zone;
    };
  }

  namespace offline
  {
    template <class Aggregator>
    void flush_aggregate(Log &log, Aggregator &aggregator)
    {
      while (log.has_unread())
      {
        Event event = log.read();
        aggregator.consume_event(event);
      }
      log.clear();
    }



    // -----------------------------------------------------------------------
    // Radix tree / trie on comparable data
    // -----------------------------------------------------------------------

    // A node is uniquely defined as a sequence of keys.
    // But we assign a 64-bit hash of the sequence as its name.
    // Detect collisions by inspecting whether parents match.

    // map arbitrary data to nodes in the tree.
    //   a different mapping data structure can be used for this part.

    // Key must be a standard layout type without padding.
    // And the alignment must be equal to that of size_t.
    template <typename Key>
    size_t hash_bytes(Key key)
    {
      return std::hash<std::string_view>{}(
          std::string_view(
            reinterpret_cast<char *>(&key),
            sizeof(key)));
    };


    // -------------------------------------------------------------------------
    //
    // Tree: Tree in which children are unsequenced.
    // Treq: Tree in which children are SEQuenced.
    // Trap: Tree in which children are labeled by unique dictionary (MAP) keys.
    //
    // -------------------------------------------------------------------------
    //
    // Treq[T] = (root: T, children: Seq[Treq[T]])
    //
    // pre_nodes: Treq[T] -> Seq[T] :=
    //   (root: T, children: Seq[Treq[T]]) -> seq(root) ++ (join map depth_pre_order children)
    //
    // post_nodes: Treq[T] -> Seq[T] :=
    //   (root: T, children: Seq[Treq[T]]) -> (join map depth_pre_order children) ++ seq(root)
    //
    // leafs: Treq[T] -> Seq[T] :=
    //   (root: T, children: Empty) -> seq(root)
    //   (root: T, children: Seq[Treq[T]]) -> join map leafs children
    //
    // levels: Treq[T] -> Seq[Seq[T]]  # zip is transpose-like, refolding
    //   (root: T, children: Empty) -> seq(seq(root))
    //   (root: T, children: Seq[Treq[T]]) -> seq(seq(root)) ++ ziplong (map levels children)
    //
    // where
    //   zip2 Seq[A] Seq[B] -> Seq[(A,B)] :=
    //     (head_a, tail_a) (head_b, tail_b) -> ((head_a, head_b), zip tail_a tail_b)
    //     Empty seq_b -> Empty
    //     seq_a Empty -> Empty
    //
    //   ziplong2 Seq[A] Seq[B] -> Seq[(A,B) or (A) or (B)] :=
    //     (head_a, tail_a) (head_b, tail_b) -> ((head_a, head_b), zip tail_a tail_b)
    //     Empty (head_b tail_b) -> (seq(head_b), zip tail_b)
    //     (head_a tail_a) Empty -> (seq(head_a), zip tail_a)
    //
    //   ziplong Seq[Seq[T]] -> Seq[Seq[T]] :=  # incidentally, 0-aligns rows
    //     seqs ->
    //       let nonemptys = filter (not eq Empty) seqs
    //       in (map head nonemptys, map (ziplong tail) nonemptys)
    //
    // -------------------------------------------------------------------------
    //
    // Tree[T] = (root: T, children: Set[Tree[T]])
    // - The same as Treq[T] but with Seq replaced by Set.
    // - leafs produces the set of leafs.
    // - pre_nodes and post_nodes degenerate to just nodes: the set of nodes.
    //
    // Trap[K->T] = (root: T, children: Map[K -> Trap[K->T]])
    // - Similar to Treq[T], degenerates to Tree[T] by (values: Map[K->X] -> Set[X])
    //
    // The compact root-first structure emphasizes nesting and preemptively disallows cycles.
    //
    // A tree could also be embedded into a map sending children to parents.
    // (Such maps have enough generality to embed forests or directed graphs.)
    //   Tree[T] ⊂ (nodes: Multiset[T], leafs: Set[nodes], parent: Map[nodes -> Maybe nodes])
    //

    // A trie is isomorphic to an associative rooted tree (Trap[T] above):
    //   Trie nodes are the chains of tree nodes starting* from the root,
    //   and trie parents are the prefixes* of tree paths.
    //
    // chains: Trap[K->T] -> Trap[K->(T, Seq[K])] :=
    //   (root: T, children: Map())
    //     -> (root: (root, Seq()), children: Map())
    //   (root: T, children: Map[K -> Trap[K->T]])
    //     -> (root: (root, Seq()), children: { k: tree_map (prefix_chain k) subtree | {k: subtree} in children })
    //
    // where
    //   treemap f (root, children) := ((f root), map (treemap f) children)
    //
    //   prefix_chain k (datum, chain) := (datum, seq(k)++chain)
    //
    //
    // *(Or, suffixes, ending at the root, to say parent(node)=tail(path).)
    //
    //  make_trie: (nodes, leafs: Set[nodes], tree_parent: Map[nodes -> Maybe nodes])
    //    -> (paths: Set[Seq[nodes]], maximal_paths: Set[paths], trie_parent: Map[paths -> Maybe paths]) :=
    //
    //    let path (n: nodes) := (seq(n)++(path(tree_parent(n)) or Empty))
    //    in
    //      paths := image path nodes
    //      maximal_paths := image path leafs
    //      trie_parent: paths -> Maybe paths :=
    //        (node, ancestors) -> ancestors
    //        Empty -> Nothing
    //
    // (Unfortunately, modeling each path as a sequence of _nodes_, rather than
    // of links, results in an off-by-one correspondence, where there is
    // a superoot in a trie for an empty tree. In practice, when constructing a
    // trie, it must be supposed that keys are stored on branches and values are
    // stored on non-root nodes.)


    template <typename Key>
    class Trie
    {
      public:
        static constexpr size_t root() { return hash_bytes(Key()); }

        static constexpr bool is_root(size_t node)
        {
          return node == root();
        }

        struct Branch { size_t parent; Key key; };
        constexpr size_t branch(Branch branch)
        {
          static_assert(alignof(decltype(Branch::parent)) == alignof(decltype(Branch::key)));
          const size_t node_hash = hash_bytes(branch);  // beware of padding!

          if (node_hash == root())
          {
            throw::std::range_error("Hash collision in Trie with root!");
          }

          if (not map.contains(node_hash))
          {
            map[node_hash] = NodeProperty{ branch.parent, branch.key };
          }
          else if (map.at(node_hash).parent != branch.parent)
          {
            throw std::range_error("Hash collision in Trie");
          }

          return node_hash;
        }

        constexpr size_t parent(size_t node) const
        {
          if (is_root(node))
            throw std::invalid_argument("Tried to obtain parent of root().");

          return map.at(node).parent;
        }

        constexpr Key key(size_t node) const
        {
          return map.at(node).key;
        }

        constexpr auto view() const
        {
          return map |
              std::views::transform(
                  [](const auto &n_v){
                      return std::make_pair(n_v.first, n_v.second.key);
                  });
        }
      private:
        struct NodeProperty { size_t parent; Key key; };  // add children for traversal
        std::map<size_t, NodeProperty> map;
    };


    // -----------------------------------------------------------------------


    class SumCalls
    {
      public:
        void consume_event(Event event);
        void print_results() const;
      private:
        struct CodePathProperty {
          long int count = 0;
          clock::duration duration = {};
          clock::time_point last_entry = {};

          void open(clock::time_point when) { ++count; last_entry = when; }
          void close(clock::time_point when) { duration += (when - last_entry); }
          // Never open twice before closing. Guaranteed by call_trie.
        };

        struct CallProperty {
          ZonePtrWData zone_w_data = {};
          long int count = 0;
          clock::duration duration = {};
          clock::duration self_duration = {};

          CallProperty & operator+=(const CallProperty &y)
          {
            zone_w_data    = y.zone_w_data;
            count         += y.count;
            duration      += y.duration;
            self_duration += y.self_duration;
            return *this;
          }
        };

        std::map<size_t, CodePathProperty> code_path_properties;

        Trie<ZonePtrWData> call_trie;
        size_t open_code_path = call_trie.root();
    };


    void SumCalls::consume_event(Event event)
    {
      // assume the input data comes on push
      if (is_push(event.mark.zone_action))
      {
        const uintptr_t zone_code = zonelog::zone_code(event.mark.zone_action);
        const uint64_t has_data = zonelog::has_data(event.mark.zone_action);
        const EventData data {event.data};
        /// const uint64_t data = 0u;
        const size_t parent = open_code_path;
        const size_t code_path =
            call_trie.branch({parent, ZonePtrWData{zone_code, data, has_data}});
        code_path_properties[code_path].open(event.mark.time_stamp);
        open_code_path = code_path;
      }
      else
      {
        const uintptr_t zone_code = zonelog::zone_code(event.mark.zone_action);
        if (zone_code != call_trie.key(open_code_path).zone_code)
          throw std::logic_error("Attempting to close a zone that is blocked or not open.");
        code_path_properties[open_code_path].close(event.mark.time_stamp);
        open_code_path = call_trie.parent(open_code_path);
      }
    }


    void SumCalls::print_results() const
    {
      const std::map<size_t, clock::duration>
      self_durations =
          [&call_trie = std::as_const(this->call_trie),
           &code_path_properties = std::as_const(this->code_path_properties)]()
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
      using Key = std::tuple<uintptr_t, RawEventData>;
      std::map<Key, CallProperty> call_properties;
      for (auto [code_path, zone_w_data] : call_trie.view())
      {
        const CodePathProperty property = code_path_properties.at(code_path);
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

        std::cout << fmt::format(
            std::locale("en_US.UTF-8"), //thousands separators
            "{:>10.2Lf}\t: {:>10.2Lf}\t/ {:>10L}\t= {:>10.2Lf}\t: {:>10.2Lf}\t",
            microseconds,
            self_microseconds,
            count,
            microseconds / count,
            self_microseconds / count);
        if (has_data)
        {
          std::cout << fmt::format("{}.{}({}-{})\n",
              zone->function, zone->name, zone_data.array[0], zone_data.array[1]);
        }
        else
        {
          std::cout << fmt::format("{}.{}\n", zone->function, zone->name);
        }
      }
    }


    // -----------------------------------------------------------------------


    static constexpr int max_depth = 64;

    using Key = std::array<uintptr_t, max_depth>;
    std::array<uintptr_t, max_depth> filled_null()
    {
      std::array<uintptr_t, max_depth> path;
      path.fill(zone_encode(nullptr));
      return path;
    }
    struct Path { std::array<uintptr_t, max_depth> path = filled_null(); int depth = 0; };
    Path push(Path p, uintptr_t zone_code) { p.path.at(p.depth++) = zone_code; return p; }
    Path pop(Path p)                    { p.path.at(--p.depth) = zone_encode(nullptr); return p; }

    struct FrameStat
    {
      long int count = 0;
      clock::duration duration = {};
      clock::time_point last_entry = {};
    };

    class SumTopDown
    {
      public:
        void consume_event(Event event)
        {
          if (is_push(event.mark.zone_action))
          {
            const uintptr_t zone_code = zonelog::zone_code(event.mark.zone_action);
            path = push(path, zone_code);
            FrameStat & frame = result[path.path];
            ++frame.count;
            frame.last_entry = event.mark.time_stamp;
          }
          else
          {
            const clock::time_point exit = event.mark.time_stamp;
            FrameStat & frame = result[path.path];
            frame.duration += (exit - frame.last_entry);
            path = pop(path);
          }
        }

        void print_results() const
        {
          for (auto it = result.cbegin(), e = result.cend(); it != e; ++it)
          {
            const Key & key = it->first;
            const FrameStat & frame_stat = it->second;

            const double microseconds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                frame_stat.duration).count() / 1000.0;

            std::cout << microseconds << "\t";
            std::cout << "/ " << frame_stat.count << "\t";
            std::cout << "= " << microseconds / frame_stat.count << "\t";

            for (uintptr_t code : key)
            {
              const Zone *zone = zone_decode(code);
              if (zone == nullptr)
                break;
              std::cout << zone->function << "\t";
            }
            std::cout << "\n";
          }
        }

      private:
        std::map<Key, FrameStat> result;
        Path path = {};
    };

    
  }
}

#endif//ZONELOG_HPP
