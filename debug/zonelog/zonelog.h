#ifndef ZONELOG_H
#define ZONELOG_H

#include <chrono>
#include <sstream>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <climits>
#include <cstdint>

// -----------------------------------------------------------------------------

namespace zonelog
{
  struct alignas(8) Zone
  {
    char const *name;
    char const *function;
    char const *pretty_function;
    char const *file;
    long int line;
  };

  class Log;
  struct Event;
  struct EventData;

  extern Log & global_log();
  //future: thread local loggers

  namespace online {
    class ZonePushPop;
    class ScopeGuard;
  }

  namespace offline
  {
    template<typename LA>
    concept LogAggregator = requires(LA &log_aggregator, Event event)
    {
      log_aggregator.consume_event(event);
    };

    template <LogAggregator LA>
    void flush_aggregate(Log &log, LA &aggregator);
  }
}

// Use as expressions (you must push and pop yourself the ZonePushPop):
#define ZONELOG_FZONE()            ZONELOG_NAMED_FZONE("")
#define ZONELOG_NAMED_FZONE(name_)  ( \
  [] (auto name, auto function, auto pretty_function, auto file, auto line) \
  -> zonelog::online::ZonePushPop { \
      static constexpr zonelog::Zone static_zone = \
          { *name, *function, *pretty_function, *file, *line }; \
      return zonelog::online::ZonePushPop{ &static_zone }; \
  }( \
    zonelog::dtl::string_tag<name_>(), \
    zonelog::dtl::string_tag<__func__>(), \
    zonelog::dtl::string_tag<__PRETTY_FUNCTION__>(), \
    zonelog::dtl::string_tag<__FILE__>(), \
    zonelog::dtl::int_tag<__LINE__>() \
   ) )
// This is the only way I know to define a static constexpr variable
// and return its address in a single expression (in C++20).
// Template arguments are constant expressions but function arguments are not.

#define ZONELOG_ZONE()            ZONELOG_NAMED_ZONE("")
#define ZONELOG_NAMED_ZONE(name_)  ( \
  [] (auto name, auto function, auto pretty_function, auto file, auto line) \
  -> zonelog::online::ZonePushPop { \
      static constexpr zonelog::Zone static_zone = \
          { *name, *function, *pretty_function, *file, *line }; \
      return zonelog::online::ZonePushPop{ &static_zone }; \
  }( \
    zonelog::dtl::string_tag<name_>(), \
    zonelog::dtl::string_tag<"">(), \
    zonelog::dtl::string_tag<"">(), \
    zonelog::dtl::string_tag<__FILE__>(), \
    zonelog::dtl::int_tag<__LINE__>() \
   ) )

// Use as anonymous scope declarations (auto constructed and destructed):
#define ZONELOG_SCOPE()          ZONELOG_NAMED_SCOPE("")
#define ZONELOG_SCOPE_DATA(data) ZONELOG_NAMED_SCOPE_DATA("", data)

#define ZONELOG_NAMED_SCOPE(name) \
  ZONELOG_NAMED_SCOPE_impl(name, __COUNTER__)

#define ZONELOG_NAMED_SCOPE_DATA(name, data) \
  ZONELOG_NAMED_SCOPE_DATA_impl(name, __COUNTER__, data)

// -----------------------------------------------------------------------------

#define ZONELOG_NAMED_SCOPE_impl(name, anon) \
  static constexpr zonelog::Zone  ZONELOG_ANON_ZONE(anon) = \
      { name, __func__, __PRETTY_FUNCTION__, __FILE__, __LINE__ }; \
  zonelog::online::ScopeGuard     ZONELOG_ANON_GUARD(anon) = \
      { & ZONELOG_ANON_ZONE(anon) };

#define ZONELOG_NAMED_SCOPE_DATA_impl(name, anon, data) \
  static constexpr zonelog::Zone  ZONELOG_ANON_ZONE(anon) = \
      { name, __func__, __PRETTY_FUNCTION__, __FILE__, __LINE__ }; \
  zonelog::online::ScopeGuard     ZONELOG_ANON_GUARD(anon) = \
      { & ZONELOG_ANON_ZONE(anon), zonelog::EventData(data) };

#define ZONELOG_ANON_ZONE(anon) ZONELOG_CAT(zl_zone_, anon)
#define ZONELOG_ANON_GUARD(anon) ZONELOG_CAT(zl_guard_, anon)

#define ZONELOG_CAT(X, Y) ZONELOG_CAT_impl(X, Y)
#define ZONELOG_CAT_impl(X, Y) X ## Y


namespace zonelog
{
  namespace dtl
  {
    // StringLiteral c++20 structural class template reproduced from
    // https://ctrpeach.io/posts/cpp20-string-literal-template-parameters/
    // under Creative Commons 4.0 (accessed 2025-03-14).
    template<size_t N>
    struct StringLiteral
    {
      constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }
      char value[N];
    };

    template <StringLiteral s>
    struct StringLiteralTag { };

    template <StringLiteral s>
    constexpr auto operator*(StringLiteralTag<s>) { return s.value; }

    template <StringLiteral s>
    constexpr auto string_tag() { return StringLiteralTag<s>(); }

    template <int i>
    struct IntTag { };

    template <int i>
    constexpr auto operator*(IntTag<i>) { return i; }

    template <int i>
    constexpr auto int_tag() { return IntTag<i>(); }
  }

  inline auto zone_encode(const Zone * zone) -> uintptr_t { return reinterpret_cast<uintptr_t>(zone); }
  inline auto zone_decode(uintptr_t code) -> const Zone * { return reinterpret_cast<const Zone *>(code); }

  struct ZoneAction
  {
    uintptr_t zone_code : sizeof(uintptr_t) * CHAR_BIT - 2;
    bool pop : 1;
    bool with_data: 1;
  };
  constexpr ZoneAction zone_push(const Zone *zone, bool with_data = false) { return { zone_encode(zone), false, with_data}; }
  constexpr ZoneAction zone_pop(const Zone *zone, bool with_data = false)  { return { zone_encode(zone), true, with_data }; }
  constexpr uintptr_t zone_code(ZoneAction za) { return za.zone_code; }
  constexpr const Zone * zone(ZoneAction za) { return zone_decode(zone_code(za)); }
  constexpr bool action_is_pop(ZoneAction za) { return za.pop; }
  constexpr bool is_push(ZoneAction za) { return not za.pop; }
  constexpr bool has_data(ZoneAction za) { return za.with_data; }

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
    EventData(uint64_t a0)              : array{{ a0 }} { }
    EventData(uint64_t a0, uint64_t a1) : array{{ a0, a1 }} { }
    EventData(std::array<uint64_t, 2> a) : array(a) { }

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

  extern const size_t preallocation;

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


  namespace online
  {
    namespace internal
    {
      inline void log_now(Log &log, ZoneAction zone_action, EventData data)
      {
        log.write(Event{{zone_action, clock::now()}, data});
      }
    }

    inline void log_push(Log &log, const Zone *zone, EventData data)
    {
      internal::log_now(log, zone_push(zone, true), data);
    }

    inline void log_pop(Log &log, const Zone *zone, EventData data)
    {
      internal::log_now(log, zone_pop(zone, true), data);
    }

    inline void log_push(Log &log, const Zone *zone)
    {
      internal::log_now(log, zone_push(zone, false), EventData{});
    }

    inline void log_pop(Log &log, const Zone *zone)
    {
      internal::log_now(log, zone_pop(zone, false), EventData{});
    }

    class ZonePushPop
    {
      public:
        constexpr ZonePushPop(const Zone *zone) : zone(zone) { }
        void push()
        {
          log_push(global_log(), zone);
          ++depth;
        }
        void pop()
        {
          --depth;
          if (depth < 0) { throw std::logic_error("Insufficient push before pop"); }
          log_pop(global_log(), zone);
        }
        void push(EventData input_data)
        {
          log_push(global_log(), zone, input_data);
          ++depth;
        }
        void pop(EventData output_data)
        {
          --depth;
          if (depth < 0) { throw std::logic_error("Insufficient push before pop"); }
          log_pop(global_log(), zone, output_data);
        }
      private:
        const Zone * const zone;
        long int depth = 0;
    };

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
    template <LogAggregator LA>
    void flush_aggregate(Log &log, LA &aggregator)
    {
      while (log.has_unread())
      {
        Event event = log.read();
        aggregator.consume_event(event);
      }
      log.clear();
    }
  }
  template <offline::LogAggregator LA>
  void flush_global(LA &aggregator)
  {
    offline::flush_aggregate(global_log(), aggregator);
  }
}

#endif//ZONELOG_H
