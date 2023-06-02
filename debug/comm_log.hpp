#ifndef DENDRO_KT_DEBUG_COMM_LOG
#define DENDRO_KT_DEBUG_COMM_LOG

#include "include/parUtils.h"

#include <unordered_map>
#include <vector>

#include <mpi.h>


#define COMMLOG_CONTEXT (::debug::CommLog::Context{__func__, __LINE__})


namespace debug
{
  class CommLog  // Uses Gather/Allgather on dup(MPI_COMM_WORLD) to make group subsets
  {
    public:
      struct Context
      {
        std::string function;
        long line;
      };

      struct Region
      {
        std::vector<MPI_Comm> comms;
        Context context;
      };

      struct [[nodiscard]] RegionBomb
      {
        RegionBomb(CommLog *logger)
          : logger(logger), incarnation(logger->incarnation())
        { }

        ~RegionBomb()
        {
          if (incarnation == logger->incarnation())
            logger->pop_region();
          else
            logger->stale_pop();
        }

        CommLog *logger;
        int incarnation;
      };

      struct CommData
      {
        std::vector<int> members;
        std::vector<Context> registration_contexts;
      };

    public:
      inline CommLog(std::ostream &out)
        : m_out(out)
      {
        MPI_Comm_dup(MPI_COMM_WORLD, &m_world);
      }

      inline ~CommLog()
      {
        MPI_Comm_free(&m_world);
      }

      inline RegionBomb declare_region(MPI_Comm comm, Context context)
      {
        this->push_region(Region{{comm}, context});
        return RegionBomb{this};
      }

      inline RegionBomb declare_region(std::initializer_list<MPI_Comm> comms, Context context)
      {
        this->push_region(Region{comms, context});
        return RegionBomb{this};
      }

      inline void clear()
      {
        m_registry.clear();
        m_region_stack.clear();

        ++m_incarnation;
      }

    public:
      inline bool registered(MPI_Comm comm) const;
      inline void register_comm(MPI_Comm comm, Context context);  // collective

    private:
      inline void push_region(Region region);
      inline void pop_region();
      inline int incarnation() const { return m_incarnation; }
      inline void stale_pop() const
      {
        const int world_rank = par::mpi_comm_rank(m_world);
        m_out << "[" + std::to_string(world_rank) + "] (stale pop)\n";
      }

    private:
      std::unordered_map<MPI_Comm, CommData> m_registry;
      std::vector<Region> m_region_stack;  //future: add "all" as root case
      std::ostream &m_out = std::cout;
      MPI_Comm m_world = MPI_COMM_NULL;
      int m_incarnation = 0;
  };




  extern CommLog *global_comm_log;





  // CommLog::registered()
  bool CommLog::registered(MPI_Comm comm) const
  {
    return m_registry.find(comm) != m_registry.end();
  }

  // CommLog::register_comm()
  void CommLog::register_comm(MPI_Comm comm, Context context)
  {
    const int world_size = par::mpi_comm_size(m_world);
    const int world_rank = par::mpi_comm_rank(m_world);

    // Unconditional Allgather.
    std::vector<char> membership(world_size, false);
    const char self_member = (comm != MPI_COMM_NULL);
    par::Mpi_Allgather(&self_member, membership.data(), 1, m_world);

    // Consolidate list of ranks that are 'active'.
    int candidate = 0;
    std::vector<int> members;
    for (bool member: membership)
    {
      if (member)
        members.push_back(candidate);
      ++candidate;
    }

    const bool is_new_communicator = not this->registered(comm);

    // logging
    const bool first_member = members.size() > 0 and members[0] == world_rank;
    if (first_member)
    {
      std::stringstream list_members;
      if (is_new_communicator)
      {
        list_members << "Registering new communicator: [ ";
        for (int rank: members)
          list_members << rank << " ";
        list_members << "]";
      }
      else
      {
        list_members << "           (old communicator)";
      }
      std::string message = list_members.str();
      message.resize(std::max<size_t>(message.size(), 50), ' ');
      message += "@:" + std::to_string(context.line) + "\n";
      m_out << message;
    }

    // Create/update CommData
    if (is_new_communicator)
      m_registry.emplace(comm, CommData{std::move(members), {}});
    m_registry[comm].registration_contexts.emplace_back(std::move(context));
  }


  namespace detail
  {
    std::string str(const CommLog::CommData &comm_data)
    {
      std::stringstream buffer;
      buffer << "( ";
      for (int rank: comm_data.members)
        buffer << rank << " ";
      buffer << ") \t (" << comm_data.registration_contexts.size() << " contexts)";
      return buffer.str();
    }
  }


  // CommLog::push_region()
  void CommLog::push_region(Region region)
  {
    const int depth = m_region_stack.size();
    const std::string prefix = std::string(2 * depth, ' ');
    std::stringstream buffer;

    const int world_size = par::mpi_comm_size(m_world);
    const int world_rank = par::mpi_comm_rank(m_world);

    for (MPI_Comm comm: region.comms)
    {
      assert(this->registered(comm));
    }

    static std::vector<char> membership;
    membership.resize(world_size, false);

    buffer << "\n";
    buffer << prefix << "[" << world_rank << "] Pushed " << region.comms.size() << " comms. Union:"
      << "\t\t\t" << region.context.function << ": " << region.context.line
      << "\n";

    for (MPI_Comm comm: region.comms)
      for (int rank: m_registry[comm].members)
        membership[rank] = true;
    buffer << prefix << "   { ";
    for (int i = 0; i < world_size; ++i)
      if (membership[i])
        buffer << i << " ";
    buffer << "}\n";

    for (MPI_Comm comm: region.comms)
      buffer << prefix << ">>> " << detail::str(m_registry[comm]) << "\n";

    m_out << buffer.str();

    m_region_stack.push_back(std::move(region));

    membership.clear();
  }

  // CommLog::pop_region()
  void CommLog::pop_region()
  {
    assert(m_region_stack.size() > 0);
    Region region = std::move(m_region_stack.back());
    m_region_stack.pop_back();

    const int depth = m_region_stack.size();
    const std::string prefix = std::string(2 * depth, ' ');
    std::stringstream buffer;

    const int world_size = par::mpi_comm_size(m_world);
    const int world_rank = par::mpi_comm_rank(m_world);

    for (MPI_Comm comm: region.comms)
    {
      assert(this->registered(comm));
    }

    static std::vector<char> membership;
    membership.resize(world_size, false);

    buffer << "\n";
    buffer << prefix << "[" << world_rank << "] Popped " << region.comms.size() << " comms. Union:"
      << "\t\t\t" << region.context.function << ": " << region.context.line
      << "\n";

    for (MPI_Comm comm: region.comms)
      for (int rank: m_registry[comm].members)
        membership[rank] = true;
    buffer << prefix << "   { ";
    for (int i = 0; i < world_size; ++i)
      if (membership[i])
        buffer << i << " ";
    buffer << "}\n";

    for (MPI_Comm comm: region.comms)
      buffer << prefix << "<<< " << detail::str(m_registry[comm]) << "\n";

    m_out << buffer.str();

    membership.clear();
  }



  // ---------------------------------------------------------------------------
  // Printing
  // ---------------------------------------------------------------------------

  class FlushOnDeath
  {
    public:
      FlushOnDeath(std::ostream &out, bool enabled)
        : out(out), enabled(enabled)
      { }

      ~FlushOnDeath()
      {
        if (enabled)
          out << buffer.str();
      }

      FlushOnDeath(FlushOnDeath &&other)
        : out(other.out),
          enabled(other.enabled),
          buffer(std::move(other.buffer))
      { }

      template <typename X>
      FlushOnDeath & operator<<(X && x)
      {
        if (enabled)
          buffer << std::forward<X>(x);
        return *this;
      }

    private:
      std::ostream &out;
      const bool enabled;
      std::stringstream buffer;
  };

  class EnablePrint
  {
    public:
      EnablePrint() = default;
      EnablePrint(bool enabled) : m_enabled(enabled) { }
      void enable(bool enabled) { m_enabled = enabled; }
      void enable() { this->enable(true); }
      void disable() { this->enable(false); }

      FlushOnDeath operator()(std::ostream &out)
      {
        return FlushOnDeath(out, m_enabled);
      }

    private:
      bool m_enabled = true;
  };

}



#endif//DENDRO_KT_DEBUG_COMM_LOG
