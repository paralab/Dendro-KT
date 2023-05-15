#ifndef DENDRO_KT_DEBUG_HPP
#define DENDRO_KT_DEBUG_HPP

#include "mpi.h"

#include <sstream>

#include <sys/prctl.h>  // prctl() to allow gdb to attach (wait_for_debugger)
#include <unistd.h>  // getpid()

// Interface
namespace dbg
{
  /**
   * @title buf
   * @usage std::cerr << (dbg::buf << "text" << data << "\n");
   * @brief Clean parallel output by collecting it into a single string first.
   */
  struct buf
  {
    std::stringstream ss;

    std::string str() const;
    std::string flush();

    template <typename T>
    buf & operator<<(T &&t);
  };
  inline std::ostream & operator<<(std::ostream &out, buf &b);
  inline std::ostream & operator<<(std::ostream &out, buf &&b);


  /** bool_str() */
  inline const char *bool_str(bool b)
  {
    return b? "true" : "false";
  }


  /**
   * @title nest_print
   * @usage (Block scope) auto nest = dbg::nest_print(__func__, ); nest << "message\n";
   * @brief Indentation for primitive nested logging.
   */
  template <int channel>
  struct NestPrint
  {
    public:
      template <typename...X>
      NestPrint(std::ostream &out, X&&...x);

      template <typename...X>
      NestPrint(X&&...x);

      template <typename X>
      std::ostream & operator<<(X &&x);

      std::ostream & operator~();

    private:
      void spacing(std::ostream &out);

      struct auto_depth
      {
        static int &depth()
        {
          static int depth = 0;
          return depth;
        }

        auto_depth() { ++depth(); }
        ~auto_depth() { --depth(); }
        int operator()() const { return depth(); }
      };

    public:
      auto_depth depth;
      std::ostream &out;
  };

  template <int channel = 0, typename...X>
  NestPrint<channel> nest_print(X&&...x)
  {
    return NestPrint<channel>(std::forward<X>(x)...);
  }

  struct Print : public NestPrint<0>
  {
    Print(std::string file, long line, std::ostream &out = std::cout)
      : NestPrint<0>(out, "[", file, ":", line, "] ")
    { }
  };


  /**
   * @title Debugging MPI programs with the GNU debugger
   * @author Tom Fogal, University of Utah
   * @author Masado Ishii
   * @date 2014-02-19
   * @modified 2019-10-30
   * @brief Utility for using gdb + mpi, based on notes by Tom Fogal.
   * @pre Building and running on Unix systems.
   * @usage Add waitForDebugger(comm, rank) right after MPI_Init().
   * @usage (If using mpich):   mpirun -np 2 -env USE_MPI_DEBUGGER 1 ./my_program
   * @usage pid=$(pgrep my_program | head -n 1) ; gdb -q -ex "attach ${pid}" -ex "set variable goAhead=1" -ex "finish"
   */
  namespace detail
  {
    inline int env_debugger_mode()
    {
      const char * value = getenv("USE_MPI_DEBUGGER");
      if (value == nullptr)
        return 0;
      return std::atoi(value);
    }
  }
  inline void wait_for_debugger(
      MPI_Comm comm, int enabled_mode = detail::env_debugger_mode());

}


// Implementation
namespace dbg
{
  //
  // struct buf
  std::string buf::str() const
  {
    return this->ss.str();
  }

  std::string buf::flush()
  {
    std::string result = std::move(this->ss).str();
    this->ss.str({});
    return result;
  }

  template <typename T>
  buf & buf::operator<<(T &&t)
  {
    this->ss << std::forward<T>(t); return *this;
  }

  inline std::ostream & operator<<(std::ostream &out, buf &b)
  {
    return out << b.flush();
  }

  inline std::ostream & operator<<(std::ostream &out, buf &&b)
  {
    return out << b;
  }
  // ///


  //
  // struct NestPrint
  template <int channel>
  template <typename...X>
  NestPrint<channel>::NestPrint(std::ostream &out, X&&...x)
  : out(out)
  {
    this->spacing(out);
    out << "> ";
    int _[] ={(out << std::forward<X>(x), 0)...};
  }

  template <int channel>
  template <typename...X>
  NestPrint<channel>::NestPrint(X&&...x)
  : NestPrint(std::cout, std::forward<X>(x)...)
  { }

  template <int channel>
  template <typename X>
  std::ostream & NestPrint<channel>::operator<<(X &&x)
  {
    this->spacing(out);
    out << "| ";
    return out << x;
  }

  template <int channel>
  std::ostream & NestPrint<channel>::operator~()
  {
    return out;
  }

  template <int channel>
  void NestPrint<channel>::spacing(std::ostream &out)
  {
    out << std::string(2 * depth(), ' ');
  }
  // ///


  //
  // wait_for_debugger()
  inline void wait_for_debugger(MPI_Comm comm, int enabled_mode)
  {
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    enum Modes { disabled = 0, attach_root = 1, attach_all = 2 };
    if (enabled_mode != disabled)
    {
      switch (enabled_mode)
      {
        case attach_root:
        {
          if (comm_rank == 0)
          {
            const unsigned long allow_trace = PR_SET_PTRACER_ANY;
            prctl(PR_SET_PTRACER, allow_trace, 0, 0, 0);

            volatile int proceed = 0;
            std::cerr << "Waiting for debugger. pid=" + std::to_string(getpid()) + "\n"
                      << "Usage (replace 'program' with your program):\n"
                      << "\n     pid=$(pgrep my_program | head -n 1) ; "
                         "gdb -q -ex \"attach ${pid}\" "
                         "-ex \"set variable proceed=1\" "
                         "-ex \"finish\""
                         "\n";
            while (proceed == 0) { /* Wait for debugger to change 'proceed'. */ }
          }
          break;
        }

        case attach_all:
        {
          // Allow attaching to all.
          const unsigned long allow_trace = PR_SET_PTRACER_ANY;
          prctl(PR_SET_PTRACER, allow_trace, 0, 0, 0);
          // Collect process ids of all ranks for the root process to print.
          unsigned long send_pid = getpid();
          std::vector<unsigned long> collect_pid;
          if (comm_rank == 0)
            collect_pid.resize(comm_size);
          MPI_Gather(&send_pid, 1, MPI_UNSIGNED_LONG,
              collect_pid.data(), 1, MPI_UNSIGNED_LONG, 0, comm);
          volatile int proceed = 0;
          if (comm_rank == 0)
          {
            std::cerr << "Waiting for debugger. pid=" + std::to_string(getpid()) + "\n";
            buf stream;
            stream << "\n    gdb -q ";
            stream << "-ex 'set pagination off' -ex 'set non-stop on' ";
            stream << "-ex 'attach " << send_pid << "' -ex 'set variable proceed=1' ";
            for (int i = 1; i < comm_size; ++i)
            {
              stream << "\\\n      "
                     << "-ex 'add-inferior' "
                     << "-ex 'inferior " << (i + 1) << "' "
                     << "-ex 'attach " << collect_pid[i] << "' "
                     << "-ex 'set variable proceed=1' "
                     << "-ex 'finish &' ";
            }
            stream << "\\\n      -ex 'inferior 1' "
                   << "-ex 'finish'\n";
            std::cerr << stream;
            while (proceed == 0) { /* Wait for debugger to change 'proceed'. */ }
          }
          else
          {
            while (proceed == 0) { /* Wait for debugger to change 'proceed'. */ }
          }
          break;
        }

        default:
          throw std::invalid_argument(
              "Unknown debug mode " + std::to_string(enabled_mode));
      }
    }
    MPI_Barrier(comm);
  }
  // ///

}

#endif//DENDRO_KT_DEBUG_HPP
