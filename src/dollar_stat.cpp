/**
 * @author Masado Ishii (University of Utah)
 * @date 2021-12-22
 */

#include <mpi.h>
#include "dollar_stat.h"
#include "serialize.hpp"

namespace dollar
{

  namespace detail
  {
    // mpi_reduce() expects [void reducer(Map &to_map, const Map &from_map)]
    template <typename ReducerT>
    std::map<DollarStat::URI, DollarStat::info> mpi_reduce(
        std::map<DollarStat::URI, DollarStat::info> totals,
        ReducerT reducer,
        MPI_Comm comm);

    void mpi_send_map(
        const std::map<DollarStat::URI, DollarStat::info> &map,
        int dest, int tag, MPI_Comm comm );

    void mpi_recv_map(
        std::map<DollarStat::URI, DollarStat::info> &map,
        int src, int tag, MPI_Comm comm, MPI_Status *status = MPI_STATUS_IGNORE );
  }



  //
  // DollarStat::DollarStat()
  //
  DollarStat::DollarStat(MPI_Comm comm)
    : m_comm(comm)
  {
    std::map<std::vector<int>, dollar::profiler::info> totals = dollar::self_totals();

    // Recover local inverse map of int->short_title.
    std::map<int, std::string> short_title;
    for (const auto &it : totals)
      short_title[it.first.back()] = it.second.short_title;

    const auto str_uri = [&short_title](const std::vector<int> &int_uri) {
      std::vector<std::string> str_uri;
      for (int i : int_uri)
        str_uri.push_back(short_title[i]);
      return str_uri;
    };

    for (const auto &it : totals)
    {
      info &info = m_totals[str_uri(it.first)];
      info.hits        = it.second.hits;
      info.total       = it.second.total;
      info.short_title = it.second.short_title;
    }
  }




  namespace detail
  {
    //
    // mpi_send_map()
    //
    void mpi_send_map( const std::map<DollarStat::URI, DollarStat::info> &map,
                       int dest, int tag, MPI_Comm comm )
    {
      serialize::Pack<std::vector<std::string>> key;
      serialize::Pack<double> hits_total;
      serialize::Pack<std::string> short_title;

      for (const auto &it : map)
      {
        key.pack(it.first);
        hits_total.pack(it.second.hits, it.second.total);
        short_title.pack(it.second.short_title);
      }

      key.mpi_send(dest, tag, comm);
      hits_total.mpi_send(dest, tag, comm);
      short_title.mpi_send(dest, tag, comm);
    }

    //
    // mpi_recv_map()
    //
    void mpi_recv_map( std::map<DollarStat::URI, DollarStat::info> &map,
                       int src, int tag, MPI_Comm comm, MPI_Status *status )
    {
      map.clear();

      serialize::Pack<std::vector<std::string>> key;
      serialize::Pack<double> hits_total;
      serialize::Pack<std::string> short_title;

      key.mpi_recv(src, tag, comm);
      hits_total.mpi_recv(src, tag, comm);
      short_title.mpi_recv(src, tag, comm);

      while (!key.end())
      {
        DollarStat::info & info = map[key.unpack()];
        hits_total.unpack(info.hits, info.total);
        info.short_title = short_title.unpack();
      }
    }


    //
    // mpi_reduce()
    //
    template <typename ReducerT>
    std::map<DollarStat::URI, DollarStat::info> mpi_reduce(
        std::map<DollarStat::URI, DollarStat::info> totals,
        ReducerT reducer,
        MPI_Comm comm)
    {
      using Map = std::map<DollarStat::URI, DollarStat::info>;
      int commSize, commRank;
      MPI_Comm_size(comm, &commSize);
      MPI_Comm_rank(comm, &commRank);

      int bit_plane = 1;
      while (bit_plane < commSize && !bool((bit_plane >> 1) & commRank))
      {
        int partner = commRank ^ bit_plane;
        if (partner < commSize)
        {
          Map partner_totals;
          const int tag = 0;

          if (partner < commRank)
          {
            mpi_send_map(totals, partner, tag, comm);
          }
          else
          {
            mpi_recv_map(partner_totals, partner, tag, comm);
            reducer(totals, const_cast<const Map &>(partner_totals));
          }
        }
        bit_plane <<= 1;
      }

      return totals;
    }
  }//namespace detail


  //
  // DollarStat::mpi_reduce_mean()
  //
  DollarStat DollarStat::mpi_reduce_mean()
  {
    using Map = std::map<URI, info>;
    const auto reducer_sum = [](Map &to_map, const Map &from_map) {
      for (const std::pair<URI, info> &it : from_map)
      {
        info & info = to_map[it.first];
        info.hits += it.second.hits;
        info.total += it.second.total;
        info.short_title = it.second.short_title;
      }
    };

    Map totals = detail::mpi_reduce(m_totals, reducer_sum, m_comm);

    int commSize;
    MPI_Comm_size(m_comm, &commSize);

    // mean
    for (auto &it : totals)
    {
      it.second.hits /= commSize;
      it.second.total /= commSize;
    }

    return DollarStat(m_comm, totals);
  }

  //
  // DollarStat::mpi_reduce_min()
  //
  DollarStat DollarStat::mpi_reduce_min()
  {
    using Map = std::map<URI, info>;
    const auto reducer_min = [](Map &to_map, const Map &from_map) {
      for (const std::pair<URI, info> &it : from_map)
      {
        const bool found = to_map.find(it.first) != to_map.end();
        info & info = to_map[it.first];
        if (found)
        {
          info.hits = fminf(info.hits, it.second.hits);
          info.total = fminf(info.total, it.second.total);
          info.short_title = it.second.short_title;
        }
        else
        {
          info = it.second;
        }
      }
    };

    return DollarStat(m_comm, detail::mpi_reduce(m_totals, reducer_min, m_comm));
  }

  //
  // DollarStat::mpi_reduce_max()
  //
  DollarStat DollarStat::mpi_reduce_max()
  {
    using Map = std::map<URI, info>;
    const auto reducer_max = [](Map &to_map, const Map &from_map) {
      for (const std::pair<URI, info> &it : from_map)
      {
        const bool found = to_map.find(it.first) != to_map.end();
        info & info = to_map[it.first];
        if (found)
        {
          info.hits = fmaxf(info.hits, it.second.hits);
          info.total = fmaxf(info.total, it.second.total);
          info.short_title = it.second.short_title;
        }
        else
        {
          info = it.second;
        }
      }
    };

    return DollarStat(m_comm, detail::mpi_reduce(m_totals, reducer_max, m_comm));
  }



  //
  // DollarStat::print()
  //
  void DollarStat::print(std::ostream &out)
  {
    int max_length = 0;
    for (const auto & it : m_totals)
    {
      const int length = it.second.short_title.size();
      max_length = (length > max_length ? length : max_length);
    }

    int width_ms = 12;
    int width_hits = 7;

    std::stringstream ss;
    char buf[1024];
    sprintf(buf, "%*s, %*s, %*s, \n",
        max_length, "name",
        width_ms, "ms",
        width_hits, "hits");
    ss << std::string(buf);

    for (const auto & it : m_totals)
    {
      const info & info = it.second;
      sprintf(buf, "%*s, %*.3f, %*.1f, \n",
          max_length, info.short_title.c_str(),
          width_ms,   info.total * 1000,
          width_hits, info.hits);
      ss << std::string(buf);
    }
    out << ss.str();
  }

  namespace detail
  {
    template<typename info>
    struct Node {
        std::string name;
        info *value;
        std::vector<Node> children;

        Node( const std::string &name, info *value = 0 ) : name(name), value(value)
        {}

        size_t tree_printer( std::string indent, bool leaf, std::ostream &out ) {
            size_t len = 0;
            {
              std::stringstream ss;
              if( leaf ) {
                  ss << indent << "+-" << name;
                  indent += "  ";
              } else {
                  ss << indent << "|-" << name;
                  indent += "| ";
              }
              name = ss.str();
              value->short_title = ss.str();
              len = name.length();
              ss << std::endl;
              out << ss.str();
            }
            for( auto end = children.size(), it = end - end; it < end; ++it ) {
                len = std::max(len, children[it].tree_printer( indent, it == (end - 1), out ));
            }
            return len;
        }
        size_t tree_printer( std::ostream &out = std::cout ) {
            return tree_printer( "", true, out );
        }
        Node&tree_recreate_branch( const std::vector<std::string> &names ) {
            auto *where = &(*this);
            for( auto &name : names ) {
                bool found = false;
                for( auto &it : where->children ) {
                    if( it.name == name ) {
                        where = &it;
                        found = true;
                        break;
                    }
                }
                if( !found ) {
                    where->children.push_back( Node(name) );
                    where = &where->children.back();
                }
            }
            return *where;
        }
        template<typename FN0, typename FN1, typename FN2>
        void tree_walker( const FN0 &method, const FN1 &pre_children, const FN2 &post_chilren  ) const {
            if( children.empty() ) {
                method( *this );
            } else {
                pre_children( *this );
                for( auto &child : children ) {
                    child.tree_walker( method, pre_children, post_chilren );
                }
                post_chilren( *this );
            }
        }
    };
  }//namespace detail

  //
  // DollarStat::print_tree() (adaptation of dollar::profiler::print())
  //
  template void DollarStat::print_tree<0>(std::ostream &, const char *, const char *) const;
  template void DollarStat::print_tree<1>(std::ostream &, const char *, const char *) const;
  template<bool for_chrome>
  void DollarStat::print_tree( std::ostream &out, const char *tab, const char *feed ) const
  {
      std::map<URI, info> totals = m_totals;

      // calculate total accumulated time
      double total = 0;
      for( auto &it : totals ) {
          total += it.second.total;
      }

      static unsigned char pos = 0;
      info dummy;
      dummy.short_title = "/";
      detail::Node<info> root( std::string() + "\\|/-"[(++pos)%4], &dummy );
      for( auto it = totals.begin(), end = totals.end(); it != end; ++it ) {
          auto &info = it->second;
          auto &node = root.tree_recreate_branch( it->first );
          node.value = &info;
      }

      if (!for_chrome)
      {
          // Tree printer
          std::stringstream ss;
          size_t maxlen = root.tree_printer( ss );

          // Add post-padding
          for( auto &cp : totals )
          {
              std::string &ti = cp.second.short_title;
              /**/ if( maxlen > ti.size() ) ti += std::string( maxlen - ti.size(), ' ' );
              else if( maxlen < ti.size() ) ti.resize( maxlen );
          }
      }

      for( auto &cp : totals ) {
          for( auto &ch : cp.second.short_title ) {
              if( ch == '\\' ) ch = '/';
          }
      }

      size_t i = 0;
      if( !for_chrome ) {
          std::string format, sep, graph, buffer(1024, '\0');
          // pre-loop
          for( auto &it : std::vector<std::string>{ "%4d.","%s","[%s]","%5.2f%% CPU","(%9.3fms)","%5.0f hits",feed } ) {
              format += sep + it;
              sep = tab;
          }
          // loop
          for( auto &it : totals ) {
              auto &info = it.second;
              double cpu = info.total * 100.0 / total;
              int width(cpu*DOLLAR_CPUMETER_WIDTH/100);
              graph = std::string( width, '=' ) + std::string( DOLLAR_CPUMETER_WIDTH - width, '.' );
#ifdef _MSC_VER
              sprintf_s( &buffer[0], 1024,
#else
              sprintf( &buffer[0],
#endif
              format.c_str(), ++i, it.second.short_title.c_str(), graph.c_str(), cpu, (float)(info.total * 1000), info.hits );
              out << &buffer[0];
          }
      } else {

          // setup
          out << "[" << std::endl;

          // json array format
          // [ref] https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
          // [ref] https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html#L54

          auto get_color = []( float pct ) {
              return pct <= 16 ? "good":
                     pct <= 33 ? "bad":
                     "terrible";
          };

          double timestamp = 0;
          int fake_pid = 0,  fake_tid = 0;
          root.tree_walker(
              [&]( const detail::Node<info> &node ) {
                  auto &info = *node.value;
                  double cpu = info.total * 100.0 / total;
                  out << "{\"name\": \"" << info.short_title << "\","
                          "\"cat\": \"" << "CPU,DOLLAR" << "\","
                          "\"ph\": \"" << 'X' << "\","
                          "\"pid\": " << fake_pid << ","
                          "\"tid\": " << fake_tid << ","
                          "\"ts\": " << (unsigned int)(timestamp * 1000 * 1000) << ","
                          "\"dur\": " << (unsigned int)(info.total * 1000 * 1000) << ","
                          "\"cname\": \"" << get_color(cpu) << "\"" "," <<
                          "\"args\": {}},\n";
                  timestamp += info.total;
              },
              [&]( const detail::Node<info> &node ) {
                  auto &info = *node.value;
                  double cpu = info.total * 100.0 / total;
                  out << "{\"name\": \"" << info.short_title << "\","
                          "\"cat\": \"" << "CPU,DOLLAR" << "\","
                          "\"ph\": \"" << 'B' << "\","
                          "\"pid\": " << fake_pid << ","
                          "\"tid\": " << fake_tid << ","
                          "\"ts\": " << (unsigned int)(timestamp * 1000 * 1000) << ","
                          "\"args\": {}},\n";
                  timestamp += info.total;
              },
              [&]( const detail::Node<info> &node ) {
                  auto &info = *node.value;
                  double cpu = info.total * 100.0 / total;
                  out << "{\"name\": \"" << info.short_title << "\","
                          "\"cat\": \"" << "CPU,DOLLAR" << "\","
                          "\"ph\": \"" << 'E' << "\","
                          "\"pid\": " << fake_pid << ","
                          "\"tid\": " << fake_tid << ","
                          "\"ts\": " << (unsigned int)((timestamp + info.total) * 1000 * 1000) << ","
                          "\"cname\": \"" << get_color(cpu) << "\"" "," <<
                          "\"args\": {}},\n";
                  timestamp += info.total;
              } );
      }
  }


}//namesapce dollar
