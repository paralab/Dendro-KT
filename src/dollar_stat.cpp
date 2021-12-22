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
        std::tie(info.hits, info.total) = hits_total.unpack<double, double>();
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

}//namesapce dollar
