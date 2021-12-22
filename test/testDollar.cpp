#include <vector>
#include <random>
#include <fstream>
#include <ostream>
#include <sstream>

#include "dollar.hpp"

#include "distTree.h"
#include "filterFunction.h"
#include "octUtils.h"
#include "tnUtils.h"
#include "treeNode.h"

const int DIM = 2;
using uint = unsigned int;
using Oct = ot::TreeNode<uint, DIM>;
using OctList = std::vector<Oct>;

OctList points(size_t n);
OctList points(DendroIntL nTotal, MPI_Comm comm);

class DollarStat
{
  struct info {
    double hits = 0;
    double total = 0;
    std::string short_title;
  };

  using URI = std::vector<std::string>;

  MPI_Comm m_comm;
  std::map<URI, info> m_totals;

  DollarStat(MPI_Comm comm, const std::map<URI, info> &totals)
    : m_comm(comm), m_totals(totals)
  {}

  template <typename ReducerT>
  std::map<URI, info> mpi_reduce(ReducerT reducer);

  public:
    DollarStat(MPI_Comm comm);

    // Note that only the root=0 has the final reduced data.
    DollarStat mpi_reduce_mean();
    DollarStat mpi_reduce_min();
    DollarStat mpi_reduce_max();

    static void mpi_send_map(
        const std::map<URI, DollarStat::info> &map,
        int dest, int tag, MPI_Comm comm );

    static void mpi_recv_map(
        std::map<URI, DollarStat::info> &map,
        int src, int tag, MPI_Comm comm, MPI_Status *status = MPI_STATUS_IGNORE );

    void print(std::ostream &out = std::cout);
};




template <typename T>
class Pack
{
  private:
    std::vector<T> m_values;
    size_t m_count = 0;
    size_t m_iter = 0;

  public:
    size_t count() const { return m_count; }
    bool end() const { return m_iter >= m_count; }

    template <typename...TS>
    void pack(TS&&...ts)
    {
      for (const T &t : {ts...})
        m_values.push_back(t);
      ++m_count;
    }

    template <typename ...TS>
    std::tuple<TS...> unpack()
    {
      return { TS(m_values[m_iter++])... };
    }

    template <typename ...TS>
    void unpack(TS&&...ts)
    {
      std::tie(ts...) = this->unpack<TS...>();
    }

    void mpi_send(int dest, int tag, MPI_Comm comm)
    {
      int sizes[3] = { (int) m_values.size(),
                       (int) m_count,
                       (int) m_iter };
      MPI_Send(sizes, 3, MPI_INT, dest, tag, comm);
      MPI_Send(&m_values[0], m_values.size(), par::Mpi_datatype<T>::value(), dest, tag, comm);
    }

    void mpi_recv(int src, int tag, MPI_Comm comm, MPI_Status *status = MPI_STATUS_IGNORE)
    {
      int sizes[3];
      MPI_Recv(sizes, 3, MPI_INT, src, tag, comm, status);
      m_values.resize(sizes[0]);
      m_count = sizes[1];
      m_iter = sizes[2];

      MPI_Recv(&m_values[0], m_values.size(), par::Mpi_datatype<T>::value(), src, tag, comm, status);
    }
};


template <>
class Pack<std::string>
{
  private:
    std::vector<char> m_chars;
    std::vector<size_t> m_lengths;
    size_t m_count = 0;
    size_t m_char_iter = 0;
    size_t m_str_iter = 0;

  public:
    size_t count() const { return m_count; }
    bool end() const { return m_str_iter >= m_count; }

    void pack(const std::string &s)
    {
      m_chars.insert(m_chars.end(), s.begin(), s.end());
      m_lengths.push_back(s.length());
      ++m_count;
    }

    std::string unpack()
    {
      std::string s(&m_chars[m_char_iter], m_lengths[m_str_iter]);
      m_char_iter += m_lengths[m_str_iter];
      m_str_iter += 1;
      return s;
    }

    void unpack(std::string &s)
    {
      s = this->unpack();
    }

    void mpi_send(int dest, int tag, MPI_Comm comm)
    {
      int sizes[5] = { (int) m_chars.size(),
                       (int) m_lengths.size(),
                       (int) m_count,
                       (int) m_char_iter,
                       (int) m_str_iter };
      MPI_Send(sizes, 5, MPI_INT, dest, tag, comm);
      MPI_Send(&m_chars[0], m_chars.size(), par::Mpi_datatype<char>::value(), dest, tag, comm);
      MPI_Send(&m_lengths[0], m_lengths.size(), par::Mpi_datatype<size_t>::value(), dest, tag, comm);
    }

    void mpi_recv(int src, int tag, MPI_Comm comm, MPI_Status *status = MPI_STATUS_IGNORE)
    {
      int sizes[5];
      MPI_Recv(sizes, 5, MPI_INT, src, tag, comm, status);
      m_chars.resize(sizes[0]);
      m_lengths.resize(sizes[1]);
      m_count = sizes[2];
      m_char_iter = sizes[3];
      m_str_iter = sizes[4];

      MPI_Recv(&m_chars[0], m_chars.size(), par::Mpi_datatype<char>::value(), src, tag, comm, status);
      MPI_Recv(&m_lengths[0], m_lengths.size(), par::Mpi_datatype<size_t>::value(), src, tag, comm, status);
    }
};


template <>
class Pack<std::vector<std::string>>
{
  private:
    Pack<std::string> m_strings;
    std::vector<size_t> m_vecsz;
    size_t m_count = 0;
    size_t m_str_iter = 0;
    size_t m_vec_iter = 0;

  public:
    size_t count() const { return m_count; }
    bool end() const { return m_vec_iter >= m_count; }

    void pack(const std::vector<std::string> &strings)
    {
      for (const std::string &str : strings)
        m_strings.pack(str);
      m_vecsz.push_back(strings.size());
      ++m_count;
    }

    std::vector<std::string> unpack()
    {
      std::vector<std::string> strings;
      size_t sz = m_vecsz[m_vec_iter];
      m_str_iter += sz;
      m_vec_iter += 1;
      while (sz-- > 0)
        strings.push_back(m_strings.unpack());
      return strings;
    }

    void unpack(std::vector<std::string> &strings)
    {
      strings = this->unpack();
    }

    void mpi_send(int dest, int tag, MPI_Comm comm)
    {
      int sizes[4] = { (int) m_vecsz.size(),
                       (int) m_count,
                       (int) m_str_iter,
                       (int) m_vec_iter };
      MPI_Send(sizes, 4, MPI_INT, dest, tag, comm);
      m_strings.mpi_send(dest, tag, comm);
      MPI_Send(&m_vecsz[0], m_vecsz.size(), par::Mpi_datatype<size_t>::value(), dest, tag, comm);
    }

    void mpi_recv(int src, int tag, MPI_Comm comm, MPI_Status *status = MPI_STATUS_IGNORE)
    {
      int sizes[4];
      MPI_Recv(sizes, 4, MPI_INT, src, tag, comm, status);
      m_vecsz.resize(sizes[0]);
      m_count = sizes[1];
      m_str_iter = sizes[2];
      m_vec_iter = sizes[3];

      m_strings.mpi_recv(src, tag, comm, status);
      MPI_Recv(&m_vecsz[0], m_vecsz.size(), par::Mpi_datatype<size_t>::value(), src, tag, comm, status);
    }
};


// ---------------------------------------


//
// main()
//
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  DendroScopeBegin();
  _InitializeHcurve(DIM);

  MPI_Comm comm = MPI_COMM_WORLD;
  int commRank, commSize;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);

  const double sfc_tol = 0.1;

  OctList octList = points(1 * 1024 * 1024, comm);

  {$
    ot::SFC_Tree<uint, DIM>::locTreeSort(octList);
  }

  {$
    ot::SFC_Tree<uint, DIM>::distTreeSort(octList, sfc_tol, comm);
  }

  DollarStat dollar_stat(comm);

  if (commRank == 0)
  {
    // Only root's data
    std::ofstream file("chrome.json");
    dollar::chrome(file);
    dollar::csv(std::cout);

    // Only root's data
    std::cout << "\n";
    dollar_stat.print(std::cout);
  }
  dollar::clear();

  // Collect mean, min, and max timings over all processes.
  DollarStat reduce_mean = dollar_stat.mpi_reduce_mean();
  DollarStat reduce_min = dollar_stat.mpi_reduce_min();
  DollarStat reduce_max = dollar_stat.mpi_reduce_max();
  if (commRank == 0)
  {
    std::cout << "\n";
    reduce_mean.print(std::cout);

    std::cout << "\n";
    reduce_min.print(std::cout);

    std::cout << "\n";
    reduce_max.print(std::cout);
  }

  _DestroyHcurve();
  DendroScopeEnd();
  MPI_Finalize();
  return 0;
}



//
// points()
//
OctList points(size_t n)
{
  OctList points;

  const double min = 0;
  const double max = 1.0 - (1.0 / (1u << m_uiMaxDepth));
  const auto clamp = [=](double x) {
    return x < min ? min : x > max ? max : x;
  };

  const auto toOctCoord = [=](double x) {
    return uint(x * (1u << m_uiMaxDepth));
  };

  std::normal_distribution<> normal{0.5, 0.2};
  std::random_device rd;
  std::mt19937_64 gen(rd());

  for (size_t i = 0; i < n; ++i)
  {
    Oct oct;
    oct.setLevel(m_uiMaxDepth);

    for (int d = 0; d < DIM; ++d)
      oct.setX(d, toOctCoord(clamp(normal(gen))));

    points.push_back(oct);
  }

  return points;
}


//
// points()  (distributed)
//
OctList points(DendroIntL nTotal, MPI_Comm comm)
{
  int commRank, commSize;
  MPI_Comm_size(comm, &commSize);
  MPI_Comm_rank(comm, &commRank);
  return points(size_t(nTotal / commSize) + size_t(commRank < nTotal % commSize));
}


// -----------------------------------------

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


// DollarStat::mpi_send_map()
void DollarStat::mpi_send_map( const std::map<URI, DollarStat::info> &map,
                   int dest, int tag, MPI_Comm comm )
{
  Pack<std::vector<std::string>> key;
  Pack<double> hits_total;
  Pack<std::string> short_title;

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


// DollarStat::mpi_recv_map()
void DollarStat::mpi_recv_map( std::map<URI, DollarStat::info> &map,
                   int src, int tag, MPI_Comm comm, MPI_Status *status )
{
  map.clear();

  Pack<std::vector<std::string>> key;
  Pack<double> hits_total;
  Pack<std::string> short_title;

  key.mpi_recv(src, tag, comm);
  hits_total.mpi_recv(src, tag, comm);
  short_title.mpi_recv(src, tag, comm);

  while (!key.end())
  {
    info & info = map[key.unpack()];
    std::tie(info.hits, info.total) = hits_total.unpack<double, double>();
    info.short_title = short_title.unpack();
  }
}


//
// DollarStat::mpi_reduce()
//
template <typename ReducerT>
std::map<DollarStat::URI, DollarStat::info>
DollarStat::mpi_reduce(ReducerT reducer)
{
  int commSize, commRank;
  MPI_Comm_size(m_comm, &commSize);
  MPI_Comm_rank(m_comm, &commRank);

  std::map<URI, info> totals = m_totals;

  int bit_plane = 1;
  while (bit_plane < commSize && !bool((bit_plane >> 1) & commRank))
  {
    int partner = commRank ^ bit_plane;
    if (partner < commSize)
    {
      std::map<URI, info> partner_totals;
      const int tag = 0;

      if (partner < commRank)
      {
        mpi_send_map(totals, partner, tag, m_comm);
      }
      else
      {
        mpi_recv_map(partner_totals, partner, tag, m_comm);
        reducer(totals, const_cast<const std::map<URI, info> &>(partner_totals));
      }
    }
    bit_plane <<= 1;
  }

  return totals;
}




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

  Map totals = this->mpi_reduce(reducer_sum);

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

  return DollarStat(m_comm, this->mpi_reduce(reducer_min));
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

  return DollarStat(m_comm, this->mpi_reduce(reducer_max));
}

