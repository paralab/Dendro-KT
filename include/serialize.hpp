/**
 * @author Masado Ishii (University of Utah)
 * @date 2021-12-22
 */

#ifndef DENDRO_KT_SERIALIZE_HPP
#define DENDRO_KT_SERIALIZE_HPP

#include <mpi.h>
#include "dtypes.h"

namespace serialize
{
  /**
   * @brief Pack converts sequence of values to/from internal vector of scalars.
   * @example
   *     std::set<int> remote, own;
   *     if (commRank % 2)  own.insert({ 1, 2, 3, 4, 5, 6 });
   *                  else  own.insert({ 2, 4, 6, 8, 10, 12 });
   *     // Union with remote set.
   *     serialize::Pack<int> p, q;
   *     for (int i : own)  p.pack(i);
   *     int tag = 0;
   *     p.mpi_send(commRank ^ 1, tag, comm);
   *     q.mpi_recv(commRank ^ 1, tag, comm);
   *     while (!q.end())  remote.insert(q.unpack1());
   *     own.insert(remote.begin(), remote.end());
   *     for (int i : own)  std::cout << i << " ";  std::cout << "\n";
   *     // 1, 2, 3, 4, 5, 6, 8, 10, 12
   */

  //
  // Pack<T> for trivially copyable type T (e.g. scalar types)
  //
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
        {
          m_values.push_back(t);
          ++m_count;
        }
      }

      template <typename T0, typename ...TS>
      void unpack(T0&& t0, TS&&...ts)
      {
        t0 = m_values[m_iter++];
        this->unpack(ts...);
      }
      // Note: Some compilers have a bug such that the braced-init-list
      //   {TS(m_values[m_iter++])...}  might not be eval'd left-to-right.

      void unpack() { }

      T unpack1() { T ret; this->unpack(ret); return ret; }

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


  //
  // Pack<std::string>
  //
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


  //
  // Pack<std::vector<std::string>>
  //
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

}//namespace serialize

#endif//DENDRO_KT_SERIALIZE_HPP
