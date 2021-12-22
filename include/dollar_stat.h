/**
 * @author Masado Ishii (University of Utah)
 * @date 2021-12-22
 */

#ifndef DENDRO_KT_DOLLAR_STAT_H
#define DENDRO_KT_DOLLAR_STAT_H

#include <dollar.hpp>

namespace dollar
{
  class DollarStat
  {
    public:
      using URI = std::vector<std::string>;

      struct info {
        double hits = 0;
        double total = 0;
        std::string short_title;
      };

    public:
      DollarStat(MPI_Comm comm);

      // Note that only the root=0 has the final reduced data.
      DollarStat mpi_reduce_mean();
      DollarStat mpi_reduce_min();
      DollarStat mpi_reduce_max();

      void print(std::ostream &out = std::cout);

    private:
      MPI_Comm m_comm;
      std::map<URI, info> m_totals;

      DollarStat(MPI_Comm comm, const std::map<URI, info> &totals)
        : m_comm(comm), m_totals(totals)
      {}
  };

}//namespace dollar

#endif//DENDRO_KT_DOLLAR_STAT_H
