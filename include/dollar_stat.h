/**
 * @author Masado Ishii (University of Utah)
 * @date 2021-12-22
 */

#ifndef DENDRO_KT_DOLLAR_STAT_H
#define DENDRO_KT_DOLLAR_STAT_H

#include <dollar.hpp>
#include <mpi.h>

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

      template<bool for_chrome = false>
      void print_tree( std::ostream &out, const char *tab = ",", const char *feed = "\n" ) const;

      void csv( std::ostream &os ) const      { this->print_tree<0>(os, ","); }
      void tsv( std::ostream &os ) const      { this->print_tree<0>(os, "\t"); }
      void markdown( std::ostream &os ) const { this->print_tree<0>(os, "|"); }
      void text( std::ostream &os ) const     { this->print_tree<0>(os, " "); }
      void chrome( std::ostream &os ) const   { this->print_tree<1>(os, ""); }

    private:
      MPI_Comm m_comm;
      std::map<URI, info> m_totals;

      DollarStat(MPI_Comm comm, const std::map<URI, info> &totals)
        : m_comm(comm), m_totals(totals)
      {}
  };

}//namespace dollar

#endif//DENDRO_KT_DOLLAR_STAT_H
