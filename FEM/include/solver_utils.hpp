
#ifndef DENDRO_KT_FEM_SOLVER_UTILS_HPP
#define DENDRO_KT_FEM_SOLVER_UTILS_HPP

#include <queue>

namespace util
{
  class ConvergenceRate
  {
    public:
      // ConvergenceRate()
      ConvergenceRate(int max_history)
        : m_max_history(max_history)
      {
        assert(max_history >= 3);
      }

      // max_history()
      int max_history() const { return m_max_history; }

      // step_count()
      int step_count() const { return m_step_count; }

      // observe_step()
      void observe_step(double value)
      {
        if (m_history.size() == this->max_history())
          m_history.pop_front();
        m_history.push_back(value);
        ++m_step_count;
      }

      // rate()
      // Real or infinity, not NaN (if last three observations are real numbers).
      double rate() const
      {
        assert(step_count() >= 2);
        double prev[3] = {
          (step_count() > 2 ?
            m_history[m_history.size() - 3] : 0.0),
          m_history[m_history.size() - 2],
          m_history[m_history.size() - 1]
        };
        if (prev[2] == 0.0 and prev[1] == 0.0)
          return 0.0;
        else if (prev[0] == 0.0)
          return prev[2] / prev[1];
        else
          return std::exp(
              std::log(prev[2]/prev[1])*0.75
              + std::log(prev[1]/prev[0])*0.25);
      }

    private:
      int m_max_history = 2;
      int m_step_count = 0;
      std::deque<double> m_history;  //future: fixed-length ring buffer
  };

}//namespace util

#endif//DENDRO_KT_FEM_SOLVER_UTILS_HPP
