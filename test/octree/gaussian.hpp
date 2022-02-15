#ifndef DENDRO_KT_TEST_GAUSSIAN_HPP
#define DENDRO_KT_TEST_GAUSSIAN_HPP

#include <vector>
#include <array>
#include <random>

extern unsigned int m_uiMaxDepth;

namespace test
{
  template <typename T, unsigned int dim, typename ArrayToX>
  std::vector<decltype(ArrayToX()(std::array<T, dim>{}, int{}))>
  gaussian(size_t n, ArrayToX array_to_x);
}

namespace test
{
  template <typename T, unsigned int dim, typename ArrayToX>
  auto gaussian(long long unsigned begin, size_t n, ArrayToX array_to_x)
      -> std::vector<decltype(array_to_x(std::array<T, dim>{}, int{}))>
  {
    using X = decltype(array_to_x(std::array<T, dim>{}, int{}));
    std::vector<X> xs;

    std::array<T, dim> upper_bounds;
    upper_bounds.fill((1u << m_uiMaxDepth) - 1);
    const auto clamp = [&upper_bounds](int d, const T c) {
        return T(fmaxf(0, fminf(upper_bounds[d], c)));
    };
    const auto clamp_lev = [](double l) {
        return int(fmaxf(0, fminf(m_uiMaxDepth, l)));
    };

    constexpr bool truly_random = false;
    long long unsigned seed = 42;
    std::random_device rd;
    long long unsigned s = (truly_random ? rd() : seed);

    std::mt19937_64 gen(s);
    gen.discard(begin * (dim + 1));

    std::normal_distribution<double> gdist;
    using GParams = std::normal_distribution<double>::param_type;

    for (size_t i = 0; i < n; ++i)
    {
      std::array<T, dim> coords;
      for (int d = 0; d < dim; ++d)
        coords[d] = clamp(d, gdist(gen, GParams(upper_bounds[d]/2.0, upper_bounds[d]/8.0)));
      int lev = clamp_lev(m_uiMaxDepth - fabs(gdist(gen, GParams(0, m_uiMaxDepth/8.0))));

      xs.push_back(array_to_x(coords, lev));
    }

    return xs;
  }
}

#endif//DENDRO_KT_TEST_GAUSSIAN_HPP
