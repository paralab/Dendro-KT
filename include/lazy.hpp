/**
 * @author Masado Ishii
 * @date 2022-03-17
 */

#ifndef DENDRO_KT_LAZY_HPP
#define DENDRO_KT_LAZY_HPP

#include <utility>
#include <assert.h>

// Maintain a bool that is updated when value is initialized.
// Initializer function not stored, so control flow readable in calling code.
// Also some initializers include a perishable reference to a containing object,
// and only the container can produce a fresh initializer for the copy.

template <typename X>
class Lazy
{
  private:
    bool m_init = false;
    X m_x;
  public:
    Lazy() = default;

    // Initialization survives copy/move.
    Lazy(const Lazy &) = default;
    Lazy(Lazy &&) = default;
    Lazy & operator=(const Lazy &) = default;
    Lazy & operator=(Lazy &&) = default;

    template <typename Y>  Lazy(Y &&y)      : m_init(true), m_x(std::forward<Y>(y)) {}
    template <typename Y>  Lazy & operator=(Y &&y)      { m_init = true;  m_x = std::forward<Y>(y);  return *this; }

    bool initialized() const { return m_init; }
    void expire()            { m_init = false; }
    X & get()                { assert(initialized());  return m_x; }
    const X & get() const    { assert(initialized());  return m_x; }
};

template <typename X>
class LazyPerishable
{
  private:
    bool m_init = false;
    X m_x;
  public:
    LazyPerishable() = default;

    // Force re-initialization on copy or assignment.
    LazyPerishable(const LazyPerishable &) { }
    LazyPerishable(LazyPerishable &&)      { }
    LazyPerishable & operator=(const LazyPerishable &) { expire();  return *this; }
    LazyPerishable & operator=(LazyPerishable &&)      { expire();  return *this; }

    template <typename Y>  LazyPerishable(Y &&y)      : m_init(true), m_x(std::forward<Y>(y)) {}
    template <typename Y>  LazyPerishable & operator=(Y &&y)      { m_init = true;  m_x = std::forward<Y>(y);  return *this; }

    bool initialized() const { return m_init; }
    void expire()            { m_init = false; }
    X & get()                { assert(initialized());  return m_x; }
    const X & get() const    { assert(initialized());  return m_x; }
};

#endif//DENDRO_KT_LAZY_HPP
