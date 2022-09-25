// ==========================================================================
// Interface
// ==========================================================================

#include <ostream>
#include <algorithm>

#define SUSPECT_STATIC(Type, var, init, Module) \
  static Type var = (spct::add<Module>((#var), var), init);

#define SUSPECT_STATIC_COUNTER(Type, var, Module) \
  { static Type var = (spct::add<Module>((#var), var), Type{}); ++var; }

namespace spct
{
  struct DefaultModuleTag;

  template <typename ModuleTag = DefaultModuleTag, typename T>
  void add(std::string name, const T &x);

  template <typename ModuleTag = DefaultModuleTag>
  void report(std::ostream &out);

  template <typename ModuleTag = DefaultModuleTag>
  size_t number_of_references();

  template <typename ModuleTag = DefaultModuleTag>
  const std::string name(size_t i);

  template <typename ModuleTag = DefaultModuleTag, typename T>
  const T & ref(size_t i, const T &deduce_type_T);

  template <typename ModuleTag = DefaultModuleTag, typename T>
  const T & ref(std::string name, const T &deduce_type_T);  // linear search

  template <typename T>
  const T & to = {};


  template<typename ModuleTag = DefaultModuleTag>
  struct Tagged
  {
    template <typename T>
    static void add(std::string name, const T &x);

    static void report(std::ostream &out);
    static size_t number_of_references();
    static const std::string name(size_t i);

    template <typename T>
    static const T & ref(size_t i, const T &deduce_type_T);

    template <typename T>
    static const T & ref(std::string name, const T &deduce_type_T);
  };

  template<typename ModuleTag>
  Tagged<ModuleTag> tagged() { return {}; }


  // Usage:
  //     void function()
  //     {
  //       static int a = (spct::add("statistic_a", a), 0);
  //       static float b = (spct::add("statistic_b", b), 1.0f);
  //       ++a;
  //       b *= 2.0f;
  //     }
  //
  //     struct MyModule;
  //     void module_function()
  //     {
  //       static int a = (spct::add<MyModule>("statistic_a", a), 0);
  //       ++c;
  //     }
  //     
  //     int main()
  //     {
  //       function();
  //       function();
  //       module_function();
  //       std::cout << "Default:\n";    spct::report(std::cout);
  //       std::cout << "MyModule:\n";   spct::report<MyModule>(std::cout);
  //     }

  // Alternative syntax if the module tag needs to be temporarily abbreviated.
  //     auto module_stats = spct::tagged<MyModuleWithAVeryDetailedName>();
  //     const int a = module_stats.ref("statistic_a", a);
  //     const float b = module_stats.ref("statistic_b", spct::to<float>);
}


// ==========================================================================
// Implementation
// ==========================================================================

#include <vector>

namespace spct
{
  template <typename T = void>
  struct CommonReference;

  template <>
  struct CommonReference<void>
  {
    virtual ~CommonReference() = default;
    friend std::ostream & operator<<(std::ostream &out, const CommonReference &v)
    {
      return v.print(out);
    }
    protected:
      virtual std::ostream & print(std::ostream &out) const = 0;
  };

  template <typename T>
  struct CommonReference : public CommonReference<void>
  {
    const T &ref;
    CommonReference(const T &ref) : ref{ref} {}  //not aggregate type due to virtual

    protected:
      virtual std::ostream & print(std::ostream &out) const
      {
        return out << ref;
      }
  };


  struct Registry
  {
    struct Record
    {
      std::string name;
      CommonReference<> *item;
    };

    Registry() = default;

    ~Registry()
    {
      for (const auto &r : m_records)
        delete r.item;
    }

    template <typename T>
    void insert(std::string name, const T &value)
    {
      m_records.push_back(Record{name, new CommonReference<T>(value)});
    }

    size_t size() const
    {
      return m_records.size();
    }

    std::string name(size_t i) const
    {
      return m_records[i].name;
    }

    CommonReference<> * item(size_t i) const
    {
      return m_records[i].item;
    }

    CommonReference<> * item(std::string name) const
    {
      const auto pos = std::find_if(m_records.begin(), m_records.end(),
          [&name](const Record &r){return r.name == name;});
      if (pos == m_records.end())
        throw std::logic_error("Name \"" + name + "\" not found in this registry.");
      return pos->item;
    }

    std::vector<Record> m_records;
  };

  template <typename ModuleTag>
  Registry registry;


  template <typename ModuleTag = DefaultModuleTag, typename T>
  void add(std::string name, const T &x)
  {
    registry<ModuleTag>.insert(name, x);
  }

  template <typename ModuleTag = DefaultModuleTag>
  void report(std::ostream &out)
  {
    for (const auto &r : registry<ModuleTag>.m_records)
    {
      out << r.name << ": " << *r.item << "\n";
    }
  }

  template <typename ModuleTag = DefaultModuleTag>
  size_t number_of_references()
  {
    return registry<ModuleTag>.size();
  }

  template <typename ModuleTag = DefaultModuleTag>
  const std::string name(size_t i)
  {
    return registry<ModuleTag>.name(i);
  }

  template <typename ModuleTag = DefaultModuleTag, typename T>
  const T & ref(size_t i, const T &deduce_type_T)
  {
    return static_cast<CommonReference<T>*>(registry<ModuleTag>.item(i))->ref;
  }

  template <typename ModuleTag = DefaultModuleTag, typename T>
  const T & ref(std::string name, const T &deduce_type_T)
  {
    return static_cast<CommonReference<T>*>(registry<ModuleTag>.item(name))->ref;
  }



  template<typename ModuleTag>
  template <typename T>
  void Tagged<ModuleTag>::add(std::string name, const T &x)
  {
    return ::spct::add<ModuleTag>(name, x);
  }

  template<typename ModuleTag>
  void Tagged<ModuleTag>::report(std::ostream &out)
  {
    return ::spct::report<ModuleTag>(out);
  }

  template<typename ModuleTag>
  size_t Tagged<ModuleTag>::number_of_references()
  {
    return ::spct::number_of_references<ModuleTag>();
  }

  template<typename ModuleTag>
  const std::string Tagged<ModuleTag>::name(size_t i)
  {
    return ::spct::name<ModuleTag>(i);
  }

  template<typename ModuleTag>
  template <typename T>
  const T & Tagged<ModuleTag>::ref(size_t i, const T &deduce_type_T)
  {
    return ::spct::ref<ModuleTag>(i, to<T>);
  }

  template<typename ModuleTag>
  template <typename T>
  const T & Tagged<ModuleTag>::ref(std::string name, const T &deduce_type_T)
  {
    return ::spct::ref<ModuleTag>(name, to<T>);
  }

}//namespace spct
