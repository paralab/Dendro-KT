#include "profiler.h"

#include "profile_matvec.h"
#include "genChannelPoints.h"

#include "treeNode.h"
#include "tsort.h"
#include "nsort.h"
#include "octUtils.h"
#include "hcurvedata.h"
#include "filterFunction.h"
#include "distTree.h"

#include "matvec.h"
#include "feMatrix.h"

#include "poissonMat.h"
#include "poissonVec.h"

#include <cstring>


namespace bench2
{
  struct ProfilerReference
  {
    ProfilerReference() = default;
    ProfilerReference(const ProfilerReference&) = default;
    ~ProfilerReference() = default;

    ProfilerReference(profiler_t & profiler, const std::string &name)
      : m_profiler(profiler), m_name(name)
    {}

    double getSeconds() const { return m_profiler.seconds / m_divisor; }
    std::string getName() const { return m_name; }
    void setDivisor(double divisor) { m_divisor = divisor; }

    profiler_t   &m_profiler;
    std::string  m_name;
    double m_divisor = 1.0;
  };

  profiler_t t_topdown;
  profiler_t t_bottomup;
  profiler_t t_elemental;
  profiler_t t_ghostread_begin;
  profiler_t t_ghostread_end;
  profiler_t t_ghostread_manage;
  profiler_t t_ghostwrite_begin;
  profiler_t t_ghostwrite_end;
  profiler_t t_ghostwrite_manage;

  std::array<ProfilerReference, 9> profilers =
  {
    ProfilerReference(t_topdown, "topdown"),
    ProfilerReference(t_bottomup, "bottomup"),
    ProfilerReference(t_elemental, "elemental"),

    ProfilerReference(t_ghostread_begin, "ghostread_begin"),
    ProfilerReference(t_ghostread_end, "ghostread_end"),
    ProfilerReference(t_ghostread_manage, "ghostread_manage"),

    ProfilerReference(t_ghostwrite_begin, "ghostwrite_begin"),
    ProfilerReference(t_ghostwrite_end, "ghostwrite_end"),
    ProfilerReference(t_ghostwrite_manage, "ghostwrite_manage"),
  };


  typedef long long unsigned Counter;

  struct CounterReference
  {
    CounterReference() = default;
    CounterReference(const CounterReference&) = default;
    ~CounterReference() = default;

    CounterReference(Counter &counter, const std::string &name)
      : m_counter(counter), m_name(name)
    {}

    void clear() { m_counter = 0; }
    double getCount() const { return double(m_counter) / m_divisor; }
    std::string getName() const { return m_name; }
    void setDivisor(double divisor) { m_divisor = divisor; }

    Counter      &m_counter;
    std::string  m_name;
    double m_divisor = 1.0;
  };

  Counter c_treeSz;
  Counter c_ghostread_sends;
  Counter c_ghostread_sendSz;
  Counter c_ghostread_recvs;
  Counter c_ghostread_recvSz;
  Counter c_ghostwrite_sends;
  Counter c_ghostwrite_sendSz;
  Counter c_ghostwrite_recvs;
  Counter c_ghostwrite_recvSz;

  std::array<CounterReference, 9> counters =
  {
    CounterReference(c_treeSz,            "treeSz"),
    CounterReference(c_ghostread_sends,   "ghostread_sends"),
    CounterReference(c_ghostread_sendSz,  "ghostread_sendSz"),
    CounterReference(c_ghostread_recvs,   "ghostread_recvs"),
    CounterReference(c_ghostread_recvSz,  "ghostread_recvSz"),
    CounterReference(c_ghostwrite_sends,  "ghostwrite_sends"),
    CounterReference(c_ghostwrite_sendSz, "ghostwrite_sendSz"),
    CounterReference(c_ghostwrite_recvs,  "ghostwrite_recvs"),
    CounterReference(c_ghostwrite_recvSz, "ghostwrite_recvSz"),

  };


  void divideProfilersByRuns(int numRuns);
  void divideCountersByRuns(int numRuns);

  void resetProfilers();
  void resetCounters();

  template <typename T>
  struct TriSummary
  {
    T m_glob_min;
    T m_glob_mean;
    T m_glob_max;
  };

  template <typename T>
  TriSummary<T> computeTriSummary(T localData, MPI_Comm comm)
  {
    TriSummary<T> g;

    par::Mpi_Reduce(&localData, &g.m_glob_min,  1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(&localData, &g.m_glob_mean, 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(&localData, &g.m_glob_max,  1, MPI_MAX, 0, comm);

    int npes, rank;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    g.m_glob_mean /= double(npes);

    if (rank != 0)
    {
      g.m_glob_min = 0;
      g.m_glob_mean = 0;
      g.m_glob_max = 0;
    }

    return g;
  }

  // ------------------------------------------------------
  int numResultFields() { return profilers.size() + counters.size(); }
  std::vector<std::string> getResultPrefixes();
  std::vector<TriSummary<double>> computeSummaries(MPI_Comm comm);
  // ------------------------------------------------------

  struct CommResult
  {
    int m_npes = 0;
    std::vector<TriSummary<double>> m_figures;
  };

  struct GlobalResult
  {
    std::vector<int> m_list_npes;
    std::vector<TriSummary<double>> m_concat_figures;
  };


  struct BenchSettings
  {
    BenchSettings() = default;
    BenchSettings(const BenchSettings &) = default;
    ~BenchSettings() = default;

    BenchSettings( int       numWarmup,
                   int       numRuns,
                   size_t    grainSz,
                   unsigned  eleOrder,
                   double    loadFlexibility,
                   int       lengthPower2,
                   bool      isAdaptive)
      :
        m_numWarmup       ( numWarmup ),
        m_numRuns         ( numRuns ),
        m_grainSz         ( grainSz ),
        m_eleOrder        ( eleOrder),
        m_loadFlexibility ( loadFlexibility ),
        m_lengthPower2    ( lengthPower2 ),
        m_isAdaptive      ( isAdaptive )
    {}

    int     m_numWarmup;
    int     m_numRuns;
    size_t  m_grainSz;
    unsigned m_eleOrder;
    double  m_loadFlexibility;
    int     m_lengthPower2;
    bool    m_isAdaptive;
  };

  CommResult benchProblem(const BenchSettings &benchSettings, MPI_Comm comm);
  GlobalResult gatherResults(const CommResult &benchCommResult, MPI_Comm commRoots);
}


//
// main()
//
int main(int argc, char** argv)
{
  // -----------------------------------------------
  // Communicators
  // -----------------------------------------------
  MPI_Init(&argc,&argv);

  MPI_Comm commGlobal = MPI_COMM_WORLD;

  int rankGlobal, npesGlobal;
  MPI_Comm_rank(commGlobal, &rankGlobal);
  MPI_Comm_size(commGlobal, &npesGlobal);

  MPI_Comm commBench;
  MPI_Comm commBenchRoots;

  // Split comms into halves.
  // If npesGlobal == pow(2, k) - 1, then all communicators are powers of 2.
  /// std::vector<int> commSizesReversed;
  int benchColor = 0;
  int remainderSize = npesGlobal;
  int lowerHalf = remainderSize / 2;
  while (rankGlobal < lowerHalf)
  {
    benchColor++;
    remainderSize = lowerHalf;
    lowerHalf = remainderSize / 2;
  }

  MPI_Comm_split(commGlobal, benchColor, rankGlobal, &commBench);
  int rankBench, npesBench;
  MPI_Comm_rank(commBench, &rankBench);
  MPI_Comm_size(commBench, &npesBench);

  const bool isBenchRoot = (rankBench == 0);
  MPI_Comm_split(commGlobal, (isBenchRoot ? 0 : MPI_UNDEFINED), rankGlobal, &commBenchRoots);
  int rankBenchRoots = 0, npesBenchRoots = 0;
  if (isBenchRoot)
  {
    MPI_Comm_rank(commBenchRoots, &rankBenchRoots);
    MPI_Comm_size(commBenchRoots, &npesBenchRoots);
  }
  // -----------------------------------------------


  //-------------------------------
  // Parse command line arguments.
  //-------------------------------
  bench2::BenchSettings benchSettings;
  {
    // Default values.
    int     numWarmup = 5;
    int     numRuns = 50;
    size_t  grainSz = 1000;
    unsigned eleOrder = 2;
    double  loadFlexibility = 0.3;
    int     lengthPower2 = 0;
    bool    isAdaptive = true;

    std::string help
        = "Usage: " + std::string(argv[0]) + " numWarmup"
                                             " numRuns"
                                             " grainSz"
                                             " eleOrder"
                                             " loadFlexibility"
                                             " lengthPower2"
                                             " isAdaptive"
                                             + "\n";
    if (argc < 1+7)
    {
      if (rankGlobal == 0)
        std::cerr << help;
      MPI_Barrier(commGlobal);
      MPI_Abort(commGlobal, 1);
    }

    numWarmup       = atoi(argv[1]);
    numRuns         = atoi(argv[2]);
    grainSz         = atol(argv[3]);
    eleOrder        = atoi(argv[4]);
    loadFlexibility = atof(argv[5]);
    lengthPower2    = atoi(argv[6]);
    isAdaptive      = atoi(argv[7]);

    benchSettings = bench2::BenchSettings(
        numWarmup,
        numRuns,
        grainSz,
        eleOrder,
        loadFlexibility,
        lengthPower2,
        isAdaptive
        );
  }
  //-------------------------------

  //--------------------------
  // Get results
  //--------------------------
  bench2::CommResult benchCommResult = bench2::benchProblem(benchSettings, commBench);
  bench2::GlobalResult globalResult;
  if (isBenchRoot)
    globalResult = bench2::gatherResults(benchCommResult, commBenchRoots);
  //--------------------------


  //--------------------------
  // Print report
  //--------------------------
  if (rankGlobal == 0)
  {
    fprintf(stdout, "npes");
    for (const std::string &fieldPrefix : bench2::getResultPrefixes())
    {
      fprintf(stdout, "\t%s.min", fieldPrefix.c_str());
      fprintf(stdout, "\t%s.mean", fieldPrefix.c_str());
      fprintf(stdout, "\t%s.max", fieldPrefix.c_str());
    }
    fprintf(stdout, "\n");

    size_t figureIdx = 0;
    for (int benchGroup = 0; benchGroup < npesBenchRoots; ++benchGroup)
    {
      fprintf(stdout, "%d", globalResult.m_list_npes[benchGroup]);
      for (int fieldIdx = 0; fieldIdx < bench2::numResultFields(); ++fieldIdx)
      {
        fprintf(stdout, "\t%f", globalResult.m_concat_figures[figureIdx].m_glob_min);
        fprintf(stdout, "\t%f", globalResult.m_concat_figures[figureIdx].m_glob_mean);
        fprintf(stdout, "\t%f", globalResult.m_concat_figures[figureIdx].m_glob_max);
        figureIdx++;
      }
      fprintf(stdout, "\n");
    }
  }
  //--------------------------

  //--------------------------
  MPI_Comm_free(&commBench);
  if (isBenchRoot)
    MPI_Comm_free(&commBenchRoots);
  MPI_Finalize();
  //--------------------------

  return 0;
}




namespace bench2
{

  // ====================================================
  // benchProblem()
  // ====================================================
  CommResult benchProblem(const BenchSettings &opt, MPI_Comm comm)
  {
    // --------------------
    int rank, npes;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);
    // --------------------


    // ---------------------------------------
    constexpr int dim = 3;
    _InitializeHcurve(dim);


    const unsigned int DOF = 1;

    using u_T = unsigned int;
    using u_TreeNode = ot::TreeNode<u_T, dim>;
    using u_TNP = ot::TNPoint<u_T, dim>;
    using u_SFC_Tree = ot::SFC_Tree<u_T, dim>;
    using u_DistTree = ot::DistTree<u_T, dim>;
    using u_DA = ot::DA<dim>;

    //
    // Set up geometry.
    //
    const ibm::DomainDecider boxDecider = bench::getBoxDecider<dim>(opt.m_lengthPower2);

    //TODO use option for isAdaptive
    std::vector<u_TreeNode> points = bench::getChannelPoints<dim>(
        opt.m_grainSz, opt.m_lengthPower2, comm);

    // Remove duplicates that could force leafs to m_uiMaxDepth.
    u_SFC_Tree::distRemoveDuplicates(points, opt.m_loadFlexibility, false, comm);

    // Create the tree based on the point distribution.
    std::vector<u_TreeNode> treeOriginal;
    u_SFC_Tree::distTreeBalancingWithFilter(boxDecider, points, treeOriginal, 1, opt.m_loadFlexibility, comm);

    // Create DistTree and get a reference to treeFiltered.
    u_DistTree dtree(treeOriginal, comm);
    dtree.filterTree(boxDecider);
    const std::vector<u_TreeNode> &treeFiltered = dtree.getTreePartFiltered();

    //
    // Create DA from DistTree and vectors from DA.
    //
    u_DA octDA(dtree, comm, opt.m_eleOrder, opt.m_grainSz, opt.m_loadFlexibility);
    std::vector<double> uSolVec, dummyVec;
    octDA.createVector(uSolVec, false, false, DOF);
    octDA.createVector(dummyVec, false, false, DOF);


    //
    // Benchmark matvec.
    //
    Point<dim> domain_min(-0.5,-0.5,-0.5);
    Point<dim> domain_max(0.5,0.5,0.5);

    PoissonEq::PoissonMat<dim> myPoissonMat(&octDA, &treeFiltered, DOF);
    myPoissonMat.setProblemDimensions(domain_min, domain_max);

    std::function<void(const double *, double*)> f_init =[](const double * xyz, double *var){
      var[0] = 1;
    };

    double *ux = &(*(uSolVec.begin()));
    double *dummy = &(*(dummyVec.begin()));

    octDA.setVectorByFunction(ux, f_init, false, false, DOF);
    octDA.setVectorByFunction(dummy, f_init, false, false, DOF);

    // Warmup
    for (int ii = 0; ii < opt.m_numWarmup; ii++)
      myPoissonMat.matVec(ux, dummy, 1.0);

    // Don't count the warmup runs.
    resetCounters();
    resetProfilers();

    // Benchmark
    for (int ii = 0; ii < opt.m_numRuns; ii++)
      myPoissonMat.matVec(ux, dummy, 1.0);

    octDA.destroyVector(uSolVec);
    octDA.destroyVector(dummyVec);

    _DestroyHcurve();
    // ---------------------------------------


    // --------------------
    // Reduce summaries for the communicator group.
    // --------------------
    divideProfilersByRuns(opt.m_numRuns);
    divideCountersByRuns(opt.m_numRuns);
    CommResult commResult;
    commResult.m_npes = npes;
    commResult.m_figures = computeSummaries(comm);
    // --------------------

    return commResult;
  }
  // ====================================================





  GlobalResult gatherResults(const CommResult &benchCommResult, MPI_Comm commRoots)
  {
    int rankRoots, npesRoots;
    MPI_Comm_size(commRoots, &npesRoots);
    MPI_Comm_rank(commRoots, &rankRoots);

    GlobalResult globalResult;

    if (rankRoots == 0)
    {
      globalResult.m_list_npes.resize(npesRoots);
      globalResult.m_concat_figures.resize(npesRoots * numResultFields());
    }

    MPI_Gather(&benchCommResult.m_npes, 1, MPI_INT,
               &globalResult.m_list_npes[0], 1, MPI_INT,
               0, commRoots);

    MPI_Gather((double*) &benchCommResult.m_figures[0],     3*numResultFields(), MPI_DOUBLE,
               (double*) &globalResult.m_concat_figures[0], 3*numResultFields(), MPI_DOUBLE,
               0, commRoots);

    return globalResult;
  }




  //
  // getResultPrefixes()
  //
  std::vector<std::string> getResultPrefixes()
  {
    std::vector<std::string> prefixes;

    for (const ProfilerReference &profilerRef : profilers)
      prefixes.push_back(profilerRef.getName());
    for (const CounterReference &counterRef : counters)
      prefixes.push_back(counterRef.getName());

    return prefixes;
  }


  //
  // devideProfilersByRuns()
  //
  void divideProfilersByRuns(int numRuns)
  {
    for (auto &profilerRef : profilers)
      profilerRef.setDivisor(numRuns);
  }


  //
  // devideCountersByRuns()
  //
  void divideCountersByRuns(int numRuns)
  {
    for (auto &counterRef : counters)
      counterRef.setDivisor(numRuns);
  }

  //
  // resetProfilers()
  //
  void resetProfilers()
  {
    for (auto &profilerRef : profilers)
      profilerRef.m_profiler.clear();
  }

  //
  // resetCounters()
  //
  void resetCounters()
  {
    for (auto &counterRef : counters)
      counterRef.clear();
  }



  //
  // computeSummaries()
  //
  std::vector<TriSummary<double>> computeSummaries(MPI_Comm comm)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<TriSummary<double>> summaries;

    for (const ProfilerReference &profilerRef : profilers)
    {
      TriSummary<double> summary = computeTriSummary<double>(profilerRef.getSeconds(), comm);
      if (rank == 0)
        summaries.push_back(summary);
    }
    for (const CounterReference &counterRef : counters)
    {
      TriSummary<double> summary = computeTriSummary<double>(counterRef.getCount(), comm);
      if (rank == 0)
        summaries.push_back(summary);
    }

    return summaries;
  }
}
