
#include <functional>

#include "mpi.h"
#include "dendro.h"

#include "treeNode.h"
#include "tsort.h"
#include "octUtils.h"
#include "distTree.h"
#include "oda.h"

#include "refel.h"
#include "poissonVec.h"
#include "poissonMat.h"
#include "gmgMat.h"

#ifdef BUILD_WITH_PETSC
  #include <petsc.h>
  #include <petscvec.h>
  #include <petscksp.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>

// =======================================================
// Parameters: Change these and the options in get_args().
// =======================================================
struct Parameters
{
  unsigned int dim;
  unsigned int maxDepth;
  unsigned int nGrids;
  double waveletTol;
  double partitionTol;
  unsigned int eleOrder;
};
// =======================================================

int DBG_COUNT;
std::ostream * DBG_FINE_RES0;
std::ostream * DBG_FINE_RES1;
std::ostream * DBG_FINE_RES2;
std::ostream * DBG_FINE_RES3;
std::ostream * DBG_COARSE_COR2;
std::ostream * DBG_COARSE_RES0;
std::ostream * DBG_COARSE_RES3;
bool DBG_WRITE_NODES = false;


struct MatlabDataSink
{
  std::ofstream m_out;

  MatlabDataSink() {}
  MatlabDataSink(const MatlabDataSink &other) = delete;

  MatlabDataSink(const std::string &pathPrefixAndSlash,
                 const std::string &baseName)
  {
    this->open(pathPrefixAndSlash, baseName);
  }

  ~MatlabDataSink()
  {
    this->close();
  }

  void open(const std::string &pathPrefixAndSlash,
            const std::string &baseName)
  {
    m_out.open(pathPrefixAndSlash + baseName + ".m", std::ios::out);
    m_out << "function data = " << baseName << "()\n"
          << "  data = [\n";
  }

  void close()
  {
    m_out << "  ];\nend\n";
    m_out.close();
  }

  operator std::ofstream&() { return m_out; }
};


struct MatlabDataRoot
{
  std::vector<std::string> m_sliceNames;
  std::vector<int> m_sliceTimes;

  void addSlice(const std::string &name, int time)
  {
    m_sliceNames.push_back(name);
    m_sliceTimes.push_back(time);
  }

  void finalize(const std::string &pathPrefixAndSlash,
                const std::string &baseName)
  {
    std::ofstream out(pathPrefixAndSlash + baseName + ".m", std::ios::out);
    out << "function data = " << baseName << "()\n";
    for (int i = 0; i < m_sliceNames.size(); i++)
      out << "  data(:,:," << m_sliceTimes[i]+1 << ") = "
          << m_sliceNames[i] << "();\n";
    out << "end\n";
    out.close();
  }
};



namespace PoissonEq
{

  template <unsigned int dim>
  class PoissonGMGMat : public gmgMat<dim, PoissonGMGMat<dim>>
  {
    // References to base class members for convenience.
    using BaseT = gmgMat<dim, PoissonGMGMat<dim>>;
    ot::MultiDA<dim> * & m_multiDA = BaseT::m_multiDA;
    /// ot::MultiDA<dim> * & m_surrogateMultiDA = BaseT::m_surrogateMultiDA;
    unsigned int & m_numStrata = BaseT::m_numStrata;
    unsigned int & m_ndofs = BaseT::m_ndofs;
    using BaseT::m_uiPtMin;
    using BaseT::m_uiPtMax;

    public:
      PoissonGMGMat(ot::MultiDA<dim> *mda, ot::MultiDA<dim> *smda, unsigned int ndofs)
        : BaseT(mda, smda, ndofs)
      {
        m_gridOperators.resize(m_numStrata);
        for (int s = 0; s < m_numStrata; ++s)
        {
          m_gridOperators[s] = new PoissonMat<dim>(&getMDA()[s], ndofs);
          m_gridOperators[s]->setProblemDimensions(m_uiPtMin, m_uiPtMax);
        }

        m_tmpRes.resize(ndofs * getMDA()[0].getLocalNodalSz());

        m_rcp_diags.resize(m_numStrata);
        for (int s = 0; s < m_numStrata; ++s)
        {
          const double scale = 1.0;  // Set appropriately.
          m_rcp_diags[s].resize(ndofs * getMDA()[s].getLocalNodalSz(), 0.0);
          m_gridOperators[s]->setDiag(m_rcp_diags[s].data(), scale);

          for (auto &a : m_rcp_diags[s])
          {
            a = 1.0f / a;
          }
        }
      }

      virtual ~PoissonGMGMat()
      {
        for (int s = 0; s < m_numStrata; ++s)
          delete m_gridOperators[s];
      }


      void setProblemDimensions(const Point<dim>& pt_min, const Point<dim>& pt_max)
      {
        BaseT::setProblemDimensions(pt_min, pt_max);
        for (int s = 0; s < m_numStrata; ++s)
          m_gridOperators[s]->setProblemDimensions(m_uiPtMin, m_uiPtMax);
      }

      // You need to define leafMatVec() and leafApplySmoother()
      // with the same signatures as gmgMat::matVec and gmgMat::smooth(),
      // in order to create a concrete class (static polymorphism).
      //
      // Do not declare matVec() and smooth(), or else some overloads of
      // gmgMat::matVec() and gmgMat::smooth() could be hidden from name lookup.

      void leafMatVec(const VECType *in, VECType *out, unsigned int stratum = 0, double scale=1.0)
      {
        m_gridOperators[stratum]->matVec(in, out, scale);  // Global matvec.
      }

      void leafApplySmoother(const VECType *res, VECType *resLeft, unsigned int stratum)
      {
        const size_t nNodes = getMDA()[stratum].getLocalNodalSz();
        const VECType * rcp_diag = m_rcp_diags[stratum].data();

        for (int ndIdx = 0; ndIdx < m_ndofs * nNodes; ++ndIdx)
          resLeft[ndIdx] = res[ndIdx] * rcp_diag[ndIdx];
      }

    protected:
      std::vector<PoissonMat<dim> *> m_gridOperators;
      std::vector<VECType> m_tmpRes;
      std::vector<std::vector<VECType>> m_rcp_diags;

      // Convenience protected accessor
      inline ot::MultiDA<dim> & getMDA() { return *m_multiDA; }
  };

}//namespace PoissonEq



// ==============================================================
// main_(): Implementation after parsing, getting dimension, etc.
// ==============================================================
template <unsigned int dim>
int main_ (Parameters &pm, MPI_Comm comm)
{
    const bool outputStatus = true;
    const bool usePetscVersion = false;

    int rProc, nProc;
    MPI_Comm_rank(comm, &rProc);
    MPI_Comm_size(comm, &nProc);

    const unsigned int m_uiDim = dim;

    m_uiMaxDepth = pm.maxDepth;
    const unsigned int nGrids = pm.nGrids;
    const double wavelet_tol = pm.waveletTol;
    const double partition_tol = pm.partitionTol;
    const unsigned int eOrder = pm.eleOrder;

    std::cout << "sizeof(VECType)==" << sizeof(VECType) << "\n";

    if (!(pm.nGrids <= pm.maxDepth + 1 && (1u << (pm.maxDepth+1 - pm.nGrids)) >= eOrder))
    {
      throw "Given maxDepth is too shallow to support given number of grids.";
    }

    double tBegin = 0, tEnd = 10, th = 0.01;

    RefElement refEl(m_uiDim,eOrder);

    // For now must be anisotropic.
    double g_min = 0.0;
    double g_max = 1.0;
    double d_min = -0.5;
    double d_max =  0.5;
    double Rg = g_max - g_min;
    double Rd = d_max - d_min;
    const Point<dim> domain_min(d_min, d_min, d_min);
    const Point<dim> domain_max(d_max, d_max, d_max);

    // sin()
    //
    /// std::function<void(const double *, double*)> f_rhs = [d_min, d_max, g_min, g_max, Rg, Rd](const double *x, double *var)
    /// {
    ///   var[0] = -dim*4*M_PI*M_PI;
    ///   for (unsigned int d = 0; d < dim; d++)
    ///     var[0] *= sin(2*M_PI*(((x[d]-g_min)/Rg)*Rd+d_min));
    /// };

    // 1
    //
    std::function<void(const double *, double*)> f_rhs = [d_min, d_max, g_min, g_max, Rg, Rd](const double *x, double *var)
    {
      var[0] = 1.0;
    };

    std::function<void(const double *, double*)> f_init =[](const double *x, double *var){
        var[0]=0;
    };

    if (!rProc && outputStatus)
      std::cout << "Generating coarseTree.\n" << std::flush;

    std::vector<ot::TreeNode<unsigned int, dim>> coarseTree;
    {
      /// // Create a tree to estimate required depth.
      /// coarseTree = ot::function2BalancedOctree<double, unsigned int, dim>(
      ///       f_rhs, 1, m_uiMaxDepth, wavelet_tol, partition_tol, eOrder, comm);

      /// // Find deepest treeNode.
      /// unsigned int effectiveDepth = 0;
      /// for (const ot::TreeNode<unsigned int, dim> &tn : coarseTree)
      ///   if (effectiveDepth < tn.getLevel())
      ///     effectiveDepth = tn.getLevel();

      unsigned int effectiveDepth = m_uiMaxDepth;
      while ((1u << (m_uiMaxDepth - effectiveDepth)) < eOrder)
        effectiveDepth--;

      unsigned int coarseDepth = effectiveDepth - (nGrids-1);

      /// // Create the actual coarse tree.
      /// coarseTree = ot::function2BalancedOctree<double, unsigned int, dim>(
      ///       f_rhs, 1, coarseDepth, wavelet_tol, partition_tol, eOrder, comm);

      ot::createRegularOctree(coarseTree, coarseDepth, comm);
    }

    if (!rProc && outputStatus)
      std::cout << "Creating grid hierarchy.\n" << std::flush;

    ot::DistTree<unsigned int, dim> dtree(coarseTree);
    ot::DistTree<unsigned int, dim> surrDTree
      = dtree.generateGridHierarchyDown(nGrids, partition_tol, comm);
    ot::MultiDA<dim> multiDA, surrMultiDA;

    if (!rProc && outputStatus)
      std::cout << "Creating multilevel ODA.\n" << std::flush;

    ot::DA<dim>::multiLevelDA(multiDA, dtree, comm, eOrder, 100, partition_tol);
    ot::DA<dim>::multiLevelDA(surrMultiDA, surrDTree, comm, eOrder, 100, partition_tol);

    ot::DA<dim> & fineDA = multiDA[0];
    ot::DA<dim> & coarseDA = multiDA[1];

    if (!rProc && outputStatus)
    {
      std::cout << "Refined DA has " << fineDA.getGlobalNodeSz() << " global nodes.\n" << std::flush;
      std::cout << "Creating poissonGMG wrapper.\n" << std::flush;
    }

    PoissonEq::PoissonGMGMat<dim> poissonGMG(&multiDA, &surrMultiDA, 1);

    if (!rProc && outputStatus)
      std::cout << "Setting up problem.\n" << std::flush;

    if (!usePetscVersion)
    {
      //
      // Non-PETSc version.
      //

      const int smoothStepsPerCycle = 1;
      const double relaxationFactor = 0.67;

      // - - - - - - - - - - -

      std::vector<VECType> _ux, _frhs, _Mfrhs;
      fineDA.createVector(_ux, false, false, 1);
      fineDA.createVector(_frhs, false, false, 1);
      fineDA.createVector(_Mfrhs, false, false, 1);

      PoissonEq::PoissonVec<dim> poissonVec(&fineDA,1);
      poissonVec.setProblemDimensions(domain_min,domain_max);

      fineDA.setVectorByFunction(_ux.data(),    f_init, false, false, 1);
      fineDA.setVectorByFunction(_Mfrhs.data(), f_init, false, false, 1);
      fineDA.setVectorByFunction(_frhs.data(),  f_rhs, false, false, 1);

      if (!rProc && outputStatus)
        std::cout << "Computing RHS.\n" << std::flush;

      poissonVec.computeVec(&(*_frhs.cbegin()), &(*_Mfrhs.begin()), 1.0);
      double normb = 0.0;
      for (VECType b : _Mfrhs)
        normb = fmax(fabs(b), normb);
      if (!rProc)
        std::cout << "normb==" << normb << "\n";

      const VECType *frhs = &(*_frhs.cbegin());
      const VECType *Mfrhs = &(*_Mfrhs.cbegin());
      VECType *ux = &(*_ux.begin());



      // Want to solve the coarse and fine problems independently.
      std::vector<VECType> _ux_2h, _frhs_2h, _Mfrhs_2h;
      fineDA.createVector(_ux_2h, false, false, 1);
      coarseDA.createVector(_frhs_2h, false, false, 1);
      coarseDA.createVector(_Mfrhs_2h, false, false, 1);

      PoissonEq::PoissonVec<dim> poissonVec_2h(&coarseDA,1);
      poissonVec_2h.setProblemDimensions(domain_min,domain_max);

      coarseDA.setVectorByFunction(_ux_2h.data(),    f_init, false, false, 1);
      coarseDA.setVectorByFunction(_Mfrhs_2h.data(), f_init, false, false, 1);
      coarseDA.setVectorByFunction(_frhs_2h.data(),  f_rhs, false, false, 1);

      if (!rProc && outputStatus)
        std::cout << "Computing RHS on coarse grid.\n" << std::flush;

      poissonVec_2h.computeVec(&(*_frhs_2h.cbegin()), &(*_Mfrhs_2h.begin()), 1.0);
      double normb_2h = 0.0;
      for (VECType b : _Mfrhs_2h)
        normb_2h = fmax(fabs(b), normb_2h);
      if (!rProc)
        std::cout << "normb_2h==" << normb_2h << "\n";

      const VECType *frhs_2h = &(*_frhs_2h.cbegin());
      const VECType *Mfrhs_2h = &(*_Mfrhs_2h.cbegin());
      VECType *ux_2h = &(*_ux_2h.begin());




      // - - - - - - - - - - -

      /// double tol=1e-6;
      double reltol=1e-3;
      unsigned int max_iter=30;

      if (!rProc && outputStatus)
      {
        std::cout << "Solving system.\n" << std::flush;
        std::cout << "    Coarse system will be solved first\n";
        std::cout << "    And then the fine system\n";
      }

      double res;

      //
      // Coarse solve
      //
      if (!rProc && outputStatus)
        std::cout << "\nSolving coarse system.\n";

      unsigned int coarseCountIter = 0;
      res = poissonGMG.residual(1, ux_2h, Mfrhs_2h, 1.0);
      if (!rProc && !(coarseCountIter & 3)) // Every 4th iteration
        std::cout << "After Jacobi iteration " << coarseCountIter
                  << ", residual == " << std::scientific << res << "\n";

      /// MatlabDataRoot coarseResRoot;
      for (coarseCountIter = 0; coarseCountIter < 100; coarseCountIter++)
      {
        DBG_COUNT = coarseCountIter;
        poissonGMG.smooth(1, ux_2h, Mfrhs_2h, 1, 1.0);

        res = poissonGMG.residual(1, ux_2h, Mfrhs_2h, 1.0);
        if (!rProc && !(coarseCountIter & 3)) // Every 4th iteration
          std::cout << "After Jacobi iteration " << coarseCountIter
                    << ", residual == " << std::scientific << res << "\n";
      }
      /// coarseResRoot.finalize("_output/", "coarseRes_root");
      if (!rProc)
        std::cout << "Final relative residual error == " << res / normb_2h << ".\n";


      //
      // Fine solve
      //
      if (!rProc && outputStatus)
        std::cout << "\nSolving fine system.\n";

      unsigned int fineCountIter = 0;
      res = poissonGMG.residual(0, ux_2h, Mfrhs_2h, 1.0);
      if (!rProc && !(fineCountIter & 3)) // Every 4th iteration
        std::cout << "After Jacobi iteration " << fineCountIter
                  << ", residual == " << std::scientific << res << "\n";

      /// MatlabDataRoot fineResRoot;
      for (fineCountIter = 0; fineCountIter < 400; fineCountIter++)
      {
        DBG_COUNT = fineCountIter;
        poissonGMG.smooth(0, ux, Mfrhs, 1, 1.0);

        res = poissonGMG.residual(0, ux, Mfrhs, 1.0);
        if (!rProc && !(fineCountIter & 3)) // Every 4th iteration
          std::cout << "After Jacobi iteration " << fineCountIter
                    << ", residual == " << std::scientific << res << "\n";
      }
      /// fineResRoot.finalize("_output/", "fineRes_root");
      if (!rProc)
        std::cout << "Final relative residual error == " << res / normb << ".\n";
    }

    else
    {

      throw "Petsc version requested but ignored!";
    }

    if(!rProc)
        std::cout<<" end of poissonGMG: "<<std::endl;

    return 0;
}
// ==============================================================


//
// get_args()
//
bool get_args(int argc, char * argv[], Parameters &pm, MPI_Comm comm)
{
  int rProc, nProc;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  // ========================
  // Set up accepted options.
  // ========================
  enum CmdOptions                           { progName, opDim, opMaxDepth, opNGrids, opWaveletTol, opPartitionTol, opEleOrder, NUM_CMD_OPTIONS };
  const char *cmdOptions[NUM_CMD_OPTIONS] = { argv[0], "dim", "maxDepth", "nGrids",  "waveletTol", "partitionTol", "eleOrder", };
  const unsigned int firstOptional = NUM_CMD_OPTIONS;  // All required.
  // ========================

  // Fill argVals.
  std::array<const char *, NUM_CMD_OPTIONS> argVals;
  argVals.fill("");
  for (unsigned int op = 0; op < argc; op++)
    argVals[op] = argv[op];

  // Check if we have the required arguments.
  if (argc < firstOptional)
  {
    if (!rProc)
    {
      std::cerr << "Usage: ";
      unsigned int op = 0;
      for (; op < firstOptional; op++)
        std::cerr << cmdOptions[op] << " ";
      for (; op < NUM_CMD_OPTIONS; op++)
        std::cerr << "[" << cmdOptions[op] << "] ";
      std::cerr << "\n";
    }
    return false;
  }

  // ================
  // Parse arguments.
  // ================
  pm.dim      = static_cast<unsigned int>(strtoul(argVals[opDim], NULL, 0));
  pm.maxDepth = static_cast<unsigned int>(strtoul(argVals[opMaxDepth], NULL, 0));
  pm.nGrids   = static_cast<unsigned int>(strtoul(argVals[opNGrids], NULL, 0));
  pm.eleOrder = static_cast<unsigned int>(strtoul(argVals[opEleOrder], NULL, 0));
  pm.waveletTol   = strtod(argVals[opWaveletTol], NULL);
  pm.partitionTol = strtod(argVals[opPartitionTol], NULL);
  // ================

  // Replay arguments.
  constexpr bool replayArguments = true;
  if (replayArguments && !rProc)
  {
    for (unsigned int op = 1; op < NUM_CMD_OPTIONS; op++)
      std::cout << YLW << cmdOptions[op] << "==" << argVals[op] << NRM << " \n";
    std::cout << "\n";
  }

  return true;
}


//
// main()
//
int main(int argc, char * argv[])
{
#ifndef BUILD_WITH_PETSC
  MPI_Init(&argc, &argv);
#else
  PetscInitialize(&argc, &argv, NULL, NULL);
#endif

  int rProc, nProc;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rProc);
  MPI_Comm_size(comm, &nProc);

  int returnCode = 1;

  Parameters pm;
  const unsigned int &dim = pm.dim;
  if (get_args(argc, argv, pm, comm))
  {
    int synchronize;
    MPI_Bcast(&synchronize, 1, MPI_INT, 0, comm);

    _InitializeHcurve(dim);

    // Convert dimension argument to template parameter.
    switch(dim)
    {
      case 2: returnCode = main_<2>(pm, comm); break;
      case 3: returnCode = main_<3>(pm, comm); break;
      case 4: returnCode = main_<4>(pm, comm); break;
      default:
        if (!rProc)
          std::cerr << "Dimension " << dim << " not currently supported.\n";
    }

    _DestroyHcurve();
  }

#ifndef BUILD_WITH_PETSC
  MPI_Finalize();
#else
  PetscFinalize();
#endif

  return returnCode;
}
