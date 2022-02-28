/**
 * @brief: Simple benchmark program to time 
 * treesort octree construction and balancing with cg node identification and 
 * building the scatter map. 
 * @date: 03/28/1019
 * @author: Milinda Fernando, School of Computing University of Utah. 
 * 
*/

#include "tsort_bench.h"
#include "distTree.h"
#include "oda.h"

namespace bench
{
    profiler_t t_sort;
    profiler_t t_con;
    profiler_t t_bal;
    profiler_t t_dtree;
    profiler_t t_da;


    void resetAllTimers()
    {
        t_sort.clear();
        t_con.clear();
        t_bal.clear();
        t_dtree.clear();
        t_da.clear();
    }

    void bench_kernel(unsigned int numPts, unsigned int numIter,unsigned int pOrder, MPI_Comm comm)
    {

        resetAllTimers();
        const unsigned int dim = 4;
        using T = unsigned int;
        using TreeNode = ot::TreeNode<T,dim>;
        using TNP = ot::TNPoint<T,dim>;
        using ScatterMap = ot::ScatterMap;
        using RecvMap = ot::GatherMap;
        const unsigned int maxPtsPerRegion = 1;
        const T leafLevel = m_uiMaxDepth;
        const unsigned int polyOrder = pOrder;

        const double loadFlexibility = 0.2;
       
        // warmpu run 
        std::vector<TreeNode> points = ot::getPts<T,dim>(numPts);
        ot::SFC_Tree<T,dim>::distTreeSort(points, loadFlexibility , comm);
       
        for(unsigned int i=0;i<numIter;i++)
        {
         
            std::vector<TreeNode> tree;
            TreeNode treeSplitterF, treeSplitterB;
            std::vector<TNP> nodeListExterior;
            /// std::vector<TNP> nodeListInterior;
            unsigned int numCGNodes;
            ScatterMap sm;
            RecvMap rm;

            // Time sorting.
            t_sort.start();    
            ot::SFC_Tree<T,dim>::distTreeSort(points, loadFlexibility , comm);
            t_sort.stop();
            
            // Time construction.
            t_con.start();    
            ot::SFC_Tree<T,dim>::distTreeConstruction(points, tree, maxPtsPerRegion, loadFlexibility, comm);
            t_con.stop();
            tree.clear();

            // Time balanced construction.
            t_bal.start();
            ot::SFC_Tree<T,dim>::distTreeBalancing(points, tree, maxPtsPerRegion, loadFlexibility, comm);
            t_bal.stop();

            // Time DistTree construction from already balanced tree.
            t_dtree.start();
            ot::DistTree<T,dim> dtree(tree, comm);
            t_dtree.stop();

            // Time DA construction
            t_da.start();
            ot::DA<dim> da(dtree, comm, polyOrder, 100, loadFlexibility);
            t_da.stop();
        }


    }


    void dump_profile_info(std::ostream& fout, profiler_t* timers, const char * const * names, unsigned int n ,MPI_Comm comm)
    {

        double stat;
        double stat_g[3*n];

        int rank, npes;
        MPI_Comm_rank(comm,&rank);
        MPI_Comm_size(comm,&npes);


        for(unsigned int i=0; i<n; i++)
        {
           stat=(timers[i].seconds) / timers[i].num_calls ;     
           
           par::Mpi_Reduce(&stat,stat_g + 3*i + 0 ,1, MPI_MIN,0,comm);
           par::Mpi_Reduce(&stat,stat_g + 3*i + 1 ,1, MPI_SUM,0,comm);
           par::Mpi_Reduce(&stat,stat_g + 3*i + 2 ,1, MPI_MAX,0,comm);

           stat_g[ 3*i + 1] = stat_g[ 3*i + 1]/(double)npes;
 
        }

        if(!rank)
        {
            for(unsigned int i=0; i<n; i++)
            {
               fout<<names[i]<<"(min)\t"<<names[i]<<"(mean)\t"<<names[i]<<"(max)\t";
            }

        }

        if(!rank)
            fout<<std::endl;

        if(!rank)
        {
            for(unsigned int i=0; i<n; i++)
            {
               fout<<stat_g[3*i + 0]<<"\t"<<stat_g[3*i + 1]<<"\t"<<stat_g[3*i+2]<<"\t";
            }
        }

        if(!rank)
            fout<<std::endl;

    }


}// end of namespace of bench



int main(int argc, char** argv)
{

    MPI_Init(&argc,&argv);
    DendroScopeBegin();

    int rank,npes;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&npes);
    
    if(argc<4)
    {
        if(!rank)
            std::cout<<"usage :  "<<argv[0]<<" pts_per_core(weak scaling) maxdepth order iter"<<std::endl;
        
        MPI_Abort(comm,0);
    }

    const unsigned int pts_per_core = atoi(argv[1]);
    m_uiMaxDepth = atoi(argv[2]);
    unsigned int pOrder =atoi(argv[3]);
    unsigned int mIter = atoi(argv[4]);

    const unsigned int dim =4;

    _InitializeHcurve(dim);

    bench::bench_kernel(pts_per_core,mIter,pOrder,comm);
    profiler_t counters []={bench::t_sort, bench::t_con, bench::t_bal, bench::t_dtree, bench::t_da};
    const char * counter_names[] ={"sort","cons","bal","dtree","da"};
    bench::dump_profile_info(std::cout,counters,counter_names,5,comm);

    _DestroyHcurve();
    DendroScopeEnd();
    MPI_Finalize();
    return 0;
}
