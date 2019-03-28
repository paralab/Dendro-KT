/**
 * @brief: Simple benchmark program to time 
 * treesort octree construction and balancing with cg node identification and 
 * building the scatter map. 
 * @date: 03/28/1019
 * @author: Milinda Fernando, School of Computing University of Utah. 
 * 
*/

#include "tsort_bench.h"

namespace bench
{
    profiler_t t_sort;
    profiler_t t_con;
    profiler_t t_bal;
    profiler_t t_cg;
    profiler_t t_sm;


    void resetAllTimers()
    {
        t_sort.clear();
        t_con.clear();
        t_bal.clear();
        t_cg.clear();
        t_sm.clear();
    }

    void bench_kernel(unsigned int numPts, unsigned int numIter, MPI_Comm comm)
    {

        resetAllTimers();
        const unsigned int dim = 4;
        using T = unsigned int;
        using TreeNode = ot::TreeNode<T,dim>;
        const unsigned int maxPtsPerRegion = 1;
        const T leafLevel = m_uiMaxDepth;

        const double loadFlexibility = 0.2;
       
        // warmpu run 
        std::vector<TreeNode> points = ot::getPts<T,dim>(numPts);
        ot::SFC_Tree<T,dim>::distTreeSort(points, loadFlexibility , comm);
       
        for(unsigned int i=0;i<numIter;i++)
        {
         
            std::vector<TreeNode> tree;
            
            t_sort.start();    
            ot::SFC_Tree<T,dim>::distTreeSort(points, loadFlexibility , comm);
            t_sort.stop();


            
            t_con.start();    
            ot::SFC_Tree<T,dim>::distTreeConstruction(points, tree, maxPtsPerRegion, loadFlexibility, comm);
            t_con.stop();
            tree.clear();

            t_bal.start();
            ot::SFC_Tree<T,dim>::distTreeBalancing(points, tree, maxPtsPerRegion, loadFlexibility, comm);
            t_bal.stop();
            tree.clear();


            // add cg point compute and scatter map compute. 


        }
        




    }

}// end of namespace of bench







int main(int argc, char ** argv)
{


    MPI_Init(&argc,&argv);

    int rank,npes;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&npes);
    
    if(argc<1)
    {
        if(!rank)
        {
            std::cout<<"usage :  "<<argv[0]<<" pts_per_core(weak scaling) maxdepth dim"<<std::endl;
        }

        MPI_Abort(comm,0);
    }

    const unsigned int pts_per_core = atoi(argv[1]);
    m_uiMaxDepth = atoi(argv[2]);
    const unsigned int dim = atoi(argv[3]);

    _InitializeHcurve(dim);


    _DestroyHcurve();
    MPI_Finalize();
    return 0;
}
