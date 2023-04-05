#ifdef BUILD_WITH_PETSC

#include "petsc.h"

#endif

#include <iostream>
#include <distTree.h>
#include <oda.h>
#include <point.h>
#include <sfcTreeLoop_matvec_io.h>
#include <sfcTreeLoop_matvec.h>
#include <octUtils.h>
#include <filterFunction.h>
#include <tnUtils.h>
#include <unordered_set>
#include <unordered_map>

constexpr unsigned int DIM = 3;
constexpr unsigned int nchild = 1u << DIM;
static double xDomainExtent;
static constexpr std::array<double, 3> domain = {1.0, 1.0, 1.0};

struct pair_hash {
    inline std::size_t operator()(const std::pair<double, double> & v) const {
        return v.first*31+v.second;
    }
};
typedef ot::TreeNode<unsigned int, DIM> TREENODE;

static const auto DomainDecider = ot::DistTree<unsigned, 3>::BoxDecider(domain);

void doLoop(ot::DA<DIM>* octDA, const std::vector<TREENODE> &treePart){
    const int ndofs = 1;
    unsigned int eleOrder = octDA->getElementOrder();
    const size_t sz = octDA->getTotalNodalSz();
    auto partFront = octDA->getTreePartFront();
    auto partBack = octDA->getTreePartBack();
    const auto tnCoords = octDA->getTNCoords();
    int elemID=0;

    std::vector<double> physcoords( DIM, 0 );
    std::vector<double> inputvals( sz, 1 );

    for( int idx = 0; idx < sz; idx++ ) {
        ot::treeNode2Physical( *( tnCoords + idx ), eleOrder + 1, &( *physcoords.begin() ) );
        inputvals.push_back( physcoords[0]*physcoords[1] );
    }

    std::vector<double> outputvals( sz, 0 );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    ot::MatvecBase<DIM, double, false> loop(sz, 1, eleOrder, tnCoords, &( *inputvals.begin() ), &(*treePart.cbegin()),treePart.size(),*partFront,*partBack);
    while(!loop.isFinished()){
        if (loop.isPre() && loop.subtreeInfo().isLeaf()) {
            
            const double *nodeCoordsFlat = loop.subtreeInfo().getNodeCoords();
            const ot::TreeNode<unsigned int, DIM>* nodeCoords = loop.subtreeInfo().readNodeCoordsIn();
            const double * nodeValsFlat = loop.subtreeInfo().readNodeValsIn();

            int  numNodes = loop.subtreeInfo().getNumNodesIn();
            auto m_bits = loop.subtreeInfo().getLeafBitsetInfo();
            auto currTree = loop.subtreeInfo().getCurrentSubtree();
            const unsigned int npe = intPow(eleOrder+1, DIM);

            std::vector<double> leafResult(ndofs*numNodes, 0.0);
            
            for( int idx = 0; idx < numNodes; idx++ ) {

                const unsigned int nodeRank = ot::TNPoint<unsigned int, DIM>::get_lexNodeRank( currTree,
                                                        nodeCoords[idx],
                                                        eleOrder + 1 );

                if( nodeRank < numNodes && nodeRank >= 0 ) {

                    std::copy_n( &nodeValsFlat[ndofs * nodeRank], ndofs, &leafResult[ndofs * nodeRank] );

                }

            }
            loop.subtreeInfo().overwriteNodeValsOut( &(*leafResult.begin()) );            
            elemID++;
            loop.next(1);

        }
        else{
            loop.step(1);
        }
    }

    size_t writtenSz = loop.finalize( &( *outputvals.begin() ) );

    bool result{true};

    for( int idx = 0; idx < outputvals.size(); idx++ ) {
        result = result && ( bool( inputvals[idx] == outputvals[idx] ) );
    }

    if( result == 1 )
        std::cout << "SUCCESS" << "\n";
    else
        std::cout << "FAILURE \n";

    if (sz > 0 && writtenSz == 0)
        std::cerr << "Warning: matvec() did not write any data! Loop misconfigured?\n";
}

/**
 * main()
 */
int main(int argc, char *argv[]) {
    typedef unsigned int DENDRITE_UINT;
    PetscInitialize(&argc, &argv, NULL, NULL);
    DendroScopeBegin() ;

        _InitializeHcurve(DIM);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        m_uiMaxDepth = 10;
        xDomainExtent = 1;
        const DENDRITE_UINT eleOrder = 1;
        if (argc < 2) {
            if (not(rank)) {
                std::cout << "Usage: level \n";
            }
            exit(EXIT_FAILURE);
        }
        const DENDRITE_UINT level = static_cast<DENDRITE_UINT>(std::atoi(argv[1]));
        std::cout << level << " " << xDomainExtent << "\n";

        MPI_Comm comm = MPI_COMM_WORLD;
        constexpr bool printTreeOn = false;  // Can print the contents of the tree vectors.
        unsigned int extents[] = {1, 1, 1};
        std::array<unsigned int, DIM> a;
        for (int d = 0; d < DIM; ++d)
            a[d] = extents[d];
        using DTree = ot::DistTree<unsigned int, DIM>;
        DTree distTree = DTree::constructSubdomainDistTree(level, DomainDecider ,
                                                           comm);
        ot::DA<DIM> *octDA = new ot::DA<DIM>(distTree, comm, eleOrder);
        // printMaxCoords(*octDA, distTree.getTreePartFiltered());
        size_t oldTreeSize = 0;
        size_t refinedTreeSize = 0;
        // Access the original tree as a list of tree nodes.
        {
            const std::vector<ot::TreeNode<unsigned int, DIM>> &treePart = distTree.getTreePartFiltered();
            oldTreeSize = treePart.size();
            /// if (printTreeOn)
            ///   printTree(treePart, level+1);
        }
        std::cout << "Old Tree \n";
        std::cout << "Num elements: " << oldTreeSize << "\n";

        for (int i = 0; i < 1; i++) {
            std::vector<ot::OCT_FLAGS::Refine> refineFlags(octDA->getLocalElementSz(),
                                                           ot::OCT_FLAGS::Refine::OCT_NO_CHANGE);
            refineFlags[0] = ot::OCT_FLAGS::Refine::OCT_REFINE;

            // distRemeshSubdomain()
            ot::DistTree<unsigned int, DIM> newDistTree, surrDistTree;
            ot::DistTree<unsigned int, DIM>::distRemeshSubdomain(distTree, refineFlags, newDistTree, surrDistTree,
                                                                 ot::RemeshPartition::SurrogateInByOut, 0.3);

            ot::DA<DIM> *newDA = new ot::DA<DIM>(newDistTree, comm, eleOrder + 1, 100, 0.3); //DistTree overload

            std::swap(octDA, newDA);
            delete newDA;
            std::swap(distTree, newDistTree);

        }

        static int subcase_id_hack = 0;
        int finest_level = 5;
        const std::vector<ot::TreeNode<unsigned int, DIM>> &treePart = distTree.getTreePartFiltered();

        
        ++subcase_id_hack;

        std::cout << distTree.getTreePartFiltered().size() << "\n";
        ot::quadTreeToGnuplot( treePart, level + 1, "output", comm);

        ot::DA<DIM> *newDA1 = new ot::DA<DIM>(distTree, 0,comm, eleOrder, 100, 0.3,1); //DistTree overload
        ot::DA<DIM> *newDA2 = new ot::DA<DIM>(distTree, 0,comm, eleOrder, 100, 0.3,0);
        // std::cout << "NewDA1 = " << newDA1->getGlobalNodeSz() << "\n";
        // std::cout << "NewDA2 = " << newDA2->getGlobalNodeSz() << "\n";
        std::cout << "NewDA1 = " << newDA1->getLocalNodalSz() << "\n";
        // std::cout << "NewDA2 = " << newDA2->getLocalNodalSz() << "\n";
        // std::cout << "NewDA2 = " << newDA2->getTotalNodalSz() << "\n";
        // std::cout << "NewDA1 = " << newDA1->getTotalNodalSz() << "\n";

        doLoop(newDA1,distTree.getTreePartFiltered());
        auto stringify = [](const std::pair<double, double>& p, std::string sep = "-")-> std::string{
            return std::to_string(p.first) + sep + std::to_string(p.second);
        };

        auto tnCoords2 = newDA2->getTNCoords();

        std::vector<double> physcoords( DIM, 0 );

        std::unordered_set<std::string> coord1str;
        std::unordered_set<std::string> coord2str;

        std::unordered_map<std::string, int> coord1map;
        std::unordered_map<std::string, int> coord2map;

        int idx = 0;
        auto tnCoords1 = newDA1->getTNCoords() + newDA1->getLocalNodeBegin();

        std::ofstream fval( "coords" + std::to_string( rank ) + ".txt" );

        for( idx = 0; idx < newDA1->getLocalNodalSz(); idx++ ) {

             ot::treeNode2Physical( *tnCoords1, eleOrder+1, &( *physcoords.begin() ) );

             fval << std::to_string( physcoords[0] ) << "," << std::to_string( physcoords[1] ) << "," <<  std::to_string( physcoords[2] ) << "\n";

//            auto key = stringify( std::make_pair( physcoords[0], physcoords[1] ) );
//
//            coord2str.insert( key );
//
//            coord2map[key] += 1;

            tnCoords1++;

        }

        fval.close();

        // for( idx = 0 ; idx < newDA1->getLocalNodalSz(); idx++ ) {

        //     ot::treeNode2Physical( *tnCoords1, eleOrder, &( *physcoords.begin() ) );

        //     auto key = stringify( std::make_pair( physcoords[0], physcoords[1] ) );

        //     // if( key == "0.062500-0.062500" ) {

        //     //     ot::treeNode2Physical( *tnCoords1, eleOrder + 1, &( *physcoords.begin() ) );
        //     //     key = stringify( std::make_pair( physcoords[0], physcoords[1] ) );

        //     //     std::cout << key << "\n";

        //     // }

        //     coord1str.insert( stringify( std::make_pair( physcoords[0], physcoords[1] ) ) );

        //     coord1map[key] += 1;

        //     tnCoords1++;

        // }

        // for( auto& strval: coord1str ) {

        //     if( coord2str.find( strval ) == coord2str.end() ) {
        //         std::cout << strval << "\n";
        //     }

        // }

        // std::cout << "end" << "\n";

        // for( auto& strval: coord2str ) {

        //     if( coord1str.find( strval ) == coord1str.end() ) {
        //         std::cout << strval << "\n";
        //     }

        // }

        // for( auto& keyval: coord1map ) {

        //     if( coord1map[keyval.first] > 1 ) {

        //         std::cout << keyval.first << "\n";

        //     }

        // }

        delete octDA;
        delete newDA1;
        delete newDA2;

        _DestroyHcurve();
    DendroScopeEnd();
    PetscFinalize();
}