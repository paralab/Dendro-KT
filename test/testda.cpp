#ifdef BUILD_WITH_PETSC

#include "petsc.h"

#endif

#include <iostream>
#include <distTree.h>
#include <oda.h>
#include <point.h>
#include <sfcTreeLoop_matvec_io.h>
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

static const auto DomainDecider = ot::DistTree<unsigned, 3>::BoxDecider(domain);


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
        ot::quadTreeToGnuplot( treePart, finest_level, "output", comm);
        // ot::quadTreeToGnuplot(sorted, finest_level, "_output/case_" + std::to_string(subcase_id_hack) + "_sorted", comm);
        ++subcase_id_hack;

        std::cout << distTree.getTreePartFiltered().size() << "\n";
        ot::DA<DIM> *newDA1 = new ot::DA<DIM>(distTree, 0,comm, eleOrder, 100, 0.3,1); //DistTree overload
        ot::DA<DIM> *newDA2 = new ot::DA<DIM>(distTree, 0,comm, eleOrder, 100, 0.3,0);
        std::cout << "NewDA1 = " << newDA1->getGlobalNodeSz() << "\n";
        std::cout << "NewDA2 = " << newDA2->getGlobalNodeSz() << "\n";
        // std::cout << "NewDA1 = " << newDA1->getLocalNodalSz() << "\n";
        // std::cout << "NewDA2 = " << newDA2->getLocalNodalSz() << "\n";

        // std::cout << "NewDA1 = " << newDA1->getTotalNodalSz() << "\n";
        // std::cout << "NewDA1 = " << newDA1->getLocalNodalSz() << "\n";
        // std::cout << "NewDA2 = " << newDA2->getTotalNodalSz() << "\n";
        // std::cout << "NewDA2 = " << newDA2->getLocalNodalSz() << "\n";

        auto stringifyPair = [](const std::pair<double, double>& p, std::string sep = "-")-> std::string{
            return std::to_string(p.first) + sep + std::to_string(p.second);
        };

        auto stringifyVECTOR = [](const std::vector<double>& p, std::string sep = "-")-> std::string{

            std::string compstring;
            int idx{0};

            std::stringstream precisionValue;
            precisionValue.precision(4);

            for( ; idx < p.size() - 1; idx++ ) {

                precisionValue << p[idx] << sep;

            }

            precisionValue << p[idx];

            return precisionValue.str();
        };

        auto tnCoords1 = newDA1->getTNCoords();
        auto tnCoords2 = newDA2->getTNCoords();

        std::vector<double> physcoords( DIM, 0 );

        std::unordered_set<std::string> coord1str;
        std::unordered_set<std::string> coord2str;

        std::unordered_map<std::string, int> coord1map;
        std::unordered_map<std::string, int> coord2map;

        int idx = 0;

        // std::cout << "New DA Coords \n";

        for( idx = 0; idx < newDA1->getLocalNodalSz(); idx++ ) {

            ot::treeNode2Physical( *tnCoords1, eleOrder + 1, &( *physcoords.begin() ) );

            // std::cout << std::to_string( physcoords[0] ) << "\t" << std::to_string( physcoords[1] ) << "\t" << std::to_string( physcoords[2] ) << "\n";

            auto key = stringifyVECTOR( physcoords );

            coord1str.insert( key );

            coord1map[key] += 1;

            tnCoords1++;

        }   
        // std::cout << "New DA Coords End \n";

        // std::cout << "Old DA Coords \n";

        for( idx = 0; idx < newDA2->getLocalNodalSz(); idx++ ) {

            ot::treeNode2Physical( *tnCoords2, eleOrder, &( *physcoords.begin() ) );

            auto key = stringifyVECTOR( physcoords );

            // std::cout << std::to_string( physcoords[0] ) << "\t" << std::to_string( physcoords[1] ) << "\t" << std::to_string( physcoords[2] ) << "\n";

            // if( key == "0.062500-0.062500" ) {

            //     ot::treeNode2Physical( *tnCoords1, eleOrder + 1, &( *physcoords.begin() ) );
            //     key = stringify( std::make_pair( physcoords[0], physcoords[1] ) );

            //     std::cout << key << "\n";

            // }

            coord2str.insert( key );

            coord2map[key] += 1;
            
            tnCoords2++;

        }

        // std::cout << "Old DA Coords End \n";

        for( auto& strval: coord1str ) {

            if( coord2str.find( strval ) == coord2str.end() ) {
                std::cout << strval << "\n";
            }

        }

        std::cout << "end" << "\n";

        for( auto& strval: coord2str ) {

            if( coord1str.find( strval ) == coord1str.end() ) {
                std::cout << strval << "\n";
            }

        }

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