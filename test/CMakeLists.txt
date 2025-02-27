
# ------------------------------------------------------------
# Directory scope commands
# ------------------------------------------------------------

link_libraries(dendroKT)
include_directories(.)


# ------------------------------------------------------------
# Executable targets
# ------------------------------------------------------------

add_executable(tstPartition testPartition.cpp)
target_link_libraries(tstPartition MPI::MPI_CXX m)

add_executable(tstConstruct testConstruct.cpp)
target_link_libraries(tstConstruct MPI::MPI_CXX m)

add_executable(tstTable testTables.cpp)
target_link_libraries(tstTable MPI::MPI_CXX m)

add_executable(tstTreeSort testTreeSort.cpp)
target_link_libraries(tstTreeSort MPI::MPI_CXX m)

add_executable(tstTreeConstruction testTreeConstruction.cpp)
target_link_libraries(tstTreeConstruction MPI::MPI_CXX m)

add_executable(tstTreeBalancing testTreeBalancing.cpp)
target_link_libraries(tstTreeBalancing MPI::MPI_CXX m)

add_executable(tstTreeVtu testTreeVtu.cpp)
target_link_libraries(tstTreeVtu MPI::MPI_CXX m)

add_executable(tstTreeJson testTreeJson.cpp)
target_link_libraries(tstTreeJson MPI::MPI_CXX m)

add_executable(tstHCurveTable testHCurveTable.cpp)
target_link_libraries(tstHCurveTable MPI::MPI_CXX m)

add_executable(tstMovingBall testMovingBall.cpp)
target_link_libraries(tstMovingBall MPI::MPI_CXX m)

add_executable(tstTSearchCmpx testTSearchCmpx.cpp)
target_link_libraries(tstTSearchCmpx MPI::MPI_CXX m)

add_executable(tstODA testODA.cpp)
target_link_libraries(tstODA MPI::MPI_CXX m)

add_executable(tstMatvec testMatvec.cpp)
target_link_libraries(tstMatvec MPI::MPI_CXX m)

add_executable(tstRefel testRefel.cpp)
target_link_libraries(tstRefel MPI::MPI_CXX m)

add_executable(tstFunction2Octree testFunction2Octree.cpp)
target_link_libraries(tstFunction2Octree MPI::MPI_CXX m)

add_executable(tstSFCTreeLoop testSFCTreeLoop.cpp)
target_link_libraries(tstSFCTreeLoop MPI::MPI_CXX m)

add_executable(tstKeepSiblingLeafsTogether testKeepSiblingLeafsTogether.cpp)
target_link_libraries(tstKeepSiblingLeafsTogether MPI::MPI_CXX m)

add_executable(tstCreateRegularOctree testCreateRegularOctree.cpp)
target_link_libraries(tstCreateRegularOctree MPI::MPI_CXX m)

add_executable(tstLocGlobSz testLocGlobSz.cpp)
target_link_libraries(tstLocGlobSz MPI::MPI_CXX m)

add_executable(tstBillionsOfElements testBillionsOfElements.cpp)
target_link_libraries(tstBillionsOfElements MPI::MPI_CXX m)

add_executable(tstMeshLoop testMeshLoop.cpp)
target_link_libraries(tstMeshLoop MPI::MPI_CXX m)

add_executable(tstGenerateGridHierarchy testGenerateGridHierarchy.cpp)
target_link_libraries(tstGenerateGridHierarchy MPI::MPI_CXX m)

add_executable(tstIntergridTransfer testIntergridTransfer.cpp)
target_link_libraries(tstIntergridTransfer MPI::MPI_CXX m)

add_executable(tstIntergridSubda testIntergridSubda.cpp)
target_link_libraries(tstIntergridSubda MPI::MPI_CXX m)

add_executable(tstRestriction testRestriction.cpp)
target_link_libraries(tstRestriction MPI::MPI_CXX m)

add_executable(tstRemesh testRemesh.cpp)
target_link_libraries(tstRemesh MPI::MPI_CXX m)

add_executable(tstCheckpoint testCheckpoint.cpp)
target_link_libraries(tstCheckpoint MPI::MPI_CXX m)

add_executable(tstBoundaryIndices testBoundaryIndices.cpp)
target_link_libraries(tstBoundaryIndices MPI::MPI_CXX m)

add_executable(tstRefineSubda testRefineSubda.cpp)
target_link_libraries(tstRefineSubda MPI::MPI_CXX m)

add_executable(tstTreeNodeOwner testTreeNodeOwner.cpp)
target_link_libraries(tstTreeNodeOwner MPI::MPI_CXX m)

add_executable(tstBoundaryRefined testBoundaryRefined.cpp)
target_link_libraries(tstBoundaryRefined MPI::MPI_CXX m)

add_executable(tstConsistentElements testConsistentElements.cpp)
target_link_libraries(tstConsistentElements MPI::MPI_CXX m)

add_executable(tstProduceHangingSubda testProduceHangingSubda.cpp)
target_link_libraries(tstProduceHangingSubda MPI::MPI_CXX m)

add_executable(tstHangingBoundary testHangingBoundary.cpp)
target_link_libraries(tstHangingBoundary MPI::MPI_CXX m)

add_executable(tstPeriodicSphereStatic testPeriodicSphereStatic.cpp)
target_link_libraries(tstPeriodicSphereStatic MPI::MPI_CXX m)

add_executable(tstPeriodicUniformNodes testPeriodicUniformNodes.cpp)
target_link_libraries(tstPeriodicUniformNodes MPI::MPI_CXX m)


add_library(tstOctreeHeaders INTERFACE octree/multisphere.h octree/gaussian.hpp)
target_include_directories(tstOctreeHeaders INTERFACE octree)

add_executable(tstIncidentGhostPoints testIncidentGhostPoints.cpp)
target_sources(tstIncidentGhostPoints PRIVATE
  ../bench/src/genChannelPoints.cpp
  ../bench/include/genChannelPoints.h)
target_include_directories(tstIncidentGhostPoints PRIVATE ../bench/include)
target_link_libraries(tstIncidentGhostPoints MPI::MPI_CXX m)

add_executable(tstOctreeNonoverlapping octree/nonoverlapping.cpp)
target_link_libraries(tstOctreeNonoverlapping tstOctreeHeaders MPI::MPI_CXX m)

add_executable(tstOctreeRemesh octree/remesh.cpp)
target_link_libraries(tstOctreeRemesh tstOctreeHeaders MPI::MPI_CXX m)

add_executable(tstIntergridRefineMany intergrid/refine_many.cpp)
target_link_libraries(tstIntergridRefineMany tstOctreeHeaders MPI::MPI_CXX m)

add_executable(tstFemCollectMat fem/collect_mat.cpp)
target_link_libraries(tstFemCollectMat tstOctreeHeaders MPI::MPI_CXX m)

add_executable(tstOctreeDistPartition octree/dist_partition.cpp)
target_link_libraries(tstOctreeDistPartition tstOctreeHeaders MPI::MPI_CXX m)

add_executable(tstParNbx par/nbx.cpp)
target_link_libraries(tstParNbx MPI::MPI_CXX m)

add_executable(tstParSendall par/sendall.cpp)
target_link_libraries(tstParSendall MPI::MPI_CXX m)

add_executable(tstParShift2 par/shift2.cpp)
target_link_libraries(tstParShift2 MPI::MPI_CXX m)

add_executable(tstDollar testDollar.cpp)
target_link_libraries(tstDollar MPI::MPI_CXX m)
