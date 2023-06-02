#ifndef DENDRO_KT_NODAL_DATA
#define DENDRO_KT_NODAL_DATA

/// #include "include/sfcTreeLoop_matvec_io.h"
#include "include/treeNode.h"
#include "include/tnUtils.h"
#include "include/oda.h"

namespace io
{

  // binary_ghosted_nodal_vectors()
  //
  //   Writes raw binary data in this format to "out" ostream:
  //     x0 x1 [x2 ...] f0 [f1 ...]
  //
  //   Returns a string specifying a record format for gnuplot binary file, eg:
  //     "%3double%double"
  //
  //   If writing on multiple MPI processes, remember to write to /scratch.
  //

  template <unsigned dim>
  inline
  std::string write_local_nodal_vectors(
      const ot::DA<dim> &da,
      const std::vector<double> &vec,
      int ndofs,
      std::ostream &out)
  {
    //future: support multiple fields with multiple dofs per field
    //fornow: arbitrary ndofs but single field of type "double".
    using X = double;  // future: arbitrary data type

    std::string format_str = "%" + std::to_string(dim) + "double";
    //future: lookup name of X with function or variable template.
    format_str += "%" + std::to_string(ndofs) + "double";

    const size_t count_nodes = da.getLocalNodalSz();
    const size_t begin_nodes = da.getLocalNodeBegin();

    const size_t count_values = count_nodes * ndofs;
    const size_t begin_values = begin_nodes * ndofs;

    const ot::TreeNode<uint32_t, dim> *ghosted_tn_coords = da.getTNCoords();
    const ot::TreeNode<uint32_t, dim> *begin_tn_coords = ghosted_tn_coords + begin_nodes;
    const ot::TreeNode<uint32_t, dim> *end_tn_coords = begin_tn_coords + count_nodes;
    const int degree = da.getElementOrder();

    //future: add scaling and displacement depending on user domain.
    //fornow: assume the domain spans the unit cube from 0.0 to 1.0.
    const auto float_coords = [=](const ot::TreeNode<uint32_t, dim> &tn)
    {
      std::array<double, dim> coords;
      ot::treeNode2Physical(tn, degree, coords.data());
      return coords;
    };

    for (size_t i = 0; i < count_nodes; ++i)
    {
      const std::array<double, dim> coords = float_coords(begin_tn_coords[i]);
      const double *values = &vec[i * ndofs];
      out.write((const char*) &coords, sizeof(coords));
      out.write((const char*) values, sizeof(*values) * ndofs);
    }

    return format_str;

    // Only use MatvecBaseIn loop if either if cell context is needed,
    // or interpolated hanging nodes are needed.
    //
    /// ot::MatvecBaseIn<dim, X>(
    ///       da->getTotalNodalSz(), ndofs, da->getElementOrder(),
    ///       extra_depth > 0, extra_depth,
    ///       da->getTNCoords(), ghosted,
    ///       da->dist_tree()->getTreePartFiltered(da->stratum()).data(),
    ///       da->dist_tree()->getTreePartFiltered(da->stratum()).size(),
    ///       *da->getTreePartFront(),
    ///       *da->getTreePartBack());
  }


  template <unsigned dim>
  inline
  void dump_nodal_data(
      const ot::DA<dim> &da,
      const std::vector<double> &vec,
      int ndofs,
      const std::string &fileprefix)
  {
    const int world_size = par::mpi_comm_size(da.getGlobalComm());
    const int world_rank = par::mpi_comm_rank(da.getGlobalComm());

    //future (maybe): mpi gather the active ranks

    // Dump binary data.
    std::ofstream rank_file(fileprefix + ".part" + std::to_string(world_rank));
    const std::string format_str = write_local_nodal_vectors(
        da, vec, ndofs, rank_file);
    rank_file.close();

    // Metadata file references other files.
    if (world_rank == 0)
    {
      const std::string meta_file_name = fileprefix + ".meta";
      std::ofstream meta_file(meta_file_name);

      meta_file << "prefix=" << std::quoted(fileprefix + ".part") << "\n";
      meta_file << "title=" << std::quoted(fileprefix) << "\n";
      meta_file << "dim=" << dim << "\n";
      meta_file << "nfields=" << ndofs << "\n";
      meta_file << "field_default=1" << "\n";
      meta_file << "comm_size=" << world_size << "\n";
      meta_file.close();

      std::cerr << "Dumped nodal vector, use `env SRC_DIR='..' ../IO/gnuplot/raw_nodal_binary.sh "
                << std::quoted(meta_file_name) << " [optional field index]`"
                << "`\n";
    }
  }

}

#endif//DENDRO_KT_NODAL_DATA
