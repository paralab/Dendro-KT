
#include "comm_hierarchy.h"
#include "dollar.hpp"

namespace par
{
  // KwayComms::KwayComms()
  KwayComms::KwayComms(const MPI_Comm root_comm)
  {
    DOLLAR("KwayComms()");

    const int kway = KWAY;
    MPI_Comm comm = root_comm;
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    m_comm_rank = comm_rank;
    int nblocks = std::min(comm_size, kway);
    KwayBlocks blockmap(comm_size, nblocks);

    m_comms.push_back(comm);
    m_children.push_back(blockmap);

    while (nblocks > 1 and comm_size > kway)
    {
      MPI_Comm new_comm;
      MPI_Comm_split(
          comm,
          blockmap.task_to_block(comm_rank),
          blockmap.task_to_block_id(comm_rank),
          &new_comm);

      comm = new_comm;
      MPI_Comm_size(comm, &comm_size);
      MPI_Comm_rank(comm, &comm_rank);
      nblocks = std::min(comm_size, kway);
      blockmap = KwayBlocks(comm_size, nblocks);

      m_comms.push_back(comm);
      m_children.push_back(blockmap);
    }
  }

  // KwayComms::~KwayComms()
  KwayComms::~KwayComms()
  {
    // Free all but root.
    for (int level = this->levels() - 1; level > 0; --level)
      MPI_Comm_free(&m_comms[level]);
  }

  // KwayComms::attach_once() : Constructs and caches a KwayComms with comm as root.
  const KwayComms & KwayComms::attach_once(MPI_Comm comm)
  {
    // Approach:
    // Construct KwayComms, store pointer in attribute, delete on comm_free.

    // Deleter is needed to create the key.
    const auto deleter = [](
        MPI_Comm comm, int comm_keyval, void *attribute_val, void *extra_state)
    {
      if (attribute_val != nullptr)
        delete (const KwayComms *) attribute_val;
      return MPI_SUCCESS;
    };

    // Key for KwayComms attribute, for all comms.
    static int key = MPI_KEYVAL_INVALID;
    if (key == MPI_KEYVAL_INVALID)
      MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, deleter, &key, (void*){});

    // Construct a KwayComms the first time being called on this comm.
    KwayComms * kway_comms = nullptr;
    int already_constructed = false;
    MPI_Comm_get_attr(comm, key, &kway_comms, &already_constructed);
    if (not already_constructed)
    {
      kway_comms = new KwayComms(comm);
      MPI_Comm_set_attr(comm, key, kway_comms);
    }
    assert(kway_comms != nullptr);
    
    return *kway_comms;
  }


}
