
#ifndef DENDRO_KT_COMM_HIERARCHY_H
#define DENDRO_KT_COMM_HIERARCHY_H

#include "mpi.h"
#include <assert.h>
#include <vector>

namespace par
{
  // KwayBlocks
  class KwayBlocks
  {
    private:
      int m_comm_size;
      int m_nblocks;

    public:
      using Task = int;
      using Blk = int;

      KwayBlocks() = delete;
      KwayBlocks(int comm_size, int nblocks)
        : m_comm_size(comm_size), m_nblocks(nblocks) {}

      // comm_size()
      int comm_size() const { return m_comm_size; }

      // nblocks()
      Blk nblocks() const { return m_nblocks; }

      // block_to_task()
      Task block_to_task(Blk blk) const {
        return blk * comm_size() / nblocks();
      };

      // block_tasks()
      Task block_tasks(Blk blk) const {
        return block_to_task(blk+1) - block_to_task(blk);
      };

      // blk_id_to_task()
      Task blk_id_to_task(Blk blk, int blk_id) const {
        return block_to_task(blk) + blk_id;
      };

      // task_to_block()
      Blk task_to_block(Task task) const {
        const int blk = ((task + 1) * nblocks() - 1) / comm_size();
        assert(block_to_task(blk) <= task and task < block_to_task(blk+1));
        return blk;
      };

      // task_to_block_id()
      int task_to_block_id(Task task) const {
        return task - block_to_task(task_to_block(task));
      }

      // task_to_next_block()
      Blk task_to_next_block(Task task) const {
        return (task * nblocks() + comm_size() - 1) / comm_size();
      };
  };


  // KwayComms : creates and frees levels >= 1
  class KwayComms
  {
    private:
      std::vector<MPI_Comm> m_comms;
      std::vector<KwayBlocks> m_children;
      int m_comm_rank = 0;

      KwayComms() = delete;
      KwayComms(const KwayComms &) = delete;
      KwayComms(KwayComms &&) = delete;
      KwayComms & operator=(const KwayComms &) = delete;
      KwayComms & operator=(KwayComms &&) = delete;

      KwayComms(const MPI_Comm root_comm);
      ~KwayComms();

    public:
      static const KwayComms & attach_once(MPI_Comm comm);

      int levels() const                    { return m_comms.size(); }
      MPI_Comm comm(int level) const        { return m_comms[level]; }
      KwayBlocks blockmap(int level) const  { return m_children[level]; }
  };
}

#endif//DENDRO_KT_COMM_HIERARCHY_H
