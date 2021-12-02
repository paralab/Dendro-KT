#ifndef DENDRO_KT_PETSC_VECTOR_H
#define DENDRO_KT_PETSC_VECTOR_H

#include <vector>
#include <algorithm>

#include "gridWrapper.h"
#include "idx.h"
#include "oda.h"

namespace ot
{
  template <typename ValT>  class PetscVector;
  template <typename ValT>  class LocalPetscVector;

  //
  // PetscVector  (wrapper)
  //
  template <typename ValT>
  class PetscVector
  {
    private:
      Vec m_data;
      size_t m_size;
      bool m_isGhosted;
      size_t m_ndofs = 1;
      unsigned long long m_globalNodeRankBegin = 0;

    public:
      // PetscVector constructor
      template <unsigned int dim>
      PetscVector(const GridWrapper<dim> &mesh, bool isGhosted, size_t ndofs, const ValT *input = nullptr)
      : m_size(
          isGhosted ?
          mesh.da()->getTotalNodalSz() * ndofs : mesh.da()->getLocalNodalSz() * ndofs),
        m_isGhosted(isGhosted),
        m_ndofs(ndofs)
      {
        mesh.da()->petscCreateVector(m_data, false, isGhosted, ndofs);
        if (input != nullptr)
        {
          const size_t localSz = ndofs * mesh.da()->getLocalNodalSz();
          const size_t localBegin = ndofs * mesh.da()->getLocalNodeBegin();
          if (isGhosted)
            input += localBegin;

          ValT *array;
          VecGetArray(m_data, &array);
          std::copy_n(input, localSz, array);
          VecRestoreArray(m_data, &array);
        }
        m_globalNodeRankBegin = mesh.da()->getGlobalRankBegin();
      }

      // vec()
      Vec vec() { return m_data; }
      const Vec vec() const { return m_data; }  // doesn't really enforce const

      // isGhosted()
      bool isGhosted() const { return m_isGhosted; }

      // ndofs()
      size_t ndofs() const { return m_ndofs; }

      // size()
      size_t size() const { return m_size; }

      // globalRankBegin();
      unsigned long long globalRankBegin() const { return m_globalNodeRankBegin; }
  };


  //
  // LocalPetscVector
  //
  template <typename ValT>
  class LocalPetscVector : public PetscVector<ValT>
  {
    public:
      template <unsigned int dim>
      LocalPetscVector(const GridWrapper<dim> &mesh, size_t ndofs, const ValT *input = nullptr)
      : PetscVector<ValT>(mesh, false, ndofs, input)
      {}
  };


  // copyToPetscVector()
  template <typename ValT>
  void copyToPetscVector(LocalPetscVector<ValT> &petscVec, const LocalVector<ValT> &vector)
  {
    ValT *array;
    VecGetArray(petscVec.vec(), &array);
    std::copy_n(vector.ptr(), petscVec.size(), array);
    VecRestoreArray(petscVec.vec(), &array);
  }

  // copyFromPetscVector()
  template <typename ValT>
  void copyFromPetscVector(LocalVector<ValT> &vector, const LocalPetscVector<ValT> &petscVec)
  {
    ValT *array;
    VecGetArray(petscVec.vec(), &array);
    std::copy_n(array, vector.size(), vector.ptr());
    VecRestoreArray(petscVec.vec(), &array);
  }
}

#endif//DENDRO_KT_PETSC_VECTOR_H
