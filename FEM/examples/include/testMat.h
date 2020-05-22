//
// Created by maksbh on 5/10/20.
//

#ifndef DENDRO_KT_TESTMAT_H
#define DENDRO_KT_TESTMAT_H
#include "oda.h"
#include "feMatrix.h"

namespace testEq {
enum AssemblyCheck{
  ElementByElement = 0,
  Overall = 1
};
template<unsigned int dim>
class testMat : public feMatrix<testMat<dim>, dim> {

 private:
  ot::DA<dim> *&m_uiOctDA = feMat<dim>::m_uiOctDA;
  Point<dim> &m_uiPtMin = feMat<dim>::m_uiPtMin;
  Point<dim> &m_uiPtMax = feMat<dim>::m_uiPtMax;

  unsigned int nPe;
  int elemCheck = 0;
  static constexpr unsigned int m_uiDim = dim;
  AssemblyCheck assemblyCheck_;

  int counterElemID = 0;

 public:
  using feMatrix<testMat<dim>, dim>::m_uiDof;
  /**@brief: constructor*/
  testMat(ot::DA<dim> *da, unsigned int dof = 1, AssemblyCheck checkType = AssemblyCheck::Overall, int elemID = 0);

  /**@brief default destructor*/
  ~testMat(){
  }

  /**@biref elemental matvec*/
  virtual void elementalMatVec(const VECType *in,
                               VECType *out,
                               unsigned int ndofs,
                               const double *coords = NULL,
                               double scale = 1.0) override;

  /**@brief things need to be performed before matvec (i.e. coords transform)*/
  bool preMatVec(const VECType *in, VECType *out, double scale = 1.0){
    counterElemID = 0;
    return true;
  }
  bool preMat(){
    counterElemID = 0;
    return true;
  }


  /**@brief things need to be performed after matvec (i.e. coords transform)*/
  bool postMatVec(const VECType *in, VECType *out, double scale = 1.0){
    return true;
  }

  virtual void getElementalMatrix(std::vector<ot::MatRecord> &records,
                                  const double *coords);
};

template<unsigned int dim>
testMat<dim>::testMat(ot::DA<dim> *da, unsigned int dof, AssemblyCheck checkType, int elemID) :  feMatrix<testMat<dim>, dim>(da, dof) {
  nPe = da->getNumNodesPerElement();
  assemblyCheck_ = checkType;
  elemCheck = elemID;
}


template<unsigned int dim>
void testMat<dim>::getElementalMatrix(std::vector<ot::MatRecord> &records,
                                      const double *coords) {

  ot::MatRecord mat;
  if((assemblyCheck_ == AssemblyCheck::Overall) or (elemCheck == counterElemID)) {
    for(int dofi = 0; dofi < m_uiDof; dofi++){
      for(int dofj = 0; dofj <m_uiDof; dofj++){
        for (int j = 0; j < nPe; j++) {
          for (int i = 0; i < nPe; i++) {
            mat.setMatValue(1.0);
            mat.setRowDim(dofi);
            mat.setColDim(dofj);
            mat.setRowID(i);
            mat.setColID(j);
            records.push_back(mat);
          }
        }
      }
    }

  }
  else{
    for (int j = 0; j < nPe; j++) {
      for (int i = 0; i < nPe; i++) {
        mat.setMatValue(0.0);
        mat.setRowDim(0);
        mat.setColDim(0);
        mat.setRowID(i);
        mat.setColID(j);
        records.push_back(mat);
      }
    }
  }
  counterElemID ++;
}
template<unsigned int dim>
void testMat<dim>::elementalMatVec(const VECType *in,
                                   VECType *out,
                                   unsigned int ndofs,
                                   const double *coords,
                                   double scale) {
  memset(out, 0, sizeof(VecType) * nPe*m_uiDof);
  if((assemblyCheck_ == AssemblyCheck::Overall) or (elemCheck == counterElemID)) {

    double *mat = new double[nPe*m_uiDof * nPe*m_uiDof];
    for (int i = 0; i < nPe*m_uiDof; i++) {
      for (int j = 0; j < nPe*m_uiDof; j++) {
        mat[i*nPe*m_uiDof + j] = 1.0;
      }
    }

    for (int i = 0; i < nPe*m_uiDof; i++) {
      for (int j = 0; j < nPe*m_uiDof; j++) {
        out[i] += in[j] * mat[i*nPe*m_uiDof + j];
      }
    }

    delete [] mat;
  }
  counterElemID ++;
}

}
#endif //DENDRO_KT_TESTMAT_H
