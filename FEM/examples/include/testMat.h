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
  /**@brief: constructor*/
  testMat(ot::DA<dim> *da, unsigned int dof = 1, AssemblyCheck checkType = AssemblyCheck::Overall, int elemID = 0);

  /**@brief default destructor*/
  ~testMat(){
  }


  bool wasActive() const;

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
                                  const double *coords,
                                  const ot::RankI *globNodeIds);
};

template<unsigned int dim>
testMat<dim>::testMat(ot::DA<dim> *da, unsigned int dof, AssemblyCheck checkType, int elemID) :  feMatrix<testMat<dim>, dim>(da, dof) {
  nPe = da->getNumNodesPerElement();
  assemblyCheck_ = checkType;
  elemCheck = elemID;
}


template<unsigned int dim>
bool testMat<dim>::wasActive() const
{
  return ((assemblyCheck_ == AssemblyCheck::Overall) or (elemCheck == counterElemID-1));
}

template<unsigned int dim>
void testMat<dim>::getElementalMatrix(std::vector<ot::MatRecord> &records,
                                      const double *coords,
                                      const ot::RankI *globNodeIds) {

  ot::MatRecord mat;
  if((assemblyCheck_ == AssemblyCheck::Overall) or (elemCheck == counterElemID)) {

    for (int j = 0; j < nPe; j++) {
      for (int i = 0; i < nPe; i++) {

        mat.setMatValue(1.0);
        mat.setColDim(0);
        mat.setRowDim(0);
        mat.setColID(globNodeIds[i]);
        mat.setRowID(globNodeIds[j]);
        records.push_back(mat);
      }
    }
  }
  else{
    for (int j = 0; j < nPe; j++) {
      for (int i = 0; i < nPe; i++) {
        mat.setMatValue(0.0);
        mat.setColDim(0);
        mat.setRowDim(0);
        mat.setColID(globNodeIds[i]);
        mat.setRowID(globNodeIds[j]);
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
  memset(out, 0, sizeof(VecType) * nPe);
  if((assemblyCheck_ == AssemblyCheck::Overall) or (elemCheck == counterElemID)) {

    /// double mat[nPe][nPe];
    /// for (int i = 0; i < nPe; i++) {
    ///   for (int j = 0; j < nPe; j++) {
    ///     mat[i][j] = 1.0;
    ///   }
    /// }

    /// for (int i = 0; i < nPe; i++) {
    ///   for (int j = 0; j < nPe; j++) {
    ///     out[i] += in[j] * mat[i][j];
    ///   }
    /// }

    double *mat = new double[nPe * nPe];
    for (int i = 0; i < nPe; i++) {
      for (int j = 0; j < nPe; j++) {
        mat[i*nPe + j] = 1.0;
      }
    }

    for (int i = 0; i < nPe; i++) {
      for (int j = 0; j < nPe; j++) {
        out[i] += in[j] * mat[i*nPe + j];
      }
    }

    delete [] mat;
  }
  counterElemID ++;
}

}
#endif //DENDRO_KT_TESTMAT_H
