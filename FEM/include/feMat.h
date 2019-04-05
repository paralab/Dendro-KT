//
// Created by milinda on 10/30/18.
//
/**
 * @author Milinda Fernando
 * @brief Abstract class for stiffness matrix computation.
 * */

#ifndef DENDRO_KT_FEMAT_H
#define DENDRO_KT_FEMAT_H

#include "oda.h"
#include "point.h"
#ifdef BUILD_WITH_PETSC
#include "petscdmda.h"
#endif

template <unsigned int dim>
class feMat {

protected:
    static constexpr unsigned int m_uiDim = dim;

    /**@brief: pointer to OCT DA*/
    ot::DA<dim>* m_uiOctDA;

    /// /**@brief: type of the DA*/  //TODO
    /// ot::DAType m_uiDaType;

    /**@brief problem domain min point*/
    Point m_uiPtMin;

    /**@brief problem domain max point*/
    Point m_uiPtMax;


#ifdef BUILD_WITH_PETSC
    /**@brief: petsc DM*/
    DM m_uiPETSC_DA;
#endif

public:
    /**@brief: feMat constructor
      * @par[in] daType: type of the DA
      * @note Does not own da.
    **/
    feMat(ot::DA<dim>* da)
    {
        m_uiOctDA=da;
    }

    /**@brief deconstructor*/
    ~feMat()
    {

    }

    /**@brief Computes the LHS of the weak formulation, normally the stifness matrix times a given vector.
     * @param [in] in input vector u
     * @param [out] out output vector Ku
     * @param [in] default parameter scale vector by scale*Ku
     * */
    virtual void matVec(const VECType * in, VECType* out,double scale=1.0)=0;

    /**@brief set the problem dimension*/
    inline void setProblemDimensions(const Point& pt_min, const Point& pt_max)
    {
        m_uiPtMin=pt_min;
        m_uiPtMax=pt_max;
    }


#ifdef BUILD_WITH_PETSC
// all PETSC function should go here.

    /** @brief Computes the LHS of the weak formulation, normally the stifness matrix times a given vector.
      * @param [in] in input vector u
      * @param [out] out output vector Ku
      * @param [in] default parameter scale vector by scale*Ku
    **/
    virtual void matVec(const Vec& in, Vec& out,double scale=1.0)=0;

    /**
     * @brief Performs the matrix assembly.
     * @param [in/out] J: Matrix assembled
     * @param [in] mtype: Matrix type
     * when the function returns, J is set to assembled matrix
     * */
    virtual bool getAssembledMatrix(Mat *J, MatType mtype) = 0;

#endif


};

#endif //DENDRO_KT_FEMAT_H
