//
// Created by milinda on 10/30/18.
//
/**
 * @file feVec.h
 * @brief feVec.h abstract interface based on Dendro 4. feVec contains an abstract interface to
 * perform FEM accumilations for a given vector.
 * @author Milinda Fernando
 *
 * **/
#ifndef DENDRO_KT_FEVEC_H
#define DENDRO_KT_FEVEC_H

#include "point.h"
#include "oda.h"

#ifdef BUILD_WITH_PETSC
#include "petscdmda.h"
#endif

template <unsigned int dim>
class feVec {

protected:
    static constexpr unsigned int m_uiDim = dim;

    /**@brief: pointer to OCT DA*/
    ot::DA<dim>* m_uiOctDA = {};

    /**@brief problem domain min point*/
    Point<dim> m_uiPtMin = Point<dim>(-1.0);

    /**@brief problem domain max point*/
    Point<dim> m_uiPtMax = Point<dim>(1.0);

#ifdef BUILD_WITH_PETSC
    /**@brief: petsc DM*/
    DM m_uiPETSC_DA;
#endif

protected:
    // Place in protected access due to polymorphism.
    feVec(const feVec &) = default;
    feVec(feVec &&) = default;
    feVec & operator=(const feVec &) = default;
    feVec & operator=(feVec &&) = default;

public:
    feVec() = default;

    /**@brief: feVec constructor
     * @par[in] daType: type of the DA
     * */
    feVec(ot::DA<dim>* da)
      : m_uiOctDA(da)
    { }

    feVec(ot::DA<dim>* da, const Point<dim>& pt_min, const Point<dim>& pt_max)
      : m_uiOctDA(da), m_uiPtMin(pt_min), m_uiPtMax(pt_max)
    { }


    // da()
    const ot::DA<dim> * da() const { return m_uiOctDA; }

    // octList()
    const std::vector<ot::TreeNode<unsigned int, dim>> * octList() const {
      return & da()->dist_tree()->getTreePartFiltered(da()->stratum());
    }


    /**
     * @brief Evaluates the RHS of the PDE at specific points (for example evaluation at the quadrature points)
     * @param [out] out : function evaluated at specific points.
     * */
    //virtual void evalVec(VECType* out,double scale=1.0)=0;

    /**
     * @brief Evaluates the right hand side of the weak formulations.
     * Typically the mass matrix multiplied with the load function.
     * @param [in] in: Input vector (f)
     * @param [out] out : Output vector (Mf)
     * */
    virtual void computeVec(const VECType* in,VECType* out,double scale=1.0)=0;


    /**@brief set the problem dimension*/
    inline void setProblemDimensions(const Point<dim>& pt_min, const Point<dim>& pt_max)
    {
        m_uiPtMin=pt_min;
        m_uiPtMax=pt_max;
    }

#ifdef BUILD_WITH_PETSC
// all PETSC function should go here.

    /**
     * @brief Evaluates the right hand side of the weak formulations.
     * Typically the mass matrix multiplied with the load function.
     * @param [in] in: Input vector (f)
     * @param [out] out : Output vector (Mf)
     * */
    virtual void computeVec(const Vec& in,Vec& out,double scale=1.0)=0;

    /**
     * @brief placeholder for non -linear solve
     * @param v the vector set as placeholder
     */
    virtual void setPlaceholder(const Vec & v)
    {
        std::cerr << "Need to override this " << __func__ << "\n";
    }
#endif


};

#endif //DENDRO_KT_FEVEC_H
