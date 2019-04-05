//
// Created by milinda on 10/30/18.
//

/**
 * @brief class that derived from abstract class feMat
 * RHS computation of the weak formulation
 * */

#ifndef DENDRO_KT_FEVECTOR_H
#define DENDRO_KT_FEVECTOR_H

#include "feVec.h"

template <typename T, unsigned int dim>
class feVector : public feVec<dim> {

protected:
    static constexpr unsigned int m_uiDim = dim;

    /**@brief number of unknowns */
    unsigned int m_uiDof;

    /**@brief element nodal vec in */
    VECType * m_uiEleVecIn;

    /***@brief element nodal vecOut */
    VECType * m_uiEleVecOut;

    /** elemental coordinates */
    double * m_uiEleCoords;


public:
    /**
     * @brief constructs an FEM stiffness matrix class.
     * @param[in] da: octree DA
     * */
    feVector(ot::DA<dim>* da,unsigned int dof=1);

    ~feVector();

    /**
     * @brief Evaluates the RHS of the PDE at specific points (for example evaluation at the quadrature points)
     * @param [out] out : function evaluated at specific points.
     * */
    //virtual void evalVec(VECType* out,double scale=1.0);


    /**
     * @brief Evaluates the right hand side of the weak formulations.
     * Typically the mass matrix multiplied with the load function.
     * @param [in] in: Input vector (f)
     * @param [out] out : Output vector (Mf)
     * */
    virtual void computeVec(const VECType* in,VECType* out,double scale=1.0);


    /**@brief evalVec for the elemental vec*/
    //virtual void elementalEvalVec(VECType* out,double scale=1.0)=0;

    /**@brief elemental compute vec which evaluate the elemental RHS of the weak formulation
     * */
    virtual void elementalComputVec(const VECType* in,VECType* out,double* coords=NULL,double scale=1.0)=0;

    #ifdef BUILD_WITH_PETSC

    /**
     * @brief Evaluates the right hand side of the weak formulations.
     * Typically the mass matrix multiplied with the load function.
     * @param [in] in: Input vector (f)
     * @param [out] out : Output vector (Mf)
     * */
    virtual void computeVec(const Vec& in,Vec& out,double scale=1.0);

    #endif


    T& asLeaf() { return static_cast<T&>(*this);}

    bool preComputeVec(const VECType* in, VECType* out,double scale=1.0) {
        return asLeaf().preComputeVec(in,out,scale);
    }

    bool postComputeVec(const VECType* in, VECType* out,double scale=1.0) {
        return asLeaf().postComputeVec(in,out,scale);
    }

    bool preEvalVec(const VECType* in, VECType* out,double scale=1.0) {
        return asLeaf().preEvalVec(in,out,scale);
    }

    bool postEvalVec(const VECType* in, VECType* out,double scale=1.0) {
        return asLeaf().postEvalVec(in,out,scale);
    }

};

template <typename T, unsigned int dim>
feVector<T,dim>::feVector(ot::DA<dim> *da,unsigned int dof) : feVec<dim>(da)
{
    m_uiDof=dof;
    const unsigned int nPe=feVec<dim>::m_uiOctDA->getNumNodesPerElement();
    m_uiEleVecIn = new  VECType[m_uiDof*nPe];
    m_uiEleVecOut = new VECType[m_uiDof*nPe];

    m_uiEleCoords= new double[m_uiDim*nPe];
}

template <typename T, unsigned int dim>
feVector<T,dim>::~feVector()
{
    delete [] m_uiEleVecIn;
    delete [] m_uiEleVecOut;

    delete [] m_uiEleCoords;
}


template <typename T, unsigned int dim>
void feVector<T,dim>::computeVec(const VECType* in,VECType* out,double scale)
{

    // todo: very simillar to matvec, but with different elemental operator. 
    /*VECType* _in=NULL;
    VECType* _out=NULL;

    if(!(m_uiOctDA->isActive()))
        return;

    preComputeVec(in,out,scale);

    m_uiOctDA->nodalVecToGhostedNodal(in,_in,false,m_uiDof);
    m_uiOctDA->createVector(_out,false,true,m_uiDof);

    VECType * val=new VECType[m_uiDof];
    for(unsigned int var=0;var<m_uiDof;var++)
        val[var]=(VECType)0;

    m_uiOctDA->setVectorByScalar(_out,val,false,true,m_uiDof);

    delete [] val;


    m_uiOctDA->readFromGhostBegin(_in,m_uiDof);

    for(m_uiOctDA->init<ot::DA_FLAGS::INDEPENDENT>();m_uiOctDA->curr()<m_uiOctDA->end<ot::DA_FLAGS::INDEPENDENT>();m_uiOctDA->next<ot::DA_FLAGS::INDEPENDENT>())
    {

        m_uiOctDA->getElementNodalValues(_in,m_uiEleVecIn,m_uiOctDA->curr(),m_uiDof);
        m_uiOctDA->getElementalCoords(m_uiOctDA->curr(),m_uiEleCoords);
        elementalComputVec(m_uiEleVecIn,m_uiEleVecOut,m_uiEleCoords,scale);
        m_uiOctDA->eleVecToVecAccumilation(_out,m_uiEleVecOut,m_uiOctDA->curr(),m_uiDof);
    }

    m_uiOctDA->readFromGhostEnd(_in,m_uiDof);

    for(m_uiOctDA->init<ot::DA_FLAGS::W_DEPENDENT>();m_uiOctDA->curr()<m_uiOctDA->end<ot::DA_FLAGS::W_DEPENDENT>();m_uiOctDA->next<ot::DA_FLAGS::W_DEPENDENT>())
    {
        m_uiOctDA->getElementNodalValues(_in,m_uiEleVecIn,m_uiOctDA->curr(),m_uiDof);
        m_uiOctDA->getElementalCoords(m_uiOctDA->curr(),m_uiEleCoords);
        elementalComputVec(m_uiEleVecIn,m_uiEleVecOut,m_uiEleCoords,scale);
        m_uiOctDA->eleVecToVecAccumilation(_out,m_uiEleVecOut,m_uiOctDA->curr(),m_uiDof);
    }

    m_uiOctDA->ghostedNodalToNodalVec(_out,out,true,m_uiDof);

    m_uiOctDA->destroyVector(_in);
    m_uiOctDA->destroyVector(_out);

    postComputeVec(in,out,scale);

    return;*/


}


#ifdef BUILD_WITH_PETSC


template <typename T, unsigned int dim>
void feVector<T,dim>::computeVec(const Vec &in, Vec &out, double scale)
{
    PetscScalar * inArry=NULL;
    PetscScalar * outArry=NULL;

    VecGetArray(in,&inArry);
    VecGetArray(out,&outArry);

    computeVec(inArry,outArry,scale);

    VecRestoreArray(in,&inArry);
    VecRestoreArray(out,&outArry);
}
#endif


#endif //DENDRO_KT_FEVECTOR_H
