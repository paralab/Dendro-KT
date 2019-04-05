//
// Created by milinda on 10/30/18.
//
/**
 * @brief class that derived from abstract class feMat
 * LHS computation of the weak formulation
 * */
#ifndef DENDRO_KT_FEMATRIX_H
#define DENDRO_KT_FEMATRIX_H

#include "feMat.h"

template <typename T, unsigned int dim>
class feMatrix : public feMat<dim> {

protected:
         static constexpr unsigned int m_uiDim = dim;

         /**@brief number of dof*/
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
        feMatrix(ot::DA<dim>* da,unsigned int dof=1);

        ~feMatrix();

        /**@brief Computes the LHS of the weak formulation, normally the stifness matrix times a given vector.
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] default parameter scale vector by scale*Ku
        * */
        virtual void matVec(const VECType* in,VECType* out, double scale=1.0);


        /**@brief Computes the elemental matvec
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] scale vector by scale*Ku
        **/
        virtual void elementalMatVec(const VECType *in, VECType *out, double *coords, double scale) = 0;



#ifdef BUILD_WITH_PETSC

        /**@brief Computes the LHS of the weak formulation, normally the stifness matrix times a given vector.
          * @param [in] in input vector u
          * @param [out] out output vector Ku
          * @param [in] default parameter scale vector by scale*Ku
        * */
        virtual void matVec(const Vec& in,Vec& out, double scale=1.0);

        /**
         * @brief Performs the matrix assembly.
         * @param [in/out] J: Matrix assembled
         * @param [in] mtype: Matrix type
         * when the function returns, J is set to assembled matrix
         **/
        virtual bool getAssembledMatrix(Mat *J, MatType mtype);


#endif

        /**@brief static cast to the leaf node of the inheritance*/
        T& asLeaf() { return static_cast<T&>(*this);}


        /**
         * @brief executed just before  the matVec loop in matvec function
         * @param[in] in : input Vector
         * @param[out] out: output vector
         * @param[in] scale: scalaing factror
         **/

        bool preMatVec(const VECType* in, VECType* out,double scale=1.0) {
           return asLeaf().preMatVec(in,out,scale);
        }


        /**@brief executed just after the matVec loop in matvec function
         * @param[in] in : input Vector
         * @param[out] out: output vector
         * @param[in] scale: scalaing factror
         * */

        bool postMatVec(const VECType* in, VECType* out,double scale=1.0) {
             return asLeaf().postMatVec(in,out,scale);
        }

        /**@brief executed before the matrix assembly */
        bool preMat() {
            return asLeaf().preMat();
        }

        /**@brief executed after the matrix assembly */
        bool postMat() {
            return asLeaf().preMat();
        }

        /// /**
        ///  * @brief Compute the elemental Matrix.
        ///  * @param[in] eleID: element ID
        ///  * @param[out] records: records corresponding to the elemental matrix.
        ///  * */
        /// void getElementalMatrix(unsigned int eleID, std::vector<ot::MatRecord>& records)
        /// {
        ///     return asLeaf().getElementalMatrix(eleID,records);
        /// }


};

template <typename T, unsigned int dim>
feMatrix<T,dim>::feMatrix(ot::DA<dim>* da,unsigned int dof) : feMat<dim>(da)
{
    m_uiDof=dof;
    const unsigned int nPe=feMat<dim>::m_uiOctDA->getNumNodesPerElement();
    m_uiEleVecIn = new  VECType[m_uiDof*nPe];
    m_uiEleVecOut = new VECType[m_uiDof*nPe];

    m_uiEleCoords= new double[m_uiDim*nPe];

}
template <typename T, unsigned int dim>
feMatrix<T,dim>::~feMatrix()
{
    delete [] m_uiEleVecIn;
    delete [] m_uiEleVecOut;
    delete [] m_uiEleCoords;

    m_uiEleVecIn=NULL;
    m_uiEleVecOut=NULL;
    m_uiEleCoords=NULL;

}

template <typename T, unsigned int dim>
void feMatrix<T,dim>::matVec(const VECType *in, VECType *out, double scale)
{

    // todo : matvec goes here. 
    /*VECType* _in=NULL;
    VECType* _out=NULL;

    if(!(m_uiOctDA->isActive()))
        return;


    preMatVec(in,out,scale);

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
        elementalMatVec(m_uiEleVecIn,m_uiEleVecOut,m_uiEleCoords,scale);
        m_uiOctDA->eleVecToVecAccumilation(_out,m_uiEleVecOut,m_uiOctDA->curr(),m_uiDof);
    }

    m_uiOctDA->readFromGhostEnd(_in,m_uiDof);


    for(m_uiOctDA->init<ot::DA_FLAGS::W_DEPENDENT>();m_uiOctDA->curr()<m_uiOctDA->end<ot::DA_FLAGS::W_DEPENDENT>();m_uiOctDA->next<ot::DA_FLAGS::W_DEPENDENT>())
    {
        m_uiOctDA->getElementNodalValues(_in,m_uiEleVecIn,m_uiOctDA->curr(),m_uiDof);
        m_uiOctDA->getElementalCoords(m_uiOctDA->curr(),m_uiEleCoords);
        elementalMatVec(m_uiEleVecIn,m_uiEleVecOut,m_uiEleCoords,scale);
        m_uiOctDA->eleVecToVecAccumilation(_out,m_uiEleVecOut,m_uiOctDA->curr(),m_uiDof);
    }



    m_uiOctDA->ghostedNodalToNodalVec(_out,out,true,m_uiDof);

    m_uiOctDA->destroyVector(_in);
    m_uiOctDA->destroyVector(_out);

    postMatVec(in,out,scale);


    return;*/

}

#ifdef BUILD_WITH_PETSC

template <typename T, unsigned int dim>
void feMatrix<T,dim>::matVec(const Vec &in, Vec &out, double scale)
{

    PetscScalar * inArry=NULL;
    PetscScalar * outArry=NULL;

    VecGetArray(in,&inArry);
    VecGetArray(out,&outArry);

    matVec(inArry,outArry,scale);

    VecRestoreArray(in,&inArry);
    VecRestoreArray(out,&outArry);

}



template <typename T, unsigned int dim>
bool feMatrix<T,dim>::getAssembledMatrix(Mat *J, MatType mtype)
{
    // todo we can skip this part for sc. 
    // Octree part ...
    /*char matType[30];
    PetscBool typeFound;
    PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-fullJacMatType", matType, 30, &typeFound);
    if(!typeFound) {
        std::cout<<"[Error]"<<__func__<<" I need a MatType for the full Jacobian matrix!"<<std::endl;
        MPI_Finalize();
        exit(0);
    }

    m_uiOctDA->createMatrix(*J, matType, 1);
    MatZeroEntries(*J);
    std::vector<ot::MatRecord> records;

    preMat();

    for(m_uiOctDA->init<ot::DA_FLAGS::WRITABLE>(); m_uiOctDA->curr() < m_uiOctDA->end<ot::DA_FLAGS::WRITABLE>();m_uiOctDA->next<ot::DA_FLAGS::WRITABLE>()) {
        getElementalMatrix(m_uiOctDA->curr(), records);
        if(records.size() > 500) {
            m_uiOctDA->petscSetValuesInMatrix(*J, records, 1, ADD_VALUES);
        }
    }//end writable
    m_uiOctDA->petscSetValuesInMatrix(*J, records, 1, ADD_VALUES);

    postMat();

    MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);

    PetscFunctionReturn(0);*/
}


#endif



#endif //DENDRO_KT_FEMATRIX_H
