/* =============================================================================
**  This file is part of the mmg software package for the tetrahedral
**  mesh modification.
**  Copyright (c) Bx INP/CNRS/Inria/UBordeaux/UPMC, 2004-
**
**  mmg is free software: you can redistribute it and/or modify it
**  under the terms of the GNU Lesser General Public License as published
**  by the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  mmg is distributed in the hope that it will be useful, but WITHOUT
**  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
**  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
**  License for more details.
**
**  You should have received a copy of the GNU Lesser General Public
**  License and of the GNU General Public License along with mmg (in
**  files COPYING.LESSER and COPYING). If not, see
**  <http://www.gnu.org/licenses/>. Please read their terms carefully and
**  use this copy of the mmg distribution only if you accept them.
** =============================================================================
*/

/**
 * \file mmg2d/API_functionsf_2d.c
 * \brief Fortran API functions for MMG2D library.
 * \author Cecile Dobrzynski (Bx INP/Inria/UBordeaux)
 * \version 5
 * \date 07 2015
 * \copyright GNU Lesser General Public License.
 * \note Please, refer to the \ref mmg2d/libmmg2d.h file for functions
 * documentation.
 *
 * Define the Fortran API functions for MMG2D library: adds function
 * definitions with upcase, underscore and double underscore to match
 * any fortran compiler.
 *
 */
#include "mmg2d.h"

/**
 * See \ref MMG2D_Init_mesh function in common/libmmgcommon.h file.
 */
FORTRAN_VARIADIC ( MMG2D_INIT_MESH, mmg2d_init_mesh,
                   (const int starter, ... ),
                   va_list argptr;
                   int ier;

                   va_start(argptr, starter);

                   ier = MMG2D_Init_mesh_var(argptr);

                   va_end(argptr);

                   if ( !ier ) exit(EXIT_FAILURE);

                   return;
  )

/**
 * See \ref MMG2D_Init_fileNames function in mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_INIT_FILENAMES, mmg2d_init_filenames,(MMG5_pMesh *mesh, MMG5_pSol *sol),
             (mesh,sol)) {
  MMG2D_Init_fileNames(*mesh,*sol);
  return;
}
/**
 * See \ref MMG2D_Init_parameters function in mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_INIT_PARAMETERS, mmg2d_init_parameters,(MMG5_pMesh *mesh),
             (mesh)) {
  MMG2D_Init_parameters(*mesh);
  return;
}
/**
 * See \ref MMG2D_Set_inputMeshName function in \ref common/libmmgcommon.h file.
 */
FORTRAN_NAME(MMG2D_SET_INPUTMESHNAME, mmg2d_set_inputmeshname,
             (MMG5_pMesh *mesh, char* meshin, int *strlen0, int* retval),
             (mesh,meshin,strlen0,retval)) {
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshin,*strlen0);
  tmp[*strlen0] = '\0';
  *retval = MMG2D_Set_inputMeshName(*mesh,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_Set_inputSolName function in \ref common/libmmgcommon.h file.
 */
FORTRAN_NAME(MMG2D_SET_INPUTSOLNAME, mmg2d_set_inputsolname,
             (MMG5_pMesh *mesh,MMG5_pSol *sol, char* solin, int* strlen0, int* retval),
             (mesh,sol,solin,strlen0,retval)) {

  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,solin,*strlen0);
  tmp[*strlen0] = '\0';
  *retval = MMG2D_Set_inputSolName(*mesh,*sol,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_Set_outputMeshName function in mmg2d/libmmg2d.h or
 * mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_OUTPUTMESHNAME,mmg2d_set_outputmeshname,
             (MMG5_pMesh *mesh, char* meshout, int* strlen0,int* retval),
             (mesh,meshout,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshout,*strlen0);
  tmp[*strlen0] = '\0';
  *retval = MMG2D_Set_outputMeshName(*mesh, tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_Set_outputSolName function in \ref common/libmmgcommon.h file.
 */
FORTRAN_NAME(MMG2D_SET_OUTPUTSOLNAME,mmg2d_set_outputsolname,
             (MMG5_pMesh *mesh,MMG5_pSol *sol, char* solout,int* strlen0, int* retval),
             (mesh,sol,solout,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,solout,*strlen0);
  tmp[*strlen0] = '\0';
  *retval = MMG2D_Set_outputSolName(*mesh,*sol,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_Set_iparameter function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_IPARAMETER,mmg2d_set_iparameter,
             (MMG5_pMesh *mesh, MMG5_pSol *sol, int *iparam, int *val, int* retval),
             (mesh,sol,iparam,val,retval)){
  *retval = MMG2D_Set_iparameter(*mesh,*sol,*iparam,*val);
  return;
}
/**
 * See \ref MMG2D_Set_dparameter function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_DPARAMETER,mmg2d_set_dparameter,
             (MMG5_pMesh *mesh, MMG5_pSol *sol, int *dparam, double *val, int* retval),
             (mesh,sol,dparam,val,retval)){
  *retval = MMG2D_Set_dparameter(*mesh,*sol,*dparam,*val);
  return;
}
/**
 * See \ref MMG2D_Set_meshSize function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_MESHSIZE,mmg2d_set_meshsize,
             (MMG5_pMesh *mesh, int *np, int *nt, int *na, int *retval),
             (mesh,np,nt,na,retval)) {
  *retval = MMG2D_Set_meshSize(*mesh,*np,*nt,*na);
  return;
}
/**
 * See \ref MMG2D_Set_solSize function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_SOLSIZE,mmg2d_set_solsize,
             (MMG5_pMesh *mesh, MMG5_pSol *sol, int* typEntity,
              int* np, int* typSol, int* retval),
             (mesh, sol, typEntity, np, typSol, retval)) {
  *retval = MMG2D_Set_solSize(*mesh,*sol,*typEntity,*np,*typSol);
  return;
}
/**
 * See \ref MMG2D_Set_solsAtVerticesSize function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_SOLSATVERTICESSIZE,mmg2d_set_solsatverticessize,
             (MMG5_pMesh *mesh, MMG5_pSol *sol,int* nsols,
              int* nentities, int* typSol, int* retval),
             (mesh, sol, nsols, nentities, typSol, retval)) {
  *retval = MMG2D_Set_solsAtVerticesSize(*mesh,sol,*nsols,*nentities,typSol);
  return;
}

/**
 * See \ref MMG2D_Get_solSize function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_SOLSIZE,mmg2d_get_solsize,
             (MMG5_pMesh *mesh, MMG5_pSol *sol, int* typEntity, int* np, int* typSol, int* retval),
             (mesh,sol,typEntity,np,typSol,retval)) {

  *retval = MMG2D_Get_solSize(*mesh,*sol,typEntity,np,typSol);
  return;
}

/**
 * See \ref MMG2D_Get_solsAtVerticesSize function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_SOLSATVERTICESSIZE,mmg2d_get_solsatverticessize,
             (MMG5_pMesh *mesh, MMG5_pSol *sol, int *nsols,
              int* nentities, int* typSol, int* retval),
             (mesh,sol,nsols,nentities,typSol,retval)) {

  *retval = MMG2D_Get_solsAtVerticesSize(*mesh,sol,nsols,nentities,typSol);
  return;
}

/**
 * See \ref MMG2D_Set_vertex function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_VERTEX,mmg2d_set_vertex,
             (MMG5_pMesh *mesh, double* c0, double* c1, int* ref,
              int* pos, int* retval),
             (mesh,c0,c1,ref,pos,retval)) {

  *retval = MMG2D_Set_vertex(*mesh,*c0,*c1,*ref,*pos);
  return;
}
/**
 * See \ref MMG2D_Set_corner function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_CORNER,mmg2d_set_corner,(MMG5_pMesh *mesh, int *k, int* retval),
             (mesh,k,retval)) {
  *retval =  MMG2D_Set_corner(*mesh,*k);
  return;
}

/**
 * See \ref MMG2D_Set_requiredVertex function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_REQUIREDVERTEX,mmg2d_set_requiredvertex,
             (MMG5_pMesh *mesh, int *k, int* retval),
             (mesh,k,retval)) {
  *retval =  MMG2D_Set_requiredVertex(*mesh,*k);
  return;
}

/**
 * See \ref MMG2D_Get_vertex function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_VERTEX,mmg2d_get_vertex,
             (MMG5_pMesh *mesh,double* c0, double* c1, int* ref,
              int* isCorner, int* isRequired, int* retval),
             (mesh,c0,c1,ref,isCorner,isRequired, retval)) {
  *retval = MMG2D_Get_vertex(*mesh,c0,c1,ref,isCorner,isRequired);
  return;
}
/**
 * See \ref MMG2D_Set_vertices function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_VERTICES,mmg2d_set_vertices,
             (MMG5_pMesh *mesh, double* vertices, int* refs, int* retval),
             (mesh,vertices,refs,retval)) {

  *retval = MMG2D_Set_vertices(*mesh,vertices,refs);
  return;
}


/**
 * See \ref MMG2D_Get_vertices function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_VERTICES,mmg2d_get_vertices,
             (MMG5_pMesh *mesh, double* vertices, int* refs,
              int* areCorners, int* areRequired, int* retval),
             (mesh,vertices,refs,areCorners,areRequired, retval)) {
  *retval = MMG2D_Get_vertices(*mesh,vertices,refs,areCorners,areRequired);
  return;
}


/**
 * See \ref MMG2D_Set_triangle function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_TRIANGLE,mmg2d_set_triangle,
             (MMG5_pMesh *mesh, int* v0, int* v1, int* v2, int* ref,int* pos,
              int* retval),
             (mesh,v0,v1,v2,ref,pos,retval)) {
  *retval = MMG2D_Set_triangle(*mesh, *v0, *v1, *v2, *ref, *pos);
  return;
}
/**
 * See \ref MMG2D_Set_requiredTriangle function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_REQUIREDTRIANGLE,mmg2d_set_requiredtriangle,
             (MMG5_pMesh *mesh, int *k, int* retval),
             (mesh,k,retval)) {
  *retval = MMG2D_Set_requiredTriangle(*mesh, *k);
  return;
}

/**
 * See \ref MMG2D_Get_triangle function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_TRIANGLE,mmg2d_get_triangle,
             (MMG5_pMesh *mesh, int* v0, int* v1, int* v2, int* ref
              ,int* isRequired, int* retval),
             (mesh,v0,v1,v2,ref,isRequired,retval)) {
  *retval = MMG2D_Get_triangle(*mesh,v0,v1,v2,ref,isRequired);
  return;
}
/**
 * See \ref MMG2D_Set_triangles function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_TRIANGLES,mmg2d_set_triangles,
             (MMG5_pMesh *mesh, int* tria, int* refs,
              int* retval),
             (mesh,tria,refs,retval)) {
  *retval = MMG2D_Set_triangles(*mesh, tria, refs);
  return;
}

/**
 * See \ref MMG2D_Get_triangles function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_TRIANGLES,mmg2d_get_triangles,
             (MMG5_pMesh *mesh, int* tria, int* refs,int* areRequired,
              int* retval),
             (mesh,tria,refs,areRequired,retval)) {
  *retval = MMG2D_Get_triangles(*mesh,tria,refs,areRequired);
  return;
}

/**
 * See \ref MMG2D_Set_edge function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_EDGE,mmg2d_set_edge,
             (MMG5_pMesh *mesh, int *v0, int *v1, int *ref, int *pos, int* retval),
             (mesh,v0,v1,ref,pos,retval)){
  *retval = MMG2D_Set_edge(*mesh,*v0,*v1,*ref,*pos);
  return;
}
/**
 * See \ref MMG2D_Set_requiredEdge function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_REQUIREDEDGE,mmg2d_set_requirededge,
             (MMG5_pMesh *mesh, int *k, int* retval),
             (mesh,k,retval)) {
  *retval = MMG2D_Set_requiredEdge(*mesh,*k);
  return;
}
/**
 * See \ref MMG2D_Set_parallelEdge function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_PARALLELEDGE,mmg2d_set_paralleledge,
             (MMG5_pMesh *mesh, int *k, int* retval),
             (mesh,k,retval)) {
  *retval = MMG2D_Set_parallelEdge(*mesh,*k);
  return;
}

/**
 * See \ref MMG2D_Get_edge function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_EDGE,mmg2d_get_edge,(MMG5_pMesh *mesh, int* e0, int* e1, int* ref
                                          ,int* isRidge, int* isRequired, int* retval),
             (mesh,e0,e1,ref,isRidge,isRequired,retval)) {
  *retval = MMG2D_Get_edge(*mesh,e0,e1,ref,isRidge,isRequired);
  return;
}
/**
 * See \ref MMG2D_Set_edges function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_EDGES,mmg2d_set_edges,
             (MMG5_pMesh *mesh, int *edges, int *refs, int* retval),
             (mesh,edges,refs,retval)){
  *retval = MMG2D_Set_edges(*mesh,edges,refs);
  return;
}

/**
 * See \ref MMG2D_Get_edges function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_EDGES,mmg2d_get_edges,(MMG5_pMesh *mesh, int* edges,
                                              int* refs,int* areRidges,
                                              int* areRequired, int* retval),
             (mesh,edges,refs,areRidges,areRequired,retval)) {
  *retval = MMG2D_Get_edges(*mesh,edges,refs,areRidges,areRequired);
  return;
}

/**
 * See \ref MMG2D_Get_meshSize function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_MESHSIZE,mmg2d_get_meshsize,
             (MMG5_pMesh *mesh, int* np, int* nt, int* na, int* retval),
             (mesh,np,nt, na,retval)) {

  *retval = MMG2D_Get_meshSize(*mesh,np,nt,na);
  return;
}
/**
 * See \ref MMG2D_Set_scalarSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_SCALARSOL,mmg2d_set_scalarsol,
             (MMG5_pSol *met, double *s, int *pos, int* retval),
             (met,s,pos,retval)) {
  *retval = MMG2D_Set_scalarSol(*met,*s,*pos);
  return;
}
/**
 * See \ref MMG2D_Get_scalarSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_SCALARSOL,mmg2d_get_scalarsol,
             (MMG5_pSol *met, double* s, int* retval),
             (met,s,retval)) {
  *retval = MMG2D_Get_scalarSol(*met,s);
  return;
}
/**
 * See \ref MMG2D_Set_scalarSols function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_SCALARSOLS,mmg2d_set_scalarsols,
             (MMG5_pSol *met, double *s, int* retval),
             (met,s,retval)) {
  *retval = MMG2D_Set_scalarSols(*met,s);
  return;
}

/**
 * See \ref MMG2D_Get_scalarSols function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_SCALARSOLS,mmg2d_get_scalarsols,
             (MMG5_pSol *met, double* s, int* retval),
             (met,s,retval)) {
  *retval = MMG2D_Get_scalarSols(*met,s);
  return;
}

/**
 * See \ref MMG2D_Set_vectorSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_VECTORSOL,mmg2d_set_vectorsol,
             (MMG5_pSol *met, double *vx, double *vy,
              int *pos, int* retval),
             (met,vx,vy,pos,retval)) {
  *retval = MMG2D_Set_vectorSol(*met,*vx,*vy,*pos);
  return;
}

/**
 * See \ref MMG2D_Get_vectorSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_VECTORSOL,mmg2d_get_vectorsol,
             (MMG5_pSol *met, double* vx,double *vy, int* retval),
             (met,vx,vy,retval)) {
  *retval = MMG2D_Get_vectorSol(*met,vx,vy);
  return;
}
/**
 * See \ref MMG2D_Set_vectorSols function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_VECTORSOLS,mmg2d_set_vectorsols,
             (MMG5_pSol *met, double *sols, int* retval),
             (met,sols,retval)) {
  *retval = MMG2D_Set_vectorSols(*met,sols);
  return;
}

/**
 * See \ref MMG2D_Get_vectorSols function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_VECTORSOLS,mmg2d_get_vectorsols,
             (MMG5_pSol *met, double* sols, int* retval),
             (met,sols,retval)) {
  *retval = MMG2D_Get_vectorSols(*met,sols);
  return;
}

/**
 * See \ref MMG2D_Set_tensorSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_TENSORSOL,mmg2d_set_tensorsol,
             (MMG5_pSol *met, double *m11, double *m12, double *m22,
              int *pos, int* retval),
             (met,m11,m12,m22,pos,retval)) {
  *retval = MMG2D_Set_tensorSol(*met,*m11,*m12,*m22,*pos);
  return;
}
/**
 * See \ref MMG2D_Get_tensorSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_TENSORSOL,mmg2d_get_tensorsol,
             (MMG5_pSol *met, double* m11,double *m12, double *m22,
              int* retval),
             (met,m11,m12,m22,retval)) {
  *retval = MMG2D_Get_tensorSol(*met,m11,m12,m22);
  return;
}
/**
 * See \ref MMG2D_Set_tensorSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_TENSORSOLS,mmg2d_set_tensorsols,
             (MMG5_pSol *met, double* sols,int* retval),
             (met,sols,retval)) {
  *retval = MMG2D_Set_tensorSols(*met,sols);
  return;
}

/**
 * See \ref MMG2D_Get_tensorSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_TENSORSOLS,mmg2d_get_tensorsols,
             (MMG5_pSol *met, double* sols, int* retval),
             (met,sols,retval)) {
  *retval = MMG2D_Get_tensorSols(*met,sols);
  return;
}
/**
 * See \ref MMG2D_Set_ithSol_solsAtVertices function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_ITHSOL_INSOLSATVERTICES,mmg2d_set_ithsol_insolsatvertices,
             (MMG5_pSol *sol, int *i,double *s,int *pos, int* retval),
             (sol,i,s,pos,retval)) {
  *retval = MMG2D_Set_ithSol_inSolsAtVertices(*sol,*i,s,*pos);
  return;
}

/**
 * See \ref MMG2D_Get_ithSol_inSolsAtVertices function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_ITHSOL_INSOLSATVERTICES,mmg2d_get_ithsol_insolsatvertices,
             (MMG5_pSol *sol, int* i,double *s,int *pos, int* retval),
             (sol,i,s,pos,retval)) {
  *retval = MMG2D_Get_ithSol_inSolsAtVertices(*sol,*i,s,*pos);
  return;
}


/**
 * See \ref MMG2D_Set_ithSols_inSolsAtVertices function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SET_ITHSOLS_INSOLSATVERTICES,mmg2d_set_ithsols_insolsatvertices,
             (MMG5_pSol *sol, int *i,double *s, int* retval),
             (sol,i,s,retval)) {
  *retval = MMG2D_Set_ithSols_inSolsAtVertices(*sol,*i,s);
  return;
}

/**
 * See \ref MMG2D_Get_ithSols_inSolsAtVertices function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_GET_ITHSOLS_INSOLSATVERTICES,mmg2d_get_ithsols_insolsatvertices,
             (MMG5_pSol *sol, int* i,double *s, int* retval),
             (sol,i,s,retval)) {
  *retval = MMG2D_Get_ithSols_inSolsAtVertices(*sol,*i,s);
  return;
}

/**
 * See \ref MMG2D_Chk_meshData function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_CHK_MESHDATA,mmg2d_chk_meshdata,
             (MMG5_pMesh *mesh,MMG5_pSol *met, int* retval),
             (mesh,met,retval)) {
  *retval = MMG2D_Chk_meshData(*mesh,*met);
  return;
}

/**
 * See \ref MMG2D_Free_all function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_VARIADIC(MMG2D_FREE_ALL,mmg2d_free_all,
                 (const int starter,...),
                 va_list argptr;
                 int     ier;

                 va_start(argptr, starter);

                 ier = MMG2D_Free_all_var(argptr);

                 va_end(argptr);

                 if ( !ier ) exit(EXIT_FAILURE);

                 return;
  )

/**
 * See \ref MMG2D_Free_structures function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_VARIADIC(MMG2D_FREE_STRUCTURES,mmg2d_free_structures,
                 (const int starter,...),
                 va_list argptr;
                 int     ier;

                 va_start(argptr, starter);

                 ier = MMG2D_Free_structures_var(argptr);

                 va_end(argptr);

                 if ( !ier ) exit(EXIT_FAILURE);

                 return;
  )

/**
 * See \ref MMG2D_Free_names function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_VARIADIC(MMG2D_FREE_NAMES,mmg2d_free_names,
                 (const int starter,...),
                 va_list argptr;
                 int     ier;

                 va_start(argptr, starter);

                 ier = MMG2D_Free_names_var(argptr);

                 va_end(argptr);

                 if ( !ier ) exit(EXIT_FAILURE);

                 return;
  )

/**
 * See \ref MMG2D_loadMesh function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_LOADMESH,mmg2d_loadmesh,(MMG5_pMesh *mesh,char* meshin,int* strlen0,int* retval),(mesh, meshin,strlen0,retval)){

  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshin,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_loadMesh(*mesh,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_loadMshMesh function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_LOADMSHMESH,mmg2d_loadmshmesh,
             (MMG5_pMesh *mesh, MMG5_pSol *sol,char* filename, int *strlen0,int* retval),
             (mesh,sol,filename,strlen0, retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,filename,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_loadMshMesh(*mesh,*sol,tmp);

  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_loadMshMesh_and_allData function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_LOADMSHMESH_AND_ALLDATA,mmg2d_loadmshmesh_and_alldata,
             (MMG5_pMesh *mesh, MMG5_pSol *sol,char* filename, int *strlen0,int* retval),
             (mesh,sol,filename,strlen0, retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,filename,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_loadMshMesh_and_allData(*mesh,sol,tmp);

  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_saveMesh function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SAVEMESH,mmg2d_savemesh,(MMG5_pMesh *mesh,char *meshin,int *strlen0, int* retval),
             (mesh,meshin,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshin,*strlen0);
  tmp[*strlen0] = '\0';
  *retval = MMG2D_saveMesh(*mesh,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_saveMshMesh function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SAVEMSHMESH,mmg2d_savemshmesh,
             (MMG5_pMesh *mesh, MMG5_pSol *sol,char* filename, int *strlen0,
              int* retval),
             (mesh,sol,filename,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,filename,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_saveMshMesh(*mesh,*sol,tmp);

  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_saveMshMesh function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SAVEMSHMESH_AND_ALLDATA,mmg2d_savemshmesh_and_alldata,
             (MMG5_pMesh *mesh, MMG5_pSol *sol,char* filename, int *strlen0,
              int* retval),
             (mesh,sol,filename,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,filename,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_saveMshMesh_and_allData(*mesh,sol,tmp);

  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_loadSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_LOADSOL,mmg2d_loadsol,
             (MMG5_pMesh *mesh,MMG5_pSol *met,char *meshin,int* strlen0,int* retval),
             (mesh,met,meshin,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshin,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_loadSol(*mesh,*met,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_loadAllSols function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_LOADALLSOLS,mmg2d_loadallsols,
             (MMG5_pMesh *mesh,MMG5_pSol *sol,char *meshin,int* strlen0,int* retval),
             (mesh,sol,meshin,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshin,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_loadAllSols(*mesh,sol,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}


/**
 * See \ref MMG2D_saveSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SAVESOL,mmg2d_savesol,(MMG5_pMesh *mesh,MMG5_pSol *met,
                                          char *meshin,int *strlen0,int* retval),
             (mesh,met,meshin,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshin,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_saveSol(*mesh,*met,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}

/**
 * See \ref MMG2D_saveSol function in \ref mmg2d/libmmg2d.h file.
 */
FORTRAN_NAME(MMG2D_SAVEALLSOLS,mmg2d_saveallsols,(MMG5_pMesh *mesh,MMG5_pSol *sol,
                                                  char *meshin,int *strlen0,int* retval),
             (mesh,sol,meshin,strlen0,retval)){
  char *tmp = NULL;

  MMG5_SAFE_MALLOC(tmp,*strlen0+1,char,return);
  strncpy(tmp,meshin,*strlen0);
  tmp[*strlen0] = '\0';

  *retval = MMG2D_saveAllSols(*mesh,sol,tmp);
  MMG5_SAFE_FREE(tmp);

  return;
}
