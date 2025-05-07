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
 * \file mmg3d/mmg3d.c
 * \brief Main file for MMG3D executable: perform 3d mesh adaptation.
 * \author Charles Dapogny (UPMC)
 * \author Cécile Dobrzynski (Bx INP/Inria/UBordeaux)
 * \author Pascal Frey (UPMC)
 * \author Algiane Froehly (Inria/UBordeaux)
 * \version 5
 * \copyright GNU Lesser General Public License.
 */

#include "mmg3d.h"

mytime         MMG5_ctim[TIMEMAX];


/**
 * Print elapsed time at end of process.
 */
static void MMG5_endcod() {
  char   stim[32];

  chrono(OFF,&MMG5_ctim[0]);
  printim(MMG5_ctim[0].gdif,stim);
  fprintf(stdout,"\n   ELAPSED TIME  %s\n",stim);
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param bdyRefs pointer toward the list of the boundary references.
 * \return npar, the number of local parameters at tetrahedra if success,
 * 0 otherwise.
 *
 * Count the local default values at tetrahedra and fill the list of the boundary
 * references.
 *
 */
static inline
int MMG5_countLocalParamAtTet( MMG5_pMesh mesh,MMG5_iNode **bdyRefs) {
  int         npar,k,ier;

  /** Count the number of different boundary references and list it */
  (*bdyRefs) = NULL;

  k = mesh->ne? mesh->tetra[1].ref : 0;

  /* Try to alloc the first node */
  ier = MMG5_Add_inode( mesh, bdyRefs, k );
  if ( ier < 0 ) {
    fprintf(stderr,"\n  ## Error: %s: unable to allocate the first boundary"
           " reference node.\n",__func__);
    return 0;
  }
  else {
    assert(ier);
    npar = 1;
  }

  for ( k=1; k<=mesh->ne; ++k ) {
    ier = MMG5_Add_inode( mesh, bdyRefs, mesh->tetra[k].ref );

    if ( ier < 0 ) {
      fprintf(stderr, "\n  ## Warning: %s: unable to list the tetra references.\n"
              "              Uncomplete parameters file.\n",__func__ );
      break;
    }
    else if ( ier ) ++npar;
  }

  return npar;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param bdryRefs pointer toward the list of the boundary references.
 * \param out pointer toward the file in which to write.
 * \return 1 if success, 0 otherwise.
 *
 * Write the local default values at tetrahedra in the parameter file.
 *
 */
static inline
int MMG5_writeLocalParamAtTet( MMG5_pMesh mesh, MMG5_iNode *bdryRefs,
                                FILE *out ) {
  MMG5_iNode *cur;

  cur = bdryRefs;
  while( cur ) {
    fprintf(out,"%d Tetrahedron %e %e %e \n",cur->val,
            mesh->info.hmin, mesh->info.hmax,mesh->info.hausd);
    cur = cur->nxt;
  }

  MMG5_Free_ilinkedList(mesh,bdryRefs);

  return 1;
}


/**
 * \param mesh pointer toward the mesh structure.
 * \return 1 if success, 0 otherwise.
 *
 * Write a DEFAULT.mmg3d file containing the default values of parameters that
 * can be locally defined.
 *
 */
static inline
int MMG3D_writeLocalParam( MMG5_pMesh mesh ) {
  MMG5_iNode  *triRefs,*tetRefs;
  int          nparTri,nparTet;
  char         *ptr,data[128];
  FILE         *out;

  /** Save the local parameters file */
  strcpy(data,mesh->namein);
  ptr = strstr(data,".mesh");
  if ( ptr ) *ptr = '\0';
  strcat(data,".mmg3d");

  if ( !(out = fopen(data,"wb")) ) {
    fprintf(stderr,"\n  ** UNABLE TO OPEN %s.\n",data);
    return 0;
  }

  fprintf(stdout,"\n  %%%% %s OPENED\n",data);

  nparTri = MMG5_countLocalParamAtTri( mesh, &triRefs);
  if ( !nparTri ) {
    fclose(out);
    return 0;
  }

  nparTet = MMG5_countLocalParamAtTet( mesh, &tetRefs);
  if ( !nparTet ) {
    fclose(out);
    return 0;
  }

  fprintf(out,"parameters\n %d\n",nparTri+nparTet);

  /** Write local param at triangles */
  if (! MMG5_writeLocalParamAtTri(mesh,triRefs,out) ) {
    fclose(out);
    return 0;
  }

  /** Write local param at tetra */
  if (! MMG5_writeLocalParamAtTet(mesh,tetRefs,out) ) {
    fclose(out);
    return 0;
  }

  fclose(out);
  fprintf(stdout,"  -- WRITING COMPLETED\n");

  return 1;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param met pointer toward a sol structure (metric).
 * \return \ref MMG5_SUCCESS if success, \ref MMG5_LOWFAILURE if failed
 * but a conform mesh is saved and \ref MMG5_STRONGFAILURE if failed and we
 * can't save the mesh.
 *
 * Program to save the local default parameter file: read the mesh and metric
 * (needed to compite the hmax/hmin parameters), scale the mesh and compute the
 * hmax/hmin param, unscale the mesh and write the default parameter file.
 *
 */
static inline
int MMG3D_defaultOption(MMG5_pMesh mesh,MMG5_pSol met) {
  mytime    ctim[TIMEMAX];
  double    hsiz;
  char      stim[32];

  MMG3D_Set_commonFunc();

  signal(SIGABRT,MMG5_excfun);
  signal(SIGFPE,MMG5_excfun);
  signal(SIGILL,MMG5_excfun);
  signal(SIGSEGV,MMG5_excfun);
  signal(SIGTERM,MMG5_excfun);
  signal(SIGINT,MMG5_excfun);

  tminit(ctim,TIMEMAX);
  chrono(ON,&(ctim[0]));

  if ( mesh->info.npar ) {
    fprintf(stderr,"\n  ## Error: %s: "
            "unable to save of a local parameter file with"
            " the default parameters values because local parameters"
            " are provided.\n",__func__);
    _LIBMMG5_RETURN(mesh,met,MMG5_LOWFAILURE);
  }


  if ( mesh->info.imprim > 0 ) fprintf(stdout,"\n  -- INPUT DATA\n");

  /* load data */
  chrono(ON,&(ctim[1]));

  if ( met->np && (met->np != mesh->np) ) {
    fprintf(stderr,"\n  ## WARNING: WRONG SOLUTION NUMBER. IGNORED\n");
    MMG5_DEL_MEM(mesh,met->m);
    met->np = 0;
  }

  chrono(OFF,&(ctim[1]));
  printim(ctim[1].gdif,stim);
  if ( mesh->info.imprim > 0 )
    fprintf(stdout,"  --  INPUT DATA COMPLETED.     %s\n",stim);

  /* analysis */
  chrono(ON,&(ctim[2]));
  MMG3D_setfunc(mesh,met);

  if ( mesh->info.imprim > 0 ) {
    fprintf(stdout,"\n  %s\n   MODULE MMG3D: IMB-LJLL : %s (%s)\n  %s\n",MG_STR,MG_VER,MG_REL,MG_STR);
    fprintf(stdout,"\n  -- DEFAULT PARAMETERS COMPUTATION\n");
  }

  /* scaling mesh + hmin/hmax computation*/
  if ( !MMG5_scaleMesh(mesh,met) ) _LIBMMG5_RETURN(mesh,met,MMG5_STRONGFAILURE);

  /* specific meshing + hmin/hmax update */
  if ( mesh->info.optim ) {
    if ( !MMG3D_doSol(mesh,met) ) {
      if ( !MMG5_unscaleMesh(mesh,met) ) _LIBMMG5_RETURN(mesh,met,MMG5_STRONGFAILURE);
      _LIBMMG5_RETURN(mesh,met,MMG5_LOWFAILURE);
    }
    MMG3D_solTruncatureForOptim(mesh,met);
  }

  if ( mesh->info.hsiz > 0. ) {
    if ( !MMG5_Compute_constantSize(mesh,met,&hsiz) ) {
     if ( !MMG5_unscaleMesh(mesh,met) ) _LIBMMG5_RETURN(mesh,met,MMG5_STRONGFAILURE);
     _LIBMMG5_RETURN(mesh,met,MMG5_STRONGFAILURE);
    }
  }

  /* unscaling mesh */
  if ( !MMG5_unscaleMesh(mesh,met) ) _LIBMMG5_RETURN(mesh,met,MMG5_STRONGFAILURE);

  /* Save the local parameters file */
  mesh->mark = 0;
  if ( !MMG3D_writeLocalParam(mesh) ) {
    fprintf(stderr,"\n  ## Error: %s: Unable to save the local parameters file.\n"
            "            Exit program.\n",__func__);
     _LIBMMG5_RETURN(mesh,met,MMG5_LOWFAILURE);
  }

  _LIBMMG5_RETURN(mesh,met,MMG5_SUCCESS);
}


/**
 * \param argc number of command line arguments.
 * \param argv command line arguments.
 * \return \ref MMG5_SUCCESS if success.
 * \return \ref MMG5_LOWFAILURE if failed but a conform mesh is saved.
 * \return \ref MMG5_STRONGFAILURE if failed and we can't save the mesh.
 *
 * Main program for MMG3D executable: perform mesh adaptation.
 *
 */
int main(int argc,char *argv[]) {

  MMG5_pMesh      mesh;
  MMG5_pSol       met,disp;
  int             ier,ierSave,msh;
  char            stim[32];

  fprintf(stdout,"  -- MMG3D, Release %s (%s) \n",MG_VER,MG_REL);
  fprintf(stdout,"     %s\n",MG_CPY);
  fprintf(stdout,"     %s %s\n",__DATE__,__TIME__);

  MMG3D_Set_commonFunc();

  /* Print timer at exit */
  atexit(MMG5_endcod);

  tminit(MMG5_ctim,TIMEMAX);
  chrono(ON,&MMG5_ctim[0]);


  /* assign default values */
  mesh = NULL;
  met  = NULL;
  disp = NULL;

  MMG3D_Init_mesh(MMG5_ARG_start,
                  MMG5_ARG_ppMesh,&mesh,MMG5_ARG_ppMet,&met,
                  MMG5_ARG_ppDisp,&disp,
                  MMG5_ARG_end);
  /* reset default values for file names */
  if ( !MMG3D_Free_names(MMG5_ARG_start,
                         MMG5_ARG_ppMesh,&mesh,MMG5_ARG_ppMet,&met,
                         MMG5_ARG_ppDisp,&disp,
                         MMG5_ARG_end) )
    return MMG5_STRONGFAILURE;


  /* Set default metric size */
  if ( !MMG3D_Set_solSize(mesh,met,MMG5_Vertex,0,MMG5_Scalar) )
    MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);

  /* command line */
  if ( !MMG3D_parsar(argc,argv,mesh,met) )  return MMG5_STRONGFAILURE;

  /* load data */
  if ( mesh->info.imprim >= 0 )
    fprintf(stdout,"\n  -- INPUT DATA\n");
  chrono(ON,&MMG5_ctim[1]);

  /* read mesh file */
  msh = 0;
  ier = MMG3D_loadMesh(mesh,mesh->namein);
  if ( !ier ) {
    if ( mesh->info.lag > -1 )
      ier = MMG3D_loadMshMesh(mesh,disp,mesh->namein);
    else
      ier = MMG3D_loadMshMesh(mesh,met,mesh->namein);
    msh = 1;
  }
  if ( ier<1 )
    MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);

  /* read displacement if any */
  if ( mesh->info.lag > -1 ) {
    if ( !msh && !MMG3D_Set_solSize(mesh,disp,MMG5_Vertex,0,MMG5_Vector) )
      MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);

    if ( !MMG3D_Set_inputSolName(mesh,disp,met->namein) )
      MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);

    if ( !msh ) {
      ier = MMG3D_loadSol(mesh,disp,disp->namein);
      if ( ier == 0 ) {
        fprintf(stderr,"\n  ## ERROR: NO DISPLACEMENT FOUND.\n");
        MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);
      }
      else if ( ier == -1 ) {
        fprintf(stderr,"\n  ## ERROR: WRONG DATA TYPE OR WRONG SOLUTION NUMBER.\n");
        MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);
      }
    }
  }
  /* read metric if any */
  else {
    if ( !msh ) {
      ier = MMG3D_loadSol(mesh,met,met->namein);

      if ( ier == -1 ) {
        fprintf(stderr,"\n  ## ERROR: WRONG DATA TYPE OR WRONG SOLUTION NUMBER.\n");
        MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);
      }
      if ( mesh->info.iso && !ier ) {
        fprintf(stderr,"\n  ## ERROR: NO ISOVALUE DATA.\n");
        MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);
      }
    }
    else {
      if ( mesh->info.iso && met->m==0 ) {
        fprintf(stderr,"\n  ## ERROR: NO ISOVALUE DATA.\n");
        MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);
      }
    }
    if ( !MMG3D_parsop(mesh,met) )
      MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_LOWFAILURE);
  }

  chrono(OFF,&MMG5_ctim[1]);
  if ( mesh->info.imprim >= 0 ) {
    printim(MMG5_ctim[1].gdif,stim);
    fprintf(stdout,"  -- DATA READING COMPLETED.     %s\n",stim);
  }

  if ( mesh->mark ) {
    /* Save a local parameters file containing the default parameters */
    ier = MMG3D_defaultOption(mesh,met);
    MMG5_RETURN_AND_FREE(mesh,met,disp,ier);
  }
  else if ( mesh->info.lag > -1 ) {
    ier = MMG3D_mmg3dmov(mesh,met,disp);
  }
  else if ( mesh->info.iso ) {
    ier = MMG3D_mmg3dls(mesh,met);
  }
  else {
    ier = MMG3D_mmg3dlib(mesh,met);
  }

  if ( ier != MMG5_STRONGFAILURE ) {
    /** Save files at medit or Gmsh format */
    chrono(ON,&MMG5_ctim[1]);
    if ( mesh->info.imprim > 0 )
      fprintf(stdout,"\n  -- WRITING DATA FILE %s\n",mesh->nameout);

    MMG5_chooseOutputFormat(mesh,&msh);

    if ( !msh )
      ierSave = MMG3D_saveMesh(mesh,mesh->nameout);
    else
      ierSave = MMG3D_saveMshMesh(mesh,met,mesh->nameout);

    if ( !ierSave )
      MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);

    if ( !msh && !MMG3D_saveSol(mesh,met,met->nameout) )
      MMG5_RETURN_AND_FREE(mesh,met,disp,MMG5_STRONGFAILURE);

    chrono(OFF,&MMG5_ctim[1]);
    if ( mesh->info.imprim > 0 )
      fprintf(stdout,"  -- WRITING COMPLETED\n");
  }

  /* free mem */
  MMG5_RETURN_AND_FREE(mesh,met,disp,ier);
}
