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
 * \file mmg2d/scalem_2d.c
 * \brief Scale and unscale mesh and solution
 * \author Charles Dapogny (UPMC)
 * \author Cécile Dobrzynski (Bx INP/Inria/UBordeaux)
 * \author Pascal Frey (UPMC)
 * \author Algiane Froehly (Inria/UBordeaux)
 * \version 5
 * \date 01 2014
 * \copyright GNU Lesser General Public License.
 **/
#include "mmg2d.h"

/**
 * \param mesh pointer toward the mesh structure.
 * \param sol pointer toward the metric or solution structure.
 * \return 1 if success, 0 if fail (computed bounding box too small
 * or one af the anisotropic input metric is not valid).
 *
 * Scale the mesh and the size informations between 0 and 1.
 * Compute a default value for the hmin/hmax parameters if needed.
 * Truncate the metric sizes between hmin/hmax
 *
 */
int MMG2D_scaleMesh(MMG5_pMesh mesh,MMG5_pSol sol) {
  //Displ     pd;
  MMG5_pPoint    ppt;
  MMG5_Info     *info;
  MMG5_pPar      ppar;
  double         dd,isqhmin,isqhmax;
  double         *m;
  double         lambda[2],v[2][2];
  int            i,k,iadr;
  char           sethmin,sethmax;
  static char    mmgWarn0=0, mmgWarn1=0;

  // pd  = mesh->disp;
  /* compute bounding box */
  if ( ! MMG5_boundingBox(mesh) ) return 0;

  info = &mesh->info;


  /* normalize coordinates and local parameters */
  dd = MMG2D_PRECI / info->delta;

  mesh->info.hausd *= dd;
  mesh->info.hsiz  *= dd;
  mesh->info.ls    *= dd;

  if ( mesh->info.npar ) {
    for (i=0; i<mesh->info.npar; i++) {
      ppar = &mesh->info.par[i];
      ppar->hmin   *= dd;
      ppar->hmax   *= dd;
      ppar->hausd  *= dd;
    }
  }

  for (k=1; k<=mesh->np; k++) {
    ppt = &mesh->point[k];
    if ( !MG_VOK(ppt) )  continue;
    ppt->c[0] = dd * (ppt->c[0] - info->min[0]);
    ppt->c[1] = dd * (ppt->c[1] - info->min[1]);
  }

  /* Check if hmin/hmax have been provided by the user and if we need to set
   * it. scale it if yes */
  sethmin = 0;
  sethmax = 0;

  if ( mesh->info.hsiz > 0. || mesh->info.optim ) {
    // We don't want to set hmin/hmax, it will be done later
    sethmin = sethmax = 1;
  }
  else {
    if ( mesh->info.hmin > 0. ) {
      mesh->info.hmin  *= dd;
      sethmin = 1;
    }
    if ( mesh->info.hmax > 0. ) {
      mesh->info.hmax  *= dd;
      sethmax = 1;
    }
  }

  /* Warning: we don't want to compute hmin/hmax from the level-set! */
  if ( mesh->info.iso || mesh->info.lag>=0 ||
       ((!sol->np) && ((!mesh->info.optim) && (mesh->info.hsiz <= 0.)) ) ) {
    /* Set default values to hmin/hmax from the bounding box if not provided by
     * the user */
    if ( !MMG5_Set_defaultTruncatureSizes(mesh,sethmin,sethmax) ) {
      fprintf(stderr,"  Exit program.\n");
      return 0;
    }
    sethmin = 1;
    sethmax = 1;
  }

  if ( !sol->np )  return 1;

  /* metric truncature and normalization and default values for hmin/hmax if not
   * provided by the user ( 0.1 \times the minimum of the metric sizes for hmin
   * and 10 \times the max of the metric sizes for hmax ). */
  switch (sol->size) {
  case 1:
    /* normalization */
    for (k=1; k<=mesh->np; k++)  {
      sol->m[k] *= dd;
      /* Check the metric */
      if (  (!mesh->info.iso) && sol->m[k] <= 0) {
        if ( !mmgWarn0 ) {
          mmgWarn0 = 1;
          fprintf(stderr,"\n  ## Error: %s: at least 1 wrong metric.\n",
                  __func__);
        }
        return 0;
      }
    }

    /* compute hmin and hmax parameters if not provided by the user */
    if ( !sethmin ) {
      mesh->info.hmin = FLT_MAX;
      for (k=1; k<=mesh->np; k++)  {
        mesh->info.hmin = MG_MIN(mesh->info.hmin,sol->m[k]);
      }
    }
    if ( !sethmax ) {
      mesh->info.hmax = 0.;
      for (k=1; k<=mesh->np; k++)  {
        mesh->info.hmax = MG_MAX(mesh->info.hmax,sol->m[k]);
      }
    }
    if ( !sethmin ) {
      mesh->info.hmin *=.1;
      /* Check that user has not given a hmax value lower that the founded
       * hmin. */
      if ( mesh->info.hmin > mesh->info.hmax ) {
        mesh->info.hmin = 0.1*mesh->info.hmax;
      }
    }
    if ( !sethmax ) {
      mesh->info.hmax *=10.;
      /* Check that user has not given a hmin value bigger that the founded
       * hmax. */
      if ( mesh->info.hmax < mesh->info.hmin ) {
        mesh->info.hmax = 10.*mesh->info.hmin;
      }
    }

    /* Truncature... if we have a metric (not a level-set) */
    if ( !mesh->info.iso ) {
      for (k=1; k<=mesh->np; k++)  {
        sol->m[k]=MG_MAX(mesh->info.hmin,sol->m[k]);
        sol->m[k]=MG_MIN(mesh->info.hmax,sol->m[k]);
      }
    }
    break;
  case 2:
    for (k=1; k<=mesh->np; k++) {
      sol->m[2*k]   *= dd;
      sol->m[2*k+1] *= dd;
    }
    break;
  case 3:
    dd = 1.0 / (dd*dd);
    /* Normalization */
    for (k=1; k<=mesh->np; k++) {
      iadr = k*sol->size;
      for (i=0; i<sol->size; i++)  sol->m[iadr+i] *= dd;
    }

    /* compute hmin and hmax parameters if not provided by the user */
    if ( !sethmin ) {
      mesh->info.hmin = FLT_MAX;
      for (k=1; k<=mesh->np; k++)  {
        iadr = k*sol->size;
        m    = &sol->m[iadr];

        /* Check the input metric */
        if ( !MMG5_eigensym(m,lambda,v) ) {
          if ( !mmgWarn0 ) {
            mmgWarn0 = 1;
            fprintf(stderr,"\n  ## Error: %s: at least 1 wrong metric.\n",
                    __func__);
          }
          return 0;
        }
        for (i=0; i<2; i++) {
          if(lambda[i]<=0) {
            if ( !mmgWarn1 ) {
              mmgWarn1 = 1;
              fprintf(stderr,"\n  ## Error: %s: at least 1 wrong metric"
                      " (eigenvalue : %e %e -- det %e\n",__func__,lambda[0],
                      lambda[1],m[0]*m[2]-m[1]*m[1]);
            }
            return 0;
          }
          mesh->info.hmin = MG_MIN(mesh->info.hmin,1./sqrt(lambda[i]));
        }
      }
    }
    if ( !sethmax ) {
      mesh->info.hmax = 0.;
      for (k=1; k<=mesh->np; k++)  {
        iadr = k*sol->size;
        m    = &sol->m[iadr];

        /* Check the input metric */
        if ( !MMG5_eigensym(m,lambda,v) ) {
          if ( !mmgWarn0 ) {
            mmgWarn0 = 1;
            fprintf(stderr,"\n  ## Error: %s: at least 1 wrong metric.\n",
                    __func__);
          }
          return 0;
        }
        for (i=0; i<2; i++) {
          if(lambda[i]<=0) {
            if ( !mmgWarn1 ) {
              mmgWarn1 = 1;
              fprintf(stderr,"\n  ## Error: %s: at least 1 wrong metric"
                      " (eigenvalue : %e %e -- det %e\n",__func__,lambda[0],
                      lambda[1],m[0]*m[2]-m[1]*m[1]);
            }
            return 0;
          }
          mesh->info.hmax = MG_MAX(mesh->info.hmax,1./sqrt(lambda[i]));
        }
      }
    }
    if ( !sethmin ) {
      mesh->info.hmin *=.1;
      /* Check that user has not given a hmax value lower that the founded
       * hmin. */
      if ( mesh->info.hmin > mesh->info.hmax ) {
        mesh->info.hmin = 0.1*mesh->info.hmax;
      }
    }
    if ( !sethmax ) {
      mesh->info.hmax *=10.;
      /* Check that user has not given a hmin value bigger that the founded
       * hmax. */
      if ( mesh->info.hmax < mesh->info.hmin ) {
        mesh->info.hmax = 10.*mesh->info.hmin;
      }
    }

    /* Truncature... if we have a metric, not a level-set */
    assert( !mesh->info.iso );
    isqhmin  = 1.0 / (mesh->info.hmin*mesh->info.hmin);
    isqhmax  = 1.0 / (mesh->info.hmax*mesh->info.hmax);
    for (k=1; k<=mesh->np; k++) {
      iadr = k*sol->size;

      m    = &sol->m[iadr];
      /* Check the input metric */
      if ( !MMG5_eigensym(m,lambda,v) ) {
        if ( !mmgWarn0 ) {
          mmgWarn0 = 1;
          fprintf(stderr,"\n  ## Error: %s: at least 1 wrong metric.\n",
                  __func__);
        }
        return 0;
      }
      for (i=0; i<2; i++) {
        if(lambda[i]<=0) {
          if ( !mmgWarn1 ) {
            mmgWarn1 = 1;
            fprintf(stderr,"\n  ## Error: %s: at least 1 wrong metric"
                    " (eigenvalue : %e %e -- det %e\n",__func__,lambda[0],
                    lambda[1],m[0]*m[2]-m[1]*m[1]);
          }
          return 0;
        }
        lambda[i]=MG_MIN(isqhmin,lambda[i]);
        lambda[i]=MG_MAX(isqhmax,lambda[i]);
      }
      m[0] = v[0][0]*v[0][0]*lambda[0] + v[1][0]*v[1][0]*lambda[1];
      m[1] = v[0][0]*v[0][1]*lambda[0] + v[1][0]*v[1][1]*lambda[1];
      m[2] = v[0][1]*v[0][1]*lambda[0] + v[1][1]*v[1][1]*lambda[1];
    }
    break;
  }
  return 1;
}

/* Unscale mesh coordinates */
int MMG2D_unscaleMesh(MMG5_pMesh mesh,MMG5_pSol sol) {
  MMG5_pPoint     ppt;
  MMG5_Info      *info;
  MMG5_pPar      ppar;
  double          dd;
  int             i,k,iadr;

  info = &mesh->info;

  /* de-normalize coordinates */
  dd = info->delta / (double)MMG2D_PRECI;
  for (k=1; k<=mesh->np; k++) {
    ppt = &mesh->point[k];
    if ( !MG_VOK(ppt) )  continue;
    ppt->c[0] = ppt->c[0] * dd + info->min[0];
    ppt->c[1] = ppt->c[1] * dd + info->min[1];
  }


  /* unscale paramter values */
  mesh->info.hmin  *= dd;
  mesh->info.hmax  *= dd;
  mesh->info.hausd *= dd;
  mesh->info.hsiz  *= dd;
  mesh->info.ls    *= dd;

  if ( mesh->info.npar ) {
    for (i=0; i<mesh->info.npar; i++) {
      ppar = &mesh->info.par[i];
      ppar->hmin   *= dd;
      ppar->hmax   *= dd;
      ppar->hausd  *= dd;
    }
  }

  /* de-normalize metric */
  if ( (!sol->np) || (!sol->m) )  return 1;

  switch (sol->size) {
  case 1:
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( !MG_VOK(ppt) )  continue;
      sol->m[k] *= dd;
    }
    break;
  case 2:
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( !MG_VOK(ppt) )  continue;
      sol->m[2*k]   *= dd;
      sol->m[2*k+1] *= dd;
    }
    break;
  case 3:
    dd = 1.0 / (dd*dd);
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( !MG_VOK(ppt) )  continue;
      iadr = k*sol->size;
      for (i=0; i<sol->size; i++)  sol->m[iadr+i] *= dd;
    }
    break;
  }

  return 1;
}
