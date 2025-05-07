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
 * \file mmg3d/chkmsh_3d.c
 * \brief Check the input mesh validity.
 * \author Charles Dapogny (UPMC)
 * \author Cécile Dobrzynski (Bx INP/Inria/UBordeaux)
 * \author Pascal Frey (UPMC)
 * \author Algiane Froehly (Inria/UBordeaux)
 * \version 5
 * \copyright GNU Lesser General Public License.
 * \todo doxygen documentation.
 */

#include "mmg3d.h"

#define  MMG5_EPSLOC   1.00005
#define  IEDG(a,b) (((a) > 0) && ((b) > 0)) ? ((a)+(b)) : (((a)+(b))-(1))

extern char ddb;

/**
 *
 * \warning Not used.
 */
void MMG5_chkvol(MMG5_pMesh mesh) {
  MMG5_pTetra    pt;
  int       k;
#ifdef DEBUG
  int       ier=1;
#endif

  for (k=1; k<=mesh->ne; k++) {
    pt = &mesh->tetra[k];
    if ( !MG_EOK(pt) )  continue;
    if ( MMG5_orvol(mesh->point,pt->v) < MMG5_NULKAL ) {
      fprintf(stderr,"\n  ## Warning: %s: tetra %d volume %e\n",__func__,
             k,MMG5_orvol(mesh->point,pt->v));
#ifdef DEBUG
      ier = 0;
#endif
    }
  }
#ifdef DEBUG
  assert(ier);
#endif
}

/**
* \return 0 if fail, 1 otherwise
 *
 * \warning Not used.
 */
int MMG5_chkmshsurf(MMG5_pMesh mesh){
  MMG5_pTria      pt;
  int        k,k1;
  int        *adja,*adja1;
  char       i,voy;

  for (k=1; k<=mesh->nt; k++) {
    pt   = &mesh->tria[k];
    adja = &mesh->adjt[3*(k-1)+1];
    for (i=0; i<3; i++) {
      if ( pt->tag[i] & MG_NOM )  continue;
      k1  = adja[i] / 3;
      voy = adja[i] % 3;

      if(!k1) continue;
      adja1 = &mesh->adjt[3*(k1-1)+1];

      if(adja1[voy] / 3 != k){
        fprintf(stderr,"\n  ## Warning: %s: wrong adjacency relation"
                " for triangles : %d %d \n",__func__,k,k1);
        return 0;
      }
    }
  }
  return 1;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param severe level of performed check
 * \param base unused argument.
 * \return 0 if fail, 1 if success.
 *
 * Check the mesh validity
 *
 */
int MMG5_mmg3dChkmsh(MMG5_pMesh mesh,int severe,int base) {
  MMG5_pTetra    pt,pt1,pt2;
  MMG5_pxTetra   pxt;
  int            *adja,*adja1,adj,adj1,k,i,iadr;
  int            iel,a0,a1,a2,b0,b1,b2;
  unsigned char  voy,voy1;
  static char    mmgErr0=0,mmgErr1=0,mmgErr2=0,mmgErr3=0,mmgErr4=0,mmgErr5=0;

  for (k=1; k<=mesh->ne; k++) {
    pt1 = &mesh->tetra[k];
    if ( !MG_EOK(pt1) || pt1->ref < 0 )   continue;
    iadr = 4*(k-1) + 1;
    adja = &mesh->adja[iadr];

    for (i=0; i<4; i++) {
      adj = adja[i];

      if ( !adj )  continue;
      adj /= 4;
      voy = adja[i] % 4;

      if ( adj == k ) {
        if ( !mmgErr0 ) {
          fprintf(stderr,"\n  ## Error: %s: 1. at least 1 wrong adjacency %d %d\n",
                  __func__,MMG3D_indElt(mesh,k),MMG3D_indElt(mesh,adj));
          fprintf(stderr,"triangle %d: %d %d %d %d\n",MMG3D_indElt(mesh,k),
                  MMG3D_indPt(mesh,pt1->v[0]),MMG3D_indPt(mesh,pt1->v[1]),
                  MMG3D_indPt(mesh,pt1->v[2]),MMG3D_indPt(mesh,pt1->v[3]));
          fprintf(stderr,"adj (%d): %d %d %d %d\n",MMG3D_indElt(mesh,k),
                  MMG3D_indElt(mesh,adja[0]/4),MMG3D_indElt(mesh,adja[1]/4),
                  MMG3D_indElt(mesh,adja[2]/4),MMG3D_indElt(mesh,adja[3]/4));
          mmgErr0 = 1;
        }
        return 0;
      }
      pt2 = &mesh->tetra[adj];
      if ( !MG_EOK(pt2) || pt2->ref < 0 ){
        if ( !mmgErr1 ) {
          fprintf(stderr,"\n  ## Error: %s: 4. at least 1 invalid adjacent %d %d\n",
                  __func__,MMG3D_indElt(mesh,adj),MMG3D_indElt(mesh,k));
          fprintf(stderr,"vertices of k   %d: %d %d %d %d\n",MMG3D_indElt(mesh,k),
                  MMG3D_indPt(mesh,pt1->v[0]),MMG3D_indPt(mesh,pt1->v[1]),
                  MMG3D_indPt(mesh,pt1->v[2]),MMG3D_indPt(mesh,pt1->v[3]));
          fprintf(stderr,"vertices of adj %d: %d %d %d %d\n",MMG3D_indElt(mesh,adj),
                  MMG3D_indPt(mesh,pt2->v[0]),MMG3D_indPt(mesh,pt2->v[1]),
                  MMG3D_indPt(mesh,pt2->v[2]),MMG3D_indPt(mesh,pt2->v[3]));
          fprintf(stderr,"adj(%d): %d %d %d %d\n",MMG3D_indElt(mesh,k),
                  MMG3D_indElt(mesh,adja[0]/4),MMG3D_indElt(mesh,adja[1]/4),
                  MMG3D_indElt(mesh,adja[2]/4),MMG3D_indElt(mesh,adja[3]/4));
          mmgErr1 = 1;
        }
        return 0;
      }
      iadr  = (adj-1)*4 + 1;
      adja1 = &mesh->adja[iadr];
      adj1  = adja1[voy] / 4;
      voy1  = adja1[voy] % 4;
      if ( adj1 != k || voy1 != i ) {
        if ( !mmgErr2 ) {
          fprintf(stderr,"\n  ## Error: %s: 2. at least 1 wrong adjacency %d %d\n",
                  __func__,MMG3D_indElt(mesh,k),MMG3D_indElt(mesh,adj1));
          fprintf(stderr,"vertices of %d: %d %d %d %d\n",MMG3D_indElt(mesh,k),
                  MMG3D_indPt(mesh,pt1->v[0]),MMG3D_indPt(mesh,pt1->v[1]),
                  MMG3D_indPt(mesh,pt1->v[2]),MMG3D_indPt(mesh,pt1->v[3]));
          fprintf(stderr,"vertices of adj %d: %d %d %d %d\n",MMG3D_indElt(mesh,adj),
                  MMG3D_indPt(mesh,pt2->v[0]),MMG3D_indPt(mesh,pt2->v[1]),
                  MMG3D_indPt(mesh,pt2->v[2]),MMG3D_indPt(mesh,pt2->v[3]));
          fprintf(stderr,"adj(%d): %d %d %d %d\n",MMG3D_indElt(mesh,k),
                  MMG3D_indElt(mesh,adja[0]/4),MMG3D_indElt(mesh,adja[1]/4),
                  MMG3D_indElt(mesh,adja[2]/4),MMG3D_indElt(mesh,adja[3]/4));
          fprintf(stderr,"adj(%d): %d %d %d %d\n",MMG3D_indElt(mesh,adj),
                  MMG3D_indElt(mesh,adja1[0]/4),MMG3D_indElt(mesh,adja1[1]/4),
                  MMG3D_indElt(mesh,adja1[2]/4),MMG3D_indElt(mesh,adja1[3]/4));
          mmgErr2 = 1;
        }
        return 0;
      }

      a0 = pt1->v[MMG5_idir[i][0]];
      a1 = pt1->v[MMG5_idir[i][1]];
      a2 = pt1->v[MMG5_idir[i][2]];

      b0 = pt2->v[MMG5_idir[voy][0]];
      b1 = pt2->v[MMG5_idir[voy][1]];
      b2 = pt2->v[MMG5_idir[voy][2]];

      if(!(((a0 == b0)&&(a1 == b1)&&(a2 ==b2))||((a0 == b0)&&(a1 == b2)&&(a2 ==b1))\
           || ((a0 == b1)&&(a1 == b0)&&(a2 ==b2)) || ((a0 == b1)&&(a1 == b2)&&(a2 ==b0)) \
           || ((a0 == b2)&&(a1 == b0)&&(a2 ==b1)) || ((a0 == b2)&&(a1 == b1)&&(a2 ==b0)) )){
        if ( !mmgErr3 ) {
          fprintf(stderr,"\n  ## Warning: %s: Inconsistent faces : tetra %d face %d;"
                  " tetra %d face %i \n",__func__,MMG3D_indElt(mesh,k),i,
                  MMG3D_indElt(mesh,adj),voy);
          fprintf(stderr,"Tet 1 : %d %d %d \n",MMG3D_indPt(mesh,a0),
                  MMG3D_indPt(mesh,a1),MMG3D_indPt(mesh,a2));
          fprintf(stderr,"Tet 2 : %d %d %d \n",MMG3D_indPt(mesh,b0),
                  MMG3D_indPt(mesh,b1),MMG3D_indPt(mesh,b2));
          mmgErr3 = 1;
        }
        return 0;
      }
    }
  }

  /* This test may have to be disactivated : check wether boundary faces (i.e. no neighbour)
     arise only when a BDY face is hit */
  for(k=1;k<=mesh->ne;k++){
    pt = &mesh->tetra[k];
    if ( !MG_EOK(pt) || pt->ref < 0 )   continue;
    adja = &mesh->adja[4*(k-1)+1];
    for(i=0;i<4;i++){
      if(!adja[i]){
        if(!pt->xt){
          if ( !mmgErr4 ) {
            mmgErr4 = 1;
            fprintf(stderr,"\n  ## Error: %s: Tetra %d: boundary face"
                    " not tagged: %d \n",__func__,MMG3D_indElt(mesh,k),i);
          }
          return 0;
        }
        else{
          pxt = &mesh->xtetra[pt->xt];
          if(!(pxt->ftag[i] & MG_BDY)){
            if ( !mmgErr4 ) {
              mmgErr4 = 1;
              fprintf(stderr,"\n  ## Error: %s: Tetra %d: boundary face"
                      " not tagged : %d \n",__func__,MMG3D_indElt(mesh,k),i);
            }
            return 0;
          }
        }
      }
    }
  }

  /* Case of an implicit surface embedded in mesh : check whether a face separating two
     different subdomains is tagged bdy */
  for(k=1; k<=mesh->ne; k++){
    pt = &mesh->tetra[k];
    if ( !MG_EOK(pt) || pt->ref < 0 )   continue;

    adja = &mesh->adja[4*(k-1)+1];
    for(i=0; i<4; i++){
      if(!adja[i]) continue;
      iel = adja[i] / 4;
      pt1 = &mesh->tetra[iel];

      if(pt->ref != pt1->ref){
        if(!pt->xt){
          if ( !mmgErr5 ) {
            mmgErr5 = 1;
            fprintf(stderr,"\n  ## Error: %s: Tetra %d face %d: common"
                    " face is a limit of two subdomains"
                    " and has not xt : %d %d %d  \n",__func__,
                    MMG3D_indElt(mesh,k),i,
                    MMG3D_indPt(mesh,pt->v[MMG5_idir[i][0]]),
                    MMG3D_indPt(mesh,pt->v[MMG5_idir[i][1]]),
                    MMG3D_indPt(mesh,pt->v[MMG5_idir[i][2]]));
          }
          return 0;
        }
        else{
          pxt = &mesh->xtetra[pt->xt];
          if(!(pxt->ftag[i] & MG_BDY)){
            if ( !mmgErr5 ) {
              mmgErr5 = 1;
              fprintf(stderr,"\n  ## Error: %s: Tetra %d %d : common"
                      " face is a limit of two subdomains"
                      " and is not tagged %d %d %d -->%d\n",__func__,
                      MMG3D_indElt(mesh,k),i,
                       MMG3D_indElt(mesh,pt->v[MMG5_idir[i][0]]),
                      MMG3D_indPt(mesh,pt->v[MMG5_idir[i][1]]),
                      MMG3D_indPt(mesh,pt->v[MMG5_idir[i][2]]), pxt->ftag[i]);
            }
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

/**
 * Search boundary faces containing point np.
 *
 * \return 0 if fail, 1 otherwise
 *
 * \warning Not used.
 **/
int MMG5_chkptonbdy(MMG5_pMesh mesh,int np){
  MMG5_pTetra      pt;
  MMG5_pxTetra     pxt;
  MMG5_pPoint      p0;
  int              k;
  char             i,j,ip;
  static char      mmgWarn0=0,mmgWarn1=0;

  for(k=1;k<=mesh->np;k++)
    mesh->point[k].flag = 0;

  /* Put flag = 1 at each point belonging to a boundary face */
  for(k=1; k<=mesh->ne; k++){
    pt = &mesh->tetra[k];
    if(!MG_EOK(pt)) continue;
    if(!pt->xt) continue;
    pxt = &mesh->xtetra[pt->xt];
    for(i=0; i<4; i++){
      if(!(pxt->ftag[i] & MG_BDY)) continue;
      for(j=0; j<3; j++){
        ip = MMG5_idir[i][j];
        if(pt->v[ip] == np) {
          if ( !mmgWarn0 ) {
            mmgWarn0 = 1;
            fprintf(stderr,"\n  ## Error: %s: point %d on face %d of tetra %d :"
                   " %d %d %d %d \n",__func__, MMG3D_indPt(mesh,pt->v[ip]),i,
                   MMG3D_indElt(mesh,k), MMG3D_indPt(mesh,pt->v[0]),
                   MMG3D_indPt(mesh,pt->v[1]),
                   MMG3D_indPt(mesh,pt->v[2]), MMG3D_indPt(mesh,pt->v[3]));
          }
        }
        p0 = &mesh->point[pt->v[ip]];
        p0->flag = 1;
      }
    }
  }


  /* Make sure that all the remaining points are not tagged BDY */
  for(k=1; k<=mesh->np; k++){
    p0 = &mesh->point[k];
    if(!MG_VOK(p0)) continue;
    if(p0->flag) continue;
    if(p0->tag & MG_BDY){
      if ( !mmgWarn1 ) {
        mmgWarn1 = 1;
        fprintf(stderr,"\n  ## Error: %s: point %d tagged bdy while belonging to no BDY face\n",
                __func__,MMG3D_indPt(mesh,k));
      }
      return 0;
    }
  }

  return 1;
}

/**
 *
 * Count how many boundary faces share point nump.
 *
 * \warning Not used.
 */
int MMG5_cntbdypt(MMG5_pMesh mesh, int nump){
  MMG5_pTetra  pt;
  MMG5_pxTetra pxt;
  int          k,nf,v0,v1,v2;
  char         i,j,ip;
  static char  mmgWarn0 = 0;

  nf = 0;

  for(k=1; k<=mesh->ne;k++){
    pt = &mesh->tetra[k];
    if(!MG_EOK(pt)) continue;
    if(!pt->xt) continue;
    pxt = &mesh->xtetra[pt->xt];
    for(i=0; i<4; i++){
      if(!(pxt->ftag[i] & MG_BDY)) continue;
      for(j=0; j<3; j++){
        ip = MMG5_idir[i][j];
        if(pt->v[ip] == nump){
          if ( !mmgWarn0 ) {
            mmgWarn0 = 1;
            v0 = pt->v[MMG5_idir[i][0]];
            v1 = pt->v[MMG5_idir[i][0]];
            v2 = pt->v[MMG5_idir[i][0]];

            fprintf(stderr,"\n  ## Error: %s: face %d %d %d in tetra : %d %d %d %d \n",
                   __func__,MMG3D_indPt(mesh,v0),MMG3D_indPt(mesh,v1),
                   MMG3D_indPt(mesh,v2),
                   MMG3D_indPt(mesh,pt->v[0]),MMG3D_indPt(mesh,pt->v[1]),
                   MMG3D_indPt(mesh,pt->v[2]),MMG3D_indPt(mesh,pt->v[3]));
          }
          nf++;
        }
      }
    }
  }
  return nf;
}

/**
 * \param mesh pointer toward the mesh structure.
 *
 * \return 0 if fail, 1 otherwise.
 *
 * Count the number of tetras that have several boundary faces, as well as the
 * number of internal edges connecting points of the boundary.
 *
 */
int MMG5_chkfemtopo(MMG5_pMesh mesh) {
  MMG5_pTetra      pt,pt1;
  MMG5_pxTetra     pxt;
  MMG5_pPoint      p0,p1;
  int              k,nf,ntet,ned,np,ischk,ilist,list[MMG3D_LMAX+2],l,np1,npchk,iel;
  char             i0,j,i,i1,ia,ier;

  ntet = ned = 0;
  for(k=1; k<=mesh->np; k++)
    mesh->point[k].flag = 0;

  /* Count elements with at least two boundary faces */
  for (k=1; k<=mesh->ne; k++) {
    pt = &mesh->tetra[k];
    if ( !MG_EOK(pt) )  continue;
    else if ( !pt->xt ) continue;
    pxt = &mesh->xtetra[pt->xt];

    nf = 0;
    for (i=0; i<4; i++) {
      if ( pxt->ftag[i] & MG_BDY )  nf++;
    }
    if ( nf >= 2 )  ntet++;
  }
  if ( mesh->info.imprim > 0 && ntet )
    printf("  *** %d tetras with at least 2 boundary faces.\n",ntet);

  /* Count internal edges connecting two points of the boundary */
  for (k=1; k<=mesh->ne; k++) {
    pt = &mesh->tetra[k];
    if ( !MG_EOK(pt) )  continue;

    for (i=0; i<4; i++) {
      np = pt->v[i];
      p0 = &mesh->point[np];
      if ( !(p0->tag & MG_BDY) )  continue;

      ischk = p0->flag % 2;
      if ( ischk )  continue;
      p0->flag += 1;

      ilist = MMG5_boulevolp(mesh,k,i,list);
      for (l=0; l<ilist; l++) {
        iel = list[l] / 4;
        i0  = list[l] % 4;
        i1  = i0;

        pt1 = &mesh->tetra[iel];
        for (j=0; j<3; j++) {
          i1  = MMG5_inxt3[i1];
          np1 = pt1->v[i1];
          if ( np1 < np )  continue;
          p1 = &mesh->point[np1];
          if ( !(p1->tag & MG_BDY) )  continue;

          ischk = p1->flag % 2;
          npchk = p1->flag / 2;
          if ( npchk == np )  continue;

          ia = IEDG(i0,i1);
          p1->flag = 2*np + ischk;

          ier = MMG5_srcbdy(mesh,iel,ia);
          if ( ier<0 ) return 0;
          else if ( !ier )  ned++;
        }
      }
    }
  }
  if ( mesh->info.imprim > 0 && ned )
    printf("  *** %d internal edges connecting boundary points.\n",ned);
  return 1;
}

/**
 *
 * Search face n0,n1,n2 in mesh, and get the support tetras, with the
 * corresponding refs.
 *
 * \warning Not used.
 */
int srcface(MMG5_pMesh mesh,int n0,int n1,int n2) {
  MMG5_pTetra  pt;
  MMG5_pxTetra pxt;
  int          k,ip0,ip1,ip2,minn,maxn,sn,mins,maxs,sum,ref;
  int16_t      tag;
  char         i;
  static char  mmgWarn0 = 0;

  minn = MG_MIN(n0,MG_MIN(n1,n2));
  maxn = MG_MAX(n0,MG_MAX(n1,n2));
  sn   = n0 + n1 + n2;
  pxt = 0;

  for(k=1; k<=mesh->ne; k++) {
    pt = &mesh->tetra[k];
    if( !MG_EOK(pt) ) continue;

    if( pt->xt ) pxt = &mesh->xtetra[pt->xt];
    for(i=0; i<4; i++) {
      ip0 = pt->v[MMG5_idir[i][0]];
      ip1 = pt->v[MMG5_idir[i][1]];
      ip2 = pt->v[MMG5_idir[i][2]];

      mins = MG_MIN(ip0,MG_MIN(ip1,ip2));
      maxs = MG_MAX(ip0,MG_MAX(ip1,ip2));
      sum  = ip0 + ip1 + ip2;
      tag  = pt->xt ? pxt->ftag[i] : 0;
      ref  = pt->xt ? pxt->ref[i] : 0;

      if( mins == minn && maxs == maxn && sum == sn ) {
        if ( !mmgWarn0 ) {
          mmgWarn0 = 1;
          fprintf(stderr,"\n  ## Error: %s: Face %d in tetra %d with ref %d:"
                  " corresponding ref %d , tag: %d\n",__func__,i,
                  MMG3D_indElt(mesh,k),pt->ref,ref,tag);
        }
      }
    }
  }


  return 1;
}
