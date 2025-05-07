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
 * \file mmgs/inout_s.c
 * \brief Input / Output Functions.
 * \author Charles Dapogny (UPMC)
 * \author Cécile Dobrzynski (Bx INP/Inria/UBordeaux)
 * \author Pascal Frey (UPMC)
 * \author Algiane Froehly (Inria/UBordeaux)
 * \version 5
 * \copyright GNU Lesser General Public License.
 * \todo doxygen documentation.
 */

#include "mmgs.h"
#include <math.h>

#define sw 4
#define sd 8

int swapbin(int sbin)
{
  int inv;
  char *p_in = (char *) &sbin;
  char *p = (char *)&inv;


  p[0] = p_in[3];
  p[1] = p_in[2];
  p[2] = p_in[1];
  p[3] = p_in[0];

  return inv;
  /*unsigned char c1, c2, c3, c4;

    c1 = sbin & 255;
    c2 = (sbin >> 8) & 255;
    c3 = (sbin >> 16) & 255;
    c4 = (sbin >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;   */

}
float swapf(float sbin)
{
  float out;
  char *p_in = (char *) &sbin;
  char *p_out = (char *) &out;
  p_out[0] = p_in[3];
  p_out[1] = p_in[2];
  p_out[2] = p_in[1];
  p_out[3] = p_in[0];

  return out;
}
double swapd(double sbin)
{
  float out;
  char *p_in = (char *) &sbin;
  char *p_out = (char *) &out;
  int i;

  for(i=0;i<8;i++)
  {
    p_out[i] = p_in[7-i];
  }

  return out;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param filename name of file.
 * \return 0 if failed, 1 otherwise.
 *
 * Read mesh data.
 *
 */
int MMGS_loadMesh(MMG5_pMesh mesh, const char *filename) {
  FILE        *inm;
  MMG5_pTria  pt1,pt2;
  MMG5_pEdge  ped;
  MMG5_pPoint ppt;
  double      *norm,*n,dd;
  float       fc;
  long         posnp,posnt,posne,posncor,posnq,posned,posnr;
  long         posnpreq,posnormal,posnc1;
  int         i,k,ia,nq,nri,ip,idn,ng,npreq;
  int         ncor,bin,iswp,nedreq,ntreq,posnedreq,bdim,binch,bpos;
  int         na,*ina,a,b,ref;
  char        *ptr,*data,chaine[128];


  posnp = posnt = posne = posncor = posnq = 0;
  posned = posnr = posnpreq = posnc1 = npreq = 0;
  posnedreq = posnormal = 0;
  ncor = nri = ng = nedreq = nq = ntreq = 0;
  bin = 0;
  iswp = 0;
  mesh->np = mesh->nt = mesh->nti = mesh->npi = 0;

  MMG5_SAFE_CALLOC(data,strlen(filename)+7,char,return 0);

  strcpy(data,filename);
  ptr = strstr(data,".mesh");
  if ( !ptr ) {
    /* data contains the filename without extension */
    strcat(data,".meshb");
    if( !(inm = fopen(data,"rb")) ) {
      /* our file is not a .meshb file, try with .mesh ext */
      ptr = strstr(data,".mesh");
      *ptr = '\0';
      strcat(data,".mesh");
      if( !(inm = fopen(data,"rb")) ) {
        fprintf(stderr,"  ** %s  NOT FOUND.\n",data);
        MMG5_SAFE_FREE(data);
        return 0;
      }
    }
    else  bin = 1;
  }
  else {
    ptr = strstr(data,".meshb");
    if ( ptr )  bin = 1;
      if( !(inm = fopen(data,"rb")) ) {
        fprintf(stderr,"  ** %s  NOT FOUND.\n",data);
        MMG5_SAFE_FREE(data);
        return 0;
      }
  }

  if ( mesh->info.imprim >= 0 )
    fprintf(stdout,"  %%%% %s OPENED\n",data);
  MMG5_SAFE_FREE(data);

  if (!bin) {
    strcpy(chaine,"D");
    while(fscanf(inm,"%127s",&chaine[0])!=EOF && strncmp(chaine,"End",strlen("End")) ) {
      if(!strncmp(chaine,"MeshVersionFormatted",strlen("MeshVersionFormatted"))) {
        fscanf(inm,"%d",&mesh->ver);
        continue;
      } else if(!strncmp(chaine,"Dimension",strlen("Dimension"))) {
        fscanf(inm,"%d",&mesh->dim);
        if(mesh->dim!=3) {
          fprintf(stderr,"BAD DIMENSION : %d\n",mesh->dim);
          return 0;
        }
        continue;
      } else if(!strncmp(chaine,"Vertices",strlen("Vertices"))) {
        fscanf(inm,"%d",&mesh->npi);
        posnp = ftell(inm);
        continue;
      } else if(!strncmp(chaine,"RequiredVertices",strlen("RequiredVertices"))) {
        fscanf(inm,"%d",&npreq);
        posnpreq = ftell(inm);
        continue;
      } else if(!strncmp(chaine,"Triangles",strlen("Triangles"))) {
        fscanf(inm,"%d",&mesh->nti);
        posnt = ftell(inm);
        continue;
      }
      else if(!strncmp(chaine,"RequiredTriangles",strlen("RequiredTriangles"))) {
        fscanf(inm,"%d",&ntreq);
        fprintf(stderr,"\n  ## Warning: %s: required triangles not yet implementd in mmgs.\n"
                " %d required triangle(s) ignored.\n",__func__,ntreq);
        continue;
      } else if(!strncmp(chaine,"Quadrilaterals",strlen("Quadrilaterals"))) {
        fscanf(inm,"%d",&nq);
        posnq = ftell(inm);
        continue;
      } else if(!strncmp(chaine,"Corners",strlen("Corners"))) {
        fscanf(inm,"%d",&ncor);
        posncor = ftell(inm);
        continue;
      } else if(!strncmp(chaine,"Edges",strlen("Edges"))) {
        fscanf(inm,"%d",&mesh->na);
        posned = ftell(inm);
        continue;
      } else if(!strncmp(chaine,"RequiredEdges",strlen("RequiredEdges"))) {
        fscanf(inm,"%d",&nedreq);
        posnedreq = ftell(inm);
        continue;
      } else if(!strncmp(chaine,"Ridges",strlen("Ridges"))) {
        fscanf(inm,"%d",&nri);
        posnr = ftell(inm);
        continue;
      } else if(!ng && !strncmp(chaine,"Normals",strlen("Normals"))) {
        fscanf(inm,"%d",&ng);
        posnormal = ftell(inm);
        continue;
      } else if(!strncmp(chaine,"NormalAtVertices",strlen("NormalAtVertices"))) {
        fscanf(inm,"%d",&mesh->nc1);
        posnc1 = ftell(inm);
        continue;
      }
    }
  } else { //binary file
    bdim = 0;
    fread(&mesh->ver,sw,1,inm);
    iswp=0;
    if(mesh->ver==16777216)
      iswp=1;
    else if(mesh->ver!=1) {
      fprintf(stdout,"BAD FILE ENCODING\n");
    }
    fread(&mesh->ver,sw,1,inm);
    if(iswp) mesh->ver = swapbin(mesh->ver);
    while(fread(&binch,sw,1,inm)!=0 && binch!=54 ) {
      if(iswp) binch=swapbin(binch);
      if(binch==54) break;
      if(!bdim && binch==3) {  //Dimension
        fread(&bdim,sw,1,inm);  //NulPos=>20
        if(iswp) bdim=swapbin(bdim);
        fread(&bdim,sw,1,inm);
        if(iswp) bdim=swapbin(bdim);
        mesh->dim = bdim;
        if(bdim!=3) {
          fprintf(stderr,"BAD MESH DIMENSION : %d\n",mesh->dim);
          return -1;
        }
        continue;
      } else if(!mesh->npi && binch==4) {  //Vertices
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&mesh->npi,sw,1,inm);
        if(iswp) mesh->npi=swapbin(mesh->npi);
        posnp = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(binch==15) {  //RequiredVertices
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&npreq,sw,1,inm);
        if(iswp) npreq=swapbin(npreq);
        posnpreq = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(!mesh->nti && binch==6) {//Triangles
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&mesh->nti,sw,1,inm);
        if(iswp) mesh->nti=swapbin(mesh->nti);
        posnt = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(binch==7) {//Quadrilaterals
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&nq,sw,1,inm);
        if(iswp) nq=swapbin(nq);
        posnq = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(!ncor && binch==13) { //Corners
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&ncor,sw,1,inm);
        if(iswp) ncor=swapbin(ncor);
        posncor = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(!mesh->na && binch==5) { //Edges
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&mesh->na,sw,1,inm);
        if(iswp) mesh->na=swapbin(mesh->na);
        posned = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(binch==16) {  //RequiredEdges
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&nedreq,sw,1,inm);
        if(iswp) nedreq=swapbin(nedreq);
        posnedreq = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(binch==14) {  //Ridges
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&nri,sw,1,inm);
        if(iswp) nri=swapbin(nri);
        posnr = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(!ng && binch==60) {  //Normals
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&ng,sw,1,inm);
        if(iswp) ng=swapbin(ng);
        posnormal = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else if(binch==20) {  //NormalAtVertices
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        fread(&mesh->nc1,sw,1,inm);
        if(iswp) mesh->nc1=swapbin(mesh->nc1);
        posnc1 = ftell(inm);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
        continue;
      } else {
        fread(&bpos,sw,1,inm); //NulPos
        if(iswp) bpos=swapbin(bpos);
        rewind(inm);
        fseek(inm,bpos,SEEK_SET);
      }
    }
  }

  if ( !mesh->npi || !mesh->nti ) {
    fprintf(stdout,"  ** MISSING DATA\n");
    return 0;
  }
  mesh->np = mesh->npi;
  mesh->nt = mesh->nti + 2*nq;

  /* mem alloc */
  if ( !MMGS_zaldy(mesh) )  return 0;

  /* read vertices */

  rewind(inm);
  fseek(inm,posnp,SEEK_SET);
  for (k=1; k<=mesh->np; k++) {
    ppt = &mesh->point[k];
    if (mesh->ver < 2) { /*float*/
      if (!bin) {
        for (i=0 ; i<3 ; i++) {
          fscanf(inm,"%f",&fc);
          ppt->c[i] = (double) fc;
        }
        fscanf(inm,"%d",&ppt->ref);
      } else {
        for (i=0 ; i<3 ; i++) {
          fread(&fc,sw,1,inm);
          if(iswp) fc=swapf(fc);
          ppt->c[i] = (double) fc;
        }
        fread(&ppt->ref,sw,1,inm);
        if(iswp) ppt->ref=swapbin(ppt->ref);
      }
    } else {
      if (!bin)
        fscanf(inm,"%lf %lf %lf %d",&ppt->c[0],&ppt->c[1],&ppt->c[2],&ppt->ref);
      else {
        for (i=0 ; i<3 ; i++) {
          fread(&ppt->c[i],sd,1,inm);
          if(iswp) ppt->c[i]=swapd(ppt->c[i]);
        }
        fread(&ppt->ref,sw,1,inm);
        if(iswp) ppt->ref=swapbin(ppt->ref);
      }
    }
    ppt->tag = MG_NUL;
  }

  /* read triangles and set seed */
  rewind(inm);
  fseek(inm,posnt,SEEK_SET);
  for (k=1; k<=mesh->nt; k++) {
    pt1 = &mesh->tria[k];
    if (!bin)
      fscanf(inm,"%d %d %d %d",&pt1->v[0],&pt1->v[1],&pt1->v[2],&pt1->ref);
    else {
      for (i=0 ; i<3 ; i++) {
        fread(&pt1->v[i],sw,1,inm);
        if(iswp) pt1->v[i]=swapbin(pt1->v[i]);
      }
      fread(&pt1->ref,sw,1,inm);
      if(iswp) pt1->ref=swapbin(pt1->ref);
    }
    for (i=0; i<3; i++) {
      ppt = &mesh->point[pt1->v[i]];
      ppt->tag &= ~MG_NUL;
    }
  }
  /* read quads */
  if ( nq > 0 ) {
    rewind(inm);
    fseek(inm,posnq,SEEK_SET);

    for (k=1; k<=nq; k++) {
      mesh->nti++;
      pt1 = &mesh->tria[mesh->nti];
      mesh->nti++;
      pt2 = &mesh->tria[mesh->nti];

      if (!bin)
        fscanf(inm,"%d %d %d %d %d",&pt1->v[0],&pt1->v[1],&pt1->v[2],&pt2->v[2],&pt1->ref);
      else {
        for (i=0 ; i<3 ; i++) {
          fread(&pt1->v[i],sw,1,inm);
          if(iswp) pt1->v[i]=swapbin(pt1->v[i]);
        }
        fread(&pt2->v[2],sw,1,inm);
        if(iswp) pt2->v[2]=swapbin(pt2->v[2]);
        fread(&pt1->ref,sw,1,inm);
        if(iswp) pt1->ref=swapbin(pt1->ref);
      }
      pt2->v[0] = pt1->v[0];
      pt2->v[1] = pt1->v[2];
      pt2->ref  = pt1->ref;
      for (i=0; i<3; i++) {
        ppt = &mesh->point[pt1->v[i]];
        ppt->tag &= ~MG_NUL;
        ppt = &mesh->point[pt2->v[i]];
        ppt->tag &= ~MG_NUL;
      }
    }
    mesh->nt = mesh->nti;
  }

  if(ncor) {
    rewind(inm);
    fseek(inm,posncor,SEEK_SET);
    for (k=1; k<=ncor; k++) {
      if(!bin)
        fscanf(inm,"%d",&i);
      else {
        fread(&i,sw,1,inm);
        if(iswp) i=swapbin(i);
      }
      if(i>mesh->np) {
        fprintf(stderr,"\n  ## Warning: %s: corner number %8d ignored.\n",
                __func__,i);
      } else {
        ppt = &mesh->point[i];
        ppt->tag |= MG_CRN;
      }
    }
  }

  /* read required vertices */
  if(npreq) {
    rewind(inm);
    fseek(inm,posnpreq,SEEK_SET);
    for (k=1; k<=npreq; k++) {
      if(!bin)
        fscanf(inm,"%d",&i);
      else {
        fread(&i,sw,1,inm);
        if(iswp) i=swapbin(i);
      }
      if(i>mesh->np) {
        fprintf(stderr,"\n  ## Warning: %s: required Vertices number %8d ignored\n",
                __func__,i);
      } else {
        ppt = &mesh->point[i];
        ppt->tag |= MG_REQ;
      }
    }
  }

  /* Read mesh edges */
  na = mesh->na;
  if ( mesh->na ) {
    rewind(inm);
    fseek(inm,posned,SEEK_SET);

    MMG5_ADD_MEM(mesh,(mesh->na+1)*sizeof(MMG5_Edge),"initial edges",return 0);
    MMG5_SAFE_CALLOC(mesh->edge,mesh->na+1,MMG5_Edge,return 0);

    /* Skip edges with MG_ISO refs */
    if( mesh->info.iso ) {
      mesh->na = 0;
      MMG5_SAFE_CALLOC(ina,na+1,int,return 0);

      for (k=1; k<=na; k++) {
        if (!bin)
          fscanf(inm,"%d %d %d",&a,&b,&ref);
        else {
          fread(&a,sw,1,inm);
          if(iswp) a=swapbin(a);
          fread(&b,sw,1,inm);
          if(iswp) b=swapbin(b);
          fread(&ref,sw,1,inm);
          if(iswp) ref=swapbin(ref);
        }
        if ( abs(ref) != MG_ISO ) {
          ped = &mesh->edge[++mesh->na];
          ped->a   = a;
          ped->b   = b;
          ped->ref = ref;
          ina[k]   = mesh->na;
        }
        else {
          /* Remove MG_REQ and MG_CRN tags on ISO edges */
          if ( MG_REQ & mesh->point[a].tag ) { mesh->point[a].tag &= ~MG_REQ; }
          if ( MG_REQ & mesh->point[b].tag ) { mesh->point[b].tag &= ~MG_REQ; }
          if ( MG_CRN & mesh->point[a].tag ) { mesh->point[a].tag &= ~MG_CRN; }
          if ( MG_CRN & mesh->point[b].tag ) { mesh->point[b].tag &= ~MG_CRN; }
        }
      }
      if( !mesh->na )
        MMG5_DEL_MEM(mesh,mesh->edge);

      else if ( mesh->na < na ) {
        MMG5_ADD_MEM(mesh,(mesh->na-na)*sizeof(MMG5_Edge),"edges",
                      fprintf(stderr,"  Exit program.\n");
                      MMG5_SAFE_FREE(ina);
                      return 0);
        MMG5_SAFE_RECALLOC(mesh->edge,na+1,(mesh->na+1),MMG5_Edge,"Edges",return 0);
      }
    }
    else {
      for (k=1; k<=mesh->na; k++) {
        if (!bin)
          fscanf(inm,"%d %d %d",&mesh->edge[k].a,&mesh->edge[k].b,&mesh->edge[k].ref);
        else {
          fread(&mesh->edge[k].a,sw,1,inm);
          if(iswp) mesh->edge[k].a=swapbin(mesh->edge[k].a);
          fread(&mesh->edge[k].b,sw,1,inm);
          if(iswp) mesh->edge[k].b=swapbin(mesh->edge[k].b);
          fread(&mesh->edge[k].ref,sw,1,inm);
          if(iswp) mesh->edge[k].ref=swapbin(mesh->edge[k].ref);
        }
        mesh->edge[k].tag |= MG_REF;
        mesh->point[mesh->edge[k].a].tag |= MG_REF;
        mesh->point[mesh->edge[k].b].tag |= MG_REF;
      }
    }

    /* get ridges */
    if ( nri ) {
      rewind(inm);
      fseek(inm,posnr,SEEK_SET);
      for (k=1; k<=nri; k++) {
        if(!bin)
          fscanf(inm,"%d",&ia);
        else {
          fread(&ia,sw,1,inm);
          if(iswp) ia=swapbin(ia);
        }
        if ( (ia>na) || (ia<0) ) {
          fprintf(stderr,"\n  ## Warning: %s: ridge number %8d ignored.\n",
                  __func__,ia);
        }
        else {
          if( mesh->info.iso ){
            if( ina[ia] == 0 ) continue;
            else
              mesh->edge[ina[ia]].tag |= MG_GEO;
          }
          else {
            mesh->edge[ia].tag |= MG_GEO;
          }
        }
      }
    }

    if ( nedreq ) {
      rewind(inm);
      fseek(inm,posnedreq,SEEK_SET);
      for (k=1; k<=nedreq; k++) {
        if(!bin)
          fscanf(inm,"%d",&ia);
        else {
          fread(&ia,sw,1,inm);
          if(iswp) ia=swapbin(ia);
        }
        if ( (ia>na) || (ia<0) ) {
          fprintf(stderr,"\n  ## Warning: %s: required edge number %8d ignored\n",
                  __func__,ia);
        }
        else {
          if( mesh->info.iso ){
            if( ina[ia] == 0 ) continue;
            else
              mesh->edge[ina[ia]].tag |= MG_REQ;
          }
          else
            mesh->edge[ia].tag |= MG_REQ;
        }
      }
    }
    if ( mesh->info.iso )
      MMG5_SAFE_FREE(ina);
  }

  /* read geometric entities */
  if ( mesh->nc1 && !ng ) {
    fprintf(stderr,"\n  ## Warning: %s: your mesh don't contains Normals but contains"
            " NormalAtVertices. The NormalAtVertices are deleted. \n",__func__);
    mesh->nc1 = 0;
  }

  if ( ng > 0 ) {
    MMG5_SAFE_CALLOC(norm,3*ng+1,double,return 0);

    rewind(inm);
    fseek(inm,posnormal,SEEK_SET);
    for (k=1; k<=ng; k++) {
      n = &norm[3*(k-1)+1];
      if ( mesh->ver == 1 ) {
        if (!bin) {
          for (i=0 ; i<3 ; i++) {
            fscanf(inm,"%f",&fc);
            n[i] = (double) fc;
          }
        } else {
          for (i=0 ; i<3 ; i++) {
            fread(&fc,sw,1,inm);
            if(iswp) fc=swapf(fc);
            n[i] = (double) fc;
          }
        }
      }
      else {
        if (!bin)
          fscanf(inm,"%lf %lf %lf",&n[0],&n[1],&n[2]);
        else {
          for (i=0 ; i<3 ; i++) {
            fread(&n[i],sd,1,inm);
            if(iswp) n[i]=swapd(n[i]);
          }
        }
      }
      dd = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
      if ( dd > MMG5_EPSD2 ) {
        dd = 1.0 / sqrt(dd);
        n[0] *= dd;
        n[1] *= dd;
        n[2] *= dd;
      }
    }

    rewind(inm);
    fseek(inm,posnc1,SEEK_SET);

    for (k=1; k<=mesh->nc1; k++) {
      if (!bin)
        fscanf(inm,"%d %d",&ip,&idn);
      else {
        fread(&ip,sw,1,inm);
        if(iswp) ip=swapbin(ip);
        fread(&idn,sw,1,inm);
        if(iswp) idn=swapbin(idn);
      }
      if ( idn > 0 && ip < mesh->np+1 )
        memcpy(&mesh->point[ip].n,&norm[3*(idn-1)+1],3*sizeof(double));
    }
    MMG5_SAFE_FREE(norm);
  }

  if ( abs(mesh->info.imprim) > 4 ) {
    fprintf(stdout,"     NUMBER OF VERTICES   %8d / %8d   CORNERS/REQ. %d / %d\n",mesh->npi,mesh->npmax,ncor,npreq);
    fprintf(stdout,"     NUMBER OF TRIANGLES  %8d / %8d\n",mesh->nti,mesh->ntmax);

    if ( mesh->na )
      fprintf(stdout,"     NUMBER OF EDGES      %8d  RIDGES %6d\n",mesh->na,nri);
  }
  fclose(inm);
  return 1;
}

int MMGS_loadMshMesh(MMG5_pMesh mesh,MMG5_pSol sol,const char *filename) {
  FILE*       inm;
  long        posNodes,posElts,*posNodeData;
  int         ier,bin,iswp,nelts,nsols;

  mesh->dim = 3;

  ier = MMG5_loadMshMesh_part1(mesh,filename,&inm,
                               &posNodes,&posElts,&posNodeData,
                               &bin,&iswp,&nelts,&nsols);
  if ( ier < 1 )  return (ier);

  if ( nsols > 1 ) {
    fprintf(stderr,"SEVERAL SOLUTION => IGNORED: %d\n",nsols);
    fclose(inm);
    MMG5_SAFE_FREE(posNodeData);
    return -1;
  }

  if ( !MMGS_zaldy(mesh) ) {
    fclose(inm);
    MMG5_SAFE_FREE(posNodeData);
    return 0;
  }

  mesh->ne = mesh->nprism = 0;

  if ( !mesh->nt ) {
    fprintf(stderr,"  ** MISSING DATA.\n");
    fprintf(stderr," Check that your mesh contains triangles.\n");
    fprintf(stderr," Exit program.\n");
    fclose(inm);
    MMG5_SAFE_FREE(posNodeData);
    return -1;
  }


  if (mesh->npmax < mesh->np || mesh->ntmax < mesh->nt ) {
    fclose(inm);
    MMG5_SAFE_FREE(posNodeData);
    return -1;
  }

  ier = MMG5_loadMshMesh_part2( mesh, &sol,&inm,
                                posNodes,posElts,posNodeData,
                                bin,iswp,nelts,nsols);
  MMG5_SAFE_FREE(posNodeData);
  if ( ier < 1 ) {
    fprintf(stderr,"  ** ERROR WHEN PARSING THE INPUT FILE\n");
    return  ier;
  }

  /* Check the metric type */
  ier = MMG5_chkMetricType(mesh,&sol->type,inm);

  return ier;
}


int MMGS_loadMshMesh_and_allData(MMG5_pMesh mesh,MMG5_pSol *sol,const char *filename) {
  FILE*       inm;
  long        posNodes,posElts,*posNodeData;
  int         ier,bin,iswp,nelts,nsols;

  mesh->dim = 3;

  ier = MMG5_loadMshMesh_part1(mesh,filename,&inm,
                               &posNodes,&posElts,&posNodeData,
                               &bin,&iswp,&nelts,&nsols);
  if ( ier < 1 )  return (ier);

  mesh->nsols = nsols;
  if ( *sol )  MMG5_DEL_MEM(mesh,*sol);
  MMG5_ADD_MEM(mesh,nsols*sizeof(MMG5_Sol),"solutions array",
                printf("  Exit program.\n"); fclose(inm);
                MMG5_SAFE_FREE(posNodeData);
                return -1);
  MMG5_SAFE_CALLOC(*sol,nsols,MMG5_Sol,return -1);

  if ( !MMGS_zaldy(mesh) ) {
    fclose(inm);
    MMG5_SAFE_FREE(posNodeData);
    return 0;
  }

  mesh->ne = mesh->nprism = 0;

  if ( !mesh->nt ) {
    fprintf(stderr,"  ** MISSING DATA.\n");
    fprintf(stderr," Check that your mesh contains triangles.\n");
    fprintf(stderr," Exit program.\n");
    fclose(inm);
    MMG5_SAFE_FREE(posNodeData);
    return -1;
  }


  if (mesh->npmax < mesh->np || mesh->ntmax < mesh->nt ) {
    fclose(inm);
    MMG5_SAFE_FREE(posNodeData);
    return -1;
  }

  ier = MMG5_loadMshMesh_part2( mesh, sol,&inm,
                                posNodes,posElts,posNodeData,
                                bin,iswp,nelts,nsols);

  if ( ier < 1 ) {
    fprintf(stderr,"  ** ERROR WHEN PARSING THE INPUT FILE\n");
  }

  MMG5_SAFE_FREE(posNodeData);

  return ier;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param filename name of file.
 * \return 0 if failed, 1 otherwise.
 *
 * Save mesh data.
 *
 * \warning you must call the \a MMGS_packMesh function before saving your
 * mesh.
 */

int MMGS_saveMesh(MMG5_pMesh mesh, const char* filename) {
  FILE         *inm;
  MMG5_pPoint  ppt;
  MMG5_pTria   pt;
  MMG5_pxPoint go;
  int          k,np,nt,nc,ng,nn,nr,nre;
  int          bin,binch,bpos;
  // int          outm;
  char         *data,*ptr,chaine[128];

  mesh->ver = 2;

  bin = 0;

  MMG5_SAFE_CALLOC(data,strlen(filename)+7,char,return 0);

  strcpy(data,filename);
  ptr = strstr(data,".mesh");
  if ( !ptr ) {
    strcat(data,".meshb");
    if( !(inm = fopen(data,"wb")) ) {
      ptr  = strstr(data,".mesh");
      *ptr = '\0';
      strcat(data,".mesh");
      if( !(inm = fopen(data,"w")) ) {
        fprintf(stderr,"  ** UNABLE TO OPEN %s.\n",data);
        MMG5_SAFE_FREE(data);
        return 0;
      }
    } else {
      bin = 1;
    }
  }
  else {
    ptr = strstr(data,".meshb");
    if( ptr ) {
      bin = 1;
      if( !(inm = fopen(data,"wb")) ) {
        fprintf(stderr,"  ** UNABLE TO OPEN %s.\n",data);
        MMG5_SAFE_FREE(data);
        return 0;
      }
    } else {
      if( !(inm = fopen(data,"w")) ) {
        fprintf(stderr,"  ** UNABLE TO OPEN %s.\n",data);
        MMG5_SAFE_FREE(data);
        return 0;
      }
    }
  }
  if ( mesh->info.imprim >= 0 )
    fprintf(stdout,"  %%%% %s OPENED\n",data);
  MMG5_SAFE_FREE(data);

  /*entete fichier*/
  if(!bin) {
    strcpy(&chaine[0],"MeshVersionFormatted 2\n");
    fprintf(inm,"%s",chaine);
    strcpy(&chaine[0],"\n\nDimension 3\n");
    fprintf(inm,"%s ",chaine);
  } else {
    binch = 1; //MeshVersionFormatted
    fwrite(&binch,sw,1,inm);
    binch = 2; //version
    fwrite(&binch,sw,1,inm);
    binch = 3; //Dimension
    fwrite(&binch,sw,1,inm);
    bpos = 20; //Pos
    fwrite(&bpos,sw,1,inm);
    binch = 3;
    fwrite(&binch,sw,1,inm);

  }
  /* vertices */
  np = nc = ng = nn = nre = 0;
  for (k=1; k<=mesh->np; k++) {
    ppt = &mesh->point[k];
    ppt->tmp = 0;
    if ( MG_VOK(ppt) ) {
      np++;
      ppt->tmp = np;
      if ( ppt->tag & MG_CRN )  nc++;
      if ( ppt->tag & MG_REQ )  nre++;
      if ( MG_EDG(ppt->tag) )   ng++;
    }
  }

  if(!bin) {
    strcpy(&chaine[0],"\n\nVertices\n");
    fprintf(inm,"%s",chaine);
    fprintf(inm,"%d\n",np);
  } else {
    binch = 4; //Vertices
    fwrite(&binch,sw,1,inm);
    bpos += 12+(1+3*mesh->ver)*4*np; //NullPos
    fwrite(&bpos,sw,1,inm);
    fwrite(&np,sw,1,inm);
  }
  for (k=1; k<=mesh->np; k++) {
    ppt = &mesh->point[k];
    if ( MG_VOK(ppt) ) {
      if(!bin) {
        fprintf(inm,"%.15lg %.15lg %.15lg %d\n",ppt->c[0],ppt->c[1],ppt->c[2],abs(ppt->ref));
      } else {
        fwrite((unsigned char*)&ppt->c[0],sd,1,inm);
        fwrite((unsigned char*)&ppt->c[1],sd,1,inm);
        fwrite((unsigned char*)&ppt->c[2],sd,1,inm);
        ppt->ref = abs(ppt->ref);
        fwrite((unsigned char*)&ppt->ref,sw,1,inm);
      }
      if ( !((ppt->tag & MG_GEO) || ppt->tag & MG_CRN) )  nn++;
    }
  }

  nt = 0;
  for (k=1; k<=mesh->nt; k++) {
    pt = &mesh->tria[k];
    if ( !MG_EOK(pt) ) continue;
    nt++;
  }

  /* write corners */
  if ( nc ) {
    if(!bin) {
      strcpy(&chaine[0],"\n\nCorners\n");
      fprintf(inm,"%s",chaine);
      fprintf(inm,"%d\n",nc);
    } else {
      binch = 13; //
      fwrite(&binch,sw,1,inm);
      bpos += 12+4*nc; //NullPos
      fwrite(&bpos,sw,1,inm);
      fwrite(&nc,sw,1,inm);
    }
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( MG_VOK(ppt) && ppt->tag & MG_CRN ) {
        if(!bin) {
          fprintf(inm,"%d\n",ppt->tmp);
        } else {
          fwrite(&ppt->tmp,sw,1,inm);
        }
      }
    }
  }
  if ( nre ) {
    if(!bin) {
      strcpy(&chaine[0],"\n\nRequiredVertices\n");
      fprintf(inm,"%s",chaine);
      fprintf(inm,"%d\n",nre);
    } else {
      binch = 15; //
      fwrite(&binch,sw,1,inm);
      bpos += 12+4*nre; //NullPos
      fwrite(&bpos,sw,1,inm);
      fwrite(&nre,sw,1,inm);
    }
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( MG_VOK(ppt) && ppt->tag & MG_REQ ) {
        if(!bin) {
          fprintf(inm,"%d\n",ppt->tmp);
        } else {
          fwrite(&ppt->tmp,sw,1,inm);
        }
      }
    }
  }

  /* write edges, ridges */
  if ( mesh->na ) {
    if(!bin) {
      strcpy(&chaine[0],"\n\nEdges\n");
      fprintf(inm,"%s",chaine);
      fprintf(inm,"%d\n",mesh->na);
    } else {
      binch = 5; //Edges
      fwrite(&binch,sw,1,inm);
      bpos += 12 + 3*4*mesh->na;//Pos
      fwrite(&bpos,sw,1,inm);
      fwrite(&mesh->na,sw,1,inm);
    }
    nre = nr = 0;
    for (k=1; k<=mesh->na; k++) {
      if(!bin) {
        fprintf(inm,"%d %d %d\n",
                mesh->edge[k].a,mesh->edge[k].b,mesh->edge[k].ref);
      } else {
        fwrite(&mesh->edge[k].a,sw,1,inm);
        fwrite(&mesh->edge[k].b,sw,1,inm);
        fwrite(&mesh->edge[k].ref,sw,1,inm);
      }
      if ( mesh->edge[k].tag & MG_REQ )  nre++;
      if ( mesh->edge[k].tag & MG_GEO )  nr++;
    }
    /* ridges */
    if ( nr ) {
      if(!bin) {
        strcpy(&chaine[0],"\n\nRidges\n");
        fprintf(inm,"%s",chaine);
        fprintf(inm,"%d\n",nr);
      } else {
        binch = 14; //Ridges
        fwrite(&binch,sw,1,inm);
        bpos += 12 + 4*nr;//Pos
        fwrite(&bpos,sw,1,inm);
        fwrite(&nr,sw,1,inm);
      }
      for (k=1; k<=mesh->na; k++) {
        if ( mesh->edge[k].tag & MG_GEO ) {
          if(!bin) {
            fprintf(inm,"%d\n",k);
          } else {
            fwrite(&k,sw,1,inm);
          }
        }
      }
    }
    if ( nre ) {
      if(!bin) {
        strcpy(&chaine[0],"\n\nRequiredEdges\n");
        fprintf(inm,"%s",chaine);
        fprintf(inm,"%d\n",nre);
      } else {
        binch = 16; //RequiredEdges
        fwrite(&binch,sw,1,inm);
        bpos += 12 + 4*nre;//Pos
        fwrite(&bpos,sw,1,inm);
        fwrite(&nre,sw,1,inm);
      }
      for (k=1; k<=mesh->na; k++)
        if ( mesh->edge[k].tag & MG_REQ )  {
          if(!bin) {
            fprintf(inm,"%d\n",k);
          } else {
            fwrite(&k,sw,1,inm);
          }
        }
    }
  }

  /* write triangles */
  if(!bin) {
    strcpy(&chaine[0],"\n\nTriangles\n");
    fprintf(inm,"%s",chaine);
    fprintf(inm,"%d\n",nt);
  } else {
    binch = 6; //Triangles
    fwrite(&binch,sw,1,inm);
    bpos += 12+16*nt; //Pos
    fwrite(&bpos,sw,1,inm);
    fwrite(&nt,sw,1,inm);
  }

  for (k=1; k<=mesh->nt; k++) {
    pt = &mesh->tria[k];
    if ( MG_EOK(pt) ) {
      if(!bin) {
        fprintf(inm,"%d %d %d %d\n",mesh->point[pt->v[0]].tmp,mesh->point[pt->v[1]].tmp
                ,mesh->point[pt->v[2]].tmp,abs(pt->ref));
      } else {
        fwrite(&mesh->point[pt->v[0]].tmp,sw,1,inm);
        fwrite(&mesh->point[pt->v[1]].tmp,sw,1,inm);
        fwrite(&mesh->point[pt->v[2]].tmp,sw,1,inm);
        pt->ref = abs(pt->ref);
        fwrite(&pt->ref,sw,1,inm);
      }
    }
  }


  if ( (!mesh->xp) || (!mesh->xpoint) ) nn = ng = 0;

  /* write normals */
  if ( nn ) {
    if(!bin) {
      strcpy(&chaine[0],"\n\nNormals\n");
      fprintf(inm,"%s",chaine);
      fprintf(inm,"%d\n",nn);
    } else {
      binch = 60; //normals
      fwrite(&binch,sw,1,inm);
      bpos += 12+(3*mesh->ver)*4*nn; //Pos
      fwrite(&bpos,sw,1,inm);
      fwrite(&nn,sw,1,inm);
    }
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( !MG_VOK(ppt) )  continue;
      else if ( !(ppt->tag & MG_GEO) && !(ppt->tag & MG_CRN) ) {
        if ( ppt->tag & MG_REF ) {
          go = &mesh->xpoint[ppt->xp];
          if(!bin) {
            fprintf(inm,"%.15lg %.15lg %.15lg \n",go->n1[0],go->n1[1],go->n1[2]);
          } else {
            fwrite((unsigned char*)&go->n1[0],sd,1,inm);
            fwrite((unsigned char*)&go->n1[1],sd,1,inm);
            fwrite((unsigned char*)&go->n1[2],sd,1,inm);
          }
        }
        else {
          if(!bin) {
            fprintf(inm,"%.15lg %.15lg %.15lg \n",ppt->n[0],ppt->n[1],ppt->n[2]);
          } else {
            fwrite((unsigned char*)&ppt->n[0],sd,1,inm);
            fwrite((unsigned char*)&ppt->n[1],sd,1,inm);
            fwrite((unsigned char*)&ppt->n[2],sd,1,inm);
          }
        }
      }
    }
    if(!bin) {
      strcpy(&chaine[0],"\n\nNormalAtVertices\n");
      fprintf(inm,"%s",chaine);
      fprintf(inm,"%d\n",nn);
    } else {
      binch = 20; //normalatvertices
      fwrite(&binch,sw,1,inm);
      bpos += 12 + 2*4*nn;//Pos
      fwrite(&bpos,sw,1,inm);
      fwrite(&nn,sw,1,inm);
    }
    nn = 0;
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( MG_VOK(ppt) && !((ppt->tag & MG_GEO) || (ppt->tag & MG_CRN) ) ) {
        if(!bin) {
          fprintf(inm,"%d %d\n",ppt->tmp,++nn);
        } else {
          fwrite(&ppt->tmp,sw,1,inm);
          ++nn;
          fwrite(&nn,sw,1,inm);
        }
      }
    }
/*
  nn = 0;
  for (k=1; k<=mesh->nt; k++) {
  pt = &mesh->tria[k];
  if ( !MG_EOK(pt) )  continue;
  for (i=0; i<3; i++) {
  ppt = &mesh->point[pt->v[i]];
  if ( ppt->tag & MG_GEO )  nn++;
  }
  }
  GmfSetKwd(outm,GmfNormalAtTriangleVertices,nn);
  for (k=1; k<=mesh->nt; k++) {
  pt = &mesh->tria[k];
  if ( !MG_EOK(pt) )  continue;
  for (i=0; i<3; i++) {
  ppt = &mesh->point[pt->v[i]];
  if ( ppt->tag & MG_GEO )  nn++;
  }
*/
  }

  /* write tangents (only if we have already analyzed the mesh =>
   * mesh->xpoint is allocated ) */
  if ( ng && mesh->xpoint ) {
    /* Write tangents */
    if(!bin) {
      strcpy(&chaine[0],"\n\nTangents\n");
      fprintf(inm,"%s",chaine);
      fprintf(inm,"%d\n",ng);
    } else {
      binch = 59; //tangent
      fwrite(&binch,sw,1,inm);
      bpos += 12+(3*mesh->ver)*4*ng; //Pos
      fwrite(&bpos,sw,1,inm);
      fwrite(&ng,sw,1,inm);
    }
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( MG_VOK(ppt) && MG_EDG(ppt->tag) ) {
        if(!bin) {
          fprintf(inm,"%.15lg %.15lg %.15lg \n",ppt->n[0],ppt->n[1],ppt->n[2]);
        } else {
          fwrite((unsigned char*)&ppt->n[0],sd,1,inm);
          fwrite((unsigned char*)&ppt->n[1],sd,1,inm);
          fwrite((unsigned char*)&ppt->n[2],sd,1,inm);
        }
      }
    }
    if(!bin) {
      strcpy(&chaine[0],"\n\nTangentAtVertices\n");
      fprintf(inm,"%s",chaine);
      fprintf(inm,"%d\n",ng);
    } else {
      binch = 61; //tangentatvertices
      fwrite(&binch,sw,1,inm);
      bpos += 12 + 2*4*ng;//Pos
      fwrite(&bpos,sw,1,inm);
      fwrite(&ng,sw,1,inm);
    }
    ng = 0;
    for (k=1; k<=mesh->np; k++) {
      ppt = &mesh->point[k];
      if ( MG_VOK(ppt) && MG_EDG(ppt->tag) ) {
        if(!bin) {
          fprintf(inm,"%d %d\n",ppt->tmp,++ng);
        } else {
          fwrite(&ppt->tmp,sw,1,inm);
          ++ng;
          fwrite(&(ng),sw,1,inm);
        }
      }
    }
  }

  if ( abs(mesh->info.imprim) > 4 ) {
    fprintf(stdout,"     NUMBER OF VERTICES   %8d  CORNERS    %6d\n",np,nc);
    fprintf(stdout,"     NUMBER OF TRIANGLES  %8d\n",nt);

    if ( mesh->na )
      fprintf(stdout,"     NUMBER OF EDGES      %8d  RIDGES     %6d\n",mesh->na,nr);
    if ( nn+ng )
      fprintf(stdout,"     NUMBER OF NORMALS    %8d  TANGENTS   %6d\n",nn,ng);
  }
  /*fin fichier*/
  if(!bin) {
    strcpy(&chaine[0],"\n\nEnd\n");
    fprintf(inm,"%s",chaine);
  } else {
    binch = 54; //End
    fwrite(&binch,sw,1,inm);
  }
  fclose(inm);
  return 1;
}

int MMGS_saveMshMesh(MMG5_pMesh mesh,MMG5_pSol sol,const char *filename) {

  return MMG5_saveMshMesh(mesh,&sol,filename,1);
}

int MMGS_saveMshMesh_and_allData(MMG5_pMesh mesh,MMG5_pSol *sol,const char *filename) {

  return MMG5_saveMshMesh(mesh,sol,filename,0);
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param met pointer toward the sol structure.
 * \param filename name of file.
 * \return -1 data invalid, 0 no file, 1 ok.
 *
 * Load metric field.
 *
 */
int MMGS_loadSol(MMG5_pMesh mesh,MMG5_pSol met,const char* filename) {

  FILE       *inm;
  long        posnp;
  int         iswp,ier,dim;
  int         k,ver,bin,np,nsols,*type;

  /** Read the file header */
  ier =  MMG5_loadSolHeader(filename,3,&inm,&ver,&bin,&iswp,&np,&dim,&nsols,
                             &type,&posnp,mesh->info.imprim);

  if ( ier < 1 ) return ier;

  if ( nsols!=1 ) {
    fprintf(stderr,"SEVERAL SOLUTION => IGNORED: %d\n",nsols);
    fclose(inm);
    MMG5_SAFE_FREE(type);
    return -1;
  }

  if ( mesh->np != np ) {
    fprintf(stderr,"  ** MISMATCHES DATA: THE NUMBER OF VERTICES IN "
            "THE MESH (%d) DIFFERS FROM THE NUMBER OF VERTICES IN "
            "THE SOLUTION (%d) \n",mesh->np,np);
    fclose(inm);
    MMG5_SAFE_FREE(type);
    return -1;
  }

  ier = MMG5_chkMetricType(mesh,type,inm);
  if ( ier <1 ) return ier;

  /* Allocate and store the header informations for each solution */
  if ( !MMGS_Set_solSize(mesh,met,MMG5_Vertex,mesh->np,type[0]) ) {
    fclose(inm);
    MMG5_SAFE_FREE(type);
    return -1;
  }
  /* For binary file, we read the verson inside the file */
  if ( ver ) met->ver = ver;

  /* Read mesh solutions */
  rewind(inm);
  fseek(inm,posnp,SEEK_SET);

  /* isotropic metric */
  if ( met->ver == 1 ) {
    /* Simple precision */
    for (k=1; k<=mesh->np; k++) {
      MMG5_readFloatSol3D(met,inm,bin,iswp,k);
    }
  }
  else {
    /* Double precision */
    for (k=1; k<=mesh->np; k++) {
      MMG5_readDoubleSol3D(met,inm,bin,iswp,k);
    }
  }

  mesh->info.inputMet = 1;

  fclose(inm);

  /* stats */
  MMG5_printMetStats(mesh,met);

  return 1;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param sol pointer toward an allocatable sol structure.
 * \param filename name of file.
 * \return -1 data invalid or we fail, 0 no file, 1 ok.
 *
 * Load a medit solution file containing 1 or more solutions.
 *
 */
int MMGS_loadAllSols(MMG5_pMesh mesh,MMG5_pSol *sol, const char *filename) {
  MMG5_pSol   psl;
  FILE       *inm;
  long        posnp;
  int         iswp,ier,dim;
  int         j,k,ver,bin,np,nsols,*type;
  char        data[10];
  static char mmgWarn = 0;

  /** Read the file header */
  ier =  MMG5_loadSolHeader(filename,3,&inm,&ver,&bin,&iswp,&np,&dim,&nsols,
                            &type,&posnp,mesh->info.imprim);
  if ( ier < 1 ) return ier;

  if ( mesh->np != np ) {
    fprintf(stderr,"  ** MISMATCHES DATA: THE NUMBER OF VERTICES IN "
            "THE MESH (%d) DIFFERS FROM THE NUMBER OF VERTICES IN "
            "THE SOLUTION (%d) \n",mesh->np,np);
    fclose(inm);
    MMG5_SAFE_FREE(type);
    return -1;
  }

  /** Sol tab allocation */
  mesh->nsols = nsols;

  if ( nsols > MMG5_NSOLS_MAX ) {
    fprintf(stderr,"\n  ## Error: %s: unexpected number of data (%d).\n",
            __func__,nsols);
    MMG5_SAFE_FREE(type);
    fclose(inm);
    return -1;
  }

  if ( *sol )  MMG5_DEL_MEM(mesh,*sol);
  MMG5_ADD_MEM(mesh,nsols*sizeof(MMG5_Sol),"solutions array",
                printf("  Exit program.\n"); fclose(inm);
                MMG5_SAFE_FREE(type);
                return -1);
  MMG5_SAFE_CALLOC(*sol,nsols,MMG5_Sol,return -1);

  for ( j=0; j<nsols; ++j ) {
    psl = *sol + j;

    /* Give an arbitrary name to the solution because the Medit format has non
     * name field */
    sprintf(data,"sol_%d",j);
    if ( !MMGS_Set_inputSolName(mesh,psl,data) ) {
      if ( !mmgWarn ) {
        mmgWarn = 1;
        fprintf(stderr,"\n  ## Warning: %s: unable to set solution name for"
                " at least 1 solution.\n",__func__);
      }
    }

    /* Allocate and store the header informations for each solution */
    if ( !MMGS_Set_solSize(mesh,psl,MMG5_Vertex,mesh->np,type[j]) ) {
      MMG5_SAFE_FREE(type);
      fclose(inm);
      return -1;
    }
    /* For binary file, we read the verson inside the file */
    if ( ver ) psl->ver = ver;
  }
  MMG5_SAFE_FREE(type);

  /* read mesh solutions */
  rewind(inm);
  fseek(inm,posnp,SEEK_SET);

  if ( (*sol)[0].ver == 1 ) {
    /* Simple precision */
    for (k=1; k<=mesh->np; k++) {
      for ( j=0; j<nsols; ++j ) {
        psl = *sol + j;
        MMG5_readFloatSol3D(psl,inm,bin,iswp,k);
      }
    }
  }
  else {
    /* Double precision */
    for (k=1; k<=mesh->np; k++) {
      for ( j=0; j<nsols; ++j ) {
        psl = *sol + j;
        MMG5_readDoubleSol3D(psl,inm,bin,iswp,k);
      }
    }
  }
  fclose(inm);

  /* stats */
  MMG5_printSolStats(mesh,sol);

  return 1;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param met pointer toward the sol structure.
 * \param filename name of file.
 * \return 0 if failed, 1 otherwise.
 *
 * Write isotropic or anisotropic metric.
 *
 */
int MMGS_saveSol(MMG5_pMesh mesh,MMG5_pSol met, const char *filename) {
  FILE*        inm;
  MMG5_pPoint  ppt;
  int          binch,bin,ier,k;

  if ( !met->m )  return -1;

  met->ver = 2;

  ier = MMG5_saveSolHeader( mesh,filename,&inm,met->ver,&bin,mesh->np,met->dim,
                            1,&met->type,&met->size);

  if ( ier < 1 )  return ier;

  for (k=1; k<=mesh->np; k++) {
    ppt = &mesh->point[k];
    if ( !MG_VOK(ppt) ) continue;

    MMG5_writeDoubleSol3D(mesh,met,inm,bin,k,1);
    fprintf(inm,"\n");
  }

  /*fin fichier*/
  if(!bin) {
    fprintf(inm,"\n\nEnd\n");
  } else {
    binch = 54; //End
    fwrite(&binch,sw,1,inm);
  }
  fclose(inm);
  return 1;
}

/**
 * \param mesh pointer toward the mesh structure.
 * \param sol pointer toward the solutions array
 * \param filename name of file.
 * \return 0 if failed, 1 otherwise.
 *
 * Write solutions array
 *
 */
int MMGS_saveAllSols(MMG5_pMesh mesh,MMG5_pSol *sol, const char *filename) {
  MMG5_pSol    psl;
  FILE*        inm;
  MMG5_pPoint  ppt;
  int          binch,bin,ier,k,j;
  int          *type,*size;

  if ( !(*sol)[0].m )  return -1;

  (*sol)[0].ver = 2;

  MMG5_SAFE_CALLOC(type,mesh->nsols,int,return 0);
  MMG5_SAFE_CALLOC(size,mesh->nsols,int,return 0);
  for (k=0; k<mesh->nsols; ++k ) {
    type[k] = (*sol)[k].type;
    size[k] = (*sol)[k].size;
  }

  ier = MMG5_saveSolHeader( mesh,filename,&inm,(*sol)[0].ver,&bin,mesh->np,
                            (*sol)[0].dim,mesh->nsols,type,size);

  MMG5_SAFE_FREE(type);
  MMG5_SAFE_FREE(size);

  if ( ier < 1 )  return ier;

  for (k=1; k<=mesh->np; k++) {
    ppt = &mesh->point[k];
    if ( !MG_VOK(ppt) ) continue;

    for ( j=0; j<mesh->nsols; ++j ) {
      psl = *sol+j;
      MMG5_writeDoubleSol3D(mesh,psl,inm,bin,k,0);
    }
    fprintf(inm,"\n");
  }

  /* End file */
  if(!bin) {
    fprintf(inm,"\n\nEnd\n");
  } else {
    binch = 54; //End
    fwrite(&binch,sw,1,inm);
  }
  fclose(inm);
  return 1;
}
