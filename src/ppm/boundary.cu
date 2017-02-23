__global__ void dev_bound_trans(smcell *cells, ybcell *cp_bac, ybcell *cp_frn, int jstart, int jend)
{
  int bd = 6;

  int ni = threadIdx.x;
  int j = cells[ni].jb;
  
  int ii = blockIdx.x+6;
  int jj;
  #if ndim == 3
  int kk = blockIdx.y+6;
  #else
  int kk = blockIdx.y;
  #endif

  int ig = cells[ni].i[ii];
  int kg = cells[ni].k[kk];

  if (j==jstart)
  {
    for (jj=0; jj<bd; jj++)
    {
      (*cp_frn).r[ig][jj][kg] = cells[ni].r[ii][jj+bd][kk];
      (*cp_frn).p[ig][jj][kg] = cells[ni].p[ii][jj+bd][kk];
      (*cp_frn).u[ig][jj][kg] = cells[ni].u[ii][jj+bd][kk];
      (*cp_frn).v[ig][jj][kg] = cells[ni].v[ii][jj+bd][kk];
      (*cp_frn).w[ig][jj][kg] = cells[ni].w[ii][jj+bd][kk];
    }
  }
  if (j==jend)
  {
    for (jj=0; jj<bd; jj++)
    {
      (*cp_bac).r[ig][jj][kg] = cells[ni].r[ii][jj+jdim][kk];
      (*cp_bac).p[ig][jj][kg] = cells[ni].p[ii][jj+jdim][kk];
      (*cp_bac).u[ig][jj][kg] = cells[ni].u[ii][jj+jdim][kk];
      (*cp_bac).v[ig][jj][kg] = cells[ni].v[ii][jj+jdim][kk];
      (*cp_bac).w[ig][jj][kg] = cells[ni].w[ii][jj+jdim][kk];
    }
  }
  return;
}

//================================================================================

__global__ void set_boundx(smcell *cells, xbcell *lft, xbcell *rgh, int istart, int iend, int incr, sdp alpha)
{
  int bd = 6;

  int np;
  int ni = threadIdx.x;
  int i = cells[ni].ib;
  int j = cells[ni].jb;
  int k = cells[ni].kb;
  
  int ii;
  #if ndim > 2
  int jj = blockIdx.x+6;
  int kk = blockIdx.y+6;  
  #elif ndim > 1
  int jj = blockIdx.x+6;
  int kk = blockIdx.y;  
  #else
  int jj = blockIdx.x;
  int kk = blockIdx.y;
  #endif
  int jg = cells[ni].j[jj];
  int kg = cells[ni].k[kk];

  if (i!=istart)
  {
    np = ni-incr;
    for (ii=0; ii<bd; ii++)
    { 
      cells[ni].r[ii][jj][kk] = cells[np].r[ii+idim][jj][kk];
      cells[ni].p[ii][jj][kk] = cells[np].p[ii+idim][jj][kk];
      cells[ni].u[ii][jj][kk] = cells[np].u[ii+idim][jj][kk];
      cells[ni].v[ii][jj][kk] = cells[np].v[ii+idim][jj][kk];
      cells[ni].w[ii][jj][kk] = cells[np].w[ii+idim][jj][kk];
    }
  }
  else
  {
    for (ii=0; ii<bd; ii++)
    {
      if (nlft==0) //reflect
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[bd+bd-ii][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[bd+bd-ii][jj][kk];
        cells[ni].u[ii][jj][kk] =-cells[ni].u[bd+bd-ii][jj][kk];
        cells[ni].v[ii][jj][kk] = cells[ni].v[bd+bd-ii][jj][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[bd+bd-ii][jj][kk];
      }
      else if (nlft==1) //outflow
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[bd][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[bd][jj][kk];
        cells[ni].u[ii][jj][kk] = cells[ni].u[bd][jj][kk];
        cells[ni].v[ii][jj][kk] = cells[ni].v[bd][jj][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[bd][jj][kk];
      }
      else if (nlft==2) //periodic
      {
        np = (iblk-1) + iblk*(j + (jblk*k));
        cells[ni].r[ii][jj][kk] = cells[np].r[ii+idim][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[np].p[ii+idim][jj][kk];
        cells[ni].u[ii][jj][kk] = cells[np].u[ii+idim][jj][kk];
        cells[ni].v[ii][jj][kk] = cells[np].v[ii+idim][jj][kk];
        cells[ni].w[ii][jj][kk] = cells[np].w[ii+idim][jj][kk];
      }
      else if (nlft==3)
      {
        cells[ni].r[ii][jj][kk] = (*lft).r[ii][jg][kg];
        cells[ni].p[ii][jj][kk] = (*lft).p[ii][jg][kg];
        cells[ni].u[ii][jj][kk] = (*lft).u[ii][jg][kg];
        cells[ni].v[ii][jj][kk] = (*lft).v[ii][jg][kg];
        cells[ni].w[ii][jj][kk] = (*lft).w[ii][jg][kg];
      }
      else if (nlft==4)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[bd][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[bd][jj][kk];
        cells[ni].u[ii][jj][kk] = cells[ni].u[bd][jj][kk];;
        cells[ni].v[ii][jj][kk] = (*lft).v[ii][jg][kg];
        cells[ni].w[ii][jj][kk] = (*lft).w[ii][jg][kg];
      }
      //printf("bound left: (%f, %f, %f)\n", cells[ni].r[ii][jj][kk], cells[ni].p[ii][jj][kk], cells[ni].u[ii][jj][kk]);
    }
  }

  if (i!=iend)
  {
    np = ni+incr;
    for (ii=iswp-bd; ii<iswp; ii++)
    {
      cells[ni].r[ii][jj][kk] = cells[np].r[ii-idim][jj][kk];
      cells[ni].p[ii][jj][kk] = cells[np].p[ii-idim][jj][kk];
      cells[ni].u[ii][jj][kk] = cells[np].u[ii-idim][jj][kk];
      cells[ni].v[ii][jj][kk] = cells[np].v[ii-idim][jj][kk];
      cells[ni].w[ii][jj][kk] = cells[np].w[ii-idim][jj][kk];
    }
  }
  else
  {
    for (ii=iswp-bd; ii<iswp; ii++)
    { 
      if (nrgh==0)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[idim+iswp-ii-2][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[idim+iswp-ii-2][jj][kk];
        cells[ni].u[ii][jj][kk] =-cells[ni].u[idim+iswp-ii-2][jj][kk];
        cells[ni].v[ii][jj][kk] = cells[ni].v[idim+iswp-ii-2][jj][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[idim+iswp-ii-2][jj][kk];
      }
      else if (nrgh==1)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[idim+bd-1][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[idim+bd-1][jj][kk];
        cells[ni].u[ii][jj][kk] = cells[ni].u[idim+bd-1][jj][kk];
        cells[ni].v[ii][jj][kk] = cells[ni].v[idim+bd-1][jj][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[idim+bd-1][jj][kk];
      }
      else if (nrgh==2)
      {
        np = 0 + iblk*(j + (jblk*k));
        cells[ni].r[ii][jj][kk] = cells[np].r[ii-idim][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[np].p[ii-idim][jj][kk];
        cells[ni].u[ii][jj][kk] = cells[np].u[ii-idim][jj][kk];
        cells[ni].v[ii][jj][kk] = cells[np].v[ii-idim][jj][kk];
        cells[ni].w[ii][jj][kk] = cells[np].w[ii-idim][jj][kk];
      }
      else if (nrgh==3)
      {
        cells[ni].r[ii][jj][kk] = (*rgh).r[ii-iswp+bd][jg][kg];
        cells[ni].p[ii][jj][kk] = (*rgh).p[ii-iswp+bd][jg][kg];
        cells[ni].u[ii][jj][kk] = (*rgh).u[ii-iswp+bd][jg][kg];
        cells[ni].v[ii][jj][kk] = (*rgh).v[ii-iswp+bd][jg][kg];
        cells[ni].w[ii][jj][kk] = (*rgh).w[ii-iswp+bd][jg][kg];
      }
      else if (nrgh==4)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[idim+bd-1][jj][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[idim+bd-1][jj][kk];
        cells[ni].u[ii][jj][kk] = (*rgh).u[ii-iswp+bd][jg][kg];
        cells[ni].v[ii][jj][kk] = (*rgh).v[ii-iswp+bd][jg][kg];
        cells[ni].w[ii][jj][kk] = (*rgh).w[ii-iswp+bd][jg][kg];
      }
      //printf("bound right: (%f, %f, %f)\n", cells[ni].r[ii][jj][kk], cells[ni].p[ii][jj][kk], cells[ni].u[ii][jj][kk]);
    }
  }

  return;
}
//================================================================================

__global__ void set_boundy(smcell *cells, ybcell *bac, ybcell *frn, int jstart, int jend, int incr)
{
  int bd = 6;

  int np;
  int ni = threadIdx.x;
  int j = cells[ni].jb;

  int ii = blockIdx.x+6;
  int jj;
  #if ndim == 3
  int kk = blockIdx.y+6;
  #else
  int kk = blockIdx.y;
  #endif

  int ig = cells[ni].i[ii];
  int kg = cells[ni].k[kk];

  if (j!=jstart)
  {
    np = ni-incr;
    for (jj=0; jj<bd; jj++)
    {   
      cells[ni].r[ii][jj][kk] = cells[np].r[ii][jj+jdim][kk];
      cells[ni].p[ii][jj][kk] = cells[np].p[ii][jj+jdim][kk];
      cells[ni].u[ii][jj][kk] = cells[np].u[ii][jj+jdim][kk];
      cells[ni].v[ii][jj][kk] = cells[np].v[ii][jj+jdim][kk];
      cells[ni].w[ii][jj][kk] = cells[np].w[ii][jj+jdim][kk];
    }
  }
  else if (jstart!=0)
  {
    for (jj=0; jj<bd; jj++)
    {
      cells[ni].r[ii][jj][kk] = (*bac).r[ig][jj][kg];
      cells[ni].p[ii][jj][kk] = (*bac).p[ig][jj][kg];
      cells[ni].u[ii][jj][kk] = (*bac).u[ig][jj][kg];
      cells[ni].v[ii][jj][kk] = (*bac).v[ig][jj][kg];
      cells[ni].w[ii][jj][kk] = (*bac).w[ig][jj][kg];
    }
  }
  else
  {
    for (jj=0; jj<bd; jj++)
    {
      if (nbac==0) //reflect
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][bd+bd-jj][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][bd+bd-jj][kk];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][bd+bd-jj][kk];
        cells[ni].v[ii][jj][kk] =-cells[ni].v[ii][bd+bd-jj][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[ii][bd+bd-jj][kk];
      }
      else if (nbac==1) //outflow
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][bd][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][bd][kk];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][bd][kk];
        cells[ni].v[ii][jj][kk] = cells[ni].v[ii][bd][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[ii][bd][kk];
      }
      else if (nbac==2)
      {
        cells[ni].r[ii][jj][kk] = (*bac).r[ig][jj][kg];
        cells[ni].p[ii][jj][kk] = (*bac).p[ig][jj][kg];
        cells[ni].u[ii][jj][kk] = (*bac).u[ig][jj][kg];
        cells[ni].v[ii][jj][kk] = (*bac).v[ig][jj][kg];
        cells[ni].w[ii][jj][kk] = (*bac).w[ig][jj][kg];
      }
    }
  }
 
  if (j!=jend)
  {
    np = ni+incr;
    for (jj=jswp-bd; jj<jswp; jj++)
    {
      cells[ni].r[ii][jj][kk] = cells[np].r[ii][jj-jdim][kk];
      cells[ni].p[ii][jj][kk] = cells[np].p[ii][jj-jdim][kk];
      cells[ni].u[ii][jj][kk] = cells[np].u[ii][jj-jdim][kk];
      cells[ni].v[ii][jj][kk] = cells[np].v[ii][jj-jdim][kk];
      cells[ni].w[ii][jj][kk] = cells[np].w[ii][jj-jdim][kk];
    }
  }
  else if (jend!=jblk-1)
  {
    for (jj=jswp-bd; jj<jswp; jj++)
    {
      cells[ni].r[ii][jj][kk] = (*frn).r[ig][jj-jswp+bd][kg];
      cells[ni].p[ii][jj][kk] = (*frn).p[ig][jj-jswp+bd][kg];
      cells[ni].u[ii][jj][kk] = (*frn).u[ig][jj-jswp+bd][kg];
      cells[ni].v[ii][jj][kk] = (*frn).v[ig][jj-jswp+bd][kg];
      cells[ni].w[ii][jj][kk] = (*frn).w[ig][jj-jswp+bd][kg];
    }
  }
  else
  {
    for (jj=jswp-bd; jj<jswp; jj++)
    { 
      if (nfrn==0)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][jdim+jswp-jj-2][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][jdim+jswp-jj-2][kk];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][jdim+jswp-jj-2][kk];
        cells[ni].v[ii][jj][kk] =-cells[ni].v[ii][jdim+jswp-jj-2][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[ii][jdim+jswp-jj-2][kk];
      }
      else if (nfrn==1)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][jdim+bd-1][kk];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][jdim+bd-1][kk];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][jdim+bd-1][kk];
        cells[ni].v[ii][jj][kk] = cells[ni].v[ii][jdim+bd-1][kk];
        cells[ni].w[ii][jj][kk] = cells[ni].w[ii][jdim+bd-1][kk];
      }
      else if (nfrn==2)
      {
        cells[ni].r[ii][jj][kk] = (*frn).r[ig][jj-jswp+bd][kg];
        cells[ni].p[ii][jj][kk] = (*frn).p[ig][jj-jswp+bd][kg];
        cells[ni].u[ii][jj][kk] = (*frn).u[ig][jj-jswp+bd][kg];
        cells[ni].v[ii][jj][kk] = (*frn).v[ig][jj-jswp+bd][kg];
        cells[ni].w[ii][jj][kk] = (*frn).w[ig][jj-jswp+bd][kg];
      }
    }
  }

  return;
}

//================================================================================
#if ndim==3
__global__ void set_boundz(smcell *cells, zbcell *udr, zbcell *top, int kstart, int kend, int incr)
{
    int bd = 6;

  int np;
  int ni = threadIdx.x;
  int j = cells[ni].jb;
  int k = cells[ni].kb;
  
  int ii = blockIdx.x+6;
  int jj = blockIdx.y+6;
  int kk;

  int ig = cells[ni].i[ii];
  int jg = cells[ni].j[jj];

  if (k!=kstart)
  {
    np = ni-incr;
    for (kk=0; kk<bd; kk++)
    { 
      cells[ni].r[ii][jj][kk] = cells[np].r[ii][jj][kk+kdim];
      cells[ni].p[ii][jj][kk] = cells[np].p[ii][jj][kk+kdim];
      cells[ni].u[ii][jj][kk] = cells[np].u[ii][jj][kk+kdim];
      cells[ni].v[ii][jj][kk] = cells[np].v[ii][jj][kk+kdim];
      cells[ni].w[ii][jj][kk] = cells[np].w[ii][jj][kk+kdim];
    }
  }
  else
  {
    for (kk=0; kk<bd; kk++)
    {
      if (nudr==0) //reflect
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][jj][bd+bd-kk-1];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][jj][bd+bd-kk-1];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][jj][bd+bd-kk-1];
        cells[ni].v[ii][jj][kk] = cells[ni].v[ii][jj][bd+bd-kk-1];
        cells[ni].w[ii][jj][kk] =-cells[ni].w[ii][jj][bd+bd-kk-1];
      }
      else if (nudr==1) //outflow
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][jj][bd];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][jj][bd];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][jj][bd];
        cells[ni].v[ii][jj][kk] = cells[ni].v[ii][jj][bd];
        cells[ni].w[ii][jj][kk] = cells[ni].w[ii][jj][bd];
      }
      else if (nudr==2) //periodic
      {
        np = (iblk-1) + iblk*(j + (jblk*k));
        cells[ni].r[ii][jj][kk] = cells[np].r[ii][jj][kk+kdim];
        cells[ni].p[ii][jj][kk] = cells[np].p[ii][jj][kk+kdim];
        cells[ni].u[ii][jj][kk] = cells[np].u[ii][jj][kk+kdim];
        cells[ni].v[ii][jj][kk] = cells[np].v[ii][jj][kk+kdim];
        cells[ni].w[ii][jj][kk] = cells[np].w[ii][jj][kk+kdim];
      }
      else if (nudr==3)
      {
        cells[ni].r[ii][jj][kk] = (*udr).r[ig][jg][kk];
        cells[ni].p[ii][jj][kk] = (*udr).p[ig][jg][kk];
        cells[ni].u[ii][jj][kk] = (*udr).u[ig][jg][kk];
        cells[ni].v[ii][jj][kk] = (*udr).v[ig][jg][kk];
        cells[ni].w[ii][jj][kk] = (*udr).w[ig][jg][kk];
      }
    }
  }

  if (k!=kend)
  {
    np = ni+incr;
    for (kk=kswp-bd; kk<kswp; kk++)
    {
      cells[ni].r[ii][jj][kk] = cells[np].r[ii][jj][kk-kdim];
      cells[ni].p[ii][jj][kk] = cells[np].p[ii][jj][kk-kdim];
      cells[ni].u[ii][jj][kk] = cells[np].u[ii][jj][kk-kdim];
      cells[ni].v[ii][jj][kk] = cells[np].v[ii][jj][kk-kdim];
      cells[ni].w[ii][jj][kk] = cells[np].w[ii][jj][kk-kdim];
    }
  }
  else
  {
    for (kk=kswp-bd; kk<kswp; kk++)
    { 
      if (ntop==0)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][jj][kdim+kswp-kk-1];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][jj][kdim+kswp-kk-1];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][jj][kdim+kswp-kk-1];
        cells[ni].v[ii][jj][kk] = cells[ni].v[ii][jj][kdim+kswp-kk-1];
        cells[ni].w[ii][jj][kk] =-cells[ni].w[ii][jj][kdim+kswp-kk-1];
      }
      else if (ntop==1)
      {
        cells[ni].r[ii][jj][kk] = cells[ni].r[ii][jj][kdim+bd-1];
        cells[ni].p[ii][jj][kk] = cells[ni].p[ii][jj][kdim+bd-1];
        cells[ni].u[ii][jj][kk] = cells[ni].u[ii][jj][kdim+bd-1];
        cells[ni].v[ii][jj][kk] = cells[ni].v[ii][jj][kdim+bd-1];
        cells[ni].w[ii][jj][kk] = cells[ni].w[ii][jj][kdim+bd-1];
      }
      else if (ntop==2)
      {
        np = iblk*(j + (jblk*k));
        cells[ni].r[ii][jj][kk] = cells[np].r[ii][jj][kk-kdim];
        cells[ni].p[ii][jj][kk] = cells[np].p[ii][jj][kk-kdim];
        cells[ni].u[ii][jj][kk] = cells[np].u[ii][jj][kk-kdim];
        cells[ni].v[ii][jj][kk] = cells[np].v[ii][jj][kk-kdim];
        cells[ni].w[ii][jj][kk] = cells[np].w[ii][jj][kk-kdim];
      }
      else if (ntop==3)
      {
        cells[ni].r[ii][jj][kk] = (*top).r[ig][jg][kk-kswp+bd];
        cells[ni].p[ii][jj][kk] = (*top).p[ig][jg][kk-kswp+bd];
        cells[ni].u[ii][jj][kk] = (*top).u[ig][jg][kk-kswp+bd];
        cells[ni].v[ii][jj][kk] = (*top).v[ig][jg][kk-kswp+bd];
        cells[ni].w[ii][jj][kk] = (*top).w[ig][jg][kk-kswp+bd];
      }
    }
  }
  return;
}
#endif
