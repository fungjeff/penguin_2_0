#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "variable_types.h"
#include "global.h"
#include "disk_profile.h"
#include "cuSafe.cu"

//#include "opacity.cu"
//#include "viscosity.cu"
//#include "kill_wave.cu"
//#include "boundary.cu"

#include "forces.cu"
#include "parabola.cu"
#include "flatten.cu"
#include "states.cu"
#include "riemann.cu"
#include "evolve.cu"
#include "remap.cu"
#include "ppmlr.cu"

//================================================================================
__global__ void sweepx(hydr_ring *rings, hydr_ring *lft, hydr_ring *rgh, sdp dt, int iblk, body planet)
{
  int i, j0, j, lim;
  int i_ring, k_ring, loop;

  i = threadIdx.x;
  j0 = blockIdx.x;
  k_ring = blockIdx.y;
  loop = (iblk/realarr) + (bool)(iblk%realarr);


  __shared__ sdp r_lt[n_pad], p_lt[n_pad], u_lt[n_pad], v_lt[n_pad], w_lt[n_pad];

  __shared__ sdp r[arrsize], p[arrsize], u[arrsize], v[arrsize], w[arrsize], e[arrsize];
  __shared__ sdp xa0[arrsize], dx0[arrsize];
  sdp dvol0, rad, azi, pol, rad_cyl;
  hydr_ring *cur;

  for (int l=0; l<loop; l++)
  {
    i_ring = (i-n_pad) + realarr*l;

    lim = blockDim.x;
    if (l==loop-1) lim = (iblk%realarr) + 2*n_pad;

    if (i_ring < iblk+n_pad)
    {
///////////////////////////////////////////////////////
      if (i_ring<0)
      {
        cur = &lft[i_ring + n_pad + n_pad*k_ring];
      }
      else if (i_ring >= iblk)
      {
        cur = &rgh[i_ring-iblk + n_pad*k_ring];
      }
      else
      {
        cur = &rings[i_ring + iblk*k_ring];
      }

      ///////////////////////////////////////////////////////

      j = j0 - (*cur).rot_j;
      if (j<0) j += jmax;
      if (j>=jmax) j -= jmax;

      if (ngeomx > 0) rad = (*cur).xc;
      else            rad = 1.0;
      azi = (*cur).yc[j0];
      #if ndim == 3
      pol = (*cur).zc;
      #else
      pol = 0.0;
      #endif

      rad_cyl = rad;
      if (ngeomz == 5) rad_cyl *= csin(pol);

      xa0[i] = (*cur).x;
      dx0[i] = (*cur).dx;
      dvol0  = (*cur).xvol;

      if (l>0 && i<n_pad)
      {
        r[i] = r_lt[i];
        p[i] = p_lt[i];
        u[i] = u_lt[i];
        v[i] = v_lt[i];
        w[i] = w_lt[i];
      }
      else
      {
        r[i] = (*cur).r[j];
        p[i] = (*cur).p[j];
        u[i] = (*cur).u[j];
        v[i] = (*cur).v[j];
        w[i] = (*cur).w[j];
        if (ngeomy > 2) v[i] *= rad_cyl;
        if (ngeomz == 5) w[i] *= rad;
      }
      #if EOS == 2
      e[i] = p[i]/(r[i]*gamm) + 0.5*((u[i]*u[i])+(v[i]*v[i]));
      #endif
      __syncthreads();

      if (i>=lim-2*n_pad && i<lim-n_pad)
      {
        r_lt[i-lim+2*n_pad] = r[i];
        p_lt[i-lim+2*n_pad] = p[i];
        u_lt[i-lim+2*n_pad] = u[i];
        v_lt[i-lim+2*n_pad] = v[i];
        w_lt[i-lim+2*n_pad] = w[i];
      }
      
      __syncthreads();

      //if (j==99)
        //printf("%i, (%i, %i, %i), (%f, %f, %f):  (%e, %e, %e, %e, %e)\n", lim, i_ring, j, k_ring, rad, azi, pol, r[i]*cpow(rad,p_alpha)-1.0, p[i]/r[i], u[i], v[i]*cpow(rad,-0.5), w[i]);
      ///////////////////////////////////////////////////////

      ppmlr(rad, azi, pol, ((*cur).rot_v-(*cur).res_v)/rad_cyl,
            r, p, u, v, w, e, 
            xa0, dx0, dvol0, 
            lim, dt, 0, planet);

      #if visc_flag == 1
      device_viscosity_r(i, dt, rad_cyl, cells, xa0, r, u, v, w, dx0, alpha);
      #endif
      __syncthreads();
      ///////////////////////////////////////////////////////

      if (i>=n_pad && i<lim-n_pad)
      {
        (*cur).r[j] = r[i];
        #if EOS == 0
        (*cur).p[j] = r[i]*get_cs2(rad_cyl);
        #elif EOS == 1
        (*cur).p[j] = get_cs2(rad_cyl)*cpow(r[i],gam)/gam;
        #else
        (*cur).p[j] = p[i];
        #endif
        (*cur).u[j] = u[i];
        if (ngeomy> 2) (*cur).v[j] = v[i]/rad_cyl;
        else           (*cur).v[j] = v[i];
        if (ngeomz==5) (*cur).w[j] = w[i]/rad;
        else           (*cur).w[j] = w[i];

      //if (j==99)
        //printf("%i, (%i, %i, %i), (%f, %f, %f):  (%f, %f, %f, %f, %f)\n", l, (int)blockIdx.x, jref, (int)blockIdx.y, rad, azi, pol, (*cur).r[j], (*cur).p[j], (*cur).u[j], (*cur).v[j], (*cur).w[j]);
      }
      ///////////////////////////////////////////////////////
    }
    __syncthreads();
  }

  return;
}

__global__ void sweepy(hydr_ring *rings, sdp dt, int iblk, body planet)
{
  int i, j0, j, jref, lim;
  int n, loop;

  i = threadIdx.x;
  n = blockIdx.x + iblk*blockIdx.y;  
  loop = (jmax/realarr) + (bool)(jmax%realarr);


  __shared__ sdp r_rg[n_pad], p_rg[n_pad], u_rg[n_pad], v_rg[n_pad], w_rg[n_pad];
  __shared__ sdp r_lt[n_pad], p_lt[n_pad], u_lt[n_pad], v_lt[n_pad], w_lt[n_pad];

  __shared__ sdp r[arrsize], p[arrsize], u[arrsize], v[arrsize], w[arrsize], e[arrsize];
  __shared__ sdp xa0[arrsize], dx0[arrsize];
  sdp dvol0, rad, azi, pol, rad_cyl;
  hydr_ring *cur = &rings[n];

  if (ngeomx > 0) rad = (*cur).xc;
  else            rad = 1.0;
  #if ndim == 3
  pol = (*cur).zc;
  #else
  pol = 0.0;
  #endif

  rad_cyl = rad;
  if (ngeomz == 5) rad_cyl *= csin(pol);

  for (int l=0; l<loop; l++)
  {
    jref = (i-n_pad) + realarr*l;

    lim = blockDim.x;
    if (l==loop-1) lim = (jmax%realarr) + 2*n_pad;

    if (jref < jmax+n_pad)
    {

      ///////////////////////////////////////////////////////    
      j0 = jref;
      if (j0<0) j0 += jmax;
      if (j0>=jmax) j0 -= jmax;

      j = j0 - (*cur).rot_j;
      if (j<0) j += jmax;
      if (j>=jmax) j -= jmax;

      azi = (*cur).yc[j0];
      xa0[i] = (*cur).y[j0];
      dx0[i] = (*cur).dy[j0];
      dvol0  = (*cur).yvol[j0];

      if (jref<0) {xa0[i] -= twopi; azi -= twopi;}
      if (jref>=jmax) {xa0[i] += twopi; azi += twopi;}
      if (ngeomy > 2) dvol0 *= rad_cyl;

      if (l>0 && i<n_pad)
      {
        r[i] = r_lt[i];
        p[i] = p_lt[i];
        u[i] = u_lt[i];
        v[i] = v_lt[i];
        w[i] = w_lt[i];
      }
      else if (l==loop-1 && i>=lim-n_pad)
      {
        r[i] = r_rg[i-lim+n_pad];
        p[i] = p_rg[i-lim+n_pad];
        u[i] = u_rg[i-lim+n_pad];
        v[i] = v_rg[i-lim+n_pad];
        w[i] = w_rg[i-lim+n_pad];
      }
      else
      {
        r[i] = (*cur).r[j];
        p[i] = (*cur).p[j];
        u[i] = (*cur).v[j];
        v[i] = (*cur).w[j];
        w[i] = (*cur).u[j];
      }      
      #if EOS == 2
      e[i] = p[i]/(r[i]*gamm) + 0.5*((u[i]*u[i])+(v[i]*v[i]));
      #endif
      __syncthreads();

      if (l==0 && i>=n_pad && i<2*n_pad)
      {
        r_rg[i-n_pad] = r[i];
        p_rg[i-n_pad] = p[i];
        u_rg[i-n_pad] = u[i];
        v_rg[i-n_pad] = v[i];
        w_rg[i-n_pad] = w[i];
      }

      if (l<loop-1 && i>=lim-2*n_pad && i<lim-n_pad)
      {
        r_lt[i-lim+2*n_pad] = r[i];
        p_lt[i-lim+2*n_pad] = p[i];
        u_lt[i-lim+2*n_pad] = u[i];
        v_lt[i-lim+2*n_pad] = v[i];
        w_lt[i-lim+2*n_pad] = w[i];
      }

      __syncthreads();
      //if (n==99)
        //printf("%i, (%i, %i, %i), (%f, %f, %f):  (%f, %f, %f, %f, %f)\n", l, (int)blockIdx.x, jref, (int)blockIdx.y, rad, azi, pol, r[i], p[i], u[i], v[i], w[i]);
      ///////////////////////////////////////////////////////

      ppmlr(rad_cyl, azi, pol, ((*cur).rot_v-(*cur).res_v)/rad_cyl,
            r, p, u, v, w, e, 
            xa0, dx0, dvol0,
            lim, dt, 1, planet);

      #if visc_flag == 1
      xa0[j] *= rad_cyl;
      device_viscosity_p(j, dt, rad_cyl, cells, xa0, r, w, u, v, dx0, alpha);
      #endif
      __syncthreads();
      ///////////////////////////////////////////////////////

      if (i>=n_pad && i<lim-n_pad)
      {
        (*cur).r[j] = r[i];
        #if EOS == 0
        (*cur).p[j] = r[i]*get_cs2(rad_cyl);
        #elif EOS == 1
        (*cur).p[j] = get_cs2(rad_cyl)*cpow(r[i],gam)/gam;
        #else
        (*cur).p[j] = p[i];
        #endif
        (*cur).u[j] = w[i];
        (*cur).v[j] = u[i];
        (*cur).w[j] = v[i];

      //if (n==99)
        //printf("%i, (%i, %i, %i), (%f, %f, %f):  (%f, %f, %f, %f, %f)\n", l, (int)blockIdx.x, jref, (int)blockIdx.y, rad, azi, pol, (*cur).r[j], (*cur).p[j], (*cur).u[j], (*cur).v[j], (*cur).w[j]);
      }

      ///////////////////////////////////////////////////////
    }
    __syncthreads();
  }

  if (i==0)
  {
    j  = (*cur).rot_j;
    j += (*cur).inc_j;
    if (j<0) j += jmax;
    if (j>=jmax) j -= jmax;
    (*cur).rot_j  = j;
    //if (blockIdx.x==0) printf("%i %i %f\n", j, (*cur).inc_j, (*cur).rot_v*cpow(rad,0.5));
  }

  return;
}

#if ndim == 3
__global__ void sweepz(smcell *cells, sdp dt, int jngr, sdp *beta, body planet, sdp alpha)
{
  int i, j, k;
  int N = blockIdx.x;
  i = blockIdx.y+6;
  j = blockIdx.z+6;
  k = threadIdx.x;

  __shared__ sdp r[arrsize], p[arrsize], u[arrsize], v[arrsize], w[arrsize], e[arrsize];
  __shared__ sdp xa0[arrsize], dx0[arrsize];
  sdp dvol0, rad, azi, pol, rad_cyl, bt;

  pol = cells[N].zc[k];
  if (ngeomy>2) rad = cells[N].xc[i];
  else          rad = 1.0;
  rad_cyl = rad;
  if (ngeomz == 5) rad_cyl *= csin(pol);
  azi = cells[N].yc[j];

  xa0[k] = cells[N].z[k];
  dx0[k] = cells[N].dz[k];
  dvol0  = cells[N].zvol[k];
  if (ngeomz > 2) dvol0 *= rad;

  r[k] = cells[N].r[i][j][k];
  p[k] = cells[N].p[i][j][k];
  u[k] = cells[N].w[i][j][k];
  v[k] = cells[N].u[i][j][k];
  if (ngeomz == 5) w[k] = cells[N].v[i][j][k]*rad_cyl;
  else             w[k] = cells[N].v[i][j][k];

  #if EOS == 2
  e  [k] = p[k]/(r[k]*gamm) + 0.5*(u[k]*u[k]) + potential(rad, azi, pol, planet);
  #endif

  __syncthreads();

  ppmlr(dt, 2, rad, azi, pol, bt, planet,
        r, p, u, v, w, e, xa0, dx0, dvol0);

  #if visc_flag == 1
  xa0[k] = cells[N].zc[k];
  if (ngeomz == 5) xa0[k] = rad*ccos(xa0[k]);
  device_viscosity_z(k, dt, rad_cyl, cells, xa0, r, v, w, u, dx0, alpha);
  #endif

  cells[N].r[i][j][k] = r[k];
  #if EOS == 0
  cells[N].p[i][j][k] = r[k]*get_cs2(rad_cyl);
  #elif EOS == 1
  cells[N].p[i][j][k] = get_cs2(rad_cyl)*cpow(r[k],gam)/gam;
  #else
  cells[N].p[i][j][k] = p[k];
  #endif
  cells[N].w[i][j][k] = u[k];
  cells[N].u[i][j][k] = v[k];
  if (ngeomz == 5) cells[N].v[i][j][k] = w[k]/rad_cyl;
  else             cells[N].v[i][j][k] = w[k];

  return;
}
#endif

//=========================================================================================

void bound_trans(GPU_plan *set)
{
  int iblk;
  for (int k=0; k<kmax; k++)
  {
    //iblk = set[0].iblk;
    //CudaSafeCall( cudaMemcpyAsync( &set[nDev-1].rgh[n_pad*k], &set[0].rings[iblk*k], n_pad*sizeof(hydr_ring), cudaMemcpyDeviceToDevice, set[n].stream) );
    for (int n=1; n<nDev; n++)
    {
      iblk = set[n].iblk;
      CudaSafeCall( cudaMemcpyAsync( &set[n-1].rgh[n_pad*k], &set[n].rings[iblk*k], n_pad*sizeof(hydr_ring), cudaMemcpyDeviceToDevice, set[n].stream) );
    }

    for (int n=0; n<nDev-1; n++)
    {
      iblk = set[n].iblk;
      CudaSafeCall( cudaMemcpyAsync( &set[n+1].lft[n_pad*k], &set[n].rings[iblk-n_pad+iblk*k], n_pad*sizeof(hydr_ring), cudaMemcpyDeviceToDevice, set[n].stream) );
    }
    //iblk = set[nDev-1].iblk;
    //CudaSafeCall( cudaMemcpyAsync( &set[0].lft[n_pad*k], &set[nDev-1].rings[iblk-1-n_pad+iblk*k], n_pad*sizeof(hydr_ring), cudaMemcpyDeviceToDevice, set[n].stream) );

    syncstreams(set);
  }

  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaStreamSynchronize(set[i].stream) );
  }
  return;
}

//=========================================================================================
void bundle_sweep2D(GPU_plan *set, sdp dt, body &planet)
{
  bound_trans(set);

  for(int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );

    sweepx<<< set[n].sx_grid , arrsize , 0 , set[n].stream >>>
          (set[n].rings, set[n].lft, set[n].rgh, dt, set[n].iblk, planet);
    CudaCheckError();

    sweepy<<< set[n].sy_grid , arrsize , 0 , set[n].stream >>>
          (set[n].rings, dt, set[n].iblk, planet);

    CudaCheckError();
  }

  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaStreamSynchronize(set[i].stream) );
  }

  return;
}

//=========================================================================================
#if ndim == 3

void bundle_sweep3D(GPU_plan *set, sdp dt, body &planet, sdp alpha)
{
  bound_trans(set);

  for(int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );

    set_boundy<<< set[n].b_grid , set[n].b_blk , 0 , set[n].stream >>>
              (set[n].cells, set[n].bac, set[n].frn, set[n].jstart, set[n].jend, set[n].iblk);
    CudaCheckError();

    sweepy<<< set[n].s_grid , set[n].s_blk , 0 , set[n].stream >>>
          (set[n].cells, dt, set[n].jblk*jdim, set[n].beta, planet, alpha);
    CudaCheckError();

    set_boundx<<< set[n].b_grid , set[n].b_blk , 0 , set[n].stream  >>>
              (set[n].cells, set[n].lft, set[n].rgh, set[n].istart, set[n].iend, 1, alpha);
    CudaCheckError();

    sweepx<<< set[n].s_grid , set[n].s_blk , 0 , set[n].stream >>>
          (set[n].cells, dt, set[n].jblk*jdim, set[n].beta, planet, alpha);
    CudaCheckError();

    set_boundz<<< set[n].b_grid , set[n].b_blk , 0 , set[n].stream >>>
              (set[n].cells, set[n].udr, set[n].top, set[n].kstart, set[n].kend, set[n].iblk*set[n].jblk);
    CudaCheckError();

    sweepz<<< set[n].s_grid , set[n].s_blk , 0 , set[n].stream >>>
          (set[n].cells, dt, set[n].jblk*jdim, set[n].beta, planet, alpha);
    CudaCheckError();

    #if kill_flag == 1
    kill_wave<<< set[n].s_grid , idim , 0 , set[n].stream >>>(set[n].cells, set[n].val, dt, xmin, xmax);
    CudaCheckError();
    #endif
  }

  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaStreamSynchronize(set[i].stream) );
  }
  return;
}

#endif
