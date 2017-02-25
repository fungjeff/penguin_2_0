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
#include "viscosity.cu"
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
__global__ void sweepx(hydr_ring *rings, hydr_ring *lft, hydr_ring *rgh, sdp dt, int iblk, body planet, sdp FrRot)
{
  int i, j0, j, lim;
  int i_ring, k_ring, loop;

  i = threadIdx.x;
  j0 = blockIdx.x;
  k_ring = blockIdx.y;
  loop = (iblk/realarr) + (bool)(iblk%realarr);

  sdp r_lt, p_lt, u_lt, v_lt, w_lt;

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
        r[i] = r_lt;
        p[i] = p_lt;
        u[i] = u_lt;
        v[i] = v_lt;
        w[i] = w_lt;
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

      if (l<loop-1 && i<n_pad)
      {
        r_lt = r[i+lim-2*n_pad];
        p_lt = p[i+lim-2*n_pad];
        u_lt = u[i+lim-2*n_pad];
        v_lt = v[i+lim-2*n_pad];
        w_lt = w[i+lim-2*n_pad];
      }
      __syncthreads();

      //if (j==99)
        //printf("%i, (%i, %i, %i), (%f, %f, %f):  (%e, %e, %e, %e, %e)\n", lim, i_ring, j, k_ring, rad, azi, pol, r[i]*cpow(rad,p_alpha)-1.0, p[i]/r[i], u[i], v[i]*cpow(rad,-0.5), w[i]);
      ///////////////////////////////////////////////////////

      ppmlr(rad, azi, pol, (*cur).res_v/rad_cyl, FrRot,
            r, p, u, v, w, e, xa0, dx0, dvol0, 
            lim, dt, 0, planet);

      #if visc_flag == 1
      device_viscosity_r(i, lim, dt, rad_cyl, xa0, r, u, v, w, dx0);
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

__global__ void sweepy(hydr_ring *rings, sdp dt, int iblk, body planet, sdp FrRot)
{
  int i, j0, j, jref, lim;
  int n, loop;

  i = threadIdx.x;
  n = blockIdx.x + iblk*blockIdx.y;
  loop = (jmax/realarr) + (bool)(jmax%realarr);


  __shared__ sdp r_rg[n_pad], p_rg[n_pad], u_rg[n_pad], v_rg[n_pad], w_rg[n_pad];
  sdp r_lt, p_lt, u_lt, v_lt, w_lt;

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
        r[i] = r_lt;
        p[i] = p_lt;
        u[i] = u_lt;
        v[i] = v_lt;
        w[i] = w_lt;
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

      if (l<loop-1 && i<n_pad)
      {
        r_lt = r[i+lim-2*n_pad];
        p_lt = p[i+lim-2*n_pad];
        u_lt = u[i+lim-2*n_pad];
        v_lt = v[i+lim-2*n_pad];
        w_lt = w[i+lim-2*n_pad];
      }

      __syncthreads();
      //if (n==99)
        //printf("%i, (%i, %i, %i), (%f, %f, %f):  (%f, %f, %f, %f, %f)\n", l, (int)blockIdx.x, jref, (int)blockIdx.y, rad, azi, pol, r[i], p[i], u[i], v[i], w[i]);
      ///////////////////////////////////////////////////////

      ppmlr(rad_cyl, azi, pol, (*cur).res_v/rad_cyl, FrRot,
            r, p, u, v, w, e, 
            xa0, dx0, dvol0,
            lim, dt, 1, planet);

      #if visc_flag == 1
      xa0[i] *= rad_cyl;
      device_viscosity_p(i, lim, dt, rad_cyl, xa0, r, w, u, v, dx0);
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

  return;
}

__global__ void rotate_rings(hydr_ring *rings, int iblk)
{
  int j;
  int n = blockIdx.x + iblk*blockIdx.y;

  j  = rings[n].rot_j;
  j += rings[n].inc_j;
  if (j<0) j += jmax;
  if (j>=jmax) j -= jmax;
  rings[n].rot_j  = j;

  //if (blockIdx.x==0) printf("%i %i %f\n", j, (*cur).inc_j, (*cur).rot_v*cpow(rad,0.5));
  return;
}

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
void bundle_sweep2D(GPU_plan *set, sdp dt, body &planet, sdp FrRot)
{
  bound_trans(set);

  for(int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );

    sweepx<<< set[n].sx_grid , arrsize , 0 , set[n].stream >>>
          (set[n].rings, set[n].lft, set[n].rgh, dt, set[n].iblk, planet, FrRot);
    CudaCheckError();

    sweepy<<< set[n].sy_grid , arrsize , 0 , set[n].stream >>>
          (set[n].rings, dt, set[n].iblk, planet, FrRot);
    CudaCheckError();

    #if FARGO_flag>0
    rotate_rings<<< set[n].sy_grid , 1 , 0 , set[n].stream >>> (set[n].rings, set[n].iblk);
    CudaCheckError();
    #endif
  }

  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaStreamSynchronize(set[i].stream) );
  }

  return;
}
